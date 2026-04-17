from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

from ccs_audio_pipeline.asset_config import (
    CLEARVOICE_SOURCE_DIR,
    CLEARVOICE_SOURCE_ZIP_URL,
    DEMUCS_BAG_FILE,
    DEMUCS_MODEL_DIR,
    DEMUCS_MODEL_FILES,
    DEMUCS_REMOTE_ROOT_URL,
    HF_ASSETS,
    assets_ready,
    format_missing_assets,
    missing_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all offline assets required by the child speech pipeline."
    )
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing assets.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify whether all offline assets are already present.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum concurrent file downloads per Hugging Face asset.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="How many times to retry a Hugging Face asset download before failing.",
    )
    args = parser.parse_args()

    if args.check_only:
        if assets_ready():
            print("All offline assets are present.")
            return
        print(format_missing_assets(), file=sys.stderr)
        raise SystemExit(1)

    if not args.hf_token and any(
        spec.token_required and spec.name in missing_assets() for spec in HF_ASSETS
    ):
        raise RuntimeError(
            "HF token is required to download gated assets. Set HF_TOKEN or pass --hf-token."
        )

    for spec in HF_ASSETS:
        download_hf_asset(
            spec,
            token=args.hf_token,
            force=args.force,
            max_workers=args.max_workers,
            retries=args.retries,
        )

    download_demucs_model(force=args.force, retries=args.retries)
    download_source_archive(
        label="ClearerVoice source",
        url=CLEARVOICE_SOURCE_ZIP_URL,
        destination_dir=CLEARVOICE_SOURCE_DIR,
        force=args.force,
        retries=args.retries,
    )

    if assets_ready():
        print("Offline assets bootstrap completed successfully.")
        return

    print(format_missing_assets(), file=sys.stderr)
    raise SystemExit(1)


def download_hf_asset(spec, token: str, force: bool, max_workers: int, retries: int) -> None:
    asset_missing = spec.name in missing_assets()
    if not force and not asset_missing:
        print(f"[skip] {spec.name}: {spec.local_dir}")
        return

    if force and spec.local_dir.exists():
        shutil.rmtree(spec.local_dir)
    spec.local_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[download] {spec.name}: {spec.repo_id} -> {spec.local_dir}")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                local_dir=str(spec.local_dir),
                token=token if spec.token_required else (token or None),
                allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
                max_workers=max_workers,
            )
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(
                f"[retry {attempt}/{retries}] {spec.name} failed with "
                f"{exc.__class__.__name__}: {exc}",
                file=sys.stderr,
            )
            if attempt >= retries:
                raise
            time.sleep(min(15 * attempt, 60))

    assert last_error is not None
    raise last_error


def download_demucs_model(force: bool, retries: int) -> None:
    asset_missing = "Demucs htdemucs_ft weights" in missing_assets()
    if not force and not asset_missing:
        print(f"[skip] Demucs htdemucs_ft weights: {DEMUCS_MODEL_DIR}")
        return

    if force and DEMUCS_MODEL_DIR.exists():
        shutil.rmtree(DEMUCS_MODEL_DIR)
    DEMUCS_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    bag_url = f"https://raw.githubusercontent.com/adefossez/demucs/main/demucs/remote/{DEMUCS_BAG_FILE}"
    download_url(bag_url, DEMUCS_MODEL_DIR / DEMUCS_BAG_FILE, retries=retries, label=DEMUCS_BAG_FILE)
    for file_name in DEMUCS_MODEL_FILES:
        download_url(
            f"{DEMUCS_REMOTE_ROOT_URL}{file_name}",
            DEMUCS_MODEL_DIR / file_name,
            retries=retries,
            label=file_name,
        )


def download_source_archive(
    label: str,
    url: str,
    destination_dir: Path,
    force: bool,
    retries: int,
) -> None:
    asset_missing = label in missing_assets()
    if not force and not asset_missing:
        print(f"[skip] {label}: {destination_dir}")
        return

    if force and destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[download] {label} -> {destination_dir}")
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        zip_path = tmpdir / "source.zip"
        download_url(url, zip_path, retries=retries, label=label)

        extract_dir = tmpdir / "extract"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        extracted_roots = [path for path in extract_dir.iterdir() if path.is_dir()]
        if len(extracted_roots) != 1:
            raise RuntimeError(f"Unexpected archive structure for {label}.")

        shutil.copytree(extracted_roots[0], destination_dir, dirs_exist_ok=True)


def download_url(url: str, destination: Path, retries: int, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url) as response, destination.open("wb") as f:
                shutil.copyfileobj(response, f)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(
                f"[retry {attempt}/{retries}] {label} failed with "
                f"{exc.__class__.__name__}: {exc}",
                file=sys.stderr,
            )
            if attempt >= retries:
                raise
            time.sleep(min(15 * attempt, 60))

    assert last_error is not None
    raise last_error


if __name__ == "__main__":
    main()

