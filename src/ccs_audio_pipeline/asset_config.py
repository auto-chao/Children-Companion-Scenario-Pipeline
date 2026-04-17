from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
MODELS_ROOT = ARTIFACTS_ROOT / "models"
VENDOR_ROOT = PROJECT_ROOT / "vendor"

DEMUCS_MODEL_NAME = "htdemucs_ft"
DEMUCS_MODEL_DIR = MODELS_ROOT / "demucs" / DEMUCS_MODEL_NAME
DEMUCS_REMOTE_ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/"
DEMUCS_MODEL_FILES = (
    "f7e0c4bc-ba3fe64a.th",
    "d12395a8-e57c48e6.th",
    "92cfc3b6-ef3bcb9c.th",
    "04573f0d-f3cf25b2.th",
)
DEMUCS_BAG_FILE = f"{DEMUCS_MODEL_NAME}.yaml"

PYANNOTE_REPO_ID = "pyannote/speaker-diarization-community-1"
PYANNOTE_REVISION = "main"
PYANNOTE_DIR = MODELS_ROOT / "pyannote" / "speaker-diarization-community-1"

CHILD_REPO_ID = "audeering/wav2vec2-large-robust-24-ft-age-gender"
CHILD_REVISION = "main"
CHILD_MODEL_DIR = MODELS_ROOT / "audeering" / "wav2vec2-large-robust-24-ft-age-gender"

SEMANTIC_REPO_ID = "BAAI/bge-m3"
SEMANTIC_REVISION = "main"
SEMANTIC_MODEL_DIR = MODELS_ROOT / "baai" / "bge-m3"

CLEARVOICE_MODEL_REPO_ID = "alibabasglab/MossFormer2_SE_48K"
CLEARVOICE_MODEL_REVISION = "main"
CLEARVOICE_MODEL_DIR = MODELS_ROOT / "clearvoice" / "MossFormer2_SE_48K"

CLEARVOICE_SOURCE_DIR = VENDOR_ROOT / "ClearerVoice-Studio" / "clearvoice"
CLEARVOICE_SOURCE_ZIP_URL = "https://github.com/modelscope/ClearerVoice-Studio/archive/refs/heads/main.zip"


@dataclass(frozen=True)
class HfAssetSpec:
    name: str
    repo_id: str
    revision: str
    local_dir: Path
    token_required: bool = False
    allow_patterns: tuple[str, ...] | None = None


HF_ASSETS = [
    HfAssetSpec(
        name="pyannote speaker diarization",
        repo_id=PYANNOTE_REPO_ID,
        revision=PYANNOTE_REVISION,
        local_dir=PYANNOTE_DIR,
        token_required=True,
        allow_patterns=(
            "config.yaml",
            "README.md",
            "diarization.gif",
            "embedding/README.md",
            "embedding/pytorch_model.bin",
            "plda/README.md",
            "plda/plda.npz",
            "plda/xvec_transform.npz",
            "segmentation/pytorch_model.bin",
        ),
    ),
    HfAssetSpec(
        name="audeering child speech classifier",
        repo_id=CHILD_REPO_ID,
        revision=CHILD_REVISION,
        local_dir=CHILD_MODEL_DIR,
        allow_patterns=(
            "config.json",
            "preprocessor_config.json",
            "README.md",
            "vocab.json",
            "pytorch_model.bin",
        ),
    ),
    HfAssetSpec(
        name="bge semantic encoder",
        repo_id=SEMANTIC_REPO_ID,
        revision=SEMANTIC_REVISION,
        local_dir=SEMANTIC_MODEL_DIR,
        allow_patterns=(
            "1_Pooling/config.json",
            "README.md",
            "colbert_linear.pt",
            "config.json",
            "config_sentence_transformers.json",
            "modules.json",
            "pytorch_model.bin",
            "sentence_bert_config.json",
            "sentencepiece.bpe.model",
            "sparse_linear.pt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    ),
    HfAssetSpec(
        name="ClearerVoice MossFormer2_SE_48K weights",
        repo_id=CLEARVOICE_MODEL_REPO_ID,
        revision=CLEARVOICE_MODEL_REVISION,
        local_dir=CLEARVOICE_MODEL_DIR,
        allow_patterns=(
            "README.md",
            "last_best_checkpoint",
            "last_best_checkpoint.pt",
        ),
    ),
]


ASSET_MARKERS = {
    "pyannote speaker diarization": [
        PYANNOTE_DIR / "config.yaml",
        PYANNOTE_DIR / "embedding" / "pytorch_model.bin",
        PYANNOTE_DIR / "plda" / "plda.npz",
        PYANNOTE_DIR / "plda" / "xvec_transform.npz",
        PYANNOTE_DIR / "segmentation" / "pytorch_model.bin",
    ],
    "audeering child speech classifier": [
        CHILD_MODEL_DIR / "config.json",
        CHILD_MODEL_DIR / "pytorch_model.bin",
    ],
    "bge semantic encoder": [
        SEMANTIC_MODEL_DIR / "config.json",
        SEMANTIC_MODEL_DIR / "pytorch_model.bin",
        SEMANTIC_MODEL_DIR / "colbert_linear.pt",
        SEMANTIC_MODEL_DIR / "sparse_linear.pt",
    ],
    "Demucs htdemucs_ft weights": [
        DEMUCS_MODEL_DIR / DEMUCS_BAG_FILE,
        *(DEMUCS_MODEL_DIR / file_name for file_name in DEMUCS_MODEL_FILES),
    ],
    "ClearerVoice source": [
        CLEARVOICE_SOURCE_DIR / "clearvoice" / "__init__.py",
        CLEARVOICE_SOURCE_DIR / "clearvoice" / "network_wrapper.py",
        CLEARVOICE_SOURCE_DIR / "clearvoice" / "config" / "inference" / "MossFormer2_SE_48K.yaml",
    ],
    "ClearerVoice MossFormer2_SE_48K weights": [
        CLEARVOICE_MODEL_DIR / "last_best_checkpoint",
        CLEARVOICE_MODEL_DIR / "last_best_checkpoint.pt",
    ],
}


def missing_assets() -> dict[str, list[Path]]:
    missing: dict[str, list[Path]] = {}
    for name, markers in ASSET_MARKERS.items():
        missing_markers = [marker for marker in markers if not marker.exists()]
        if missing_markers:
            missing[name] = missing_markers
    return missing


def assets_ready() -> bool:
    return not missing_assets()


def format_missing_assets() -> str:
    missing = missing_assets()
    if not missing:
        return "All offline assets are present."

    lines = ["Missing offline assets:"]
    for name, markers in missing.items():
        lines.append(f"- {name}")
        for marker in markers:
            lines.append(f"  - {marker}")
    lines.append("Run `python scripts/bootstrap_assets.py --hf-token <your_token>` first.")
    return "\n".join(lines)

