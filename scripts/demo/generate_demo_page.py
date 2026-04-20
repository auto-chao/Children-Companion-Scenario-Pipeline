#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从含 turns + tts_audio 的 JSONL 生成静态 demo_page/index.html。

音频 URL 为相对 index.html 的路径；样本数据以 UTF-8 JSON 内嵌于 application/json 脚本块。

同一 ``manifest_line`` 在 JSONL 中出现多次时（重跑追加），只保留**最后一次**记录，
避免页面顶部仍展示无 ``tts_audio`` 的旧行。
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[3]


_ROOT = _repo_root()


def _rel_url_from_demo(repo_rel: str) -> str:
    """demo_page/index.html -> ../../outputs/..."""
    p = repo_rel.strip().replace("\\", "/")
    if not p.startswith("outputs/"):
        p = f"outputs/{p}" if not p.startswith("/") else p.lstrip("/")
    return "../../" + p


def _dedupe_samples_by_manifest_line(samples: list[dict]) -> list[dict]:
    """同一 manifest_line 多行时保留文件中最晚出现的一条（最新助手/TTS 结果）。"""
    by_ml: dict[int, dict] = {}
    no_ml: list[dict] = []
    for row in samples:
        ml = row.get("manifest_line")
        if isinstance(ml, int):
            by_ml[ml] = row
        else:
            no_ml.append(row)
    out = [by_ml[k] for k in sorted(by_ml.keys())]
    out.extend(no_ml)
    return out


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>儿童陪伴 Demo</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: { sans: ['"Noto Sans SC"', 'system-ui', 'sans-serif'] },
          colors: {
            surface: { DEFAULT: 'rgba(15, 23, 42, 0.72)', raised: 'rgba(30, 41, 59, 0.55)' },
          },
          boxShadow: {
            glow: '0 0 0 1px rgba(148, 163, 184, 0.08), 0 25px 50px -12px rgba(0, 0, 0, 0.45)',
          },
        },
      },
    };
  </script>
  <style>
    .audio-skin audio { height: 2.5rem; border-radius: 0.75rem; }
    .grain {
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    }
  </style>
</head>
<body class="min-h-screen text-slate-100 antialiased grain" style="background: linear-gradient(165deg, #0c1222 0%, #111827 40%, #0f172a 100%);">
  <div class="pointer-events-none fixed inset-0 overflow-hidden">
    <div class="absolute -left-40 top-0 h-96 w-96 rounded-full bg-cyan-500/10 blur-[100px]"></div>
    <div class="absolute -right-32 bottom-0 h-80 w-80 rounded-full bg-violet-600/10 blur-[90px]"></div>
  </div>
  <div class="relative mx-auto max-w-3xl px-4 py-10 sm:px-6 sm:py-14">
    <header class="mb-10 text-center sm:mb-12 sm:text-left">
      <p class="mb-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-cyan-400/90">Companion · Voice · TTS</p>
      <h1 class="bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-3xl font-bold tracking-tight text-transparent sm:text-4xl">
        儿童语音对照试听
      </h1>
      <p class="mx-auto mt-3 max-w-xl text-sm leading-relaxed text-slate-400 sm:mx-0">
        下方为每条样本的模型回复与音频；请用本地 HTTP 打开页面以正常加载 <code class="rounded bg-slate-800/80 px-1.5 py-0.5 font-mono text-[11px] text-cyan-300/90">outputs/</code> 资源。
      </p>
      <p class="mt-4 inline-flex flex-wrap items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-400 backdrop-blur-sm">
        <span class="text-slate-500">数据</span>
        <code class="max-w-[min(100%,28rem)] truncate font-mono text-[11px] text-emerald-300/90">__SOURCE_PATH__</code>
      </p>
    </header>
    <div id="app" class="space-y-8"></div>
  </div>
  <script type="application/json" id="payload-json">__JSON_PAYLOAD__</script>
  <script>
    function esc(s) {
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }
    function encodeRelPath(href) {
      if (!href) return href;
      var s = String(href).split(String.fromCharCode(92)).join('/');
      return s.split('/').map(function (seg) {
        if (seg === '' || seg === '.' || seg === '..') return seg;
        return encodeURIComponent(seg);
      }).join('/');
    }
    /** file:// 保持 ../../outputs；http(s) 下改为 /outputs/... 避免相对路径在子目录页面下解析异常 */
    function resolveMediaHref(href) {
      if (!href) return href;
      var s = String(href).split(String.fromCharCode(92)).join('/');
      if (location.protocol === 'file:') return s;
      while (s.indexOf('../') === 0) {
        s = s.slice(3);
      }
      if (s.indexOf('outputs/') === 0) {
        return '/' + s;
      }
      return s;
    }
    function audioTagForHref(href) {
      const src = esc(encodeRelPath(resolveMediaHref(href)));
      const lower = String(href).toLowerCase();
      let mime = '';
      if (lower.endsWith('.m4a')) mime = 'audio/mp4';
      else if (lower.endsWith('.wav')) mime = 'audio/wav';
      const base = 'class="h-10 w-full accent-cyan-500" controls preload="metadata"';
      if (mime) {
        return '<audio ' + base + '>' +
          '<source src="' + src + '" type="' + mime + '" /></audio>';
      }
      return '<audio ' + base + ' src="' + src + '"></audio>';
    }
    const SAMPLES = JSON.parse(document.getElementById('payload-json').textContent);
    const app = document.getElementById('app');
    if (!SAMPLES.length) {
      app.innerHTML = '<div class="rounded-2xl border border-dashed border-white/15 bg-white/[0.03] p-12 text-center text-slate-500">' +
        '<p class="text-sm">暂无样本，请先生成 JSONL 并重新运行 generate_demo_page.py</p></div>';
    }
    SAMPLES.forEach((sample, si) => {
      const card = document.createElement('article');
      card.className = 'rounded-3xl border border-white/[0.08] bg-surface shadow-glow backdrop-blur-xl';
      const ml = sample.manifest_line != null ? sample.manifest_line : (si + 1);
      const badge = sample.dialog_type || '对话';
      const badgeCls = badge.indexOf('多轮') >= 0
        ? 'border-violet-500/30 bg-violet-500/15 text-violet-200'
        : 'border-emerald-500/30 bg-emerald-500/15 text-emerald-200';
      const ref = sample.recording_dialogue_ref || '';
      const refBlock = ref
        ? '<details class="group mb-5 rounded-2xl border border-white/10 bg-black/20 open:bg-black/25">' +
          '<summary class="cursor-pointer list-none px-4 py-3 text-sm font-medium text-slate-300 marker:content-none ' +
          'after:ml-2 after:inline-block after:text-slate-500 after:transition group-open:after:rotate-90 after:content-[\'›\']">' +
          '<span class="text-cyan-400/90">录音对话参考</span>' +
          '<span class="ml-2 text-xs font-normal text-slate-500">（manifest · 点击展开）</span></summary>' +
          '<div class="border-t border-white/5 px-4 py-3 text-xs leading-relaxed text-slate-400 whitespace-pre-wrap">' +
          esc(ref) + '</div></details>'
        : '';
      card.innerHTML =
        '<div class="flex flex-col gap-3 border-b border-white/5 px-5 pb-4 pt-5 sm:flex-row sm:items-center sm:justify-between">' +
        '<div class="flex items-baseline gap-3">' +
        '<span class="text-3xl font-bold tabular-nums text-white/90">#' + esc(String(ml)) + '</span>' +
        '<span class="text-sm text-slate-500">样本</span></div>' +
        '<span class="inline-flex w-fit items-center rounded-full border px-3 py-1 text-xs font-medium ' + badgeCls + '">' +
        esc(badge) + '</span></div>' +
        '<div class="px-5 pb-5">' + refBlock + '<div class="space-y-5 turns"></div></div>';
      const turnsEl = card.querySelector('.turns');
      (sample.turns || []).forEach((turn, ti) => {
        const turnIdx = turn.turn_index != null ? turn.turn_index : (ti + 1);
        const bubble = document.createElement('div');
        bubble.className = 'relative overflow-hidden rounded-2xl border border-white/[0.06] bg-surface-raised';
        const pt = turn.plain_text || '';
        const sem = turn.semantic_content || '';
        const ae = turn.acoustic_emotion || '';
        const childHref = turn.child_audio_href || '';
        const ttsHref = turn.tts_audio_href || '';
        const ttsErr = turn.tts_error || '';
        let childBlock = '<p class="py-2 text-center text-xs text-slate-500">无儿童音频路径</p>';
        if (childHref) {
          childBlock = '<div class="audio-skin">' + audioTagForHref(childHref) + '</div>';
        }
        let ttsBlock = '<p class="py-2 text-center text-xs text-amber-200/70">' +
          (ttsErr ? ('TTS：' + esc(ttsErr)) : '尚未生成 TTS') + '</p>';
        if (ttsHref) {
          ttsBlock = '<div class="audio-skin">' + audioTagForHref(ttsHref) + '</div>';
        }
        const chips = [];
        if (sem) {
          chips.push('<span class="inline-flex max-w-full rounded-lg bg-cyan-950/50 px-2 py-1 text-[11px] leading-snug text-cyan-100/90 ring-1 ring-cyan-500/20">' +
            '<span class="shrink-0 text-cyan-500/80">语义</span><span class="ml-1.5 text-slate-300">' + esc(sem) + '</span></span>');
        }
        if (ae) {
          chips.push('<span class="inline-flex max-w-full rounded-lg bg-violet-950/40 px-2 py-1 text-[11px] leading-snug text-violet-100/90 ring-1 ring-violet-500/20">' +
            '<span class="shrink-0 text-violet-400/80">声学</span><span class="ml-1.5 text-slate-300">' + esc(ae) + '</span></span>');
        }
        const chipsHtml = chips.length
          ? '<div class="mb-3 flex flex-col gap-2 sm:flex-row sm:flex-wrap">' + chips.join('') + '</div>'
          : '';
        bubble.innerHTML =
          '<div class="absolute left-0 top-0 h-full w-1 bg-gradient-to-b from-cyan-400/80 to-violet-500/60"></div>' +
          '<div class="pl-5 pr-4 pb-4 pt-4 sm:pl-6">' +
          '<div class="mb-3 flex items-center gap-2">' +
          '<span class="flex h-7 w-7 items-center justify-center rounded-lg bg-white/10 text-xs font-semibold text-white">' +
          esc(String(turnIdx)) + '</span>' +
          '<span class="text-xs font-medium uppercase tracking-wider text-slate-500">Turn</span></div>' +
          chipsHtml +
          '<blockquote class="mb-5 border-l-2 border-cyan-400/40 pl-4 text-[15px] font-medium leading-relaxed text-slate-100">' +
          esc(pt) + '</blockquote>' +
          '<div class="grid gap-4 sm:grid-cols-2">' +
          '<div class="rounded-xl bg-black/25 p-3 ring-1 ring-white/5">' +
          '<div class="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">' +
          '<span class="h-1.5 w-1.5 rounded-full bg-cyan-400"></span>儿童原声</div>' + childBlock + '</div>' +
          '<div class="rounded-xl bg-black/25 p-3 ring-1 ring-white/5">' +
          '<div class="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">' +
          '<span class="h-1.5 w-1.5 rounded-full bg-violet-400"></span>助手 TTS</div>' + ttsBlock + '</div>' +
          '</div></div>';
        turnsEl.appendChild(bubble);
      });
      app.appendChild(card);
    });
  </script>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description="生成 demo_page/index.html")
    p.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "outputs" / "assistant_responses_with_tts.jsonl",
        help="富 JSONL（含 turns、tts_audio 可选）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "demo_page",
        help="输出目录（写入 index.html）",
    )
    p.add_argument(
        "--fallback-input",
        type=Path,
        default=_ROOT / "outputs" / "assistant_responses_multiturn.jsonl",
        help="若 --input 不存在则尝试此文件",
    )
    p.add_argument(
        "--manifest-lines",
        type=str,
        default="",
        help="只展示指定 manifest_line（逗号分隔，如 2）；留空则展示全部行",
    )
    args = p.parse_args()

    inp = args.input
    if not inp.is_file():
        inp = args.fallback_input
    if not inp.is_file():
        print(f"输入不存在: {args.input} / {args.fallback_input}", file=sys.stderr)
        return 1

    try:
        src_disp = str(inp.relative_to(_ROOT))
    except ValueError:
        src_disp = str(inp)

    samples: list[dict] = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    samples = _dedupe_samples_by_manifest_line(samples)

    keep_lines: set[int] | None = None
    if args.manifest_lines.strip():
        keep_lines = set()
        for part in args.manifest_lines.split(","):
            part = part.strip()
            if part.isdigit():
                keep_lines.add(int(part))

    enriched: list[dict] = []
    for row in samples:
        if keep_lines is not None:
            ml = row.get("manifest_line")
            if isinstance(ml, int) and ml not in keep_lines:
                continue
            if isinstance(ml, str) and ml.isdigit() and int(ml) not in keep_lines:
                continue
        row = dict(row)
        turns = row.get("turns")
        if not isinstance(turns, list):
            enriched.append(row)
            continue
        new_turns = []
        for t in turns:
            if not isinstance(t, dict):
                new_turns.append(t)
                continue
            t = dict(t)
            a = t.get("audio")
            if isinstance(a, str) and a:
                t["child_audio_href"] = _rel_url_from_demo(f"child_dataset/{a}")
            ta = t.get("tts_audio")
            if isinstance(ta, str) and ta:
                t["tts_audio_href"] = _rel_url_from_demo(ta)
            new_turns.append(t)
        row["turns"] = new_turns
        row["dialog_type"] = "多轮对话" if len(new_turns) > 1 else "单轮对话"
        enriched.append(row)

    payload_json = json.dumps(enriched, ensure_ascii=False)
    # 避免 HTML 解析器把 JSON 里的 </ 当成 </script> 结束标签
    payload_json = payload_json.replace("</", "<\\/")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "index.html"

    page = PAGE_TEMPLATE.replace("__SOURCE_PATH__", html.escape(src_disp)).replace(
        "__JSON_PAYLOAD__", payload_json
    )

    out_html.write_text(page, encoding="utf-8")
    print(f"已写入 {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
