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
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-slate-950 text-slate-100">
  <div class="max-w-3xl mx-auto px-4 py-8">
    <header class="mb-8">
      <h1 class="text-2xl font-bold text-white tracking-tight">儿童语音 · 助手回复 · TTS</h1>
      <p class="text-slate-400 text-sm mt-2">数据来源：<code class="text-cyan-400">__SOURCE_PATH__</code></p>
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
      if (mime) {
        return '<audio class="w-full" controls preload="metadata">' +
          '<source src="' + src + '" type="' + mime + '" /></audio>';
      }
      return '<audio class="w-full" controls preload="metadata" src="' + src + '"></audio>';
    }
    const SAMPLES = JSON.parse(document.getElementById('payload-json').textContent);
    const app = document.getElementById('app');
    SAMPLES.forEach((sample, si) => {
      const card = document.createElement('article');
      card.className = 'rounded-2xl border border-slate-800 bg-slate-900/80 p-5 shadow-xl';
      const ml = sample.manifest_line != null ? sample.manifest_line : (si + 1);
      const badge = sample.dialog_type || '对话';
      const badgeCls = badge.indexOf('多轮') >= 0
        ? 'bg-violet-500/20 text-violet-300'
        : 'bg-emerald-500/20 text-emerald-300';
      const ref = sample.recording_dialogue_ref || '';
      const refBlock = ref
        ? '<div class="mb-4 rounded-lg border border-slate-700 bg-slate-900/60 p-3 text-xs text-slate-400 whitespace-pre-wrap">' +
          '<div class="text-slate-500 mb-1">录音对话参考（manifest）</div>' +
          esc(ref) +
          '</div>'
        : '';
      card.innerHTML =
        '<div class="flex items-center justify-between gap-3 mb-4">' +
        '<span class="text-lg font-semibold text-white">样本 #' + esc(String(ml)) + '</span>' +
        '<span class="text-xs font-medium px-2 py-1 rounded-full ' + badgeCls + '">' + esc(badge) + '</span>' +
        '</div>' +
        refBlock +
        '<div class="space-y-5 turns"></div>';
      const turnsEl = card.querySelector('.turns');
      (sample.turns || []).forEach((turn, ti) => {
        const turnIdx = turn.turn_index != null ? turn.turn_index : (ti + 1);
        const bubble = document.createElement('div');
        bubble.className = 'rounded-xl border border-slate-700/80 bg-slate-800/50 p-4';
        const pt = turn.plain_text || '';
        const sem = turn.semantic_content || '';
        const ae = turn.acoustic_emotion || '';
        const childHref = turn.child_audio_href || '';
        const ttsHref = turn.tts_audio_href || '';
        const ttsErr = turn.tts_error || '';
        let childBlock = '<p class="text-slate-500 text-sm">无音频路径</p>';
        if (childHref) {
          childBlock = audioTagForHref(childHref);
        }
        let ttsBlock = '<p class="text-amber-200/80 text-sm">' +
          (ttsErr ? ('TTS: ' + esc(ttsErr)) : '未生成 TTS') + '</p>';
        if (ttsHref) {
          ttsBlock = audioTagForHref(ttsHref);
        }
        const metaBits = [];
        if (sem) metaBits.push('语义：' + esc(sem));
        if (ae) metaBits.push('情绪：' + esc(ae));
        const metaHtml = metaBits.length
          ? '<div class="text-xs text-slate-500 mb-2 space-y-1">' + metaBits.join('<br/>') + '</div>'
          : '';
        bubble.innerHTML =
          '<div class="text-xs text-slate-500 mb-2">轮次 ' + esc(String(turnIdx)) + '</div>' +
          metaHtml +
          '<div class="prose prose-invert prose-sm max-w-none mb-4 whitespace-pre-wrap text-slate-200">' +
          esc(pt) + '</div>' +
          '<div class="grid sm:grid-cols-2 gap-4">' +
          '<div><div class="text-xs font-medium text-slate-400 mb-1">儿童输入音频</div>' + childBlock + '</div>' +
          '<div><div class="text-xs font-medium text-slate-400 mb-1">助手 TTS</div>' + ttsBlock + '</div>' +
          '</div>';
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
