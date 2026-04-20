# 儿童陪伴场景 · 语音数据集与对话 Demo

从原始亲子对话音频中抽取**儿童说话片段**，构建 **manifest**，再经第三方多模态 API 生成**陪伴式回复**，并用 **CosyVoice** 合成语音，最后在浏览器中查看 **静态 Demo 页**。  
主要产物在运行后生成于 `outputs/`（clone 后可能为空，属正常）。

## 你需要准备什么

- **操作系统**：Linux / macOS / Windows（脚本示例以 **Git Bash** 为准）。
- **Python**：3.10+，推荐使用 **conda** 环境（下文以 `ccs` 为例）。
- **ffmpeg**：系统可执行文件在 `PATH` 中。
- **NVIDIA GPU**：强烈推荐（数据集与 TTS 均会快很多）；CPU 也可跑，见下文 TTS 说明。
- **网络**：首次下载模型权重、克隆 CosyVoice、调用助手 API 时需要。

## 安装

```bash
conda activate ccs
pip install -r constraints.txt
pip install -e .
```

**CUDA 版 PyTorch**（与 `constraints.txt` 中版本一致；新显卡请按 [PyTorch 官网](https://pytorch.org/) 选择对应 cu 版本）示例：

```bash
pip install --upgrade "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
```

使用 **pyannote** 等模型前，请在 Hugging Face 网页上接受对应模型的使用条款。

## 首次下载离线资产

```bash
conda activate ccs
export HF_TOKEN=你的_huggingface_token
python scripts/bootstrap_assets.py --hf-token "$HF_TOKEN"
```

检查：

```bash
python scripts/bootstrap_assets.py --check-only
```

## 一键跑全流程

将示例音频放在 `data/audio/`（默认使用其中的 `*.m4a`）。设置**第三方 Gemini 兼容代理**的密钥（由你的代理服务商提供，不是 Google AI Studio 官方 key）。**模块 1 数据集转写与模块 2 助手**均使用该密钥（见 `src/ccs_audio_pipeline/asr_gemini_proxy.py`）。

```bash
conda activate ccs
export GEMINI_PROXY_API_KEY=你的代理密钥
export HF_TOKEN=你的_huggingface_token   # 若尚未 bootstrap 或需首次部署 CosyVoice
export ASSISTANT_WORKERS=4               # 助手步骤并行 worker（默认 4，可按配额调小）
bash main.sh
```

`main.sh` 会依次：检查离线资产 → 构建儿童数据集 → 生成助手回复 →（若尚无 CosyVoice 虚拟环境则）运行 `scripts/deploy_cosyvoice.py` → TTS → 生成 `demo_page/index.html`。

**仅构建数据集、不调用助手 API**时，可执行：

```bash
bash build_child_dataset.sh
```

### 模块 1：用工具提取片段、ASR 调 API 与 CPU 占用

- **处理顺序**（单文件内）：`load_audio_mono` → `DialogueFrontend.build_foreground_dialogue_view`（Demucs `htdemucs_ft` + ClearVoice `MossFormer2_SE_48K`）→ `extract_child_query_turns`（pyannote）→ 对**读入后的原始 mono 波形**按轮切片做 `ChildVoiceDetector` → 通过阈值的片段经 **`GeminiProxyAsr`** 得到转写与句向量（`SentenceTransformer` / `BAAI/bge-m3`）→ `build_dialogs` 聚链 → `cut_audio`（ffmpeg）与 `write_manifest`（家长间隙再次 `GeminiProxyAsr`）。细节见 [`pipeline.py`](src/ccs_audio_pipeline/pipeline.py)。
- **ASR（调 API）**：儿童片段与家长间隙的转写**不跑本地 ASR 模型**，统一通过 **Gemini 兼容 HTTP** `generateContent`（[`GeminiProxyAsr`](src/ccs_audio_pipeline/asr_gemini_proxy.py)），默认模型名为 `gemini-3-flash-preview`（可用 **`GEMINI_ASR_MODEL`** 覆盖）；需 **`GEMINI_PROXY_API_KEY`**（或 `GEMINI_API_KEY`）；可选 **`GEMINI_PROXY_BASE`**、**`GEMINI_ASR_PROMPT`**。单独跑 `bash build_child_dataset.sh` 前也需设置上述密钥。
- **减轻 CPU 占满**：模块 1 仍有 Demucs、pyannote、儿童判定、BGE 等本地计算。可调低 `build_child_dataset.sh` 中的 **`--num-threads`**（如 `4` 或 `2`），并令 OpenMP/BLAS 与之一致，例如 `export OMP_NUM_THREADS=4` 与 `export MKL_NUM_THREADS=4`，避免与 PyTorch 线程叠乘把机器打满。

## 输出在哪里

| 路径 | 说明 |
|------|------|
| `outputs/child_dataset/manifest.jsonl` | 多轮对话样本；除儿童片段 ASR（`user`/`user_*`）外，含相邻两轮之间（及片尾）**家长说话 ASR**（`assistant`/`assistant_*`），可选片头 `recording_prefix_adult` |
| `outputs/child_dataset/audios/*.m4a` | 儿童片段音频 |
| `outputs/assistant_responses_multiturn.jsonl` | 助手回复（含 `plain_text`、`semantic_content`、`acoustic_emotion`；多轮时历史轮以文本摘要注入；`recording_dialogue_ref` 为按轮次截断的亲子转录参考文本，多轮时每轮可含 `recording_dialogue_ref`） |
| `outputs/tts_generated/*.wav` | 合成语音 |
| `outputs/assistant_responses_with_tts.jsonl` | 带 `tts_audio` 路径的汇总 |
| `demo_page/index.html` | 浏览器对照收听；**推荐**用 `bash demo_page/local_http.sh start` 起本地 HTTP 后打开提示的 URL（`file://` 直接打开可能无法播放音频）。`local_http.sh` 会自动探测 `PYTHON` / `python3` / `py`（含真实 `sys.executable`）/ `python`（跳过 Windows Store 占位），必要时用 `where.exe` 与 cmd 侧 PATH 对齐；仍失败可设置 `PYTHON` |

## TTS：GPU 与 CPU

- **默认**：`run_tts.sh` / `main.sh` 中的 TTS 在可用时使用 **GPU**（不设置 `CUDA_VISIBLE_DEVICES`）。
- **RTX 50 系列（sm_120）建议**：先在 CosyVoice venv 里升级 GPU 版 torch（脚本已内置）：

```bash
python scripts/deploy_cosyvoice.py --skip-clone --skip-download
```

- **强制 CPU**（无 NVIDIA 驱动、或需避免 GPU 时）：

```bash
COSYVOICE_FORCE_CPU=1 bash main.sh
# 或
COSYVOICE_FORCE_CPU=1 bash run_tts.sh
```

PowerShell 写法：

```powershell
$env:COSYVOICE_FORCE_CPU="1"; bash .\run_tts.sh
```

CosyVoice 使用独立虚拟环境 `artifacts/cosyvoice/.venv`（由 `deploy_cosyvoice.py` 创建）。若整机拷贝仓库到新机器，建议在新环境中重新执行 `python scripts/deploy_cosyvoice.py` 以重建 venv。

**合成逻辑**：CosyVoice 每轮仅朗读 JSON 中的 **`plain_text`**（zero-shot，需参考音频与 prompt；见 `run_tts.sh` / `batch_cosyvoice_tts.py`）。

## 流水线概览

下图给出端到端技术路线，与 [`run_pipeline`](src/ccs_audio_pipeline/pipeline.py) / 助手脚本一致：**模块 1** 对每段输入 `*.m4a` 做读入 → `DialogueFrontend`（Demucs 人声 + ClearVoice 增强）→ pyannote 轮次 → 在**原始波形**上儿童概率过滤 → **`GeminiProxyAsr`** 转写儿童片段与家长间隙 → BGE 向量与 NetworkX 聚链 → ffmpeg 导出切片并写 **manifest**；**模块 2** 中豆包听评仅为**离线** Prompt 迭代（**未**接入 `main.sh`），定稿后的系统指令由 `generate_assistant_responses.py` 批量调用 **`generateContent`**；**模块 3** 朗读 `plain_text` 合成语音并生成 Demo。全量跑通使用上文「一键跑全流程」中的 `bash main.sh`。

### 模块 2：多模态 API 如何构建 response（请求里放了什么）

助手步骤由 [`scripts/assistant/generate_assistant_responses.py`](scripts/assistant/generate_assistant_responses.py) 调用 **Gemini 兼容** `generateContent`，在请求体 `contents` 中拼装多轮对话；**当前轮以儿童片段音频为主输入**，manifest 里的 `user` 等字段主要用于归档与生成**按轮截断的亲子转写参考**（与代码中 `recording_dialogue_ref` / `_dialogue_ref_text_for_turn` 一致）。**历史轮（下表「历史多轮」、流程图中 `i_hist`）**在同一样本内由脚本把**上一轮及更早轮次**的 API 返回写入 `contents`，再与本轮请求拼接；首轮 `history_turns` 为空，仅有系统指令、可选转写参考与本轮音频。

| 信息类型 | 说明 |
|----------|------|
| **系统指令（任务与 JSON 格式）** | 人设、安全与互动要求，以及模型必须输出的字段 `semantic_content`、`acoustic_emotion`、`plain_text`（见脚本内 `SYSTEM_INSTRUCTION` / `_full_task_text()`）。 |
| **亲子转写参考（可选）** | 片头家长话（若有）+ 截至当前轮的「孩子 / 家长(录音)」ASR 文本，冠以「历史对话&&当前轮输入语音的转录文本（仅供参考，本轮输入以语音为准）」；**多轮时不含未来轮与当前轮之后家长话**。 |
| **历史多轮（仅 `--mode multi`）** | 已完成轮次按 **user → model** 交替写入：user 侧为上一轮模型已产出的语义/声学摘要（`【历史轮·孩子语音理解（文本摘要）】`），model 侧为该轮 `plain_text`。 |
| **当前轮儿童音频** | 本轮 `*.m4a` 经 Base64 放入 `inline_data`（默认 MIME `audio/mp4`），与上述文本在同一条 user `parts` 中一并提交。 |
| **生成配置** | `generation_config.response_mime_type = application/json`（`_build_payload(..., json_mode=True)`）；若代理不支持可回退为非 JSON 模式再解析。可选 **`--with-google-search`** 时在请求体中增加 `tools: [{google_search: {}}]`。 |

**API 返回**：模型文本经解析、校验后写入 `outputs/assistant_responses_multiturn.jsonl`（或单轮输出文件名），每轮包含 `plain_text`、`semantic_content`、`acoustic_emotion` 等。

```mermaid
flowchart TB
  subgraph mod_seg [模块1：语音切分与数据集 · run_pipeline]
    s1["亲子对话音频 · data/audio/*.m4a"] --> m_load["读入 mono 波形<br/>load_audio_mono · librosa 优先 失败则 ffmpeg PCM"]
    m_load --> m_fg["前景 DialogueFrontend<br/>Demucs htdemucs_ft vocals → ClearVoice MossFormer2_SE_48K<br/>见 dialogue_frontend.py 与 asset_config 中 CLEARVOICE_*"]
    m_fg --> m_pya["说话人轮次 · extract_child_query_turns<br/>pyannote.audio Pipeline · speaker-diarization-community-1"]
    m_pya --> m_child["儿童轮次筛选 ChildVoiceDetector<br/>audeering/wav2vec2-large-robust-24-ft-age-gender · p_child ≥ child_threshold<br/>切片来自读入后的原始波形 非增强轨"]
    m_child --> m_asr["儿童片段转写 GeminiProxyAsr<br/>generateContent · 默认 gemini-3-flash-preview"]
    m_asr --> m_bge["句向量 · SentenceTransformer<br/>BAAI/bge-m3 本地权重"]
    m_bge --> m_link["会话链 build_dialogs / build_link_graph<br/>sklearn cosine_similarity + 间隔与说话人加权 · NetworkX 连通分量 · max_turns 切块"]
    m_link --> s2["manifest.jsonl + audios/*.m4a<br/>cut_audio ffmpeg AAC · write_manifest 家长间隙 ASR 同 GeminiProxyAsr"]
  end

  subgraph mod_resp [模块2：陪伴式回复构建]
    subgraph loop_doubao [离线研究：豆包参与 Prompt 迭代 · 未接入 main.sh]
      r1[设计初始 Prompt] --> r2[多模态生成回复]
      r2 --> r3[试听儿童与助手语音]
      r3 --> r4[豆包评判与优化建议]
      r4 --> r5{满意?}
      r5 -->|否| r6[据建议修订 Prompt]
      r6 --> r2
      r5 -->|是| r7[Prompt 定稿]
    end
    subgraph api_req [批量推理 · generate_assistant_responses.py]
      i_audio["当前轮儿童片段 inline_data<br/>MIME 默认 audio/mp4"]
      i_sys["SYSTEM_INSTRUCTION + _full_task_text<br/>JSON 三字段约束"]
      i_rec["亲子转写参考 可选<br/>_RECORDING_CTX_HEADER + _dialogue_ref_text_for_turn"]
      i_hist["多轮 history_turns · 仅 multi<br/>_history_user_text + 上轮 plain_text"]
      r8["POST .../v1beta/models/{model}:generateContent<br/>与 ASR 同源代理 GEMINI_PROXY_BASE"]
    end
    r7 --> i_sys
    i_sys --> r8
    s2 --> i_audio
    s2 --> i_rec
    s2 --> i_hist
    i_audio --> r8
    i_rec --> r8
    i_hist --> r8
  end

  subgraph mod_tts [模块3：语音合成]
    t1[朗读 plain_text] --> t2[合成语音]
  end

  r8 --> t1
  t2 --> d1[静态 Demo 页]

```

实现与默认参数见 [`src/ccs_audio_pipeline/pipeline.py`](src/ccs_audio_pipeline/pipeline.py)、[`dialogue_frontend.py`](src/ccs_audio_pipeline/dialogue_frontend.py)、[`asset_config.py`](src/ccs_audio_pipeline/asset_config.py)；助手请求体拼装见 [`scripts/assistant/generate_assistant_responses.py`](scripts/assistant/generate_assistant_responses.py)。上文「模块 2」表格与各 `i_*` 节点对应。

## 第三方模型与许可

本仓库代码以 **Apache-2.0** 发布（见 [`LICENSE`](LICENSE)）。依赖的 **Demucs、pyannote、CosyVoice、Sentence-Transformers、BGE** 等第三方权重各有原始许可证与条款；用于研究或产品前请自行阅读并遵守。生成内容不代表任何机构观点。
