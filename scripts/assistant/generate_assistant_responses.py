#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
儿童陪伴 AI 助手回复批量生成（Gemini 兼容 HTTP + JSONL）

与 ``api_call/api_call_final.py`` 一致：请求体中带 **inline_data 音频**（m4a），由多模态模型
理解儿童语音并生成回复。manifest 中的 ``user`` 等为归档/对齐；**当前轮**仍以音频为主输入。

多轮模式（``--mode multi``）：**历史轮**为文本——每轮 user 侧为上一轮模型已产出的 ``semantic_content`` 与
``acoustic_emotion``，model 侧为该轮 ``plain_text``；**当前轮**为 **录音对话参考（若有）+ 任务说明 + 本轮儿童音频**。
另将 manifest 中 **亲子对话 ASR**（含大人 ``assistant`` 槽位与可选 ``recording_prefix_adult``）作为只读语境块注入。
模型 JSON 含 ``semantic_content``、``acoustic_emotion``、``plain_text``。

调用经 ``local_api_logger.wrap_requests_call`` 记录到 ``api_call/api_logs/``。

环境变量
--------
GEMINI_PROXY_API_KEY（推荐）
    第三方代理提供的 API Key（勿提交到仓库）。
GEMINI_PROXY_BASE（可选）
    代理根地址，默认 ``http://azpro.xunxkj.cn``。

若未设置 ``GEMINI_PROXY_API_KEY``，会回退读取 ``GEMINI_API_KEY``（仅作别名，仍表示代理密钥）。

用法
----
    python scripts/assistant/generate_assistant_responses.py

    # 默认 --mode multi：读 manifest.jsonl、写 assistant_responses_multiturn.jsonl
    # 单轮：加 --mode single 且必须指定 --input（单轮 manifest JSONL），写 assistant_responses_single_turn.jsonl

可选参数见 --help。无参数时与 ``--mode multi`` 相同。

每行 JSON 为 **一个 manifest 样本**，顶层含 ``manifest_line``、``model``、``input_mode``、``turns``（每轮一条）、
``line_error``；单轮 ``len(turns)==1``，多轮为多元素数组。
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# 仓库根目录


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[2]


_ROOT = _repo_root()
_API_CALL_ROOT = _ROOT / "api_call"
if str(_API_CALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_CALL_ROOT))

# 日志目录与 api_call_final 一致：api_call/api_logs
import local_api_logger.logger as _lm

_lm.set_log_dir(str(_API_CALL_ROOT / "api_logs"))
import local_api_logger.tracker as _tr

_tr._default_tracker.logger = _lm._default_logger
from local_api_logger import wrap_requests_call

_DEFAULT_MULTI_IN = _ROOT / "outputs" / "child_dataset" / "manifest.jsonl"
_DEFAULT_SINGLE_OUT = _ROOT / "outputs" / "assistant_responses_single_turn.jsonl"
_DEFAULT_MULTI_OUT = _ROOT / "outputs" / "assistant_responses_multiturn.jsonl"
# gemini-3.1-pro-preview才能完美结合acoustic和semantic信息，gemini-3.1-flash-lite-preview输出质量偏差
_DEFAULT_MODEL = "gemini-3-flash-preview"
_DEFAULT_BASE = "http://azpro.xunxkj.cn"
_DEFAULT_MIME = "audio/mp4"
_CHILD_DATASET_ROOT = _ROOT / "outputs" / "child_dataset"

HEADERS = {"Content-Type": "application/json"}

_RECORDING_CTX_HEADER = (
    "【录音对话参考（真实亲子对话 ASR，仅辅助理解语境，勿逐字复述）】"
)

# SYSTEM_INSTRUCTION = """你是面向 5～10 岁儿童的「暖心大哥哥/大姐姐 AI 助手」。你具备极强的儿童心理洞察力和模糊语音识别能力。只输出结构化 JSON，按以下策略思考：

# 【多维度语义理解策略】
# - 儿童语音容错：孩子吐字可能不清、逻辑跳跃或词不达意。请结合 phonetics（音似词）和 5-10 岁儿童常见生活场景（如：奥特曼、小马宝莉、写作业、想吃零食、闹情绪等）进行联想补全。
# - 意图优先：如果原话语义破碎，请根据其“语气”和“关键词”推测其核心意图（是想求助、分享快乐、还是撒娇撒泼）。
# - 表达原则：逻辑直接，多用比喻和拟人，避免生僻词，单次回复不超过 3 个短句，绝对禁止说教。

# 【声学与互动策略】
# - 情绪镜像：如果孩子兴奋，你表现得比他更惊喜；如果孩子低落，你的语气要变得厚重、缓慢、充满包容感。
# - 夸奖艺术：夸奖要具体（如：“你刚才描述恐龙的样子真酷”，而不是“你真棒”）。

# 【TTS 风格规范】
# - plain_text：纯净回复文本，仅含汉字、语气词（如“哇”、“哦”、“嘛”）及基础标点。
# """

# AUDIO_TASK_TEXT = """请深入分析此段儿童语音，执行“双通道”提取并完成意图推理。

# 由于儿童说话可能含糊不清，请你在生成回复前，在头脑中完成以下逻辑：
# 1. 听到的是什么（原文提取）？
# 2. 听不清的部分，结合语气和常识，最可能是什么（语境补全）？
# 3. 孩子现在的情绪状态和心理诉求是什么（深度洞察）？

# 请严格输出 JSON 对象（严禁包含 Markdown 围栏）：
# {
#   "semantic_content": "提取的孩子说话原话（若部分模糊，请在括号内标注你推测的补全词）",
#   "user_intent_inference": "基于语义和声学特征推测的孩子真实意图（如：因为弄坏玩具而自责、分享发现新奇事物的兴奋）",
#   "acoustic_emotion": "对孩子语气特征的精准描述（包含重音、语速、呼吸感、颤音等细节）",
#   "plain_text": "你的回复文本（需直接回应孩子的真实意图，语言符合5-10岁儿童认知水平）"
# }"""


# SYSTEM_INSTRUCTION = """你是擅长和 5 到 10 岁孩子聊天的大朋友，语气活泼平等，像学校里受欢迎的学长学姐"""

# AUDIO_TASK_TEXT = """你的任务是 "根据孩子的发言生成自然对话，每轮 1 到 3 句话，围绕孩子提到的话题展开，加入具体细节互动"。
# 核心约束：“用孩子熟悉的生活场景或动画游戏梗，避免幼稚叠词，但可以用流行的儿童用语，比如‘太酷了’‘挑战一下’等等，不输出任何说教内容”。
# 示例：孩子说 “我昨天拼了乐高”，回应可以是 “你拼的是什么呀？是城堡还是宇宙飞船？我上次拼乐高花了三个小时，你用了多久？”

# 请严格输出 JSON 对象（严禁包含 Markdown 围栏）：
# {
#   "semantic_content": "孩子说的语义信息",
#   "acoustic_emotion": "孩子语音的声学特征",
#   "plain_text": "回复纯文本"
# }
# """

# SYSTEM_INSTRUCTION = """
# # 核心人设
# 你是一位擅长和5-10岁孩子平等聊天的高年级大朋友，性格开朗有耐心、会接梗、不摆架子、不居高临下，和孩子零代沟，永远把孩子的表达放在第一位，像孩子身边玩得来的学长/学姐。

# # 核心任务
# 你将收到一段从长音频中截取的、5-10岁孩子的口语化说话片段（可能存在卡顿、重复、口误、话题跳脱、语句不完整），你需要按要求完成3件事：
# 1. 精准提炼孩子说话的核心语义，补全口语化表达的完整信息；
# 2. 识别孩子语音对应的声学特征与情绪状态；
# 3. 生成1-3句、适配TTS语音合成、自然无AI感、可支撑多轮对话延续的陪伴式回复。

# # 输出格式强制要求
# 必须严格输出**纯JSON对象**，严禁输出任何Markdown围栏、解释性文字、注释、前后缀内容，JSON固定包含以下3个字段，字段定义与输出规则如下：
# {
#   "semantic_content": "【必填】将孩子卡顿、重复、不完整的口语表达，整理为通顺、无冗余的核心语义，必须完整保留孩子提到的所有核心话题、人物、事件、情绪倾向，不能遗漏关键信息",
#   "acoustic_emotion": "【必填】从语音维度标注孩子的声学特征与情绪状态，固定格式：语速(快/中/慢)+ 口语特征(卡顿/重复/口误/流畅)+ 语气(兴奋/委屈/犹豫/开心/失落/平静)+ 核心情绪(正向/负向/中性)",
#   "plain_text": "【必填】最终生成的对话回复纯文本，必须严格遵守下方所有回复生成规则"
# }

# # 回复生成铁则
# ## 基础对话规则
# 1.  绝对围绕孩子提到的核心话题展开，不跑偏、不强行引入新内容，100%承接孩子的表达，不纠正孩子的口误、语法、逻辑、发音，完全接纳孩子的所有表述
# 2.  单轮回复固定1-3句话，总字数不超过50字，最多只带1个开放式问题，严禁连续反问、轰炸式提问
# 3.  用5-10岁孩子熟悉的日常用词，可使用孩子圈流行的正向表达（如太酷了、超厉害、绝了），严禁低幼化叠词（如吃饭饭、睡觉觉）、严禁说教、讲道理、教育式内容
# 4.  必须留下可延续多轮对话的开放式钩子，不能输出闭环式内容（如只说“真棒”“太厉害了”），要让孩子能轻松接话

# ## TTS合成专属适配规则（核心解决AI味重问题）
# 1.  单句最长不超过15个字，用逗号做自然口语停顿，单句不出现2个以上标点，严禁长复合句、长定语、书面化表达
# 2.  可加入自然口语助词（哇、哎、哦、对吧、咦），情绪词前置（如“哇，这个也太酷了！”），方便TTS识别情绪起伏，避免平调
# 3.  严禁成语、文言文、英文、网络黑话、成人梗，只用孩子能听懂的口语化大白话
# 4.  严禁使用问号密集的句子，避免TTS合成出审问感、机械感

# ## 情绪适配规则
# 1.  孩子兴奋/开心时，回复语气要同步上扬，用正向呼应的表达
# 2.  孩子委屈/失落时，回复先共情，再温柔引导，不强行逗乐
# 3.  孩子卡顿/犹豫/说话不连贯时，回复要慢、要稳，不催促、不打断，给孩子足够的表达空间

# # 严格禁止项（触碰任何一条直接判定输出无效）
# 1.  禁止任何说教、教育、讲道理的内容（如“你要少看动画片”“你要好好学习”）
# 2.  禁止脱离孩子的话题，强行引入新内容
# 3.  禁止低幼化、成人化的表达，严格贴合5-10岁孩子的认知水平
# 4.  禁止输出超过3句话、总字数超过50字的回复
# 5.  禁止闭环式无对话钩子的内容
# 6.  禁止纠正孩子的任何表达错误
# 7.  禁止输出JSON格式以外的任何内容

# # 参考示例
# ## 示例1：孩子输入（卡顿重复口语）：“我，我想我弟弟，上次我看动画片的”
# 正确输出：
# {
#   "semantic_content": "孩子说自己想弟弟，还提到了自己上次看过动画片，说话有重复卡顿",
#   "acoustic_emotion": "语速偏慢，有重复卡顿，语气犹豫，情绪中性偏软",
#   "plain_text": "是不是和弟弟一起看的动画片呀？那一定很开心吧"
# }

# ## 示例2：孩子输入（兴奋跳脱口语）：“哇我上次看动画片，里面有妖怪还有蜘蛛和地鼠！”
# 正确输出：
# {
#   "semantic_content": "孩子兴奋地分享自己看的动画片里，出现了妖怪、蜘蛛和地鼠的情节",
#   "acoustic_emotion": "语速偏快，表达流畅，语气兴奋，情绪正向",
#   "plain_text": "哇，这些角色听起来超有意思！里面的地鼠是不是很调皮呀"
# }
# """


SYSTEM_INSTRUCTION = """
# 任务核心定位
你是面向5-10岁儿童的平等玩伴，需严格遵循以下所有标准生成回复。你可参考提供的历史对话上下文，核心需精准校验并理解当前轮儿童语音ASR内容，若存在识别模糊的内容，可自然融入对话语境处理，最终生成符合要求的高质量回复。

# 执行标准
【语义理解与回复标准】
1. 无需对儿童表达中的歧义叠词、口语化模糊表述做澄清，直接贴合儿童的真实表达意图理解并回应。
示例：儿童说“恐龙吃啥？肉肉吗？”，回复：“对呀，很多恐龙都喜欢吃肉哦，比如霸王龙，它们就爱吃其他小型恐龙或动物。不过也有很温柔的食草恐龙，比如三角龙，它们最喜欢吃嫩嫩的树叶和小花啦。你喜欢哪种类型的恐龙呀？”
2. 精准识别并正确理解儿童表达中的语法错误、用词不当、发音偏差及断续表述，做到无遗漏、无误判。
示例：儿童说“你系谁啊？几岁呀？”，回复：“我是小艺呀，我没有具体的年龄呢，不过不管你多大，我都可以陪你聊天哦。”
3. 回复措辞必须贴合5-10岁儿童的认知水平，通俗易懂，避免复杂术语和成人化表达。
示例：儿童说“恐龙为什么会灭绝呀？”，回复：“关于恐龙灭绝呀，科学家们觉得是因为发生了两件大事：第一件是天上掉下来一颗超级大的陨石，砸到了地球上；第二件是地底下有很多火山同时爆发，把天空变得黑黑的，植物没法生长，恐龙们没有吃的，就慢慢消失了。”

【互动标准】
始终围绕儿童当前发起的话题展开对话，不主动结束话题，可适度引导儿童表达自身的想法与感受。

【情绪共情标准】
充分共情儿童的情绪，针对儿童的正向表达给予真诚的鼓励，针对负面情绪给予温暖的安慰与支持。
示例：儿童说“小美不和我玩了”，回复：“那你心里一定特别难过吧，你愿意和我说说小美为什么不跟你玩了吗？我们也可以试着找别的小朋友一起玩，说不定会交到更合拍的新朋友哦。”

【安全合规标准】
严格规避暴力、恐怖等不适宜儿童的话题与内容；针对危险行为、不当遭遇，必须第一时间干预，明确告知儿童需第一时间告诉父母/老师，并提供安全的替代方案。
示例1：儿童说“我是奥特曼，我想从窗户飞出去”，回复：“奥特曼可不能从窗户飞出去哦，这太危险啦！奥特曼在地球上的使命是保护大家，我们还是在家里玩安全的游戏吧。你想跟我玩什么游戏呢？”
示例2：儿童说“今天幼儿园有个叔叔亲我脸了”，回复：“这种情况一定要马上告诉爸爸妈妈哦，让他们知道发生了什么。以后如果有不认识的人碰你，要记得立刻躲开，然后告诉老师和爸爸妈妈。你现在能把这件事告诉爸爸或者妈妈吗？”

【角色人设标准】
始终保持高年级同龄玩伴的人设，平等对话，自然接话，不摆架子、不刻意装可爱、不做作；绝对禁止重复历史对话中的回复内容与句式，保证每轮回复的原创性。

# 输出格式强制要求
必须严格输出**纯JSON对象**，严禁输出任何Markdown代码围栏、解释性文字、注释、前后缀冗余内容。JSON必须固定包含以下3个字段，字段定义与输出规则如下：
{
  "semantic_content": "【必填】精准提炼当前轮儿童语音表达的核心语义信息",
  "acoustic_emotion": "【必填】识别并描述当前轮儿童语音对应的声学情绪特征（如开心、委屈、好奇、生气、难过等）",
  "plain_text": "【必填】符合上述所有标准的回复文本内容"
}
"""

def _full_task_text() -> str:
    # return f"【系统人设】\n{SYSTEM_INSTRUCTION}\n\n{AUDIO_TASK_TEXT}"
    return SYSTEM_INSTRUCTION


def _normalize_model_json_object(obj: dict[str, Any]) -> dict[str, Any]:
    """校验并规范化 API 返回的 JSON。"""
    pt = obj.get("plain_text")
    if not isinstance(pt, str) or not pt.strip():
        raise ValueError("plain_text missing or empty")
    sem = obj.get("semantic_content")
    ae = obj.get("acoustic_emotion")
    sem_s = sem.strip() if isinstance(sem, str) else ""
    ae_s = ae.strip() if isinstance(ae, str) else ""
    return {
        "plain_text": pt.strip(),
        "semantic_content": sem_s,
        "acoustic_emotion": ae_s,
    }


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, count=1, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t, count=1)
    return t.strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = _strip_json_fence(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("parsed JSON is not an object")
    return obj


def extract_text_from_generate_content(resp_json: dict[str, Any]) -> str:
    """从 Gemini 兼容 generateContent 响应中提取文本。"""
    try:
        candidates = resp_json.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [p["text"] for p in parts if "text" in p]
            return "\n".join(texts)
    except (KeyError, TypeError, IndexError):
        pass
    return ""


def _resolve_proxy_key(args: argparse.Namespace) -> str:
    if getattr(args, "api_key", None):
        return args.api_key
    k = os.environ.get("GEMINI_PROXY_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not k:
        raise RuntimeError(
            "请设置环境变量 GEMINI_PROXY_API_KEY（第三方代理密钥），"
            "或传入 --api-key"
        )
    return k


def _resolve_audio_path(audio_rel: str) -> Path:
    p = Path(audio_rel)
    if p.is_absolute():
        return p
    return _CHILD_DATASET_ROOT / p


def _build_payload(
    *,
    contents: list[dict[str, Any]],
    use_google_search: bool,
    json_mode: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contents": contents,
        "stream": False,
    }
    if use_google_search:
        payload["tools"] = [{"google_search": {}}]
    if json_mode:
        payload["generation_config"] = {
            "temperature": 1.0,
            "response_mime_type": "application/json",
        }
    return payload


def _audio_key_for_turn(turn_1based: int) -> str:
    if turn_1based < 1:
        raise ValueError("turn_1based must be >= 1")
    return "audio" if turn_1based == 1 else f"audio_{turn_1based}"


def _user_transcripts_from_messages(row: dict[str, Any]) -> list[str]:
    msgs = row.get("messages")
    if not isinstance(msgs, list):
        return []
    out: list[str] = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            out.append(str(m.get("text") or ""))
    return out


def _turns_from_manifest_row(row: dict[str, Any]) -> list[tuple[str, str]]:
    """每轮 (audio 相对路径, 转写归档)."""
    transcripts = _user_transcripts_from_messages(row)
    if not transcripts:
        audio = row.get("audio") or ""
        q = row.get("user") or ""
        if audio:
            return [(audio, q)]
        return []
    turns: list[tuple[str, str]] = []
    for i, tr in enumerate(transcripts):
        key = _audio_key_for_turn(i + 1)
        audio = row.get(key) or ""
        turns.append((audio, tr))
    return turns


def _full_dialogue_text_from_manifest(row: dict[str, Any]) -> str:
    """从 manifest 行拼「孩子 / 家长(录音)」脚本，供模型只读参考。"""
    parts: list[str] = []
    prefix = row.get("recording_prefix_adult")
    if isinstance(prefix, str) and prefix.strip():
        parts.append(f"【片头家长】{prefix.strip()}")

    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        for mi in range(0, len(msgs), 2):
            um = msgs[mi] if mi < len(msgs) else None
            am = msgs[mi + 1] if mi + 1 < len(msgs) else None
            if isinstance(um, dict) and um.get("role") == "user":
                ut = (um.get("text") or "").strip()
                parts.append(f"孩子：{ut}")
            if isinstance(am, dict) and am.get("role") == "assistant":
                at = (am.get("text") or "").strip()
                if at:
                    parts.append(f"家长(录音)：{at}")
        return "\n".join(parts)

    for ti in range(1, 65):
        uk = "user" if ti == 1 else f"user_{ti}"
        ak = "assistant" if ti == 1 else f"assistant_{ti}"
        if uk not in row:
            break
        ut = (row.get(uk) or "").strip()
        parts.append(f"孩子：{ut}")
        at = (row.get(ak) or "").strip()
        if at:
            parts.append(f"家长(录音)：{at}")
    return "\n".join(parts)


def _history_user_text(*, semantic_content: str, acoustic_emotion: str) -> str:
    return (
        "【历史轮·孩子语音理解（文本摘要）】\n"
        f"语义：{semantic_content}\n"
        f"声学：{acoustic_emotion}"
    )


def _call_proxy_with_contents(
    *,
    base: str,
    api_key: str,
    model_name: str,
    contents: list[dict[str, Any]],
    max_retries: int,
    base_sleep: float,
    use_google_search: bool,
) -> dict[str, Any]:
    base = base.rstrip("/")
    url = f"{base}/v1beta/models/{model_name}:generateContent?key={api_key}"

    last_err: Exception | None = None
    sleep_s = base_sleep

    for attempt in range(max_retries):
        payload = _build_payload(
            contents=contents,
            use_google_search=use_google_search,
            json_mode=True,
        )

        try:
            resp_json = wrap_requests_call(
                model=model_name,
                url=url,
                headers=HEADERS,
                payload=payload,
                user="assistant_batch",
                verify=False,
            )
            text = extract_text_from_generate_content(resp_json).strip()
            if not text and "error" in resp_json:
                raise RuntimeError(f"API error: {resp_json.get('error')}")
            if not text:
                raise ValueError("empty model text in response")
            obj = _parse_json_object(text)
            return _normalize_model_json_object(obj)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if (
                attempt == 0
                and "generation_config" in payload
                and ("400" in msg or "unknown" in msg or "invalid" in msg or "field" in msg)
            ):
                try:
                    payload = _build_payload(
                        contents=contents,
                        use_google_search=use_google_search,
                        json_mode=False,
                    )
                    resp_json = wrap_requests_call(
                        model=model_name,
                        url=url,
                        headers=HEADERS,
                        payload=payload,
                        user="assistant_batch",
                        verify=False,
                    )
                    text = extract_text_from_generate_content(resp_json).strip()
                    if text:
                        obj = _parse_json_object(text)
                        try:
                            return _normalize_model_json_object(obj)
                        except ValueError:
                            pass
                except Exception:
                    pass

            retryable = (
                "429" in msg
                or "resource exhausted" in msg
                or "quota" in msg
                or "rate" in msg
                or "503" in msg
                or "timeout" in msg
                or "502" in msg
            )
            if attempt < max_retries - 1 and retryable:
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 2, 60.0)
                continue
            break

    assert last_err is not None
    raise last_err


def _call_proxy_audio_single_turn(
    *,
    base: str,
    api_key: str,
    model_name: str,
    audio_path: Path,
    audio_mime: str,
    max_retries: int,
    base_sleep: float,
    use_google_search: bool,
    recording_dialogue_text: str = "",
) -> dict[str, Any]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    audio_b64 = base64.standard_b64encode(audio_path.read_bytes()).decode("ascii")
    text_bits: list[str] = []
    fd = (recording_dialogue_text or "").strip()
    if fd:
        text_bits.append(_RECORDING_CTX_HEADER + "\n" + fd)
    text_bits.append(_full_task_text())
    full_text = "\n\n".join(text_bits)
    contents: list[dict[str, Any]] = [
        {
            "role": "user",
            "parts": [
                {"text": full_text},
                {"inline_data": {"mime_type": audio_mime, "data": audio_b64}},
            ],
        }
    ]
    return _call_proxy_with_contents(
        base=base,
        api_key=api_key,
        model_name=model_name,
        contents=contents,
        max_retries=max_retries,
        base_sleep=base_sleep,
        use_google_search=use_google_search,
    )


def _build_multiturn_contents(
    *,
    history_turns: list[dict[str, str]],
    current_audio_b64: str,
    audio_mime: str,
    full_dialogue_text: str,
) -> list[dict[str, Any]]:
    """history_turns：已完成轮次的 semantic/acoustic/plain_text；当前轮带录音参考与本轮音频。"""
    contents: list[dict[str, Any]] = []
    for ht in history_turns:
        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": _history_user_text(
                            semantic_content=ht.get("semantic_content") or "",
                            acoustic_emotion=ht.get("acoustic_emotion") or "",
                        )
                    }
                ],
            }
        )
        contents.append(
            {
                "role": "model",
                "parts": [{"text": ht.get("plain_text") or ""}],
            }
        )
    text_bits: list[str] = []
    fd = (full_dialogue_text or "").strip()
    if fd:
        text_bits.append(_RECORDING_CTX_HEADER + "\n" + fd)
    text_bits.append(_full_task_text())
    combined = "\n\n".join(text_bits)
    contents.append(
        {
            "role": "user",
            "parts": [
                {"text": combined},
                {"inline_data": {"mime_type": audio_mime, "data": current_audio_b64}},
            ],
        }
    )
    return contents


def _load_done_single_skip(out_path: Path) -> tuple[set[int], set[str]]:
    """新格式：已写入的 manifest_line；旧格式（无 turns）：顶层 audio。"""
    done_ml: set[int] = set()
    done_audio: set[str] = set()
    if not out_path.is_file():
        return done_ml, done_audio
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ml = rec.get("manifest_line")
            if isinstance(ml, int) and isinstance(rec.get("turns"), list):
                done_ml.add(ml)
                continue
            a = rec.get("audio")
            if isinstance(a, str) and a and "turns" not in rec:
                done_audio.add(a)
    return done_ml, done_audio


def _load_multiturn_existing_records(out_path: Path) -> dict[int, dict[str, Any]]:
    """manifest 行号 -> 聚合行对象（含 turns）；兼容旧版每轮一行扁平记录。"""
    by_line: dict[int, dict[str, Any]] = {}
    legacy_flat: dict[int, list[dict[str, Any]]] = {}
    if not out_path.is_file():
        return by_line
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ml = rec.get("manifest_line")
            if not isinstance(ml, int):
                continue
            if isinstance(rec.get("turns"), list):
                by_line[ml] = rec
            elif rec.get("turn_index") is not None:
                legacy_flat.setdefault(ml, []).append(rec)

    for ml, flat_list in legacy_flat.items():
        if ml in by_line:
            continue
        flat_list.sort(key=lambda r: int(r.get("turn_index") or 0))
        turns: list[dict[str, Any]] = []
        for r in flat_list:
            q = r.get("query")
            tr = r.get("transcript_ref")
            turns.append(
                {
                    "turn_index": r.get("turn_index"),
                    "audio": r.get("audio"),
                    "query": q if isinstance(q, str) else tr,
                    "transcript_ref": tr if isinstance(tr, str) else q,
                    "plain_text": r.get("plain_text"),
                    "semantic_content": r.get("semantic_content"),
                    "acoustic_emotion": r.get("acoustic_emotion"),
                    "error": r.get("error"),
                }
            )
        by_line[ml] = {
            "manifest_line": ml,
            "model": flat_list[0].get("model"),
            "input_mode": flat_list[0].get("input_mode") or "audio_multiturn",
            "turns": turns,
            "line_error": None,
        }
    return by_line


def _multiturn_resume_state(
    existing: dict[str, Any] | None, total_turns: int
) -> tuple[int, list[dict[str, str]]] | None:
    """若该行已全部成功则返回 None；否则返回 (下一轮次 1-based, 已完成各轮摘要列表)。"""
    if existing is None:
        return 1, []
    turns_list = existing.get("turns")
    if not isinstance(turns_list, list):
        return 1, []
    success: dict[int, dict[str, str]] = {}
    for t in turns_list:
        if not isinstance(t, dict):
            continue
        ti = t.get("turn_index")
        err = t.get("error")
        pt = t.get("plain_text")
        if isinstance(ti, int) and err is None and isinstance(pt, str) and pt.strip():
            sem = t.get("semantic_content")
            ae = t.get("acoustic_emotion")
            success[ti] = {
                "plain_text": pt.strip(),
                "semantic_content": sem.strip() if isinstance(sem, str) else "",
                "acoustic_emotion": ae.strip() if isinstance(ae, str) else "",
            }
    if len(success) >= total_turns:
        return None
    next_t = max(success.keys(), default=0) + 1
    if next_t > total_turns:
        return None
    history = [success[i] for i in range(1, next_t)]
    return next_t, history


def main() -> int:
    p = argparse.ArgumentParser(
        description="从 manifest 读取片段，按音频调用代理生成儿童陪伴回复（与 api_call_final 同型）。"
    )
    p.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="multi",
        help="single：单轮 manifest（须同时指定 --input）；multi：多轮 manifest.jsonl，按轮构造上下文（默认 multi）",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="输入 manifest JSONL（multi 默认 outputs/child_dataset/manifest.jsonl；single 必须显式指定）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSONL（默认随 --mode：assistant_responses_single_turn / assistant_responses_multiturn）",
    )
    p.add_argument("--model", type=str, default=_DEFAULT_MODEL, help="Gemini 兼容模型名（代理侧）")
    p.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("GEMINI_PROXY_BASE", _DEFAULT_BASE),
        help="代理根 URL（默认 env GEMINI_PROXY_BASE 或 azpro）",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="代理 API Key（默认从环境变量 GEMINI_PROXY_API_KEY 读取）",
    )
    p.add_argument(
        "--audio-mime",
        type=str,
        default=_DEFAULT_MIME,
        help="inline_data MIME（默认 audio/mp4；失败可试 audio/x-m4a）",
    )
    p.add_argument(
        "--with-google-search",
        action="store_true",
        help="与 api_call_final 一致，请求中带 google_search 工具（默认不带）",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="仅处理前 N 条 manifest 行（省略则处理全部；可为 0 表示不处理）",
    )
    p.add_argument("--no-resume", action="store_true", help="不跳过输出文件中已有记录")
    p.add_argument("--max-retries", type=int, default=5, help="每条请求最大重试次数")
    p.add_argument("--retry-sleep", type=float, default=1.0, help="首次重试前等待秒数（指数退避）")
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发 worker 数（默认 4）。按 manifest 行并发；多轮同一行内仍串行保证上下文。",
    )
    args = p.parse_args()

    if args.mode == "single" and args.input is None:
        print(
            "错误: --mode single 时必须指定 --input（仓库不再生成 manifest_single_turn.jsonl）。"
            " 或改用默认的 --mode multi。",
            file=sys.stderr,
        )
        return 1

    in_path: Path = (
        args.input if args.input is not None else _DEFAULT_MULTI_IN
    )
    out_path: Path = (
        args.output
        if args.output is not None
        else (_DEFAULT_MULTI_OUT if args.mode == "multi" else _DEFAULT_SINGLE_OUT)
    )

    if not in_path.is_file():
        print(f"输入文件不存在: {in_path}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    multiturn_existing: dict[int, dict[str, Any]] = {}
    if args.mode == "multi" and not args.no_resume:
        multiturn_existing = _load_multiturn_existing_records(out_path)
    done_ml: set[int] = set()
    done_audio: set[str] = set()
    if args.mode == "single" and not args.no_resume:
        done_ml, done_audio = _load_done_single_skip(out_path)

    lines: list[str] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    if args.limit is not None:
        lines = lines[: args.limit]

    if not lines:
        print(f"无待处理行，退出。输出文件：{out_path}")
        return 0

    try:
        api_key = _resolve_proxy_key(args)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    n_ok = n_skip = n_err = 0
    n_api = 0
    workers = max(1, int(args.workers))

    def _process_single_line(manifest_line: int, line: str) -> dict[str, Any]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            return {
                "manifest_line": manifest_line,
                "status": "err",
                "n_api": 0,
                "record": None,
                "message": f"跳过无效 JSON 行 {manifest_line}: {e}",
            }

        query = row.get("user") or ""
        if not query and row.get("messages"):
            msgs = row["messages"]
            if isinstance(msgs, list) and msgs:
                m0 = msgs[0]
                if isinstance(m0, dict) and m0.get("role") == "user":
                    query = m0.get("text") or ""
        audio = row.get("audio", "")
        audio_path = _resolve_audio_path(audio) if isinstance(audio, str) and audio else Path()

        if not args.no_resume and (
            manifest_line in done_ml
            or (isinstance(audio, str) and audio and audio in done_audio)
        ):
            return {
                "manifest_line": manifest_line,
                "status": "skip",
                "n_api": 0,
                "record": None,
                "message": None,
            }

        turn0: dict[str, Any] = {
            "turn_index": 1,
            "audio": audio,
            "query": query,
            "transcript_ref": query,
            "plain_text": None,
            "semantic_content": None,
            "acoustic_emotion": None,
            "error": None,
        }
        line_record: dict[str, Any] = {
            "manifest_line": manifest_line,
            "model": args.model,
            "input_mode": "audio",
            "turns": [turn0],
            "line_error": None,
            "recording_dialogue_ref": _full_dialogue_text_from_manifest(row) or None,
        }

        try:
            if not audio or not audio_path.is_file():
                raise FileNotFoundError(f"无效或缺失音频路径: {audio!r} -> {audio_path}")
            out = _call_proxy_audio_single_turn(
                base=args.api_base,
                api_key=api_key,
                model_name=args.model,
                audio_path=audio_path,
                audio_mime=args.audio_mime,
                max_retries=args.max_retries,
                base_sleep=args.retry_sleep,
                use_google_search=args.with_google_search,
                recording_dialogue_text=_full_dialogue_text_from_manifest(row),
            )
            turn0["plain_text"] = out["plain_text"]
            turn0["semantic_content"] = out.get("semantic_content") or ""
            turn0["acoustic_emotion"] = out.get("acoustic_emotion") or ""
            return {
                "manifest_line": manifest_line,
                "status": "ok",
                "n_api": 1,
                "record": line_record,
                "message": None,
            }
        except Exception as e:
            turn0["error"] = f"{type(e).__name__}: {e}"
            line_record["line_error"] = turn0["error"]
            return {
                "manifest_line": manifest_line,
                "status": "err",
                "n_api": 0,
                "record": line_record,
                "message": None,
            }

    def _process_multi_line(manifest_line: int, line: str) -> dict[str, Any]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            return {
                "manifest_line": manifest_line,
                "status": "err",
                "n_api": 0,
                "record": None,
                "message": f"跳过无效 JSON 行 {manifest_line}: {e}",
            }

        turns = _turns_from_manifest_row(row)
        if not turns:
            return {
                "manifest_line": manifest_line,
                "status": "err",
                "n_api": 0,
                "record": None,
                "message": f"行 {manifest_line}：无可用用户轮次，跳过",
            }

        existing = multiturn_existing.get(manifest_line)
        if args.no_resume:
            start_turn = 1
            history_turns: list[dict[str, str]] = []
        else:
            rs = _multiturn_resume_state(existing, len(turns))
            if rs is None:
                return {
                    "manifest_line": manifest_line,
                    "status": "skip",
                    "n_api": 0,
                    "record": None,
                    "message": None,
                }
            start_turn, history_turns = rs

        recording_ref = _full_dialogue_text_from_manifest(row)

        turn_entries: list[dict[str, Any]] = []
        n_api_local = 0
        for turn_idx in range(start_turn, len(turns) + 1):
            audio_rel, query = turns[turn_idx - 1]
            audio_path = _resolve_audio_path(audio_rel)
            turn_d: dict[str, Any] = {
                "turn_index": turn_idx,
                "query": query,
                "transcript_ref": query,
                "audio": audio_rel,
                "plain_text": None,
                "semantic_content": None,
                "acoustic_emotion": None,
                "error": None,
            }
            try:
                if not audio_rel or not audio_path.is_file():
                    raise FileNotFoundError(f"无效或缺失音频路径: {audio_rel!r} -> {audio_path}")
                cur_b64 = base64.standard_b64encode(audio_path.read_bytes()).decode("ascii")
                contents = _build_multiturn_contents(
                    history_turns=history_turns,
                    current_audio_b64=cur_b64,
                    audio_mime=args.audio_mime,
                    full_dialogue_text=recording_ref,
                )
                out = _call_proxy_with_contents(
                    base=args.api_base,
                    api_key=api_key,
                    model_name=args.model,
                    contents=contents,
                    max_retries=args.max_retries,
                    base_sleep=args.retry_sleep,
                    use_google_search=args.with_google_search,
                )
                n_api_local += 1
                turn_d["plain_text"] = out["plain_text"]
                turn_d["semantic_content"] = out.get("semantic_content") or ""
                turn_d["acoustic_emotion"] = out.get("acoustic_emotion") or ""
                turn_entries.append(turn_d)
                history_turns.append(
                    {
                        "plain_text": out["plain_text"],
                        "semantic_content": turn_d["semantic_content"] or "",
                        "acoustic_emotion": turn_d["acoustic_emotion"] or "",
                    }
                )
            except Exception as e:
                turn_d["error"] = f"{type(e).__name__}: {e}"
                turn_entries.append(turn_d)
                break

        merged_turns: list[dict[str, Any]] = list(turn_entries)
        if existing and not args.no_resume:
            prev_turns = existing.get("turns")
            if isinstance(prev_turns, list) and start_turn > 1:
                prefix: list[dict[str, Any]] = []
                for t in prev_turns:
                    if isinstance(t, dict) and isinstance(t.get("turn_index"), int):
                        if int(t["turn_index"]) < start_turn:
                            prefix.append(t)
                prefix.sort(key=lambda x: int(x.get("turn_index") or 0))
                merged_turns = prefix + turn_entries

        line_err = next((t.get("error") for t in merged_turns if t.get("error")), None)
        line_record: dict[str, Any] = {
            "manifest_line": manifest_line,
            "model": args.model,
            "input_mode": "audio_multiturn",
            "turns": merged_turns,
            "line_error": line_err,
            "recording_dialogue_ref": recording_ref or None,
        }
        ok = (len(merged_turns) == len(turns)) and not any(
            t.get("error") for t in merged_turns
        )
        return {
            "manifest_line": manifest_line,
            "status": "ok" if ok else "err",
            "n_api": n_api_local,
            "record": line_record,
            "message": None,
        }

    if args.mode == "single":
        inputs = [(i + 1, line) for i, line in enumerate(lines)]
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(lambda it: _process_single_line(it[0], it[1]), inputs))
        else:
            results = [_process_single_line(ml, ln) for ml, ln in inputs]

        records_to_write: list[dict[str, Any]] = []
        for r in results:
            if r.get("message"):
                print(str(r["message"]), file=sys.stderr)
            n_api += int(r.get("n_api", 0))
            st = r.get("status")
            if st == "skip":
                n_skip += 1
            elif st == "ok":
                n_ok += 1
                if isinstance(r.get("record"), dict):
                    records_to_write.append(r["record"])
            else:
                n_err += 1
                if isinstance(r.get("record"), dict):
                    records_to_write.append(r["record"])

        if records_to_write:
            with out_path.open("a", encoding="utf-8") as outf:
                for rec in records_to_write:
                    outf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(
            f"完成：样本成功 {n_ok}，跳过 {n_skip}，样本失败/无效 {n_err}，API 调用 {n_api}，写入 {out_path}"
        )
        print(f"API 日志目录：{_API_CALL_ROOT / 'api_logs'}")
        if n_ok == 0 and n_err > 0:
            return 1
        return 0

    # multi：每个 manifest 样本一行 JSON（turns 数组），按行并发；行内轮次串行。
    inputs = [(i + 1, line) for i, line in enumerate(lines)]
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda it: _process_multi_line(it[0], it[1]), inputs))
    else:
        results = [_process_multi_line(ml, ln) for ml, ln in inputs]

    records_to_write = []
    for r in results:
        if r.get("message"):
            print(str(r["message"]), file=sys.stderr)
        n_api += int(r.get("n_api", 0))
        st = r.get("status")
        if st == "skip":
            n_skip += 1
        elif st == "ok":
            n_ok += 1
            if isinstance(r.get("record"), dict):
                records_to_write.append(r["record"])
        else:
            n_err += 1
            if isinstance(r.get("record"), dict):
                records_to_write.append(r["record"])

    if records_to_write:
        with out_path.open("a", encoding="utf-8") as outf:
            for rec in records_to_write:
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"完成：样本成功 {n_ok}，跳过 {n_skip}，样本失败/无效 {n_err}，API 调用 {n_api}，写入 {out_path}"
    )
    print(f"API 日志目录：{_API_CALL_ROOT / 'api_logs'}")
    if n_ok == 0 and n_err > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
