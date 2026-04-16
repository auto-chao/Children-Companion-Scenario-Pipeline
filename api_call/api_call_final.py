import base64
from pathlib import Path

from local_api_logger import wrap_requests_call

API_KEY = "sk-ragFTLE01dU6dZPTgOKSfhPdW66jKVRY2PDfX7QQCLX4uo0F"
MODEL = "gemini-3-flash-preview"
URL = f"http://azpro.xunxkj.cn/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
HEADERS = {"Content-Type": "application/json"}

_ROOT = Path(__file__).resolve().parents[1]

# ============ 配置区 ============
# 相对项目根目录的本地音频（m4a）；ASR 由多模态模型完成
AUDIO_REL = Path("outputs") / "child_dataset" / "audios" / (
    "video2_放学的路上，聊天把妹妹聊哭了_0_0004_37257_53609.m4a"
)
AUDIO_MIME = "audio/mp4"  # 若网关报错可改为 audio/x-m4a
ASR_PROMPT = "请转写这段音频中的语音内容，按时间顺序输出文字，不要添加无关说明。"
# ================================
STREAM_KEY = False


def extract_text(resp_json):
    """从 Gemini 响应 JSON 中提取文本内容"""
    try:
        candidates = resp_json.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [p["text"] for p in parts if "text" in p]
            return "\n".join(texts)
    except Exception as e:
        return f"[解析失败] {e}\n原始响应: {resp_json}"
    return str(resp_json)


audio_path = _ROOT / AUDIO_REL
if not audio_path.is_file():
    raise FileNotFoundError(f"音频文件不存在: {audio_path}")

audio_b64 = base64.standard_b64encode(audio_path.read_bytes()).decode("ascii")

payload = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {"text": ASR_PROMPT},
                {
                    "inline_data": {
                        "mime_type": AUDIO_MIME,
                        "data": audio_b64,
                    }
                },
            ],
        }
    ],
    "tools": [
        {"google_search": {}}
    ],
    "stream": STREAM_KEY,
}

# 自动处理流式响应并记录日志
response = wrap_requests_call(
    model=MODEL,
    url=URL,
    headers=HEADERS,
    payload=payload,
    user="my_app",
    verify=False,
)


print("回复:\n", extract_text(response))
