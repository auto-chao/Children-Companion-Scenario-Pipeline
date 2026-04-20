import base64
import time
from pathlib import Path
import sys

# 强制将标准输出流设置为 UTF-8 编码 (解决 Windows 输出包含 emoji 报错的问题)
sys.stdout.reconfigure(encoding='utf-8')

# 假设你的本地请求封装器
from local_api_logger import wrap_requests_call

API_KEY = "sk-ragFTLE01dU6dZPTgOKSfhPdW66jKVRY2PDfX7QQCLX4uo0F"
URL = "http://azpro.xunxkj.cn/v1/chat/completions" # 统一的中转接口地址

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# ============ 实验配置区 ============
MODELS_TO_TEST =[
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-mini-tts",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5.2",
    "gpt-5.4",
    "qwen3.5-122b-a10b",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "qwen3.5-397b-a17b",
    "qwen3.5-flash",
    "qwen3.5-omni-flash",
    "qwen3.5-omni-plus",
    "qwen3.5-plus",
    "qwen3.6-plus"
]

_ROOT = Path(__file__).resolve().parents[1]
AUDIO_REL = Path("outputs") / "child_dataset" / "audios" / (
    "video2_放学的路上，聊天把妹妹聊哭了_0_0004_38961_57034.m4a"
)

# 两个阶段的测试 Prompt
TEXT_PROMPT = "这是一条纯文本连通性测试。请你直接回复我“文本测试成功”这六个字，不需要任何额外说明。"
ASR_PROMPT = "请转写这段音频中的语音内容，并用语音回复我你的总结。"

# 请求控制配置
MAX_RETRIES = 3
RETRY_DELAY = 2.0
SLEEP_BETWEEN_TESTS = 2.0 # 同一模型的文本和音频测试间隙
SLEEP_BETWEEN_MODELS = 3.0 # 不同模型之间的间隙
# ====================================

def get_family(model_name):
    if model_name.startswith("gpt"):
        return "gpt"
    elif model_name.startswith("qwen"):
        return "qwen"
    return "unknown"

def build_payload(family, model, test_type, prompt, audio_b64=None, audio_path=None):
    """根据测试类型（text/audio）构建对应的请求体"""
    
    # 1. 纯文本测试 Payload (所有模型通用标准格式)
    if test_type == "text":
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        
    # 2. 音频测试 Payload
    if test_type == "audio":
        ext = audio_path.suffix.lower().replace(".", "") if audio_path else "mp3"
        if ext not in ["mp3", "wav"]:
            ext = "mp3" 

        if family == "gpt":
            return {
                "model": model,
                "modalities": ["text", "audio"],
                "audio": {"voice": "alloy", "format": "wav"},
                "messages":[
                    {
                        "role": "user",
                        "content":[
                            {"type": "text", "text": prompt},
                            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": ext}}
                        ]
                    }
                ],
                "stream": False
            }
        elif family == "qwen":
            mime_type = f"audio/{ext}" if ext != "mp3" else "audio/mpeg"
            audio_uri = f"data:{mime_type};base64,{audio_b64}"
            return {
                "model": model,
                "messages":[
                    {
                        "role": "user",
                        "content":[
                            {"type": "text", "text": prompt},
                            {"type": "audio_url", "audio_url": {"url": audio_uri}}
                        ]
                    }
                ],
                "modalities":["text", "audio"], 
                "stream": False
            }
        else:
            return {
                "model": model,
                "messages":[{"role": "user", "content": prompt}],
                "stream": False
            }

def extract_result(resp_json, output_audio_path=None):
    result = {
        "text_reply": "",
        "has_audio_out": False,
        "is_error": False,
        "error_msg": ""
    }

    try:
        if "error" in resp_json:
            result["is_error"] = True
            result["error_msg"] = str(resp_json["error"])
            return result

        choices = resp_json.get("choices", [])
        if not choices:
            result["is_error"] = True
            result["error_msg"] = "响应中缺少 choices 字段"
            return result
            
        message = choices[0].get("message", {})
        result["text_reply"] = message.get("content", "")
        
        audio_data = message.get("audio", {})
        if audio_data and "data" in audio_data and output_audio_path:
            audio_bytes = base64.b64decode(audio_data["data"])
            with open(output_audio_path, "wb") as f:
                f.write(audio_bytes)
            result["has_audio_out"] = True

    except Exception as e:
        result["is_error"] = True
        result["error_msg"] = f"解析异常: {e}"
        
    return result

def send_with_retry(model, url, headers, payload, max_retries, delay, test_label="test"):
    for attempt in range(1, max_retries + 1):
        try:
            # print(f"    ➜ 发送{test_label}请求 ({attempt}/{max_retries})...")
            response = wrap_requests_call(
                model=model,
                url=url,
                headers=headers,
                payload=payload,
                user=f"batch_test_{test_label}",
                verify=False,
            )
            
            if not response or isinstance(response, str):
                raise ValueError(f"网关返回无效格式: {str(response)[:100]}")
                
            return response 
            
        except Exception as e:
            if attempt < max_retries:
                time.sleep(delay)
            else:
                return {"error": {"message": f"请求失败: {e}"}}


# ============ 主程序 ============
def main():
    audio_path = _ROOT / AUDIO_REL
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    audio_bytes = audio_path.read_bytes()
    audio_b64 = base64.standard_b64encode(audio_bytes).decode("ascii")

    print("🚀 开始双重验证 (文本模态 & 音频模态)...")
    print(f"📁 测试音频: {AUDIO_REL.name}\n")
    
    summary_results =[]

    for i, model in enumerate(MODELS_TO_TEST, 1):
        print("="*60)
        print(f"[{i}/{len(MODELS_TO_TEST)}] 正在测试模型: {model}")
        family = get_family(model)
        
        model_stats = {
            "model": model,
            "text_io_ok": False,
            "audio_in_ok": False,
            "audio_out_ok": False,
            "error_detail": ""
        }

        # ---------------------------------------------------------
        # [阶段 1]: 纯文本连通性测试 (Text In -> Text Out)
        # ---------------------------------------------------------
        print("  [阶段 1] 测试纯文本输入输出...")
        payload_text = build_payload(family, model, "text", TEXT_PROMPT)
        resp_text = send_with_retry(model, URL, HEADERS, payload_text, MAX_RETRIES, RETRY_DELAY, "text")
        res_text = extract_result(resp_text)
        
        if res_text["is_error"]:
            print(f"    ❌ 文本测试报错: {res_text['error_msg']}")
            model_stats["error_detail"] = "文本测试失败"
        else:
            print(f"    ✅ 文本回复: {res_text['text_reply'].strip()}")
            if "测试成功" in res_text["text_reply"]:
                model_stats["text_io_ok"] = True
            else:
                # 即使没严格按照要求回复，只要返回了非空文本，也可视为基础连通
                model_stats["text_io_ok"] = True if res_text["text_reply"] else False

        time.sleep(SLEEP_BETWEEN_TESTS)

        # ---------------------------------------------------------
        # [阶段 2]: 音频模态测试 (Audio In -> Text/Audio Out)
        # ---------------------------------------------------------
        print("  [阶段 2] 测试音频输入输出...")
        output_audio_path = f"reply_{model}.wav"
        payload_audio = build_payload(family, model, "audio", ASR_PROMPT, audio_b64, audio_path)
        resp_audio = send_with_retry(model, URL, HEADERS, payload_audio, MAX_RETRIES, RETRY_DELAY, "audio")
        res_audio = extract_result(resp_audio, output_audio_path)
        
        if res_audio["is_error"]:
            print(f"    ❌ 音频测试报错/被拒绝: {res_audio['error_msg']}")
            if not model_stats["error_detail"]:
                model_stats["error_detail"] = "音频测试报被拒绝"
        else:
            model_stats["audio_in_ok"] = True
            text_preview = res_audio["text_reply"].replace('\n', ' ')[:40] + "..."
            print(f"    📝 伴随文本回复: {text_preview}")
            
            if res_audio["has_audio_out"]:
                print(f"    🎙️ 音频生成成功! -> {output_audio_path}")
                model_stats["audio_out_ok"] = True
            else:
                print("    ⚠️ 未生成音频 (仅返回文本)")

        summary_results.append(model_stats)
        
        if i < len(MODELS_TO_TEST):
            time.sleep(SLEEP_BETWEEN_MODELS)

    # ============ 打印最终汇总表格 ============
    print("\n\n" + "#"*80)
    print(" " * 25 + "📊 全局模态支持测试汇总报告")
    print("#"*80)
    # 对齐中文字符比较麻烦，尽量用英文状态描述保持整齐
    print(f"{'模型名称':<22} | {'Text In/Out':<12} | {'Audio In':<10} | {'Audio Out':<10} | {'详情简述'}")
    print("-" * 80)
    
    for res in summary_results:
        m_name = res["model"]
        
        # 状态图标
        txt_status = "✅ 支持" if res["text_io_ok"] else "❌ 失败"
        aud_in_status = "✅ 接收" if res["audio_in_ok"] else "❌ 失败"
        aud_out_status = "✅ 支持" if res["audio_out_ok"] else "❌ 不支持"
        
        err_desc = res["error_detail"] if res["error_detail"] else "正常"
            
        print(f"{m_name:<22} | {txt_status:<12} | {aud_in_status:<10} | {aud_out_status:<10} | {err_desc}")

    print("-" * 80)
    print("📝 说明：")
    print(" 1. Text In/Out: 发送纯文本能否成功获取回复。")
    print(" 2. Audio In: 接口是否允许传入音频参数且不报 400 错误 (注:需看控制台日志核实转写内容是否瞎编)。")
    print(" 3. Audio Out: 模型是否真正返回了语音流并保存为 wav 文件。")
    print("#"*80)

if __name__ == "__main__":
    main()