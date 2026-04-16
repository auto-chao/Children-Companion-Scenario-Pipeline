import json

def collect_stream_data(stream_generator):
    """Collect reasoning and content from SSE stream generator"""
    reasoning_parts, content_parts = [], []

    for line in stream_generator:
        # 生成器已经返回了包含换行符的行，需要去除空白
        line = line.strip()

        if not line or not line.startswith('data: '):
            continue

        data = line[6:].strip()
        if data == '[DONE]':
            continue

        try:
            parsed = json.loads(data)

            # 安全检查 choices 数组
            if not parsed.get('choices') or len(parsed['choices']) == 0:
                continue

            delta = parsed['choices'][0].get('delta', {})

            if reasoning := delta.get('reasoning_content'):
                reasoning_parts.append(reasoning)

            if content := delta.get('content'):
                content_parts.append(content)
                # breakpoint()
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # 调试：打印解析失败的数据
            print(f"解析失败: {e}, 数据: {data[:100]}")
            continue

    return ''.join(reasoning_parts), ''.join(content_parts)

def collect_non_stream_data(stream):
    return stream["choices"][0]["message"]["content"]