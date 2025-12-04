import json
import os
import requests

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# One-API 地址
ONEAPI_URL = os.getenv("ONE_API_BASE_URL")
print(str(ONEAPI_URL))

# 如果需要 API Key，可以放在 headers 中
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + os.getenv("ONE_API_KEY"),  # 如果 One-API 配置了密钥
}

print(str(HEADERS))

# 构造请求体
data = {
    "model": os.getenv("ONE_API_MODEL"),  # 根据你 One-API 部署支持的模型修改
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，分析当前模型哪家供应商最强"},
    ],
    "stream": True,  # 关键参数：开启流式输出
}

print(str(data))

with requests.post(ONEAPI_URL, json=data, headers=HEADERS, stream=True) as response:
    print(response)
    if response.status_code != 200:
        print("请求失败，状态码：", response.status_code)
        print("返回内容：", response.text)
    else:
        print("Assistant:", end="", flush=True)
        buffer = b""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if line.startswith(b"data: "):
                        line_data = line[len(b"data: ") :].strip()
                        if line_data == b"[DONE]":
                            break
                        try:
                            # 使用 UTF-8 解码
                            chunk_json = json.loads(line_data.decode("utf-8"))
                            content = chunk_json["choices"][0]["delta"].get("content")
                            if content:
                                print(content, end="", flush=True)
                        except Exception:
                            continue
        print()