from openai import OpenAI
import os

# 1. 配置 Gemini API Key
# 建议通过环境变量设置，而不是硬编码在代码中
# 确保你已经从 Google AI Studio 获取了你的 GEMINI_API_KEY
# 假设环境变量名为 GEMINI_API_KEY
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # ⚠️ 风险提示：硬编码 API Key 存在安全风险，仅用于快速测试。
    # 建议使用 os.environ.get("GEMINI_API_KEY")
    print("⚠️ 警告：请设置 GEMINI_API_KEY 环境变量以提高安全性。")
    # 替换为你的实际 Gemini API 密钥
    # GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
    # 如果没有设置环境变量且未在此处硬编码，代码将无法运行。
    pass

# 2. 初始化 OpenAI Client
# 核心：将 base_url 设置为 Gemini 兼容的 API 接口
# 注意：此 URL 适用于 Google Generative AI API（非 Vertex AI）
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://api.gemini.dev/v1",
)

# 3. 进行 API 调用
try:
    # 使用 Gemini 模型名称，例如 gemini-2.5-flash
    model_name = "gemini-2.5-flash"

    print(f"🚀 正在调用 Gemini 模型: {model_name}...")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一位资深的嵌入式C语言架构师。"},
            {"role": "user", "content": "请用C语言写一个简单的 LED 闪烁程序片段，用于STM32F4系列微控制器。"}
        ],
        temperature=0.7,
        max_tokens=500
    )

    # 4. 打印结果
    # 响应结构与 OpenAI 的 API 响应结构保持一致
    if response.choices:
        print("\n--- 智源的回答（通过 OpenAI 接口获取） ---")
        print(response.choices[0].message.content)
        print("-------------------------------------------\n")
    else:
        print("未收到有效的模型响应。")

except Exception as e:
    print(f"❌ 调用失败：{e}")