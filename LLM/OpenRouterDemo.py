import os
import logging
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量(如果使用.env文件)
load_dotenv(find_dotenv())

# OpenRouter配置
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", os.getenv('OPENROUTER_API_KEY'))
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "x-ai/grok-4.1-fast"  # 使用可用的模型

# 初始化OpenAI客户端,使用OpenRouter配置
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

logger.info(f"OpenRouter客户端初始化完成 - Model: {OPENROUTER_MODEL}")


def test_simple_chat():
    """
    测试基本的聊天功能
    
    Returns:
        bool: 请求是否成功
    """
    logger.info("开始测试简单聊天功能")
    print("\n" + "="*50)
    print("【测试1: 简单聊天】")
    print("="*50)
    
    try:
        # 发送API请求
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Hello! Please introduce yourself briefly."
                }
            ],
            temperature=0.7
        )
        
        # 提取响应内容
        content = response.choices[0].message.content
        
        # 成功输出
        logger.info(f"✅ 请求成功 - 响应长度: {len(content)} 字符")
        print(f"\n✅ 请求成功!")
        print(f"Assistant: {content}")
        print(f"\n使用的模型: {response.model}")
        print(f"总tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
        
        return True
        
    except Exception as e:
        # 捕获并详细记录错误
        error_type = type(e).__name__
        error_msg = str(e)
        
        logger.error(f"❌ 请求失败: {error_type} - {error_msg}")
        print(f"\n❌ 请求失败!")
        print(f"错误类型: {error_type}")
        print(f"错误信息: {error_msg}")
        
        # 如果有HTTP响应信息,打印详细内容
        if hasattr(e, 'response'):
            logger.error(f"HTTP状态码: {e.response.status_code}")
            logger.error(f"响应内容: {e.response.text}")
            print(f"HTTP状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text}")
        
        return False


def test_reasoning_chat():
    """
    测试带推理功能的聊天(如果模型支持)
    
    Returns:
        bool: 请求是否成功
    """
    logger.info("开始测试推理聊天功能")
    print("\n" + "="*50)
    print("【测试2: 推理聊天(需要支持推理的模型)】")
    print("="*50)
    
    try:
        # 第一次API调用 - 带推理
        logger.info("发送第一次推理请求")
        response = client.chat.completions.create(
            model="openai/o1-mini",  # 使用支持推理的模型
            messages=[
                {
                    "role": "user",
                    "content": "How many r's are in the word 'strawberry'?"
                }
            ],
            # extra_body={"reasoning": {"enabled": True}}  # 某些模型支持
        )
        
        # 提取助手消息
        assistant_msg = response.choices[0].message
        
        logger.info(f"✅ 第一次请求成功")
        print(f"\n✅ 第一次请求成功!")
        print(f"Assistant: {assistant_msg.content}")
        
        # 检查是否有推理详情
        if hasattr(assistant_msg, 'reasoning_details') and assistant_msg.reasoning_details:
            print(f"\n推理详情: {assistant_msg.reasoning_details}")
            logger.info("检测到推理详情")
        
        # 第二次API调用 - 继续对话
        logger.info("发送第二次推理请求")
        messages = [
            {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
            {
                "role": "assistant",
                "content": assistant_msg.content
            },
            {"role": "user", "content": "Are you sure? Think carefully."}
        ]
        
        # 如果有推理详情,保留它
        if hasattr(assistant_msg, 'reasoning_details') and assistant_msg.reasoning_details:
            messages[1]["reasoning_details"] = assistant_msg.reasoning_details
        
        response2 = client.chat.completions.create(
            model="openai/o1-mini",
            messages=messages,
            # extra_body={"reasoning": {"enabled": True}}
        )
        
        logger.info(f"✅ 第二次请求成功")
        print(f"\n✅ 第二次请求成功!")
        print(f"Assistant: {response2.choices[0].message.content}")
        print(f"\n总tokens: {response2.usage.total_tokens if response2.usage else 'N/A'}")
        
        return True
        
    except Exception as e:
        # 捕获并详细记录错误
        error_type = type(e).__name__
        error_msg = str(e)
        
        logger.error(f"❌ 推理请求失败: {error_type} - {error_msg}")
        print(f"\n❌ 推理请求失败!")
        print(f"错误类型: {error_type}")
        print(f"错误信息: {error_msg}")
        
        # 打印详细错误信息
        if hasattr(e, 'response'):
            logger.error(f"HTTP状态码: {e.response.status_code}")
            logger.error(f"响应内容: {e.response.text}")
            print(f"HTTP状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text}")
        
        # 如果是模型不支持的错误,提供建议
        if "model" in error_msg.lower() or "not found" in error_msg.lower():
            print("\n💡 提示: 该模型可能不支持或不存在,请检查模型名称")
            logger.warning("模型不支持或不存在")
        
        return False


def test_streaming_chat():
    """
    测试流式输出功能
    
    Returns:
        bool: 请求是否成功
    """
    logger.info("开始测试流式输出功能")
    print("\n" + "="*50)
    print("【测试3: 流式输出】")
    print("="*50)
    
    try:
        # 发送流式请求
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Tell me a short story about AI in 3 sentences."
                }
            ],
            stream=True,  # 启用流式输出
            temperature=0.7
        )
        
        logger.info("✅ 流式请求启动成功")
        print("\n✅ 流式请求成功!")
        print("Assistant: ", end="", flush=True)
        
        # 接收流式数据
        full_content = ""
        chunk_count = 0
        
        for chunk in response:
            chunk_count += 1
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_content += content
                print(content, end="", flush=True)
        
        print()  # 换行
        logger.info(f"流式输出完成 - 共接收 {chunk_count} 个数据块,总长度: {len(full_content)} 字符")
        print(f"\n总字符数: {len(full_content)}")
        print(f"数据块数: {chunk_count}")
        
        return True
        
    except Exception as e:
        # 捕获并详细记录错误
        error_type = type(e).__name__
        error_msg = str(e)
        
        logger.error(f"❌ 流式请求失败: {error_type} - {error_msg}")
        print(f"\n❌ 流式请求失败!")
        print(f"错误类型: {error_type}")
        print(f"错误信息: {error_msg}")
        
        # 打印详细错误信息
        if hasattr(e, 'response'):
            logger.error(f"HTTP状态码: {e.response.status_code}")
            logger.error(f"响应内容: {e.response.text}")
            print(f"HTTP状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text}")
        
        return False


def main():
    """主函数 - 运行所有测试"""
    logger.info("OpenRouter演示程序启动")
    print("\n" + "#"*50)
    print("# OpenRouter API 测试演示")
    print("#"*50)
    print(f"\nAPI地址: {OPENROUTER_BASE_URL}")
    print(f"默认模型: {OPENROUTER_MODEL}")
    
    # 运行所有测试
    results = {
        "简单聊天": test_simple_chat(),
        "流式输出": test_streaming_chat(),
        "推理聊天": test_reasoning_chat(),
    }
    
    # 总结测试结果
    print("\n" + "="*50)
    print("【测试结果汇总】")
    print("="*50)
    
    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{test_name}: {status}")
        logger.info(f"{test_name}: {'成功' if success else '失败'}")
    
    # 计算成功率
    success_count = sum(results.values())
    total_count = len(results)
    success_rate = (success_count / total_count) * 100
    
    print(f"\n成功率: {success_count}/{total_count} ({success_rate:.1f}%)")
    logger.info(f"程序执行完成 - 成功率: {success_rate:.1f}%")


if __name__ == "__main__":
    main()
