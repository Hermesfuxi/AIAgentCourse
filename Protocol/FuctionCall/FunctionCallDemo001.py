import json
import os
import logging
from typing import Dict, Any, Callable

from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

class WeatherAssistant:
    """智能天气助手类"""
    
    def __init__(self):
        """初始化助手"""
        self.client = self._init_client()
        self.tool_map = {
            "get_weather": self.get_weather,
            "dress_advice": self.dress_advice
        }
        self.tools = self._define_tools()
        
    def _init_client(self) -> OpenAI:
        """初始化OpenAI客户端"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
        
        return OpenAI(
            api_key=api_key, 
            base_url="https://api.deepseek.com"
        )
    
    def _define_tools(self) -> list:
        """定义工具配置"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "根据城市名称返回模拟天气情况",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "城市名，例如：北京、上海"}
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "dress_advice",
                    "description": "根据天气情况给出穿衣建议",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "weather": {"type": "string", "description": "天气信息，例如：晴，28°C"}
                        },
                        "required": ["weather"]
                    }
                }
            }
        ]
    
    @staticmethod
    def get_weather(city: str) -> str:
        """获取天气信息（模拟数据）"""
        dummy_weather = {
            "北京": "晴，28°C",
            "上海": "多云，22°C",
            "广州": "小雨，19°C",
            "长沙": "阴，25°C",
            "深圳": "晴，30°C",
            "杭州": "多云，24°C"
        }
        result = dummy_weather.get(city, f"抱歉，暂无{city}的天气信息")
        logger.info(f"获取{city}天气: {result}")
        return result
    
    @staticmethod
    def dress_advice(weather: str) -> str:
        """根据天气给出穿衣建议"""
        advice_map = {
            "雨": "记得带伞，穿防水外套和防滑鞋。",
            "晴": "适合轻便服装，注意防晒，可穿短袖。",
            "多云": "建议穿长袖薄衫，可准备薄外套。",
            "阴": "适合春秋装，建议穿薄外套或卫衣。"
        }
        
        # 根据天气关键词匹配建议
        for keyword, advice in advice_map.items():
            if keyword in weather:
                # 根据温度调整建议
                if "30" in weather or "3" in weather.split("°")[0][-1:]:
                    if "晴" in weather:
                        advice += " 温度较高，多补充水分。"
                elif "1" in weather.split("°")[0] or "2" in weather.split("°")[0][:2]:
                    advice += " 温度适中，注意保暖。"
                
                logger.info(f"天气'{weather}'的穿衣建议: {advice}")
                return advice
        
        return "根据当前体感和个人喜好选择合适的服装。"
    
    def _execute_tool_call(self, call) -> str:
        """执行工具调用"""
        try:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            
            if name not in self.tool_map:
                error_msg = f"未知工具：{name}"
                logger.error(error_msg)
                return error_msg
            
            result = self.tool_map[name](**args)
            logger.info(f"工具调用成功: {name}({args}) -> {result}")
            return result
            
        except json.JSONDecodeError as e:
            error_msg = f"解析工具参数失败: {e}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"工具调用异常: {e}"
            logger.error(error_msg)
            return error_msg
    
    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """与助手对话"""
        messages = [
            {"role": "system", "content": "你是一个智能天气助手，能够根据用户需求调用工具完成任务。请提供准确、实用的建议。"},
            {"role": "user", "content": user_message}
        ]
        
        iteration = 0
        
        while iteration < max_iterations:
            try:
                # 向模型发起请求
                resp = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.7
                )
                
                msg = resp.choices[0].message
                
                # 如果没有调用工具，说明得到最终回答
                if not msg.tool_calls:
                    logger.info("对话完成，返回最终答案")
                    return msg.content
                
                # 添加助手消息到历史
                messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
                
                # 处理所有工具调用
                for call in msg.tool_calls:
                    result = self._execute_tool_call(call)
                    
                    # 将工具调用结果添加到消息历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
                
                iteration += 1
                
            except Exception as e:
                error_msg = f"API调用失败: {e}"
                logger.error(error_msg)
                return f"抱歉，服务暂时不可用：{error_msg}"
        
        return "抱歉，对话轮次过多，请重新开始。"


def main():
    """主函数"""
    try:
        assistant = WeatherAssistant()
        
        # 示例对话
        user_query = "我今天要去上海，明天要去长沙，后天要去北京，大后天去广州，该怎么穿衣服？"
        
        print(f"用户问题: {user_query}")
        print("\n" + "="*50)
        
        result = assistant.chat(user_query)
        
        print(f"\n最终回答: {result}")
        
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        print(f"程序运行失败: {e}")


if __name__ == "__main__":
    main()