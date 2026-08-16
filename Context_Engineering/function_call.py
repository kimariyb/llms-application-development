import json
import os
import time

import requests

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# .env 在项目根目录（Context_Engineering 的上一级）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    max_retries=0,  # 关闭 SDK 内置重试，由 create_with_retry 统一处理
)
MODEL = os.getenv("OPENAI_MODEL")

# 工具定义（JSON Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_data",
            "description": "查询指定城市今日天气，返回最高温、最低温、星期和天气类型。当用户询问天气、气温、下雨等情况时调用此函数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，如：北京、广州",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

# ─── 城市编码 ───
CITY_CODES = {
    "北京": "101010100",
    "广州": "101280101",
}


# ─── 函数实现 ───
def get_weather_data(city: str) -> str:
    """查询指定城市今日天气，返回 JSON 字符串（供 LLM function calling 使用）。

    Args:
        city: 城市名，必须是 CITY_CODES 中已登记的城市。

    Returns:
        JSON 字符串。code=1 成功，result 含天气数据；
        code=0 失败，result 为空，message 含错误原因。
    """
    if city not in CITY_CODES:
        return json.dumps(
            {"city": city, "code": 0, "result": {}, "message": "未登记的城市"},
            ensure_ascii=False,
        )

    url = f"http://t.weather.itboy.net/api/weather/city/{CITY_CODES[city]}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        forecast = response.json()["data"]["forecast"][0]
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        return json.dumps(
            {"city": city, "code": 0, "result": {}, "message": str(e)},
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "city": city,
            "code": 1,
            "result": {
                "high_temperature": forecast["high"],
                "low_temperature": forecast["low"],
                "week": forecast["week"],
                "type": forecast["type"],
            },
        },
        ensure_ascii=False,
    )


# 注册映射：函数名 → 可调用对象
available_functions = {
    "get_weather_data": get_weather_data,
}


# 限流重试：固定间隔等待
def create_with_retry(messages: list, **kwargs):
    """调用 LLM，遇到 429 限流时等待重试。

    注意：被拒的请求同样计入 RPM 配额，快速重试会让限流窗口
    永远处于满载状态，因此等待间隔要 >= 60/RPM 秒（3 RPM → 25s）。
    """
    for attempt in range(4):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, **kwargs
            )
        except RateLimitError:
            wait = 25
            print(f"[限流] 第 {attempt + 1} 次被限，等待 {wait}s 重试...")
            time.sleep(wait)
    raise RuntimeError("重试 4 次后仍被限流，请稍后再试")


# 单轮工具循环：处理「模型 → 工具 → 模型」直到给出最终回答
def run_tool_loop(messages: list) -> str:
    """执行一轮带工具调用的推理，返回最终回答。

    messages 就地更新：assistant / tool 消息会被追加进去，
    从而保留完整历史供后续轮次使用。
    """
    while True:
        resp = create_with_retry(messages, tools=tools)
        msg = resp.choices[0].message

        # 分支：模型不需要工具，记录回复并返回
        if not msg.tool_calls:
            messages.append(msg)
            return msg.content

        # 分支：模型决定调函数，逐个执行
        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            fn = available_functions.get(fn_name)

            if fn is None:
                result = json.dumps(
                    {"code": 0, "message": f"未注册的函数: {fn_name}"},
                    ensure_ascii=False,
                )
            else:
                result = fn(**fn_args)

            print(f"[工具调用] {fn_name}({fn_args}) → {result}")

            # 执行结果以 tool 角色回传
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )
        # 循环回到 while 顶部，让模型看结果后决定是否继续调工具或给出最终回答


# 多轮对话主循环：messages 跨轮次持久化
def chat() -> None:
    """交互式多轮对话。输入 exit/quit 退出。"""
    messages = [
        {
            "role": "system",
            "content": "你是一个天气播报小助手，根据用户提供的城市回答当地天气情况。如果用户提供的信息不明确，提示用户明确输入，不要编造内容。",
        }
    ]

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            print("再见！")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        answer = run_tool_loop(messages)
        print(f"助手: {answer}")


if __name__ == "__main__":
    chat()
