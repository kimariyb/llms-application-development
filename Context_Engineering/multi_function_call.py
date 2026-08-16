import json
import os
import time

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# .env 在项目根目录（Context_Engineering 的上一级）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    max_retries=0,
)
MODEL = os.getenv("OPENAI_MODEL")


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_plane_number",
            "description": "根据出发地、目的地和日期，查询对应日期的航班号。当用户询问某两地之间有什么航班、航班号是多少时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "description": "出发地，如：北京、郑州",
                        "type": "string",
                    },
                    "end": {
                        "description": "目的地，如：深圳、北京",
                        "type": "string",
                    },
                    "date": {
                        "description": "日期，格式：YYYY-MM-DD",
                        "type": "string",
                    },
                },
                "required": ["start", "end", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_plane_ticket_price",
            "description": "查询指定航班号在指定日期的票价。需要先知道航班号才能调用；如果用户只提供路线不提供航班号，请先调用 get_plane_number 查询航班号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "description": "航班号",
                        "type": "string",
                    },
                    "date": {
                        "description": "日期，格式：YYYY-MM-DD",
                        "type": "string",
                    },
                },
                "required": ["number", "date"],
            },
        },
    },
]


# ─── 函数实现 ───
PLANE_NUMBERS = {
    ("北京", "深圳"): "ZH9126",
    ("北京", "广州"): "CA1356",
    ("郑州", "北京"): "CZ1123",
    ("郑州", "天津"): "MU3661",
}


def get_plane_number(start: str, end: str, date: str) -> str:
    """查询指定路线和日期的航班号，返回 JSON 字符串。

    Args:
        start: 出发地。
        end: 目的地。
        date: 日期，格式 YYYY-MM-DD。

    Returns:
        JSON 字符串。code=1 成功，result 含航班号；
        code=0 失败，message 含错误原因。
    """
    number = PLANE_NUMBERS.get((start, end))
    if number is None:
        return json.dumps(
            {"code": 0, "result": {}, "message": f"未开通航线: {start} → {end}"},
            ensure_ascii=False,
        )

    return json.dumps(
        {"code": 1, "result": {"date": date, "plane_number": number}},
        ensure_ascii=False,
    )


def get_plane_ticket_price(number: str, date: str) -> str:
    """查询指定航班在指定日期的票价，返回 JSON 字符串。

    Args:
        number: 航班号。
        date: 日期，格式 YYYY-MM-DD。

    Returns:
        JSON 字符串。code=1 成功，result 含票价（元）。
    """
    return json.dumps(
        {"code": 1, "result": {"number": number, "date": date, "ticket_price": 668}},
        ensure_ascii=False,
    )


# 注册映射：函数名 → 可调用对象
available_functions = {
    "get_plane_number": get_plane_number,
    "get_plane_ticket_price": get_plane_ticket_price,
}


# 限流重试：固定间隔等待（被拒请求也计入 RPM，间隔须 >= 60/RPM）
def create_with_retry(messages: list, **kwargs):
    """调用 LLM，遇到 429 限流时等待重试。"""
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


# 完整调用循环
def chat_with_tools(user_input: str) -> str:
    """带工具调用能力的对话函数，支持串行/并行多工具调用。"""
    messages = [
        {
            "role": "system",
            "content": "现在你是一个航班查询助手，将根据用户问题提供答案，但是不要假设或猜测传入函数的参数值。如果用户的描述不明确，请要求用户提供必要信息。",
        },
        {"role": "user", "content": user_input},
    ]

    # 外层循环：模型可能连续多轮调用工具（串行依赖：上一步结果是下一步参数）
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
                try:
                    result = fn(**fn_args)
                except Exception as e:
                    # 执行异常也以 code=0 回传，让模型自行决定下一步
                    result = json.dumps(
                        {"code": 0, "message": f"函数执行出错: {e}"},
                        ensure_ascii=False,
                    )

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


if __name__ == "__main__":
    answer = chat_with_tools(
        "帮我查询 2025 年 7 月 30 日郑州到北京的航班票价，如果你不知道请调用工具查询"
    )
    print(f"\n最终回答: {answer}")
