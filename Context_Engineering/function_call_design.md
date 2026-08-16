# Function Calling 设计原则与定义方法

> 记录 LLM Function Calling 的核心概念、设计原则、定义方法与完整实现模式。

---

## 1. 什么是 Function Calling

Function Calling 是让 LLM 调用外部工具的机制：**模型不执行函数，只输出"我想调用哪个函数、参数是什么"的结构化 JSON，实际执行由你的代码完成，执行结果再喂回模型**。

完整循环：

```
用户提问 ──► LLM 判断需要工具 ──► 返回函数名 + 参数（JSON）
                                        │
                                        ▼
                              你的代码真正执行函数
                                        │
                                        ▼
执行结果塞回上下文 ──► LLM 基于结果生成最终回答 ──► 用户
```

本质上是 Context Engineering 的一部分：工具定义和执行结果都是注入上下文窗口的内容。

---

## 2. 设计原则

### 2.1 单一职责

一个函数只做一件事。不要让一个函数同时干查询 + 格式化 + 发送，拆成独立函数让模型自行组合。

- ✅ `get_weather` / `format_report` / `send_email`
- ❌ `get_weather_format_and_send`

### 2.2 命名：动词 + 宾语，自解释

函数名和描述是模型选工具的唯一依据。模型看到的是字符串，不是代码。

- 命名用 `snake_case` 动宾结构：`search_flights`、`send_email`、`get_stock_price`
- 避免含糊名：`handle_data`、`process` ❌
- 名字里带约束更稳：`get_todays_weather`（明确"今天"）优于 `get_weather`（模型会问：哪天？）

### 2.3 描述写给模型看，不是写给人看

`description` 是提示词的一部分，直接影响调用准确率：

```json
{
  "name": "get_weather",
  "description": "查询指定城市今日天气，返回最高温、最低温和天气类型。当用户询问天气、气温、下雨等情况时调用此函数。"
}
```

要点：
- **干什么** + **返回什么** + **何时用**，三件事都说
- 参数也写 `description`，注明格式和默认值
- 别写实现细节（"通过 HTTP 请求第三方 API"），模型不需要

### 2.4 参数设计：少而精

- 参数数量控制在 3–5 个以内，超过就该拆函数
- 必选参数越少越好——模型每填一个参数就多一分出错可能
- 能推导的不要让模型填：查"今天天气"就不该有 `date` 参数
- 参数类型用最简单的 JSON Schema 类型：`string` / `number` / `boolean` / `enum`
- 值域有限的参数用 `enum` 约束，别让模型自由发挥

```json
{
  "type": "object",
  "properties": {
    "city": {
      "type": "string",
      "description": "城市名，如：北京、上海"
    },
    "unit": {
      "type": "string",
      "enum": ["celsius", "fahrenheit"],
      "description": "温度单位，默认 celsius"
    }
  },
  "required": ["city"]
}
```

### 2.5 返回值：结构化、可判定、可恢复

Function calling 场景下，返回值会被塞回上下文给模型读，必须约定统一结构：

```json
{
  "code": 1,
  "result": { },
  "message": ""
}
```

| 字段 | 含义 |
|------|------|
| `code` | 状态码：1 成功，0 失败 |
| `result` | 成功时含数据，失败时为空 |
| `message` | 失败时含原因，让模型能向用户解释 |

三条铁律：
- **永远不要抛异常终止流程**——网络失败也返回 `code:0` + 错误信息，模型拿到错误才能向用户解释或换方案
- **字段名稳定**——模型依赖字段名读取数据，字段名变了调用就废了
- **返回紧凑**——只返回模型需要的字段，别把原始 API 响应整个塞回去（浪费 token，还会淹没信号）

### 2.6 幂等与副作用

- 查询类函数（读）天然安全，可以随意重试
- 写操作类函数（发邮件、下单）要考虑：参数校验前置、防重复执行、必要时让模型先确认再调用

---

## 3. 定义方法

### 3.1 工具定义：JSON Schema 格式

以 OpenAI 兼容接口为例（Kimi / DeepSeek / Qwen 等均兼容）：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市今日天气。当用户询问天气、气温时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，如：北京、上海",
                    },
                },
                "required": ["city"],
            },
        },
    }
]
```

### 3.2 函数实现：规范模板（Google 风格 docstring）

```python
def get_weather(city: str) -> str:
    """查询指定城市今日天气，返回 JSON 字符串（供 LLM function calling 使用）。

    Args:
        city: 城市名。

    Returns:
        JSON 字符串。code=1 成功，result 含天气数据；
        code=0 失败，result 为空，message 含错误原因。
    """
    url = f"https://api.example.com/weather/{city}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, KeyError, ValueError) as e:
        return json.dumps({"code": 0, "message": str(e)}, ensure_ascii=False)

    return json.dumps({
        "code": 1,
        "result": {
            "high_temperature": data["high"],
            "low_temperature": data["low"],
            "weather_type": data["type"],
        },
    }, ensure_ascii=False)
```

实现规范要点：
- f-string 拼 URL（不用 `os.path.join`）
- `timeout` 必设，防止永久阻塞
- 所有异常捕获后转为 `code:0` 的正常返回
- `ensure_ascii=False` 保证中文可读

### 3.3 注册映射：函数名 → 可调用对象

```python
available_functions = {
    "get_weather": get_weather,
}
```

### 3.4 完整调用循环

```python
def chat_with_tools(user_input: str) -> str:
    messages = [{"role": "user", "content": user_input}]

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,        # 注入工具定义
    )
    msg = resp.choices[0].message

    # 分支 1：模型决定调函数
    if msg.tool_calls:
        messages.append(msg)  # 先把模型的调用意图存入历史
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            result = available_functions[fn_name](**fn_args)   # 真正执行

            # 执行结果以 tool 角色塞回上下文
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # 分支 2：带着结果再问一次模型，拿最终自然语言回答
        final = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        return final.choices[0].message.content

    # 分支 3：模型不需要工具，直接回答
    return msg.content
```

关键点：
- `tools` 参数传入定义，模型自动判断是否需要
- 模型可能一次调用多个工具（`tool_calls` 是列表），逐个执行
- 执行结果必须带 `tool_call_id` 回传，模型靠它对应"哪个结果来自哪次调用"
- **这是一个 Loop**：模型看完结果后还可能再要求调用其他工具，生产代码应外层套 `while` 循环直到模型不再发起 `tool_calls`

---

## 4. 常见坑

| 坑 | 后果 | 规避 |
|----|------|------|
| 工具描述含糊 | 模型乱选函数或瞎填参数 | description 写全"干什么+返回什么+何时用" |
| 函数内部抛异常 | 整个对话崩溃 | 捕获所有异常，转为 code:0 返回 |
| 无 timeout | 请求挂起永久阻塞 | `requests.get(url, timeout=5)` |
| 返回原始 API 大 JSON | token 浪费、模型被噪声干扰 | 只返回模型需要的字段 |
| 参数让模型自由填枚举值 | 模型编造不合法的值 | enum 列出合法值 |
| 执行结果不带 tool_call_id | 模型无法对应结果，报错 | 严格回传 id |
| 只支持一轮工具调用 | 复杂任务中断 | 外层 while 循环直到无 tool_calls |

---

## 5. 与相关概念的关系

- **Function Calling ⊂ Context Engineering**：工具定义占用上下文 token，执行结果也注入上下文——本质还是在管理"模型看到什么"
- **MCP 是 Function Calling 的标准化封装**：把工具定义、发现、执行协议化，解决 N 个应用 × M 个工具的集成问题
- **Function Calling 是 Agent 的地基**：ReAct 循环（推理→行动→观察）中的"行动"就是 function call；Loop 工程里每一轮迭代都靠它接触外部世界

---

## 参考

- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use Guide: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
