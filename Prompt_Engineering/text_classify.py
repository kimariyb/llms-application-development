import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# .env 在项目根目录（Prompt_Engineering 的上一级）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("OPENAI_MODEL")

# 分类标签
LABELS = ['新闻报道', '财务报告', '公司公告', '分析师报告']

# 少样本示例：每个类别一条典型文本
CLASS_EXAMPLES = {
    '新闻报道': '今日，股市经历了一轮震荡，受到宏观经济数据和全球贸易紧张局势的影响。投资者密切关注美联储可能的政策调整，以适应市场的不确定性。',
    '财务报告': '本公司年度财务报告显示，去年公司实现了稳步增长的盈利，同时资产负债表呈现强劲的状况。经济环境的稳定和管理层的有效战略执行为公司的健康发展奠定了基础。',
    '公司公告': '本公司高兴地宣布成功完成最新一轮并购交易，收购了一家在人工智能领域领先的公司。这一战略举措将有助于扩大我们的业务领域，提高市场竞争力',
    '分析师报告': '最新的行业分析报告指出，科技公司的创新将成为未来增长的主要推动力。云计算、人工智能和数字化转型被认为是引领行业发展的关键因素，投资者应关注这些趋势',
}

# ─── 系统提示词：声明角色 ───
SYSTEM_PROMPT = """<role>
你是一位金融文本分类助手，具备以下专业能力：
- 精准识别金融文本的类别（新闻报道、财务报告、公司公告、分析师报告）
- 理解金融领域术语与文本结构特征
- 严格遵循预设分类体系，不自行编造类别
你的回答需体现该角色的专业判断力与严谨性。
</role>"""

# ─── 用户提示词模板：任务 + 示例 + 输入 + 输出格式 ───
USER_PROMPT_TEMPLATE = """<task>
根据输入的金融文本内容，从以下预设类别中选择唯一一个最匹配的类别。
可选类别：{labels}
任务目标：返回一个类别名称字符串，且必须是上述类别之一。
任务约束：只返回类别名称，不返回任何解释、推理或额外文字。
</task>

<examples>
以下示例展示期望的输入输出对应关系：

<example_1>
输入：{ex1_text}
输出：{ex1_label}
</example_1>

<example_2>
输入：{ex2_text}
输出：{ex2_label}
</example_2>

<example_3>
输入：{ex3_text}
输出：{ex3_label}
</example_3>

<example_4>
输入：{ex4_text}
输出：{ex4_label}
</example_4>
</examples>

<input>
待分类文本：
###
{content}
###
</input>

<output_format>
只输出一个类别名称，必须是以下四个之一：{labels}
不要输出任何其他内容。
</output_format>"""


def classify_news(content):
    """调用 LLM 对金融文本进行分类（stream 模式），边接收边打印，返回完整类别字符串"""
    labels_str = '、'.join(LABELS)
    examples = list(CLASS_EXAMPLES.items())

    prompt = USER_PROMPT_TEMPLATE.format(
        labels=labels_str,
        content=content,
        ex1_text=examples[0][1], ex1_label=examples[0][0],
        ex2_text=examples[1][1], ex2_label=examples[1][0],
        ex3_text=examples[2][1], ex3_label=examples[2][0],
        ex4_text=examples[3][1], ex4_label=examples[3][0],
    )

    # 使用流式运行
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        extra_body={"thinking": {"type": "disabled"}},
    )

    result = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            result += delta
            print(delta, end="", flush=True)  # 逐字打印，不换行
    print()  # 收尾换行
    return result


if __name__ == "__main__":
    test_text = "为保持银行体系流动性充裕，2024年12月26日人民银行以固定利率、数量招标方式开展了1063亿元逆回购操作"
    result = classify_news(test_text)
    print(f"分类结果: {result}")
