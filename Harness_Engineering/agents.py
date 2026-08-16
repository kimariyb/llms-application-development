"""CrewAI 多智能体示例：作家 → 内容编辑 → 寄信人 串行协作。"""

import os

from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

from tools import store_poetry_to_txt, send_message

# .env 在项目根目录（Harness_Engineering 的上一级）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ─── LLM（LiteLLM 封装，OpenAI 兼容接口）───
# model 前缀 "openai/" 告诉 LiteLLM 使用 OpenAI 兼容协议，base_url 指向 Moonshot
# kimi-k2.6 是推理模型，只允许 temperature=1，不设置则默认 1
llm = LLM(
    model=f"openai/{os.getenv('OPENAI_MODEL')}",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# ─── Agent 定义 ───
poet = Agent(
    role="作家",
    goal="根据用户需求，创作出情感丰富的文章（不超过 300 字）。",
    backstory="你是一名著名的作家，拥有千万级别的粉丝，最擅长写情感类型的文章。",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

letter_writer = Agent(
    role="内容编辑",
    goal="对作家撰写的文章内容进行精心编辑。",
    backstory=(
        "你是一名经验丰富的编辑，在书信编辑方面有多年的专业经验。"
        "你需要将作家写的文章内容整理编排成书信的样式，"
        "并使用提供的工具将书信内容存储到本地磁盘上。"
        "当文件成功保存时返回 '书信已保存'。"
    ),
    verbose=True,
    allow_delegation=False,
    tools=[store_poetry_to_txt],
    llm=llm,
)

sender = Agent(
    role="寄信人",
    goal="将编辑好的书信以邮件的形式发送给心仪的人。",
    backstory=(
        "你是一名勤恳的信使，专注于将书信传递给每个人。"
        "你必须使用提供的工具读取本地书信文件并发送到收件人邮箱。"
        "如果成功发送，返回 '信件已发送'。"
    ),
    verbose=True,
    allow_delegation=False,
    tools=[send_message],
    llm=llm,
)


def run(content: str) -> str:
    """运行多智能体工作流，返回最终结果。

    Args:
        content: 用户的创作需求。

    Returns:
        Crew 执行结果字符串。
    """
    task1 = Task(
        description=f"用户需求: {content}。你最后给出的答案必须是一份富含情感的情书。",
        agent=poet,
        expected_output="一份不超过 300 字的情书",
    )

    task2 = Task(
        description=(
            "检查语法错误，进行编辑和格式化。"
            "使用工具将内容保存到本地磁盘。"
            "你最后的答案必须是文件是否已成功保存。"
        ),
        agent=letter_writer,
        expected_output="书信已保存",
    )

    task3 = Task(
        description=(
            "读取本地保存的书信文件，通过工具将邮件发送给收件人。"
            "你最后的答案必须是邮件是否已成功发送。"
        ),
        agent=sender,
        expected_output="信件已发送",
    )

    crew = Crew(
        agents=[poet, letter_writer, sender],
        tasks=[task1, task2, task3],
        verbose=True,
        process=Process.sequential,
    )

    return crew.kickoff()


if __name__ == "__main__":
    user_input = input("请输入你的需求：\n")
    result = run(user_input)
    print("\n######################")
    print(result)
