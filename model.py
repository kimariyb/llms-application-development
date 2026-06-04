from openai import OpenAI
from dotenv import load_dotenv
import os


# 加载 .env 文件（自动读取配置）
load_dotenv()


# 从配置文件安全获取密钥和配置
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL")


# 创建客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)


# 提示词工程
prompt = """

"""


def model_chat(content: str, model: str = model) -> str:
    if not isinstance(model, str) and not isinstance(model, str):
        raise TypeError("Model must be of type str or None")
    if model is None:
        raise ValueError("Model must be of type str or None")

    try:
        response = client.chat.completions.create(
          model=model,
          messages=[
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": content}
          ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise e


def classify(content: str) -> str:
    prompt.format(content=content, label=['', '', '', ''])
    return model_chat(content=content)