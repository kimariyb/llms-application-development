"""CrewAI 多智能体工具集：书信保存 + 邮件发送。"""

import os
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path

from crewai.tools import BaseTool

# 书信文件路径（项目根目录 / output / letter.txt）
LETTER_PATH = Path(__file__).resolve().parent.parent / "output" / "letter.txt"


class StorePoetryTool(BaseTool):
    """将编辑后的书信文本保存到本地 txt 文件。"""

    name: str = "store_poetry_to_txt"
    description: str = "将编辑后的书信文本内容保存到本地 txt 文件中。"

    def _run(self, content: str) -> str:
        """保存文本到文件。

        Args:
            content: 书信正文文本。

        Returns:
            保存成功返回文件路径，失败返回错误信息。
        """
        try:
            LETTER_PATH.parent.mkdir(parents=True, exist_ok=True)
            LETTER_PATH.write_text(content, encoding="utf-8")
            return f"书信已保存至 {LETTER_PATH}"
        except Exception as e:
            return f"保存失败: {e}"


class SendMessageTool(BaseTool):
    """读取本地书信文件，以邮件形式发送给收件人。"""

    name: str = "send_message"
    description: str = "读取本地书信文件，以邮件形式发送给收件人。"

    def _run(self) -> str:
        """发送邮件。

        Returns:
            发送成功返回 "信件已发送"，失败返回错误信息。
        """
        from_addr = os.getenv("EMAIL_FROM_ADDR")
        from_name = os.getenv("EMAIL_FROM_NAME", "小可爱")
        from_pwd = os.getenv("EMAIL_PWD")
        to_addr = os.getenv("EMAIL_TO_ADDR")
        smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.163.com")
        subject = os.getenv("EMAIL_SUBJECT", "一封书信")

        # 参数校验
        missing = [
            k for k, v in {
                "EMAIL_FROM_ADDR": from_addr,
                "EMAIL_PWD": from_pwd,
                "EMAIL_TO_ADDR": to_addr,
            }.items() if not v
        ]
        if missing:
            return f"邮件配置缺失，请在 .env 中设置: {', '.join(missing)}"

        # 读取书信内容
        if not LETTER_PATH.exists():
            return f"书信文件不存在: {LETTER_PATH}"
        body = LETTER_PATH.read_text(encoding="utf-8")

        # 构建邮件
        msg = MIMEText(body, "plain", "utf-8")
        msg["From"] = formataddr([from_name, from_addr])
        msg["To"] = to_addr
        msg["Subject"] = subject

        # 发送（with 语句自动管理连接生命周期）
        try:
            with smtplib.SMTP_SSL(smtp_server, 465, timeout=10) as srv:
                srv.login(from_addr, from_pwd)
                srv.sendmail(from_addr, [to_addr], msg.as_string())
            return "信件已发送"
        except Exception as e:
            return f"发送失败: {e}"


# 工具实例（供 agents.py 导入）
store_poetry_to_txt = StorePoetryTool()
send_message = SendMessageTool()
