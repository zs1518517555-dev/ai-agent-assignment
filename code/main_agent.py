import os
import json
import re
import requests
from dotenv import load_dotenv
from rich.console import Console

# 导入你麾下的两员大将
from medical_agent import MedicalDataMinerAgent
from expert_qa_agent import MedicalExpertAgent
from path_config import KNOWLEDGE_BASE_DIR

console = Console()
load_dotenv()
API_KEY = os.getenv("API_KEY")


class RouterAgent:
    def __init__(self):
        console.print("[bold cyan]🔄 正在唤醒系统底层...[/bold cyan]")
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        # 实例化两个子 Agent
        self.medical_worker = MedicalDataMinerAgent()

        # 这里已经填好了你之前的真实 PartA 路径
        knowledge_path = str(KNOWLEDGE_BASE_DIR)
        self.qa_expert = MedicalExpertAgent(knowledge_path)

    def classify_intent(self, user_input):
        """核心意图识别引擎"""
        prompt = f"""
        你是一个中央路由大管家。请判断用户输入的意图：

        1. 如果用户是在提问、请教知识、探讨概念（例如：“什么是...”、“...的难点”、“解释一下...”） -> 返回 JSON: {{"intent": "QA"}}
        2. 如果用户是在要求执行动作、处理数据集、清洗数据（例如：“帮我处理...”、“清洗并提取...”） -> 返回 JSON: {{"intent": "ACTION"}}

        用户输入：“{user_input}”
        请只返回严格的 JSON，不要包含任何其他文字。
        """

        payload = {"model": "qwen-max", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            reply = response.json()["choices"][0]["message"]["content"]

            # 正则暴力提取 JSON
            match = re.search(r'\{.*\}', reply, re.DOTALL)
            if match:
                return json.loads(match.group(0)).get("intent", "QA")
            return "QA"  # 默认当作问答处理
        except Exception:
            return "QA"

    def run(self):
        console.print("\n[bold green]=====================================================[/bold green]")
        console.print("[bold green]🧠 医疗影像全能 AI 助理已上线！（支持数据处理与学术问答）[/bold green]")
        console.print("[bold green]=====================================================[/bold green]")

        while True:
            user_input = console.input("\n[bold yellow]👤 请输入您的指令或问题 (输入 '退出' 结束)：[/bold yellow] ")
            if user_input.strip() == "": continue
            if user_input.strip() == "退出": break

            # 1. 瞬间意图识别
            intent = self.classify_intent(user_input)

            # 2. 智能路由分发
            if intent == "ACTION":
                console.print(
                    "[bold magenta]⚡ 管家判定：这是一个【数据挖掘处理】任务。正在唤醒医疗数据矿工...[/bold magenta]")
                self.medical_worker.chat_and_execute(user_input)
            else:
                console.print("[bold cyan]📚 管家判定：这是一个【学术文献问答】任务。正在唤醒学术专家...[/bold cyan]")
                self.qa_expert.answer_question(user_input)


if __name__ == "__main__":
    master_agent = RouterAgent()
    master_agent.run()