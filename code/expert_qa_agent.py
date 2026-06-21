import os
import json
import re
import requests
from dotenv import load_dotenv

from rich.console import Console
from rich.markdown import Markdown
from path_config import BENCHMARK_FILE, ENHANCED_BENCHMARK_FILE

# 加载 API Key
load_dotenv()
API_KEY = os.getenv("API_KEY")


class MedicalExpertAgent:
    def __init__(self, knowledge_base_dir):
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        self.knowledge_base_dir = knowledge_base_dir
        self.knowledge_docs = self._load_knowledge_base()
        self.qa_items = self._load_qa_benchmark()

    def _load_knowledge_base(self):
        """核心数据挖掘动作：加载你刚刚生成的所有本地知识资产"""
        docs = []
        print("📚 正在加载本地学术知识库...")

        if not os.path.exists(self.knowledge_base_dir):
            print("⚠️ 找不到知识库目录，请检查路径！")
            return docs

        # 遍历 PartA/data 下的所有 kebab-case 文件夹
        for folder_name in os.listdir(self.knowledge_base_dir):
            folder_path = os.path.join(self.knowledge_base_dir, folder_name)
            if os.path.isdir(folder_path):
                content_path = os.path.join(folder_path, "content.txt")
                source_path = os.path.join(folder_path, "source.json")

                # 读取内容和来源
                if os.path.exists(content_path) and os.path.exists(source_path):
                    with open(content_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    with open(source_path, "r", encoding="utf-8") as f:
                        source = f.read().strip()

                    docs.append({
                        "topic": folder_name,
                        "content": content,
                        "source": source
                    })
        print(f"✅ 成功将 {len(docs)} 篇顶级学术论文装载进 Agent 大脑！\n")
        return docs

    def _load_qa_benchmark(self):
        """加载本地 QA 题库作为辅助检索索引，不作为外部知识源。"""
        qa_path = ENHANCED_BENCHMARK_FILE if ENHANCED_BENCHMARK_FILE.exists() else BENCHMARK_FILE

        if not qa_path.exists():
            print("⚠️ 未找到 QA 评测基准文件。")
            return []

        try:
            with open(qa_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"✅ 成功加载 {len(data)} 道 QA 评测题作为辅助检索索引。\n")
            return data
        except Exception as e:
            print(f"⚠️ QA 评测基准读取失败: {e}")
            return []

    def _extract_query_terms(self, query):
        """提取英文、领域短语与中文 n-gram，避免把整句中文当成一个关键词。"""
        query_lower = query.lower()

        terms = re.findall(r"[a-z0-9][a-z0-9\-+.#]*", query_lower)
        domain_terms = [
            "图像处理", "机器学习", "深度学习", "数据挖掘", "医学图像", "医学图像分割",
            "图像分类", "语义分割", "特征提取", "预处理", "归一化", "训练", "测试",
            "评估", "流程", "cnn", "rnn", "unet", "u-net", "mamba", "transformer",
            "diffusion", "pytorch", "mapreduce", "信息检索"
        ]

        for term in domain_terms:
            if term.lower() in query_lower or term in query:
                terms.append(term.lower())

        for chunk in re.findall(r"[\u4e00-\u9fa5]+", query):
            for n in (2, 3, 4):
                for i in range(max(0, len(chunk) - n + 1)):
                    terms.append(chunk[i:i + n])

        return list(dict.fromkeys([term for term in terms if len(term.strip()) >= 2]))

    def _score_text(self, text, terms):
        text_lower = text.lower()
        score = 0
        for term in terms:
            term_lower = term.lower()
            if term_lower in text_lower:
                weight = 3 if len(term_lower) >= 4 else 1
                score += text_lower.count(term_lower) * weight
        return score

    def _retrieve_relevant_context(self, query, top_k=3):
        """本地 RAG 检索：知识资产正文 + 本地 QA 题库辅助索引。"""
        terms = self._extract_query_terms(query)
        doc_by_topic = {doc["topic"]: doc for doc in self.knowledge_docs}
        topic_scores = {doc["topic"]: 0 for doc in self.knowledge_docs}

        for doc in self.knowledge_docs:
            searchable_text = " ".join([
                doc.get("topic", ""),
                doc.get("content", ""),
                doc.get("source", "")
            ])
            topic_scores[doc["topic"]] += self._score_text(searchable_text, terms)

        for item in self.qa_items:
            reference = item.get("reference", {})
            topic = reference.get("topic") or item.get("reference_topic")
            if not topic or topic not in doc_by_topic:
                continue

            qa_text = " ".join([
                item.get("question", ""),
                item.get("answer", ""),
                topic,
                " ".join(reference.get("keywords", [])) if isinstance(reference.get("keywords"), list) else "",
                reference.get("citation_apa", "")
            ])
            qa_score = self._score_text(qa_text, terms)
            if qa_score > 0:
                topic_scores[topic] += qa_score * 2

        ranked_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_by_topic[topic] for topic, score in ranked_topics[:top_k] if score > 0]


    def answer_question(self, user_input):
        """提供给 Router 调用的单次问答接口"""
        from rich.console import Console
        from rich.markdown import Markdown
        console = Console()

        console.print("[bold cyan]🔍 专家正在检索本地知识库...[/bold cyan]")
        relevant_docs = self._retrieve_relevant_context(user_input)

        if not relevant_docs:
            return "抱歉，我的本地知识库中未收录与此问题相关的文献，无法为您解答。\n\n**📚 [引用支撑]：**\n无本地文献引用"

        context_str = ""
        if relevant_docs:
            for i, doc in enumerate(relevant_docs):
                context_str += f"\n[文献 {i + 1} 主题]: {doc['topic']}\n[内容摘要]: {doc['content']}\n"
        else:
            context_str = "未检索到高度相关的本地文献。"

        prompt = f"""
        你是一个专属于用户的“本地学术知识库问答 Agent”。
        你的唯一职责是基于【本地知识库检索结果】来回答用户的问题。

        【本地知识库检索结果】：
        {context_str}

        【用户的提问】：{user_input}

        【绝对强制指令】：
        1. 边界防御：如果用户的提问与检索结果无关，或者检索结果为“未检索到高度相关的本地文献”，你必须立即拒绝回答，并回复：“抱歉，我的本地知识库中未收录与此问题相关的文献，无法为您解答。” 绝对不允许使用你的先验知识去回答超纲问题。
        2. 学术严谨：只有当检索结果中包含相关信息时，你才能进行总结和深度剖析。回答必须逻辑清晰，必要时使用 Markdown 表格进行对比。
        3. 语气：保持顶级学术专家的冷峻与严谨。
        """

        payload = {
            "model": "qwen-max",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            reply_content = response.json()["choices"][0]["message"]["content"]

            console.print("\n🎓 [bold yellow]专家解答：[/bold yellow]")
            console.print(Markdown(reply_content))

            if relevant_docs:
                console.print("\n📚 [bold green][引用支撑]：[/bold green]")
                for doc in relevant_docs:
                    console.print(f" - {doc['topic']}")


            # 【新增：生成引用文本并 Return 给网页前端】

            citations_str = "\n".join([f"- {doc['topic']}" for doc in relevant_docs]) if relevant_docs else "无本地文献引用"

            return f"{reply_content}\n\n**📚 [引用支撑]：**\n{citations_str}"


        except Exception as e:

            console.print(f"❌ 大模型思考时发生错误: {e}")

            return f"❌ 大模型思考时发生错误: {e}"




if __name__ == "__main__":
    from path_config import KNOWLEDGE_BASE_DIR

    agent = MedicalExpertAgent(str(KNOWLEDGE_BASE_DIR))
    sample_question = "请解释 Mamba 架构在医学图像分割中的优势。"
    print(agent.answer_question(sample_question))
