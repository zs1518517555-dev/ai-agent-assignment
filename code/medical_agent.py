import os
import json
import requests
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from dotenv import load_dotenv
from path_config import resolve_project_path

# 加载 API Key
load_dotenv()
API_KEY = os.getenv("API_KEY")


class MedicalDataMinerAgent:
    def __init__(self):
        # 如果你使用的是智谱，请替换为智谱的 URL 和模型名
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

    # --------- 数据挖掘底层工具 1: 单个 NIfTI 处理 ---------
    def preprocess_nifti(self, file_path, target_shape=(128, 128, 64), is_label=False):
        """核心特征工程：重采样与归一化（区分图像与标签）"""
        try:
            img = nib.load(file_path)
            data = img.get_fdata()

            # 1. 计算重采样比例
            resize_factor = (
                target_shape[0] / data.shape[0],
                target_shape[1] / data.shape[1],
                target_shape[2] / data.shape[2]
            )

            # 2. 执行空间重采样
            if is_label:
                # 标签必须用最近邻插值(order=0)，防止类别标签被破坏
                resampled_data = zoom(data, resize_factor, order=0)
                final_data = resampled_data
            else:
                # 图像用线性插值(order=1)
                resampled_data = zoom(data, resize_factor, order=1)
                # 图像执行 Z-score 归一化
                mean_val = np.mean(resampled_data)
                std_val = np.std(resampled_data)
                final_data = (resampled_data - mean_val) / (std_val + 1e-8)

            # 3. 保存预处理后的特征张量
            output_name = file_path.replace('.nii.gz', '_processed.npy').replace('.nii', '_processed.npy')
            np.save(output_name, final_data)
            return True

        except Exception as e:
            print(f"❌ [处理失败] {file_path}: {e}")
            return False

    # --------- 数据挖掘底层工具 2: 批量流水线 ---------
    def batch_process_dataset(self, dataset_dir):
        """遍历所有病人文件夹，自动识别并处理数据"""
        print(f"\n📂 [自动化流水线启动] 正在扫描数据集目录: {dataset_dir}")
        if not os.path.exists(dataset_dir):
            print("❌ 找不到指定的文件夹，请检查路径。")
            return

        patient_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
        print(f"🔍 共发现 {len(patient_folders)} 个病人数据文件夹。")

        success_count = 0
        for patient_id in patient_folders:
            patient_path = os.path.join(dataset_dir, patient_id)
            print(f"\n⚙️ 正在处理病人 [{patient_id}] 的数据...")

            # 遍历病人文件夹内的所有 nii/nii.gz 文件
            for file_name in os.listdir(patient_path):
                if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                    file_path = os.path.join(patient_path, file_name)

                    # 智能判断：如果文件名包含 label, mask, seg 等字眼，则判定为标签
                    is_label = any(keyword in file_name.lower() for keyword in ['label', 'mask', 'seg'])

                    type_str = "标签(Label)" if is_label else "图像(Image)"
                    print(f"   -> 发现{type_str}: {file_name}")

                    if self.preprocess_nifti(file_path, is_label=is_label):
                        print(f"   ✅ {type_str} 处理并保存成功。")
                        success_count += 1

        print(f"\n🎉 [流水线结束] 共成功处理并生成了 {success_count} 个张量特征文件！")

    # --------- Agent 大脑交互逻辑 ---------
    def chat_and_execute(self, user_instruction):
        print(f"\n👤 收到指令: {user_instruction}")

        prompt = f"""
        你是一个专门负责医学图像数据挖掘的 AI Agent。
        用户指令："{user_instruction}"

        请分析用户的意图：
        1. 如果用户要求处理整个数据集（比如处理某个文件夹下的所有病人数据），提取该文件夹名称，返回 JSON：
           {{"intent": "BATCH_PREPROCESS", "dataset_path": "提取的文件夹路径"}}
        2. 如果用户的指令不是关于数据处理的（比如询问知识），返回：
           {{"intent": "CHAT", "reply": "你的自然语言回复"}}

        请仅返回 JSON，不要任何多余文字。
        """

        payload = {
            "model": "qwen-max",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }

        try:
            print("🧠 正在请求云端大模型决策方案...")
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            reply_content = response.json()["choices"][0]["message"]["content"]

            # 清洗并解析大模型返回的 JSON
            json_str = reply_content.replace('```json', '').replace('```', '').strip()
            decision = json.loads(json_str)

            if decision.get("intent") == "BATCH_PREPROCESS":
                dataset_path = resolve_project_path(decision.get("dataset_path", ""))
                self.batch_process_dataset(dataset_path)
            else:
                print(f"🤖 Agent 回复: {decision.get('reply')}")

        except Exception as e:
            print(f"❌ 大模型解析错误: {e}")
            if 'response' in locals():
                print(f"详细错误: {response.text}")


if __name__ == "__main__":
    agent = MedicalDataMinerAgent()

    instruction = "帮我批量清洗和预处理当前目录下的 npc_dataset_nii 数据集，提取出可供深度学习训练的特征张量。"
    agent.chat_and_execute(instruction)