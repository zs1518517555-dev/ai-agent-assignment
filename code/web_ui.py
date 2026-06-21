import json
import os
import re
from pathlib import Path

import gradio as gr
import numpy as np
import requests
from dotenv import load_dotenv

from expert_qa_agent import MedicalExpertAgent
from medical_agent import MedicalDataMinerAgent
from model_adapters import MedSAMAdapter, UnconfiguredSegmenter
from path_config import DEFAULT_NPC_DATASET_DIR, KNOWLEDGE_BASE_DIR
from segmentation_visualizer import (
    clamp_slice_index,
    list_available_cases,
    load_case,
    render_case,
    slice_count,
)

load_dotenv(Path(__file__).resolve().parent / ".env")
API_KEY = os.getenv("API_KEY")

print("🔄 正在唤醒底层引擎与本地知识库...")
medical_worker = MedicalDataMinerAgent()
knowledge_path = str(KNOWLEDGE_BASE_DIR)
qa_expert = MedicalExpertAgent(knowledge_path)
model_segmenter = MedSAMAdapter()


def router_and_execute(user_message, history):
    """大管家：中央路由与意图分发。"""
    prompt = f"""
    你是一个中央路由大管家。请判断用户输入的意图：
    1. 提问、请教知识、探讨概念（如：解释什么是...） -> 返回 JSON: {{"intent": "QA"}}
    2. 执行动作、处理数据集、清洗数据（如：帮我处理npc_dataset_nii数据集） -> 返回 JSON: {{"intent": "ACTION"}}
    用户输入：“{user_message}”
    """

    payload = {"model": "qwen-max", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        intent = json.loads(match.group(0)).get("intent", "QA") if match else "QA"
    except Exception:
        intent = "QA"

    if intent == "ACTION":
        medical_worker.chat_and_execute(user_message)
        return (
            "🚀 **[系统通知：成功唤醒数据矿工]**\n\n"
            "✅ 医疗影像预处理与特征工程流水线已触发！\n\n"
            "请查看本地代码窗口的终端日志获取详细进度，特征张量文件将自动保存至本地文件夹。"
        )

    expert_reply = qa_expert.answer_question(user_message)
    return f"🎓 **[本地知识库专家]**\n\n{expert_reply}"


def user_chat(user_msg, chat_history):
    if not user_msg or not user_msg.strip():
        return "", chat_history
    bot_response = router_and_execute(user_msg, chat_history)
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": bot_response})
    return "", chat_history


def _blank_image(size=256):
    return np.zeros((size, size, 3), dtype=np.uint8)


def _metrics_markdown(result, mask_type):
    metrics = result["metrics"]
    missing_note = ""
    if result["missing_mask"]:
        missing_note = f"\n\n> 当前病例未找到“{mask_type}”文件，已显示空 mask。"
    return f"""
### Dice / IoU

| 对比范围 | Dice | IoU |
|---|---:|---:|
| 当前切片 | {metrics["slice_dice"]:.4f} | {metrics["slice_iou"]:.4f} |
| 3D 体数据 | {metrics["volume_dice"]:.4f} | {metrics["volume_iou"]:.4f} |

病例体数据尺寸：`{result["shape"]}`  
当前切片编号：`{result["index"]}` / `{result["max_index"]}`{missing_note}
"""


def render_visualization(dataset_dir, case_id, axis, slice_index, mask_type, alpha):
    try:
        result = render_case(
            case_id=case_id,
            axis=axis,
            index=slice_index,
            mask_type=mask_type,
            alpha=alpha,
            dataset_dir=dataset_dir,
        )
        status = f"✅ 已显示病例 {case_id} 的 {axis} 切片。"
        return result["image"], result["mask"], result["overlay"], _metrics_markdown(result, mask_type), status
    except Exception as exc:
        message = f"❌ 可视化失败：{exc}"
        return _blank_image(), _blank_image(), _blank_image(), "### Dice / IoU\n暂无可用指标。", message


def refresh_case_list(dataset_dir):
    cases = list_available_cases(dataset_dir)
    if not cases:
        return gr.update(choices=[], value=None), "⚠️ 没有找到同时包含 image_processed.npy 和 mask_processed.npy 的病例。"
    return gr.update(choices=cases, value=cases[0]), f"✅ 已发现 {len(cases)} 个可视化病例。"


def update_case_or_axis(dataset_dir, case_id, axis, mask_type, alpha):
    try:
        image, _ = load_case(case_id, dataset_dir)
        max_index = slice_count(image.shape, axis) - 1
        value = clamp_slice_index(image.shape, axis, max_index // 2)
        result = render_case(case_id, axis=axis, index=value, mask_type=mask_type, alpha=alpha, dataset_dir=dataset_dir)
        status = f"✅ 已切换到病例 {case_id}，方向 {axis}。"
        return (
            gr.update(maximum=max_index, value=value),
            result["image"],
            result["mask"],
            result["overlay"],
            _metrics_markdown(result, mask_type),
            status,
        )
    except Exception as exc:
        return (
            gr.update(maximum=0, value=0),
            _blank_image(),
            _blank_image(),
            _blank_image(),
            "### Dice / IoU\n暂无可用指标。",
            f"❌ 切换失败：{exc}",
        )


def _raw_slice(volume, axis, index):
    axis = (axis or "axial").lower()
    if axis == "axial":
        return volume[:, :, index]
    if axis == "coronal":
        return volume[:, index, :]
    return volume[index, :, :]


def _assign_raw_slice(volume, axis, index, slice_data):
    axis = (axis or "axial").lower()
    if axis == "axial":
        volume[:, :, index] = slice_data
    elif axis == "coronal":
        volume[:, index, :] = slice_data
    else:
        volume[index, :, :] = slice_data


def run_model_prediction(dataset_dir, case_id, axis, slice_index, alpha):
    try:
        if not model_segmenter.available():
            return (
                gr.update(),
                _blank_image(),
                _blank_image(),
                _blank_image(),
                "### Dice / IoU\n暂无可用指标。",
                f"⚠️ {model_segmenter.last_error}\n\n请先运行 MedSAM 依赖与权重下载脚本，然后再使用模型预测。",
            )

        image, gt_mask = load_case(case_id, dataset_dir)
        axis = (axis or "axial").lower()
        index = clamp_slice_index(image.shape, axis, slice_index)
        image_slice = _raw_slice(image, axis, index)
        gt_slice = _raw_slice(gt_mask, axis, index)

        box = model_segmenter.box_from_mask(gt_slice)
        if box is None:
            box = model_segmenter.default_box(image_slice)

        pred_slice = model_segmenter.predict_slice(image_slice, box=box)
        pred_volume = np.zeros_like(gt_mask, dtype=np.uint8)
        _assign_raw_slice(pred_volume, axis, index, pred_slice)

        root = Path(dataset_dir) if dataset_dir else Path(DEFAULT_NPC_DATASET_DIR)
        output_path = root / str(case_id) / "prediction_medsam_slice.npy"
        np.save(output_path, pred_volume)

        result = render_case(
            case_id=case_id,
            axis=axis,
            index=index,
            mask_type="模型预测",
            alpha=alpha,
            dataset_dir=dataset_dir,
        )
        status = f"✅ MedSAM 已完成当前切片预测，box={box}，结果保存为 {output_path.name}。"
        return (
            gr.update(value="模型预测"),
            result["image"],
            result["mask"],
            result["overlay"],
            _metrics_markdown(result, "模型预测"),
            status,
        )
    except Exception as exc:
        return (
            gr.update(),
            _blank_image(),
            _blank_image(),
            _blank_image(),
            "### Dice / IoU\n暂无可用指标。",
            f"❌ MedSAM 推理失败：{exc}",
        )


case_choices = list_available_cases(DEFAULT_NPC_DATASET_DIR)
initial_case = case_choices[0] if case_choices else None
initial_slider_max = 0
initial_slider_value = 0
if initial_case:
    try:
        initial_image, _ = load_case(initial_case, DEFAULT_NPC_DATASET_DIR)
        initial_slider_max = slice_count(initial_image.shape, "axial") - 1
        initial_slider_value = initial_slider_max // 2
    except Exception:
        initial_slider_max = 0
        initial_slider_value = 0


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; padding: 18px;">
            <h1>🧠 3D医疗影像智能全能助理 (Medical AI Agent)</h1>
            <h3>本地知识库 RAG + NIfTI 数据处理 + 分割可视化</h3>
            <p>系统支持学术问答、医学影像预处理、分割标签可视化、Dice/IoU 评估与模型预测。</p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("智能问答与数据处理"):
            with gr.Row():
                with gr.Column(scale=8):
                    chatbot = gr.Chatbot(
                        height=500,
                        avatar_images=("👤", "🤖"),
                        label="全能 AI 助理交互终端",
                    )
                    msg = gr.Textbox(
                        label="请输入您的学术问题或数据处理指令：",
                        placeholder="例如：解释什么是 Mamba 架构？或：帮我批量清洗 npc_dataset_nii 数据集。",
                    )

                    with gr.Row():
                        submit_btn = gr.Button("🚀 发送指令", variant="primary")
                        clear_btn = gr.ClearButton([msg, chatbot], value="🗑️ 清除对话记录")

                with gr.Column(scale=2):
                    gr.Markdown(
                        f"""
                        ### ⚙️ 系统运行状态
                        - **路由大管家**: <span style="color:green">在线 (Online)</span>
                        - **RAG 专家引擎**: <span style="color:green">已挂载 {len(qa_expert.knowledge_docs)} 个知识资产</span>
                        - **评测题索引**: <span style="color:green">已加载 {len(qa_expert.qa_items)} 道 QA</span>
                        - **数据挖掘矿工**: <span style="color:green">待机中 (Standby)</span>
                        - **分割可视化**: <span style="color:green">已发现 {len(case_choices)} 个可用病例</span>
                        """
                    )

            msg.submit(user_chat, [msg, chatbot], [msg, chatbot])
            submit_btn.click(user_chat, [msg, chatbot], [msg, chatbot])

        with gr.Tab("分割可视化与模型预测"):
            with gr.Row():
                with gr.Column(scale=3):
                    dataset_box = gr.Textbox(
                        value=str(DEFAULT_NPC_DATASET_DIR),
                        label="数据集目录",
                    )
                    refresh_btn = gr.Button("🔄 刷新病例列表")
                    case_dropdown = gr.Dropdown(
                        choices=case_choices,
                        value=initial_case,
                        label="病例编号",
                    )
                    axis_radio = gr.Radio(
                        choices=["axial", "coronal", "sagittal"],
                        value="axial",
                        label="切片方向",
                    )
                    slice_slider = gr.Slider(
                        minimum=0,
                        maximum=initial_slider_max,
                        value=initial_slider_value,
                        step=1,
                        label="切片编号",
                    )
                    mask_type_radio = gr.Radio(
                        choices=["真实 mask", "模型预测"],
                        value="真实 mask",
                        label="mask 类型",
                    )
                    alpha_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.45,
                        step=0.05,
                        label="叠加透明度",
                    )

                    model_predict_btn = gr.Button("🧩 运行模型预测")
                    status_md = gr.Markdown("等待选择病例。")

                with gr.Column(scale=7):
                    with gr.Row():
                        image_out = gr.Image(label="原图切片", type="numpy", height=300)
                        mask_out = gr.Image(label="mask 切片", type="numpy", height=300)
                        overlay_out = gr.Image(label="overlay 叠加图", type="numpy", height=300)
                    metrics_md = gr.Markdown("### Dice / IoU\n暂无可用指标。")

            refresh_btn.click(refresh_case_list, [dataset_box], [case_dropdown, status_md])

            case_dropdown.change(
                update_case_or_axis,
                [dataset_box, case_dropdown, axis_radio, mask_type_radio, alpha_slider],
                [slice_slider, image_out, mask_out, overlay_out, metrics_md, status_md],
            )
            axis_radio.change(
                update_case_or_axis,
                [dataset_box, case_dropdown, axis_radio, mask_type_radio, alpha_slider],
                [slice_slider, image_out, mask_out, overlay_out, metrics_md, status_md],
            )

            for component in [slice_slider, mask_type_radio, alpha_slider]:
                component.change(
                    render_visualization,
                    [dataset_box, case_dropdown, axis_radio, slice_slider, mask_type_radio, alpha_slider],
                    [image_out, mask_out, overlay_out, metrics_md, status_md],
                )

            model_predict_btn.click(
                run_model_prediction,
                [dataset_box, case_dropdown, axis_radio, slice_slider, alpha_slider],
                [mask_type_radio, image_out, mask_out, overlay_out, metrics_md, status_md],
            )

            demo.load(
                update_case_or_axis,
                [dataset_box, case_dropdown, axis_radio, mask_type_radio, alpha_slider],
                [slice_slider, image_out, mask_out, overlay_out, metrics_md, status_md],
            )


if __name__ == "__main__":
    demo.launch(inbrowser=True)

