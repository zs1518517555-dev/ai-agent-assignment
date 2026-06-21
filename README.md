# 3D Medical Image Intelligent Agent

Course project submission for **Deep Learning & Life Sciences — AI Agent Course Project**.

## Topic

`medical-image-segmentation`

This project builds a local AI agent for 3D medical image analysis. It combines:

- evidence-grounded medical knowledge Q&A,
- NIfTI medical image preprocessing,
- segmentation visualization and Dice/IoU evaluation,
- MedSAM model adapter integration.

## Assignment Completion Summary

| Required Part | Requirement | Completed |
| --- | --- | --- |
| Knowledge Assets | At least 20 valid knowledge sets | 75 knowledge sets |
| Evaluation Benchmark | At least 20 QA test questions | 81 QA items |
| Knowledge Q&A Project | Optional modules with total difficulty >= 4 | Total selected difficulty = 4 |

## Optional Modules Selected

| Category | Module | Level | Evidence in this project |
| --- | --- | ---: | --- |
| Memory | Expand Knowledge Assets | 1 | 75 curated knowledge assets with coverage analysis |
| Memory | Agentic RAG | 1 | Local retrieval-based expert QA with benchmark-assisted context |
| Orchestration | Basic Static Orchestration | 1 | Gradio UI routes tasks to QA, data processing, visualization, and MedSAM prediction modules |
| Skills | Deploy / Integrate Tool or Skill | 1 | MedSAM adapter and segmentation utility tools integrated into the workflow |

Total difficulty: `1 + 1 + 1 + 1 = 4`.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── code/
│   ├── web_ui.py
│   ├── expert_qa_agent.py
│   ├── medical_agent.py
│   ├── segmentation_visualizer.py
│   ├── path_config.py
│   └── model_adapters/
├── knowledge_assets/
│   └── data/
├── benchmark/
│   ├── qa_dataset.json
│   └── qa_dataset_enhanced.json
├── docs/
├── results/
├── screenshots/
└── demo/
```

## How to Run

1. Install Python 3.11.

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Create `code/.env` from `code/.env.example`:

```text
API_KEY=your_api_key_here
```

4. Start the web UI:

```bash
cd code
python web_ui.py
```

5. Open the local Gradio URL printed in the terminal, usually:

```text
http://127.0.0.1:7860
```

## Main Features

- **Expert QA Agent**: loads local knowledge assets and QA examples for evidence-grounded answers.
- **NIfTI Processing**: converts raw medical image and mask volumes into standardized NumPy arrays.
- **Segmentation Workspace**: supports slice browsing, mask overlay, and Dice/IoU metrics.
- **MedSAM Adapter**: supports optional MedSAM checkpoint inference when `medsam_vit_b.pth` is available.

## Data and Model Weight Policy

Large medical imaging files and model checkpoints are intentionally excluded from the Git submission.

Excluded examples:

- `code/npc_dataset_nii/`
- `model_weights/`
- `*.nii.gz`
- `*.npy`
- `*.pth`

The code supports these files when placed locally, but they should not be committed to GitHub.

## Validation

See:

- `results/validation_report.md`
- `results/demo_log.md`
- `docs/optional_modules.md`
- `docs/coverage_analysis.md`

## Demo Materials

- Editable presentation: `docs/Medical_AI_Agent_Editable_With_Video_Placeholder.pptx`
- Bilingual speech script: `docs/Medical_AI_Agent_Bilingual_Speech.txt`
- Demo video: `demo/Medical_AI_Agent_Demo_20260621_195154.mp4`
- Screenshots: `screenshots/`
