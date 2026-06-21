# Demo Log

## Environment

- OS: Windows
- Python: 3.11
- UI framework: Gradio
- Project topic: `medical-image-segmentation`

## Observed Runtime Behavior

The local application starts from:

```bash
cd code
python web_ui.py
```

The Gradio UI runs locally at:

```text
http://127.0.0.1:7860
```

Startup logs observed during testing:

```text
Loading local academic knowledge base...
Successfully loaded 75 knowledge assets.
Successfully loaded 81 QA benchmark items.
Running on local URL: http://127.0.0.1:7860
```

## Demonstrated Functions

- Local knowledge Q&A
- NIfTI preprocessing workflow
- Segmentation slice browsing
- Mask overlay visualization
- Dice / IoU reporting
- MedSAM adapter status and inference path

## Demo Assets

- Video: `demo/Medical_AI_Agent_Demo_20260621_195154.mp4`
- Screenshots: `screenshots/`
