# Optional Module Selection

The assignment requires the selected optional modules to have a total difficulty of at least 4.

This project selects four Lv.1 modules:

| Category | Module | Level | Implementation Evidence |
| --- | --- | ---: | --- |
| Memory | Expand Knowledge Assets | 1 | 75 curated knowledge assets, exceeding the 20-set foundation requirement by 55 sets |
| Memory | Agentic RAG | 1 | `expert_qa_agent.py` retrieves local knowledge and QA examples before generation |
| Orchestration | Basic Static Orchestration | 1 | `web_ui.py` routes user tasks to multiple specialized modules through a fixed workflow |
| Skills | Deploy / Integrate Tool or Skill | 1 | `model_adapters/medsam_adapter.py` and `segmentation_visualizer.py` expose reusable domain tools |

Total difficulty:

```text
1 + 1 + 1 + 1 = 4
```

## Why These Modules Match the Project

### Expand Knowledge Assets

The project contains 75 knowledge assets under `knowledge_assets/data/`. Each asset has:

- `content.txt`
- `keywords.json`
- `source.json`

### Agentic RAG

The expert agent loads:

- local knowledge assets,
- QA benchmark examples,
- source metadata.

It retrieves relevant local evidence before asking the model to answer.

### Basic Static Orchestration

The Gradio UI provides a fixed task workflow:

1. Expert QA
2. NIfTI preprocessing
3. Segmentation visualization
4. MedSAM prediction

### Tool / Skill Integration

The project integrates reusable domain-specific tools:

- NIfTI preprocessing utility,
- segmentation renderer,
- MedSAM adapter.
