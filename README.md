# Deep Learning & Life Sciences — AI Agent Course Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A course project assignment for "Deep Learning and Life Sciences", focusing on building an AI Agent-based knowledge Q&A platform for protein corona research.

## 📋 Overview

This repository contains the complete assignment specifications for a course project where students build an intelligent Q&A system using multi-agent collaboration. The project is divided into **mandatory foundational modules** and **optional advanced modules** with three difficulty levels.

**Total Score: 100 points**
- Knowledge Assets (35 points)
- Evaluation Benchmark (35 points)
- Knowledge Q&A Project (30 points)

## 📁 Repository Structure

```
.
├── README.md                          # This file
├── ch/                                # Chinese version
│   ├── 评分标准.md                     # Grading criteria
│   ├── 基础模块/
│   │   ├── 知识资产.md
│   │   ├── 评测基准.md
│   │   └── 知识问答项目.md
│   └── 可选模块/
│       ├── 技能.md
│       ├── 编排.md
│       └── 记忆.md
│
└── en/                                # English version
    ├── grading_criteria.md
    ├── foundational_modules/
    │   ├── knowledge_asset.md
    │   ├── benchmark.md
    │   └── knowledge_qa_project.md
    └── optional_modules/
        ├── skills.md
        ├── orchestration.md
        └── memory.md
```

## 🎯 Foundational Modules (Mandatory)

All students must complete the following three modules:

### 1. Knowledge Assets (35 points)
Collect and organize ≥20 knowledge sets related to computational biology and deep learning. Each knowledge set includes:
- `content/` — Original content (papers, web pages, etc.)
- `keywords.json` — Keywords and topic classification
- `source.json` — Source metadata (DOI, URL, etc.)

**Research Topic Themes**: Students can select one of the following topics as their research direction:

| Topic | Description |
|------|------|
| `protein-corona` | Protein Corona |
| `single-cell-omics` | Single-cell Omics |
| `genomics-sequencing` | Genomics and Sequencing |
| `gene-perturbation` | Gene Perturbation |
| `drug-discovery` | Drug Discovery |
| `protein-structure` | Protein Structure |
| `neural-network` | Neural Networks |
| `graph-neural-network` | Graph Neural Networks |
| `generative-model` | Generative Models |
| `foundation-model` | Foundation Models |
| `multi-omics` | Multi-omics Integration |
| `spatial-biology` | Spatial Biology |
| `epigenomics` | Epigenomics |
| `biomedical-nlp` | Biomedical Natural Language Processing |
| `molecular-dynamics` | Molecular Dynamics |
| `medical-image-reconstruction` | Medical Image Reconstruction |
| `medical-image-segmentation` | Medical Image Segmentation |
| `medical-image-classification` | Medical Image Classification/Recognition |
| `medical-image-registration` | Medical Image Registration |
| `medical-image-synthesis` | Medical Image Synthesis |

Students can define custom keywords (using kebab-case format), but topics should align with the selected research direction.

### 2. Evaluation Benchmark (35 points)
Build a test suite with ≥20 questions covering the knowledge base. Each question includes:
- Question text and standard answer
- Source citation
- Topic classification
- Difficulty rating

### 3. Knowledge Q&A Project (30 points)
Implement an intelligent Q&A system by selecting optional modules.

**Module Selection Rule**: The sum of difficulty values of selected modules must be **≥ 4**.

Examples of valid combinations:
- One Lv.2 + one Lv.2 (2 + 2 = 4)
- One Lv.1 + one Lv.3 (1 + 3 = 4)
- Two Lv.1 + one Lv.2 (1 + 1 + 2 = 4)
- One Lv.1 + two Lv.2 + one Lv.3 (1 + 2 + 2 + 3 = 8) — exceeds minimum, allowed

The final score is calculated as:
```
Weighted Total = Σ(Module Score × Module Weight)
Knowledge Q&A Score = Weighted Total × 0.3
```

## 🚀 Optional Modules (Self-selected)

Students choose tasks from three categories. Each module has a difficulty level (1=Basic, 2=Intermediate, 3=Challenge). **The sum of selected modules' difficulty values must be at least 4.**

### 🛠️ Skill System (`技能.md` / `skills.md`)
| Module | Level | Description |
|--------|-------|-------------|
| Deploy MCP Tool/Skill | Lv.1 | Integrate an existing MCP tool into the platform |
| Build Domain MCP/Skill | Lv.2 | Create MCP interfaces for bioinformatics tools (UniProt, PDB, BLAST, etc.) |
| Large-scale Skill Routing | Lv.2 | Implement a router for ≥2000 skills with semantic search |
| Self-Evolving Skill System | Lv.2 | Extract patterns from usage logs to auto-generate new skills |

### 🔀 Workflow Orchestration (`编排.md` / `orchestration.md`)
| Module | Level | Description |
|--------|-------|-------------|
| Basic Static Orchestration | Lv.1 | Fixed multi-agent collaboration workflow (≥4 agents) |
| Basic Dynamic Orchestration | Lv.1 | Dynamic agent creation and parallel execution |
| Advanced Static Orchestration | Lv.2 | Pre-defined workflow templates with JSON/YAML config |
| Advanced Dynamic Orchestration | Lv.3 | Auto-generated workflows with planning and error recovery |

### 🧠 Memory System (`记忆.md` / `memory.md`)
| Module | Level | Description |
|--------|-------|-------------|
| Expand Knowledge Assets | Lv.1 | Add ≥30 new knowledge sets with coverage analysis |
| Structured Knowledge Base | Lv.1 | Build entity-relation graphs for knowledge retrieval |
| Agentic RAG | Lv.1 | Implement iterative retrieval with self-correction |
| Persistent Memory System | Lv.1 | Multi-layer memory (working/episodic/semantic) |
| Multi-hop Reasoning | Lv.2 | Complex multi-step retrieval with verification loops |
| Multimodal Knowledge Base | Lv.2 | Support text/image/table modalities |
| Self-Evolving Memory | Lv.3 | Auto-extract experience units and optimize memory structure |

## 📊 Grading

See [评分标准](ch/评分标准.md) or [grading_criteria](en/grading_criteria.md) for detailed grading criteria.

**Key Principles:**
- Foundational modules focus on completeness and quality
- Optional modules focus on functionality and demonstrable results
- Higher difficulty modules carry more weight in the final score
- Code must be runnable; documentation must be complete

## 🛠️ Tech Stack (Recommended)

- **Agent Framework**: Agno, LangChain, LangGraph, CrewAI, or AutoGen
- **Knowledge Base**: Vector DB (Chroma, Weaviate, Milvus) or Graph DB (Neo4j)
- **MCP Protocol**: For tool integration and skill standardization
- **Evaluation**: Custom benchmark suite with automated scoring

## 📚 References

Key papers referenced across modules:
- **SkillRouter** (2026) — Large-scale skill routing
- **AAFLOW** (2026) — Agentic workflow patterns
- **Graph RAG** (2024) / **LinearRAG** (2025) — Knowledge retrieval
- **Agentic RAG Survey** (2025) — Iterative retrieval methods
- **Foundation Agents Memory Survey** (2026) — Memory architectures
- **MSTAR** (2026) — Self-evolving memory systems

## 📝 Submission

1. Fork this repository
2. Complete the foundational modules in your fork
3. Select and implement optional modules
4. Submit a Pull Request with:
   - All code and documentation
   - Evaluation benchmark results
   - Demo screenshots or logs

## 📧 Contact

For questions about the assignment, please open an issue in this repository.
