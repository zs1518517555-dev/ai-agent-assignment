from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_assets" / "data"
BENCHMARK_FILE = PROJECT_ROOT / "benchmark" / "qa_dataset.json"
ENHANCED_BENCHMARK_FILE = PROJECT_ROOT / "benchmark" / "qa_dataset_enhanced.json"
DEFAULT_NPC_DATASET_DIR = CODE_DIR / "npc_dataset_nii"


def resolve_project_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidates = [CODE_DIR / path, PROJECT_ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(CODE_DIR / path)
