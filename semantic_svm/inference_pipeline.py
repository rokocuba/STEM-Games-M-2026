from __future__ import annotations

import json
from pathlib import Path

import joblib

try:
    from .embedding_extractor import (
        DEFAULT_MODEL_NAME,
        EmbeddingExtractor,
        resolve_model_name,
    )
    from .svm_classifier import AI_CLASS_ID
except ImportError:
    from embedding_extractor import (
        DEFAULT_MODEL_NAME,
        EmbeddingExtractor,
        resolve_model_name,
    )
    from svm_classifier import AI_CLASS_ID


COMPONENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = COMPONENT_DIR / "artifacts" / "semantic_svm.joblib"


class SemanticRageAnalyzer:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        embedding_model_name: str | None = None,
        device: str | None = None,
        model_cache_dir: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.classifier = joblib.load(self.model_path)
        metadata = self._load_metadata()
        model_name = resolve_model_name(
            str(
                embedding_model_name
                or metadata.get("embedding_model", DEFAULT_MODEL_NAME)
            )
        )
        self.extractor = EmbeddingExtractor(
            model_name=model_name, device=device, cache_folder=model_cache_dir
        )

    def _load_metadata(self) -> dict[str, object]:
        metadata_path = self.model_path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def predict_probability(self, text: str) -> float:
        embedding = self.extractor.encode([text], show_progress=False)
        classes = list(self.classifier.classes_)
        ai_index = classes.index(AI_CLASS_ID)
        return float(self.classifier.predict_proba(embedding)[0, ai_index])
