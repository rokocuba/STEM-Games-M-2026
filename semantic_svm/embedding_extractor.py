from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

FAST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BGE_SMALL_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LARGE_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
DEFAULT_MODEL_NAME = FAST_MODEL_NAME
MODEL_PRESETS = {
    "fast": FAST_MODEL_NAME,
    "minilm": FAST_MODEL_NAME,
    "bge-small": BGE_SMALL_MODEL_NAME,
    "bge": BGE_SMALL_MODEL_NAME,
    "large": LARGE_MODEL_NAME,
    "mixedbread": LARGE_MODEL_NAME,
}


def resolve_model_name(model_name: str) -> str:
    return MODEL_PRESETS.get(model_name.lower(), model_name)


def _batched_texts(texts: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    for start in range(0, len(texts), batch_size):
        yield list(texts[start : start + batch_size])


def _text_chunks(
    csv_path: Path,
    text_column: str,
    chunksize: int,
    max_rows: int | None,
) -> Iterator[list[str]]:
    rows_left = max_rows

    for chunk in pd.read_csv(csv_path, usecols=[text_column], chunksize=chunksize):
        if rows_left is not None:
            if rows_left <= 0:
                break
            chunk = chunk.head(rows_left)
            rows_left -= len(chunk)

        texts = chunk[text_column].fillna("").astype(str).tolist()
        if texts:
            yield texts


def count_rows(
    csv_path: str | Path,
    text_column: str,
    chunksize: int,
    max_rows: int | None = None,
    progress: bool = True,
) -> int:
    csv_path = Path(csv_path)
    total = 0

    with tqdm(
        desc=f"Counting rows in {csv_path.name}",
        unit="rows",
        dynamic_ncols=True,
        disable=not progress,
    ) as bar:
        for texts in _text_chunks(csv_path, text_column, chunksize, max_rows):
            total += len(texts)
            bar.update(len(texts))

    return total


class EmbeddingExtractor:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        cache_folder: str | Path | None = None,
    ) -> None:
        self.model_name = resolve_model_name(model_name)
        cache = str(cache_folder) if cache_folder is not None else None
        self.model = SentenceTransformer(
            self.model_name, device=device, cache_folder=cache
        )
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension

        if hasattr(self.model, "get_embedding_dimension"):
            dimension = self.model.get_embedding_dimension()
        else:
            dimension = self.model.get_sentence_embedding_dimension()

        if dimension is None:
            dimension = self.model.encode([""], convert_to_numpy=True).shape[1]

        self._dimension = int(dimension)
        return self._dimension

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )
        return embeddings.astype(np.float32, copy=False)

    def embed_csv(
        self,
        csv_path: str | Path,
        cache_path: str | Path,
        text_column: str = "comment",
        batch_size: int = 64,
        chunksize: int = 4096,
        max_rows: int | None = None,
        normalize: bool = True,
        overwrite: bool = False,
        progress: bool = True,
    ) -> np.ndarray:
        csv_path = Path(csv_path)
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and not overwrite:
            print(f"Using cached embeddings: {cache_path}", flush=True)
            return np.load(cache_path, mmap_mode="r")

        total_rows = count_rows(
            csv_path, text_column, chunksize, max_rows, progress=progress
        )
        if total_rows == 0:
            raise ValueError(f"No rows found in {csv_path}")

        temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        if temp_path.exists():
            temp_path.unlink()

        embeddings = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, self.dimension),
        )

        offset = 0
        with tqdm(
            total=total_rows,
            desc=f"Embedding {csv_path.name}",
            unit="rows",
            dynamic_ncols=True,
            disable=not progress,
        ) as bar:
            for chunk_index, texts in enumerate(
                _text_chunks(csv_path, text_column, chunksize, max_rows), start=1
            ):
                chunk_batches = max(1, (len(texts) + batch_size - 1) // batch_size)
                for batch_index, batch_texts in enumerate(
                    _batched_texts(texts, batch_size), start=1
                ):
                    bar.set_postfix_str(
                        f"chunk {chunk_index}, batch {batch_index}/{chunk_batches}, batch_size {len(batch_texts)}"
                    )
                    encoded = self.encode(
                        batch_texts, batch_size=batch_size, normalize=normalize
                    )
                    next_offset = offset + len(encoded)
                    embeddings[offset:next_offset] = encoded
                    offset = next_offset
                    bar.update(len(encoded))

        embeddings.flush()
        del embeddings
        temp_path.replace(cache_path)
        return np.load(cache_path, mmap_mode="r")
