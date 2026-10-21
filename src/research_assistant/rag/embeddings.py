"""Embedding backends behind one small protocol.

* :class:`MiniLMEmbeddings`: all-MiniLM-L6-v2 through ONNX Runtime, the default. It needs
  no PyTorch, which keeps the image a few hundred MB instead of a few GB.
* :class:`HashingEmbeddings`: a deterministic bag-of-words hash for tests and ``--dry-run``,
  so everything runs with no model download.

Both return L2-normalized float32 vectors, so a dot product is the cosine similarity.
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path
from typing import List, Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Anything that can turn a list of texts into a normalised matrix."""

    name: str
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array of unit vectors."""
        ...


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class HashingEmbeddings:
    """Deterministic feature-hashing embeddings (no external model).

    Uses the signed hashing trick: each token is hashed to a bucket and a
    sign, which keeps collisions roughly unbiased. Good enough for tests and
    demos; not a substitute for a real embedding model.
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return bucket, sign

    def embed(self, texts: List[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                bucket, sign = self._hash(token)
                matrix[row, bucket] += sign
        return _normalize_rows(matrix)


MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
MINILM_FILES = ("onnx/model.onnx", "tokenizer.json")
MAX_TOKENS = 256


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_minilm(dest: Path, base: str = f"https://huggingface.co/{MINILM_ID}/resolve/main") -> Path:
    """Fetch the ONNX export and tokenizer; refuse weights that aren't the official ones."""
    dest = Path(dest)
    for name in MINILM_FILES:
        target = dest / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            urllib.request.urlretrieve(f"{base}/{name}", tmp)
            tmp.replace(target)
    got = _sha256(dest / "onnx" / "model.onnx")
    if got != MINILM_SHA256:
        raise RuntimeError(f"model.onnx has sha256 {got}, not the official {MINILM_SHA256}")
    return dest


class MiniLMEmbeddings:
    """all-MiniLM-L6-v2 through ONNX Runtime: no PyTorch, weights pinned by SHA-256.

    A text becomes a vector the way sentence-transformers does it for this model: lowercased
    WordPiece tokens between [CLS] and [SEP], cut at 256 tokens, mean pooling over the
    attention mask, L2 normalization.
    """

    name = "all-MiniLM-L6-v2"
    dim = 384

    def __init__(self, model_dir: Path, threads: int = 1, batch_size: int = 32) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        if not (model_dir / "onnx" / "model.onnx").exists():
            download_minilm(model_dir)
        path = model_dir / "onnx" / "model.onnx"
        if _sha256(path) != MINILM_SHA256:
            raise RuntimeError(f"{path} is not the official {MINILM_ID} export")
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # The published tokenizer.json truncates and pads to 128; this does its own.
        self._tok.no_truncation()
        self._tok.no_padding()
        cls_id, sep_id = self._tok.token_to_id("[CLS]"), self._tok.token_to_id("[SEP]")
        if cls_id is None or sep_id is None:
            raise RuntimeError(f"{model_dir / 'tokenizer.json'} has no [CLS] or [SEP] token")
        self._cls: int = cls_id
        self._sep: int = sep_id
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> np.ndarray:
        seqs = [[self._cls, *self._tok.encode(t, add_special_tokens=False).ids[: MAX_TOKENS - 2], self._sep]
                for t in texts]
        out = np.zeros((len(seqs), self.dim), dtype=np.float32)
        for start in range(0, len(seqs), self.batch_size):
            batch = seqs[start : start + self.batch_size]
            width = max(len(x) for x in batch)
            ids = np.zeros((len(batch), width), dtype=np.int64)
            mask = np.zeros((len(batch), width), dtype=np.int64)
            for row, x in enumerate(batch):
                ids[row, : len(x)] = x
                mask[row, : len(x)] = 1
            hidden = self._sess.run(None, {"input_ids": ids, "attention_mask": mask,
                                           "token_type_ids": np.zeros_like(ids)})[0]
            m = mask[..., None].astype(np.float32)
            out[start : start + len(batch)] = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return _normalize_rows(out)


def get_embeddings(backend: str = "minilm", model_dir: Path | str = "models/all-MiniLM-L6-v2",
                   hashing_dim: int = 512) -> EmbeddingBackend:
    backend = (backend or "").lower()
    if backend in {"hashing", "hash"}:
        return HashingEmbeddings(dim=hashing_dim)
    if backend in {"minilm", "all-minilm-l6-v2"}:
        return MiniLMEmbeddings(Path(model_dir))
    raise ValueError(f"unknown embedding backend {backend!r}; use 'minilm' or 'hashing'")
