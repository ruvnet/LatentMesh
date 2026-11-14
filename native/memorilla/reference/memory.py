"""The Memorilla memory module.

``MemoryModule`` compresses a variable number of document embeddings into a fixed set of memory vectors, conditioned
on the question, and projects them into the decoder's input embedding space.
"""

from collections.abc import Iterator
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from memorilla.utils import CHECKPOINT_FILE, CONFIG_FILE, checkpoint_file, load_memory_state

DEFAULT_NUM_MEMORIES = 16
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_SELF_ATTN_LAYERS = 0
DEFAULT_NUM_CROSS_ATTN_LAYERS = 2
DEFAULT_DROPOUT = 0.1
FFN_EXPANSION = 4
CACHE_CHUNK_SIZE = 65_536  # documents projected at a time by MemoryModule.write_cache


@dataclass(frozen=True)
class MemoryConfig:
    """Constructor arguments of a ``MemoryModule``.

    Attributes:
        embedding_dim: Width of the document and question embeddings (encoder hidden size).
        output_dim: Width of the memory tokens (decoder hidden size).
        num_memories: Number of memory tokens produced per example.
        num_heads: Attention heads in every attention layer.
        num_self_attn_layers: Transformer encoder layers applied to the documents (0 disables them).
        num_cross_attn_layers: Number of cross-attention + feed-forward refinement blocks.
        dropout: Dropout probability.
        retrieval_init: Whether the slots start from the most question-similar documents instead of learned vectors.
    """

    embedding_dim: int
    output_dim: int
    num_memories: int = DEFAULT_NUM_MEMORIES
    num_heads: int = DEFAULT_NUM_HEADS
    num_self_attn_layers: int = DEFAULT_NUM_SELF_ATTN_LAYERS
    num_cross_attn_layers: int = DEFAULT_NUM_CROSS_ATTN_LAYERS
    dropout: float = DEFAULT_DROPOUT
    retrieval_init: bool = False

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "MemoryConfig":
        """Infer the architecture from a state dict.

        The number of attention heads and the dropout rate leave no trace in the weights and take their defaults, as
        does the number of memory slots of a ``retrieval_init`` module.

        Args:
            state: A ``MemoryModule`` state dict.

        Returns:
            The inferred configuration.
        """
        retrieval_init = "memory_queries" not in state
        return cls(
            embedding_dim=state["query_cond_norm.weight"].shape[0],
            output_dim=state["memory_projection.weight"].shape[0],
            num_memories=DEFAULT_NUM_MEMORIES if retrieval_init else state["memory_queries"].shape[1],
            num_self_attn_layers=_count_layers(state, r"encoder\.layers\.(\d+)\."),
            num_cross_attn_layers=_count_layers(state, r"cross_attn_layers\.(\d+)\."),
            retrieval_init=retrieval_init,
        )


@dataclass
class MemoryCache:
    """The question-independent part of the memory module for a document collection, written once and read many times.

    Attributes:
        layers: Per cross-attention layer, keys ``[collections, heads, head_dim, num_docs]`` and values
            ``[collections, heads, num_docs, head_dim]``.
        padding_mask: ``[collections, num_docs]`` with True at padding positions, or None.
    """

    layers: list[tuple[torch.Tensor, torch.Tensor]]
    padding_mask: torch.Tensor | None = None

    @property
    def num_collections(self) -> int:
        """Number of collections; a single collection can serve a batch of questions."""
        return self.layers[0][0].size(0)

    @property
    def nbytes(self) -> int:
        """Bytes held by the keys and values."""
        return sum(tensor.numel() * tensor.element_size() for layer in self.layers for tensor in layer)


def _count_layers(state: dict[str, torch.Tensor], pattern: str) -> int:
    """Count the distinct layer indices matched by ``pattern`` in the state dict keys.

    Args:
        state: A state dict.
        pattern: Regular expression with one group capturing the layer index.

    Returns:
        The number of distinct layer indices.
    """
    indices = {match.group(1) for key in state if (match := re.match(pattern, key))}
    return len(indices)


def _multihead_attention(embedding_dim: int, num_heads: int, dropout: float) -> nn.MultiheadAttention:
    """Build a batch-first multi-head attention layer.

    Args:
        embedding_dim: Model width.
        num_heads: Number of heads.
        dropout: Attention dropout.

    Returns:
        The attention layer.
    """
    return nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)


class MemoryModule(nn.Module):
    """Question-conditioned compression of document embeddings into ``num_memories`` memory tokens.

    The forward pass:

    1. Initial slots: ``num_memories`` learned vectors, or with ``retrieval_init`` the documents most similar to the
       question (cosine similarity, zero-padded when there are fewer documents than slots).
    2. Question conditioning: slots cross-attend to the question embedding (residual + LayerNorm).
    3. Optional document self-attention encoder with ``num_self_attn_layers`` Transformer layers.
    4. ``num_cross_attn_layers`` refinement blocks: pre-norm cross-attention from the slots to the normalised
       documents, then a pre-norm feed-forward network, both residual.
    5. Final LayerNorm and a linear projection into the decoder embedding space.
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        num_memories: int = DEFAULT_NUM_MEMORIES,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_self_attn_layers: int = DEFAULT_NUM_SELF_ATTN_LAYERS,
        num_cross_attn_layers: int = DEFAULT_NUM_CROSS_ATTN_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        retrieval_init: bool = False,
    ) -> None:
        """Build the module.

        Args:
            embedding_dim: Width of the document and question embeddings (encoder hidden size).
            output_dim: Width of the memory tokens (decoder hidden size).
            num_memories: Number of memory tokens produced per example.
            num_heads: Attention heads in every attention layer.
            num_self_attn_layers: Transformer encoder layers applied to the documents (0 disables them).
            num_cross_attn_layers: Number of cross-attention + feed-forward refinement blocks.
            dropout: Dropout probability.
            retrieval_init: Initialise the slots from the most question-similar documents instead of learned vectors.
        """
        super().__init__()
        self.config = MemoryConfig(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            num_memories=num_memories,
            num_heads=num_heads,
            num_self_attn_layers=num_self_attn_layers,
            num_cross_attn_layers=num_cross_attn_layers,
            dropout=dropout,
            retrieval_init=retrieval_init,
        )

        if not retrieval_init:
            self.memory_queries = nn.Parameter(torch.randn(1, num_memories, embedding_dim))
            nn.init.xavier_uniform_(self.memory_queries)

        self.query_cond_attn = _multihead_attention(embedding_dim, num_heads, dropout)
        self.query_cond_norm = nn.LayerNorm(embedding_dim)

        if num_self_attn_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * FFN_EXPANSION,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_self_attn_layers)
        self.doc_norm = nn.LayerNorm(embedding_dim)

        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_cross_attn_layers)])
        self.cross_attn_layers = nn.ModuleList(
            [_multihead_attention(embedding_dim, num_heads, dropout) for _ in range(num_cross_attn_layers)]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_cross_attn_layers)])
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim * FFN_EXPANSION),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embedding_dim * FFN_EXPANSION, embedding_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_cross_attn_layers)
            ]
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.memory_projection = nn.Linear(embedding_dim, output_dim)
        nn.init.xavier_uniform_(self.memory_projection.weight)

    @property
    def num_memories(self) -> int:
        """Number of memory tokens produced per example."""
        return self.config.num_memories

    def forward(
        self,
        doc_embeds: torch.Tensor,
        doc_padding_mask: torch.Tensor | None,
        question_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compress documents into memory tokens.

        Args:
            doc_embeds: Document embeddings ``[batch, num_docs, embedding_dim]``.
            doc_padding_mask: ``[batch, num_docs]`` with True at padding positions, or None when nothing is padded.
            question_embeds: Question embeddings ``[batch, embedding_dim]``.

        Returns:
            Memory tokens ``[batch, num_memories, output_dim]``.
        """
        question = question_embeds.unsqueeze(1)
        slots = self._initial_slots(doc_embeds, doc_padding_mask, question_embeds)

        conditioned, _ = self.query_cond_attn(slots, question, question)
        slots = self.query_cond_norm(slots + conditioned)

        if self.config.num_self_attn_layers > 0:
            doc_embeds = self.encoder(doc_embeds, src_key_padding_mask=doc_padding_mask)
        docs = self.doc_norm(doc_embeds)

        for cross_attn_norm, cross_attn, ffn_norm, ffn in zip(
            self.cross_attn_norms, self.cross_attn_layers, self.ffn_norms, self.ffn_layers, strict=True
        ):
            attended, _ = cross_attn(cross_attn_norm(slots), docs, docs, key_padding_mask=doc_padding_mask)
            slots = slots + self.attn_dropout(attended)
            slots = slots + ffn(ffn_norm(slots))

        return self.memory_projection(self.layer_norm(slots))

    def _initial_slots(
        self,
        doc_embeds: torch.Tensor,
        doc_padding_mask: torch.Tensor | None,
        question_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Return the slots before question conditioning.

        Args:
            doc_embeds: Document embeddings ``[batch, num_docs, embedding_dim]``.
            doc_padding_mask: ``[batch, num_docs]`` padding mask or None.
            question_embeds: Question embeddings ``[batch, embedding_dim]``.

        Returns:
            Initial slots ``[batch, num_memories, embedding_dim]``.
        """
        batch_size = doc_embeds.size(0)
        if not self.config.retrieval_init:
            return self.memory_queries.expand(batch_size, -1, -1)

        docs = F.normalize(doc_embeds, p=2, dim=-1)
        question = F.normalize(question_embeds, p=2, dim=-1)
        scores = torch.bmm(docs, question.unsqueeze(-1)).squeeze(-1)
        if doc_padding_mask is not None:
            scores = scores.masked_fill(doc_padding_mask, float("-inf"))

        k = min(self.num_memories, scores.size(1))
        top_indices = scores.topk(k, dim=-1).indices
        slots = torch.gather(doc_embeds, 1, top_indices.unsqueeze(-1).expand(-1, -1, doc_embeds.size(-1)))
        if k < self.num_memories:
            padding = slots.new_zeros(batch_size, self.num_memories - k, doc_embeds.size(-1))
            slots = torch.cat([slots, padding], dim=1)
        return slots

    @property
    def supports_cache(self) -> bool:
        """Whether ``write_cache`` applies: learned slots and no document self-attention (the released modules)."""
        return not self.config.retrieval_init and self.config.num_self_attn_layers == 0

    def project_documents(
        self, doc_embeds: torch.Tensor, chunk_size: int = CACHE_CHUNK_SIZE
    ) -> Iterator[tuple[slice, list[tuple[torch.Tensor, torch.Tensor]]]]:
        """Yield the cross-attention keys and values of the documents, a chunk of documents at a time.

        Args:
            doc_embeds: Document embeddings ``[collections, num_docs, embedding_dim]``.
            chunk_size: Documents per chunk.

        Yields:
            The chunk's document positions and, per layer, keys and values ``[collections, heads, chunk, head_dim]``.
        """
        collections, num_docs, dim = doc_embeds.shape
        heads = self.config.num_heads
        for start in range(0, num_docs, chunk_size):
            docs = self.doc_norm(doc_embeds[:, start : start + chunk_size])
            shape = (collections, docs.size(1), heads, dim // heads)
            layers = []
            for attn in self.cross_attn_layers:
                keys = F.linear(docs, attn.in_proj_weight[dim : 2 * dim], attn.in_proj_bias[dim : 2 * dim])
                values = F.linear(docs, attn.in_proj_weight[2 * dim :], attn.in_proj_bias[2 * dim :])
                layers.append((keys.view(shape).transpose(1, 2), values.view(shape).transpose(1, 2)))
            yield slice(start, start + docs.size(1)), layers

    @torch.no_grad()
    def write_cache(
        self,
        doc_embeds: torch.Tensor,
        doc_padding_mask: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        chunk_size: int = CACHE_CHUNK_SIZE,
    ) -> MemoryCache:
        """Compute the question-independent keys and values of a document collection once.

        ``read_cache`` on the result equals ``forward`` up to floating-point rounding. The documents are projected a
        chunk at a time, so writing needs little memory beyond the cache itself.

        Args:
            doc_embeds: Document embeddings ``[collections, num_docs, embedding_dim]``.
            doc_padding_mask: ``[collections, num_docs]`` with True at padding positions, or None.
            dtype: Dtype of the cached keys and values (default: that of ``doc_embeds``).
            chunk_size: Documents projected at a time.

        Returns:
            The cache.

        Raises:
            ValueError: If the module does not support caching (see ``supports_cache``).
        """
        if not self.supports_cache:
            raise ValueError("Caching needs learned memory slots and no document self-attention.")
        collections, num_docs, dim = doc_embeds.shape
        heads = self.config.num_heads
        options = {"dtype": dtype or doc_embeds.dtype, "device": doc_embeds.device}
        layers = [
            (
                torch.empty(collections, heads, dim // heads, num_docs, **options),
                torch.empty(collections, heads, num_docs, dim // heads, **options),
            )
            for _ in self.cross_attn_layers
        ]
        for rows, projected in self.project_documents(doc_embeds, chunk_size):
            for (keys, values), (chunk_keys, chunk_values) in zip(layers, projected, strict=True):
                keys[..., rows] = chunk_keys.transpose(2, 3)
                values[:, :, rows] = chunk_values
        return MemoryCache(layers, doc_padding_mask)

    def read_cache(self, cache: MemoryCache, question_embeds: torch.Tensor) -> torch.Tensor:
        """Produce memory tokens from a cache, as ``forward`` would from the documents.

        Args:
            cache: Output of ``write_cache``, with one collection per question or one for all of them.
            question_embeds: Question embeddings ``[batch, embedding_dim]``.

        Returns:
            Memory tokens ``[batch, num_memories, output_dim]``.

        Raises:
            ValueError: If the cache holds neither one collection nor one per question.
        """
        batch = question_embeds.size(0)
        if cache.num_collections not in (1, batch):
            raise ValueError(f"The cache holds {cache.num_collections} collections for {batch} questions.")
        question = question_embeds.unsqueeze(1)
        slots = self.memory_queries.expand(batch, -1, -1)
        conditioned, _ = self.query_cond_attn(slots, question, question)
        slots = self.query_cond_norm(slots + conditioned)

        for (keys, values), cross_attn_norm, cross_attn, ffn_norm, ffn in zip(
            cache.layers, self.cross_attn_norms, self.cross_attn_layers, self.ffn_norms, self.ffn_layers, strict=True
        ):
            attended = self._attend_cached(cross_attn, cross_attn_norm(slots), keys, values, cache.padding_mask)
            slots = slots + self.attn_dropout(attended)
            slots = slots + ffn(ffn_norm(slots))
        return self.memory_projection(self.layer_norm(slots))

    def _attend_cached(
        self,
        attn: nn.MultiheadAttention,
        slots: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Cross-attention of ``attn`` from the slots ``[batch, num_memories, dim]`` over cached keys and values.

        Written as two matrix products, since the head width (2560 / 8 = 320 for the released modules) is above what
        the fused attention kernels support. When one collection serves the batch, all slots attend in one product.
        """
        batch, num_slots, dim = slots.shape
        heads = self.config.num_heads
        head_dim = dim // heads
        queries = F.linear(slots, attn.in_proj_weight[:dim], attn.in_proj_bias[:dim]).to(keys.dtype)
        queries = queries.view(batch, num_slots, heads, head_dim).transpose(1, 2)
        shared = keys.size(0) == 1 and batch > 1
        if shared:
            queries = queries.transpose(0, 1).reshape(1, heads, batch * num_slots, head_dim)
        scores = torch.matmul(queries, keys).float() * head_dim**-0.5
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        attended = torch.matmul(torch.softmax(scores, dim=-1).to(values.dtype), values)
        if shared:
            attended = attended.view(heads, batch, num_slots, head_dim).transpose(0, 1)
        return attn.out_proj(attended.transpose(1, 2).reshape(batch, num_slots, dim).to(slots.dtype))

    def save(self, directory: str | Path) -> Path:
        """Write ``memory.pt`` (state dict) and ``config.json`` (constructor arguments).

        Args:
            directory: Output directory, created if needed.

        Returns:
            The output directory.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / CHECKPOINT_FILE)
        with open(directory / CONFIG_FILE, "w") as handle:
            json.dump(asdict(self.config), handle, indent=2)
            handle.write("\n")
        return directory

    @classmethod
    def from_pretrained(cls, path: str | Path, **overrides: int | float | bool) -> "MemoryModule":
        """Load a module from a checkpoint directory or ``memory.pt`` file.

        The architecture comes from ``config.json`` next to the weights when present and is otherwise inferred from
        the state dict. Weights are loaded strictly.

        Args:
            path: Checkpoint directory or weights file.
            **overrides: Constructor arguments that replace the stored or inferred ones (e.g. ``num_heads``).

        Returns:
            The loaded module, in evaluation mode, with the dtype of the stored weights.
        """
        weights = checkpoint_file(path)
        state = load_memory_state(weights)
        config_path = weights.parent / CONFIG_FILE
        if config_path.exists():
            with open(config_path) as handle:
                config = MemoryConfig(**json.load(handle))
        else:
            config = MemoryConfig.from_state_dict(state)

        module = cls(**{**asdict(config), **overrides})
        dtype = next(iter(state.values())).dtype
        module.to(dtype=dtype).load_state_dict(state, strict=True)
        return module.eval()
