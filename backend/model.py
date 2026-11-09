"""Small, explicit causal transformer. All captured values come from this forward pass."""
import math
import re
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    layers: int = 3
    heads: int = 4
    d_model: int = 64
    context_limit: int = 48
    seed: int = 23


class Tokenizer:
    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.lookup = {token: index for index, token in enumerate(vocabulary)}

    @staticmethod
    def split(text):
        return re.findall(r"\w+|[^\w\s]", text.lower())

    @classmethod
    def build(cls, texts):
        return cls(cls.specials + sorted({word for text in texts for word in cls.split(text)}))

    def encode(self, text, eos=False):
        return [1] + [self.lookup.get(word, 3) for word in self.split(text)] + ([2] if eos else [])

    def decode(self, ids):
        words = [self.vocabulary[index] for index in ids if index not in (0, 1, 2)]
        return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(words))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.width = config.d_model // config.heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x, ablate_head=None):
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [item.reshape(batch, length, self.heads, self.width).transpose(1, 2) for item in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.width)
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        probabilities = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        heads = probabilities @ v
        if ablate_head is not None:
            gate = torch.ones(self.heads, device=x.device)
            gate[ablate_head] = 0
            heads = heads * gate[None, :, None, None]
        return self.proj(heads.transpose(1, 2).reshape(batch, length, channels)), probabilities


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = Attention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(nn.Linear(config.d_model, 4 * config.d_model), nn.GELU(), nn.Linear(4 * config.d_model, config.d_model))

    def forward(self, x, ablate_head=None):
        mixed, weights = self.attention(self.ln1(x), ablate_head)
        x = x + mixed
        x = x + self.mlp(self.ln2(x))
        return x, weights


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, config=Config()):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.position = nn.Embedding(config.context_limit, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size, bias=False)

    def forward(self, ids, capture=False, ablate=None):
        if ids.ndim != 2 or ids.shape[1] > self.config.context_limit:
            raise ValueError("Expected batch/token IDs within context limit")
        x = self.embedding(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        layers = []
        for index, block in enumerate(self.blocks):
            head = ablate[1] if ablate is not None and ablate[0] == index else None
            x, attention = block(x, head)
            if capture:
                layers.append((attention.detach(), x.detach()))
        return self.output(self.norm(x)), layers
