"""Reproduce the original educational fixture: python scripts/train.py."""
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F

from backend.corpus import sentences
from backend.engine import ROOT
from backend.model import Config, TinyTransformer, Tokenizer


def main():
    torch.set_num_threads(1)
    torch.manual_seed(Config().seed)
    torch.use_deterministic_algorithms(True)
    random.seed(Config().seed)
    texts = sorted(set(sentences()))
    tokenizer = Tokenizer.build(texts)
    shuffled = texts.copy()
    random.shuffle(shuffled)
    heldout = shuffled[::7]
    training = [text for index, text in enumerate(shuffled) if index % 7]
    def batch(texts):
        sequences = [tokenizer.encode(text, eos=True) for text in texts]
        padded = torch.zeros((len(sequences), max(map(len, sequences))), dtype=torch.long)
        for index, sequence in enumerate(sequences): padded[index, :len(sequence)] = torch.tensor(sequence)
        return padded[:, :-1], padded[:, 1:]
    train_x, train_y = batch(training)
    eval_x, eval_y = batch(heldout)
    model = TinyTransformer(len(tokenizer.vocabulary))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    def evaluate(x, y):
        with torch.inference_mode():
            logits, _ = model(x)
            loss = float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=0))
            mask = y != 0
            accuracy = float(((logits.argmax(-1) == y) & mask).sum() / mask.sum())
        return {"loss": loss, "accuracy": accuracy}
    before = evaluate(eval_x, eval_y)
    for step in range(500):
        indices = torch.randint(len(train_x), (64,))
        logits, _ = model(train_x[indices])
        loss = F.cross_entropy(logits.flatten(0, 1), train_y[indices].flatten(), ignore_index=0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0: print(f"step={step} train_batch_loss={float(loss):.4f}", flush=True)
    model.eval()
    final_eval = evaluate(eval_x, eval_y)
    final_train = evaluate(train_x, train_y)
    counts = torch.bincount(train_y[train_y != 0], minlength=len(tokenizer.vocabulary))
    baseline_token = int(counts.argmax())
    baseline_accuracy = float((eval_y[eval_y != 0] == baseline_token).float().mean())
    corpus = "\n".join(texts) + "\n"
    directory = ROOT / "examples"
    directory.mkdir(exist_ok=True)
    (directory / "corpus.txt").write_text(corpus, encoding="utf-8")
    checkpoint = directory / "tiny-transformer.pt"
    torch.save(model.state_dict(), checkpoint)
    metadata = {
        "id": "cortexlab-tiny-v1", "name": "Cortex Tiny · original grammar fixture", "layers": 3, "heads": 4,
        "d_model": 64, "context_limit": 48, "vocab_size": len(tokenizer.vocabulary),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "trained": True, "seed": Config().seed,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "tokenizer": {"id": "cortexlab-word-v1", "vocabulary": tokenizer.vocabulary, "unknown": "<unk>", "rules": "Lowercase words, numbers and punctuation; prepends <bos>; unsupported words become <unk>"},
        "training": {"dataset": "Original authored color/object/verb grammar plus five model-inspection sentences", "license": "MIT", "scope": "Educational grammar fixture; not a general language model", "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(), "train_sequences": len(training), "heldout_sequences": len(heldout), "split": "Sorted unique sentences, seeded shuffle, every seventh held out", "steps": 500, "batch_size": 64, "optimizer": "AdamW lr=0.003 weight_decay=0.01", "initial_heldout": before, "final_heldout": final_eval, "final_train": final_train, "baseline": {"kind": "Most frequent training target token", "token": tokenizer.vocabulary[baseline_token], "heldout_accuracy": baseline_accuracy}, "torch": torch.__version__, "device": "cpu", "determinism": "Seeded CPU run; exact retraining across Torch/platform versions not guaranteed"},
    }
    (directory / "model.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "demo-config.json").write_text(json.dumps({"prompt": "a comet is cyan . the comet", "max_new_tokens": 6, "ablate": None}, indent=2), encoding="utf-8")
    print(json.dumps(metadata["training"], indent=2), flush=True)
    if final_eval["accuracy"] <= baseline_accuracy:
        raise RuntimeError("Model did not beat the declared baseline")


if __name__ == "__main__": main()
