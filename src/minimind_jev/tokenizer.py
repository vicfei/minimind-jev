"""Own byte-level BPE tokenizer (the minimind route: BBPE works for Chinese subwords
and round-trips exactly). Trained on the task corpus; ~8K vocab."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def train_bpe(corpus: Iterable[str], vocab_size: int, out_path: Path,
              special_tokens: tuple[str, ...] = ("[PAD]", "[MASK]")) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(corpus, trainer)
    tok.decoder = decoders.ByteLevel()  # byte-level decode makes round-trips exact
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))
