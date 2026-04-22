"""
Convert raw NER annotations into BERT-ready tensors.

The core problem: BERT uses wordpieces, not words.
  "overdue" → ["over", "##due"]   (2 tokens)
  "INV-0042" → ["IN", "##V", "-", "00", "##42"]  (5 tokens)

But our labels are per WORD, not per wordpiece.
We need to align them — that's what this file does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


# ── Label schema ─────────────────────────────────────────────────────────────

# BIO tagging scheme — the standard for NER
# B = Beginning of entity  "Acme" in "Acme Corp"
# I = Inside entity        "Corp" in "Acme Corp"
# O = Outside any entity   "from", "is", "the"
#
# WHY BIO and not just entity/not-entity:
# Without BIO, "Acme Corp" would be two separate PERSON predictions.
# BIO lets the model learn that "Corp" continues "Acme" — one entity.

LABEL2ID = {
    "O":            0,
    "B-PERSON":     1,
    "I-PERSON":     2,
    "B-ORG":        3,
    "I-ORG":        4,
    "B-DATE":       5,
    "I-DATE":       6,
    "B-INVOICE_ID": 7,
    "I-INVOICE_ID": 8,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


# ── Raw annotation format ─────────────────────────────────────────────────────

@dataclass
class NERExample:
    """One labeled sentence.

    words:  ["Invoice", "INV-0042", "from", "Acme", "Corp", "is", "overdue"]
    labels: ["O",       "B-INVOICE_ID", "O",  "B-ORG", "I-ORG", "O", "O"]

    WHY store as word-level: this is how humans annotate.
    The dataset.py joba is to convert this to token-level for BERT.
    """
    words: list[str]
    labels: list[str]
    doc_id: Optional[str] = None

    def __post_init__(self):
        if len(self.words) != len(self.labels):
            raise ValueError(
                f"words ({len(self.words)}) and labels "
                f"({len(self.labels)}) must have same length"
            )


# ── The alignment problem ─────────────────────────────────────────────────────

def align_labels_with_tokens(
    words: list[str],
    word_labels: list[str],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 128,
) -> dict:
    """Tokenise words and align word-level labels to token-level.

    The core challenge — one word can become multiple tokens:
      word:   ["Acme",   "Corp"]
      tokens: ["Ac", "##me", "Corp"]   ← BERT splits "Acme"
      labels: ["B-ORG",  "I-ORG" ]
      aligned:["B-ORG", "I-ORG", "I-ORG"]  ← "##me" inherits from "Ac"

    Two strategies for sub-word tokens:
      1. Label all sub-words with the word's label (what we do here)
      2. Label only the first sub-word, mark rest as -100

    WHY -100 is special: PyTorch CrossEntropyLoss ignores positions
    where the label is -100. So marking sub-word continuations as -100
    means the model is only penalised on the first sub-word of each word.
    Both strategies work — strategy 2 is slightly cleaner mathematically.
    We use strategy 1 here for simplicity.
    """
    tokenized = tokenizer(
        words,
        is_split_into_words=True,   # WHY: tells tokenizer input is
                                    # already split by spaces — don't
                                    # re-split on punctuation
        truncation=True,
        max_length=max_length,
        padding="max_length",       # pad all sequences to max_length
        return_tensors="pt",
    )

    # word_ids() maps each token back to its word index
    # [CLS] → None, "Ac" → 0, "##me" → 0, "Corp" → 1, [SEP] → None
    word_ids = tokenized.word_ids(batch_index=0)

    aligned_labels = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # [CLS] and [SEP] special tokens — ignore in loss
            aligned_labels.append(-100)

        elif word_id != prev_word_id:
            # First token of a new word — use the word's label
            label_str = word_labels[word_id]
            aligned_labels.append(LABEL2ID[label_str])

        else:
            # Sub-word continuation (e.g. "##me" from "Acme")
            # Convert B- to I- so "##me" doesn't look like a new entity start
            label_str = word_labels[word_id]
            if label_str.startswith("B-"):
                label_str = "I-" + label_str[2:]
            aligned_labels.append(LABEL2ID[label_str])

        prev_word_id = word_id

    tokenized["labels"] = torch.tensor([aligned_labels])
    return tokenized


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class NERDataset(Dataset):
    """PyTorch Dataset wrapping NERExamples.

    WHY inherit Dataset: PyTorch's DataLoader expects this interface.
    It handles batching, shuffling, and multiprocess loading for free
    once we implement __len__ and __getitem__.
    """

    def __init__(
        self,
        examples: list[NERExample],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 128,
    ):
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._encodings = [
            align_labels_with_tokens(
                ex.words, ex.labels, tokenizer, max_length
            )
            for ex in examples
        ]

    def __len__(self) -> int:
        return len(self._encodings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one training example as a dict of tensors.

        DataLoader will stack these into batches automatically.
        """
        enc = self._encodings[idx]
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         enc["labels"].squeeze(0),
        }


# ── Sample data factory ───────────────────────────────────────────────────────

def make_sample_examples() -> list[NERExample]:
    """10 hand-labeled SmartDesk examples for quick smoke tests.

    In production these come from your annotation tool
    (Label Studio, Prodigy, Doccano etc.)
    """
    return [
        NERExample(
            words=["Invoice", "INV-0042", "from", "Acme", "Corp", "is", "overdue"],
            labels=["O", "B-INVOICE_ID", "O", "B-ORG", "I-ORG", "O", "O"],
        ),
        NERExample(
            words=["Please", "pay", "INV-9981", "by", "March", "15", "2024"],
            labels=["O", "O", "B-INVOICE_ID", "O", "B-DATE", "I-DATE", "I-DATE"],
        ),
        NERExample(
            words=["John", "Smith", "approved", "the", "payment", "to", "TechCorp"],
            labels=["B-PERSON", "I-PERSON", "O", "O", "O", "O", "B-ORG"],
        ),
        NERExample(
            words=["INV-0100", "from", "GlobalTech", "Ltd", "due", "January", "2024"],
            labels=["B-INVOICE_ID", "O", "B-ORG", "I-ORG", "O", "B-DATE", "I-DATE"],
        ),
        NERExample(
            words=["Sarah", "Connor", "submitted", "invoice", "INV-5523"],
            labels=["B-PERSON", "I-PERSON", "O", "O", "B-INVOICE_ID"],
        ),
    ]