"""
Production fine-tuning using HuggingFace Trainer.

WHY Trainer over manual loop:
  - Handles distributed training automatically
  - Mixed precision (fp16) out of the box
  - Auto-checkpointing — saves best model by F1
  - Built-in logging to TensorBoard / WandB
  - Gradient accumulation for large effective batch sizes
  - Early stopping to prevent overfitting

Everything in Stage 2 still happens — Trainer just
wraps it so you don't maintain it yourself.
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field

from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
import evaluate

from smartdesk.ml.finetuning.dataset import (
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    NERExample,
    align_labels_with_tokens,
)

logger = logging.getLogger(__name__)


# ── Seqeval metric ────────────────────────────────────────────────────────────

"""
WHY seqeval over simple token accuracy:

Simple accuracy counts every token equally.
  Sentence: "Invoice INV-0042 from Acme Corp is overdue"
  7 tokens. 5 are "O". Model predicts "O" for everything.
  Accuracy = 5/7 = 71% — looks fine. But NER is completely broken.

Seqeval measures entity-level F1:
  Did the model get the FULL entity right?
  "Acme Corp" is one ORG entity — both words must be correct.
  Getting "Acme" right but "Corp" wrong = entire entity is wrong.

This is what production NER actually cares about.
"""

seqeval = evaluate.load("seqeval")


def compute_metrics(eval_preds):
    """Convert model outputs to seqeval entity-level F1.

    Called by Trainer after every eval step automatically.
    """
    logits, labels = eval_preds

    # logits shape: (batch, seq_len, num_labels)
    # argmax → predicted label id per token
    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    true_preds  = []

    for pred_row, label_row in zip(predictions, labels):
        true_label_row = []
        true_pred_row  = []

        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == -100:
                continue   # skip [CLS], [SEP], padding

            true_label_row.append(ID2LABEL[label_id])
            true_pred_row.append(ID2LABEL[pred_id])

        true_labels.append(true_label_row)
        true_preds.append(true_pred_row)

    results = seqeval.compute(
        predictions=true_preds,
        references=true_labels,
    )

    # Return flat dict — Trainer logs these automatically
    return {
        "precision": round(results["overall_precision"], 4),
        "recall":    round(results["overall_recall"],    4),
        "f1":        round(results["overall_f1"],        4),
        "accuracy":  round(results["overall_accuracy"],  4),
    }


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ProductionFinetuneConfig:
    model_name: str        = "bert-base-uncased"
    output_dir: str        = "outputs/ner_model"
    max_length: int        = 128
    batch_size: int        = 16
    eval_batch_size: int   = 32
    epochs: int            = 5
    learning_rate: float   = 2e-5
    warmup_ratio: float    = 0.1
    weight_decay: float    = 0.01
    frozen_layers: int     = 10

    # Mixed precision — use fp16 if GPU available
    # WHY fp16: cuts memory in half, runs 2x faster on modern GPUs.
    # Weights are stored as 16-bit floats instead of 32-bit.
    # Tiny accuracy loss — not measurable on most NER tasks.
    fp16: bool             = False   # set True if CUDA available

    # Early stopping — stop if F1 doesn't improve for N evals
    # WHY: prevents overfitting on small datasets
    early_stopping_patience: int = 3

    # Gradient accumulation
    # WHY: if batch_size=16 is too large for GPU memory,
    # set batch_size=4 and gradient_accumulation_steps=4.
    # Effective batch = 4 × 4 = 16. Same math, less memory.
    gradient_accumulation_steps: int = 1

    save_strategy: str     = "epoch"
    eval_strategy: str     = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str   = "f1"


# ── Dataset converter ─────────────────────────────────────────────────────────

def examples_to_hf_dataset(
    examples: list[NERExample],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """Convert NERExample list → HuggingFace Dataset.

    WHY HuggingFace Dataset over plain PyTorch Dataset:
    HF Trainer expects HF Dataset format. It also gives us
    free caching, memory-mapping for large datasets, and
    easy .map() for parallel preprocessing.
    """
    records = []
    for ex in examples:
        enc = align_labels_with_tokens(
            ex.words, ex.labels, tokenizer, max_length
        )
        records.append({
            "input_ids":      enc["input_ids"].squeeze(0).tolist(),
            "attention_mask": enc["attention_mask"].squeeze(0).tolist(),
            "labels":         enc["labels"].squeeze(0).tolist(),
        })

    return Dataset.from_list(records)


# ── Main production trainer ───────────────────────────────────────────────────

class ProductionNERTrainer:
    """Fine-tune BERT for NER using HuggingFace Trainer.

    The Trainer handles:
      - Training loop with gradient accumulation
      - Mixed precision (fp16)
      - Checkpointing best model by F1
      - Early stopping
      - Logging to console / TensorBoard
      - Evaluation after every epoch
    """

    def __init__(self, config: ProductionFinetuneConfig):
        self._config = config

        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        self._model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

        self._freeze_layers(config.frozen_layers)

    def _freeze_layers(self, n: int) -> None:
        """Same freezing logic as Stage 2."""
        for param in self._model.bert.embeddings.parameters():
            param.requires_grad = False

        for i, layer in enumerate(self._model.bert.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False

        trainable = sum(
            p.numel() for p in self._model.parameters()
            if p.requires_grad
        )
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "Trainable: %s / %s params (%.1f%%)",
            f"{trainable:,}", f"{total:,}",
            100 * trainable / total,
        )

    def train(
        self,
        train_examples: list[NERExample],
        eval_examples: list[NERExample],
    ) -> dict:
        """Run production fine-tuning. Returns best eval metrics."""

        # Convert to HuggingFace datasets
        train_ds = examples_to_hf_dataset(
            train_examples, self._tokenizer, self._config.max_length
        )
        eval_ds = examples_to_hf_dataset(
            eval_examples, self._tokenizer, self._config.max_length
        )

        # DataCollator handles dynamic padding per batch
        # WHY dynamic padding: Stage 2 padded all sequences to max_length.
        # Dynamic padding pads each BATCH to its longest sequence.
        # A batch of 5-word sentences doesn't waste compute on 123 padding tokens.
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self._tokenizer,
            padding=True,
        )

        training_args = TrainingArguments(
            output_dir=self._config.output_dir,

            # Training schedule
            num_train_epochs=self._config.epochs,
            per_device_train_batch_size=self._config.batch_size,
            per_device_eval_batch_size=self._config.eval_batch_size,
            learning_rate=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
            warmup_ratio=self._config.warmup_ratio,
            gradient_accumulation_steps=self._config.gradient_accumulation_steps,

            # Evaluation and saving
            eval_strategy=self._config.eval_strategy,
            save_strategy=self._config.save_strategy,
            load_best_model_at_end=self._config.load_best_model_at_end,
            metric_for_best_model=self._config.metric_for_best_model,
            greater_is_better=True,

            # Performance
            fp16=self._config.fp16,

            # Logging
            logging_steps=10,
            report_to="none",   # set "tensorboard" or "wandb" in production

            # Reproducibility
            seed=42,
            data_seed=42,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self._tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self._config.early_stopping_patience
                )
            ],
        )

        logger.info("Starting training...")
        trainer.train()

        # Evaluate final model
        metrics = trainer.evaluate()
        logger.info("Final metrics: %s", metrics)

        # Save final model
        trainer.save_model(self._config.output_dir)
        self._tokenizer.save_pretrained(self._config.output_dir)
        logger.info("Model saved to %s", self._config.output_dir)

        return metrics


def create_production_trainer(
    config: ProductionFinetuneConfig | None = None,
) -> ProductionNERTrainer:
    """Factory — same pattern as every other phase."""
    return ProductionNERTrainer(config or ProductionFinetuneConfig())