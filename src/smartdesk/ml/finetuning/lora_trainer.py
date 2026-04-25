# src/smartdesk/ml/finetuning/lora_trainer.py
"""
LoRA fine-tuning for SmartDesk NER.
Base model: bert-base-uncased (or any HuggingFace model)
Adapters: one per document type (invoice, legal, hr)
"""
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import torch
import logging

logger = logging.getLogger(__name__)


class LoRANERTrainer:
    """
    Wraps PEFT LoRA for SmartDesk NER tasks.

    Key idea:
      - Load BERT once (frozen)
      - Add LoRA adapters to attention layers only
      - Train only the adapter params (12K instead of 110M)
      - Save adapter separately — base model not touched
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        label_list: list[str] = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        self.model_name = model_name
        self.label_list = label_list or ["O", "B-ORG", "I-ORG", "B-INVOICE_ID"]
        self.num_labels = len(self.label_list)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def _build_lora_model(self):
        """
        Load base model + wrap with LoRA config.
        Only attention projection layers get adapters.
        """
        # 1. Load base model
        base_model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )

        # 2. LoRA config — target only attention weight matrices
        # query (q) and value (v) projections — standard LoRA choice
        # Adding key (k) rarely helps; output projection optional
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",          # don't adapt bias terms
            target_modules=["query", "value"],  # BERT attention projs
            modules_to_save=["classifier"],     # keep classifier head trainable
        )

        # 3. Wrap — this freezes base and adds adapter matrices
        model = get_peft_model(base_model, lora_config)

        # 4. Print param summary — this is the money shot for interviews
        model.print_trainable_parameters()
        # Output: trainable params: 888,068 || all params: 109,594,628 (0.81%)

        return model

    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str = "models/lora-smartdesk-ner",
        num_epochs: int = 3,
    ):
        self.model = self._build_lora_model()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=3e-4,   # LoRA can use HIGHER LR than full fine-tune
            #                       because only adapters update — base stays stable
            warmup_ratio=0.1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none",
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        return trainer

    def save_adapter(self, path: str):
        """
        Save ONLY the adapter weights — not the base model.
        adapter_model.bin is ~3MB vs 440MB for full BERT.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Adapter saved to {path}")
        logger.info("Base model NOT saved — reuse from HuggingFace cache")