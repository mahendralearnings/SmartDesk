# src/smartdesk/ml/finetuning/qlora_trainer.py
"""
QLoRA = Quantize base model to 4-bit FIRST, then add LoRA.
Use when: GPU < 8GB, or model is 7B+ params.
"""
"""
peft ==parameter efficient fine-tuning== is HuggingFace's library for LoRA and other adapter methods.

"""


from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForTokenClassification, BitsAndBytesConfig
import torch


def load_qlora_model(
    model_name: str,
    num_labels: int,
    lora_r: int = 8,
    lora_alpha: int = 16,
):
    """
    QLoRA in 3 steps:
      1. Load base model in 4-bit (bitsandbytes NF4 format)
      2. Prepare for k-bit training (freezes + adds gradient checkpointing)
      3. Wrap with LoRA adapters
    """

    # ── STEP 1: 4-bit quantization config ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",        # NormalFloat4 — better than int4 for weights
        bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16, store in 4bit
        bnb_4bit_use_double_quant=True,   # quantize the quantization constants too
        #                                   this saves another ~0.4 bits per param
    )

    base_model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # ── STEP 2: Prepare for k-bit training ──
    # This does 3 things automatically:
    #   a) casts LayerNorm to fp32 (stability)
    #   b) enables gradient checkpointing (memory saving)
    #   c) enables input gradients (required for backprop through frozen layers)
    model = prepare_model_for_kbit_training(base_model)

    # ── STEP 3: Add LoRA adapters ──
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model