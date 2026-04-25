# src/smartdesk/ml/finetuning/adapter_manager.py
"""
The killer feature of LoRA: one base model, many adapters.
Swap at runtime — no reloading the 440MB base model.
"""
from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer


class AdapterManager:
    """
    SmartDesk production pattern:
      - Load base BERT once at startup (440MB, done once)
      - Load adapters on demand (~3MB each, instant)
      - Swap between invoice / legal / HR adapters per request
    """

    ADAPTER_PATHS = {
        "invoice": "models/adapters/invoice",
        "legal":   "models/adapters/legal",
        "hr":      "models/adapters/hr",
    }

    def __init__(self, base_model_name: str = "bert-base-uncased", num_labels: int = 9):
        print("Loading base model once...")
        self.base_model = AutoModelForTokenClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self._active_adapter = None
        self._peft_model = None
        print("Base model loaded. Adapters will load on demand.")

    def load_adapter(self, domain: str):
        """Load a domain-specific adapter onto the base model."""
        if domain not in self.ADAPTER_PATHS:
            raise ValueError(f"Unknown domain: {domain}. Choose from {list(self.ADAPTER_PATHS)}")

        path = self.ADAPTER_PATHS[domain]
        print(f"Loading {domain} adapter from {path}...")

        self._peft_model = PeftModel.from_pretrained(self.base_model, path)
        self._active_adapter = domain
        print(f"Active adapter: {domain}")
        return self._peft_model

    def swap_adapter(self, domain: str):
        """
        Switch to a different domain adapter.
        Base model weights stay in memory — only adapter matrices swap.
        This is ~10ms vs ~3s for reloading the full model.
        """
        if self._peft_model is None:
            return self.load_adapter(domain)

        self._peft_model.load_adapter(self.ADAPTER_PATHS[domain], adapter_name=domain)
        self._peft_model.set_adapter(domain)
        self._active_adapter = domain
        print(f"Swapped to {domain} adapter")
        return self._peft_model

    def predict(self, text: str) -> list[dict]:
        """Run NER with currently active adapter."""
        if self._peft_model is None:
            raise RuntimeError("No adapter loaded. Call load_adapter() first.")

        inputs = self.tokenizer(text, return_tensors="pt")
        import torch
        with torch.no_grad():
            outputs = self._peft_model(**inputs)

        predictions = outputs.logits.argmax(-1)[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return [{"token": t, "pred_id": p} for t, p in zip(tokens, predictions)]