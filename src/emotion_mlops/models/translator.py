from typing import List

import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer


def load_mt(model_name: str, device: str):
    """Load MarianMT translation model."""
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    return tokenizer, model


@torch.no_grad()
def translate_batch(
    sentences: List[str], tokenizer, model, device: str, bs: int = 16
) -> List[str]:
    """Translate English sentences to Dutch."""
    outputs = []
    for i in tqdm(range(0, len(sentences), bs), desc="[MT] Translating EN->NL"):
        batch = sentences[i : i + bs]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        generated = model.generate(**encoded, max_new_tokens=256)
        decoded = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
        outputs.extend(decoded)
    return outputs
