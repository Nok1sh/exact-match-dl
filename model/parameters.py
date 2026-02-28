import torch
from transformers import AutoTokenizer

class Params:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TOKENIZER = AutoTokenizer.from_pretrained("MilyaShams/rubert-russian-qa-sberquad", return_offsets_mapping=True)