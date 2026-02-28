import torch
import json

from torch.utils.data import Dataset
from model.parameters import Params

tokenizer = Params.TOKENIZER

class ExtractMatchDataset(Dataset):

    def __init__(self, max_length=512, stride=128):
        super().__init__()

        self.dataset = []

        with open("data/train.json", "r") as f:
            examples = json.load(f)
        
        for data in examples:
            text = data["text"]
            label = data["label"]

            start_ind = int(data["extracted_part"]["answer_start"][0])
            end_ind = int(data["extracted_part"]["answer_end"][0])

            tokenized = tokenizer(
                label,
                text,
                truncation="only_second",
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt"
            )

            for i in range(len(tokenized["input_ids"])):
                offsets = tokenized["offset_mapping"][i]
                inputs = tokenized["input_ids"][i]
                token_types = tokenized["token_type_ids"][i]

                token_start = 0
                token_end = 0

                for id, (start, end) in enumerate(offsets):
                    if start <= start_ind < end:
                        token_start = id
                    if start < end_ind <= end:
                        token_end = id
                
                if token_start == 0 or token_end == 0:
                    token_start = token_end = 0
                
                self.dataset.append({
                    "input_ids": inputs,
                    "attention_mask": tokenized["attention_mask"][i],
                    "token_type_ids": token_types,
                    "start_positions": torch.tensor(token_start),
                    "end_positions": torch.tensor(token_end)
                })
        

        self.dataset_size = len(self.dataset)
    

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return self.dataset[index]

    def get_data(self):
        return self.dataset
