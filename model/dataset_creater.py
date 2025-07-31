################################################################################
# filename: dataset_creater.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 31/07,2025
################################################################################

import os

import PyPDF2
import torch
from torch.utils.data import Dataset, DataLoader

################################################################################

class DatasetCVTransformer(Dataset):
    def __init__(self, datas_to_use : list[dict[str, str | int]], tokenizer, max_length=512):
        self.input_vectors = []
        self.labels = []
        for data in datas_to_use:
            input_ids, attention_mask = get_vect_from_cv_tag(
                                                            cv_path=data["cv_path"],
                                                            tag = data["tag"],
                                                            tokenizer = tokenizer,
                                                            max_length = max_length
                                                        )
            self.input_vectors.append((input_ids, attention_mask))
            self.labels.append(data["scoring"])

    def __len__(self):
        return len(self.input_vectors)
    
    def __getitem__(self, idx):
        input_ids, attention_mask = self.input_vectors[idx]
        label = self.labels[idx]
        return {
            "input_ids": input_ids.squeeze(0),  # supprime la dim batch ajoutée par tokenizer
            "attention_mask": attention_mask.squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32)  # si score réel
        }

################################################################################

def get_vect_from_cv_tag(cv_path, tag, tokenizer, max_length=512):
    text_cv = get_text_from_cv(cv_path)

    text_to_encode = f"Tag: {tag} CV: {text_cv}"

    encoded = tokenizer(
        text_to_encode,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    return encoded["input_ids"], encoded["attention_mask"]

################################################################################

def get_text_from_cv(cv_path):
    text = ""
    with open(cv_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

################################################################################

def create_dataloader(datas_to_use : list[dict[str, str | int]], tokenizer, batch_size=8, max_length=512):
    dataset = DatasetCVTransformer(
        datas_to_use = datas_to_use,
        tokenizer    = tokenizer,
        max_length   = max_length,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers = os.cpu_count() // 2,
    )

################################################################################
# End of File
################################################################################