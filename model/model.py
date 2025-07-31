################################################################################
# filename: model.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 31/07,2025
################################################################################

import torch
import torch.nn as nn
from transformers import AutoModel

################################################################################
#CONSTANTS

hidden_size_mlp = 256
output_size_mlp = 1

################################################################################

class ClassificatorCVTransformer(nn.Module):
    def __init__(self, type="prajjwal1/bert-tiny"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(type)
        input_size_mlp = self.bert.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(input_size_mlp, hidden_size_mlp),
            nn.ReLU(),
            nn.Linear(hidden_size_mlp, output_size_mlp)  # Output: pertinence (float between 0–5)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        score = self.mlp(cls_output)
        return score

################################################################################
# End of File
################################################################################