import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from transformers import DistilBertModel
from .. import config

class SNGPClassifier(nn.Module):
    def __init__(self, n_classes, model_name='distilbert-base-uncased'):
        super(SNGPClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        
        self.classifier = spectral_norm(
            nn.Linear(self.bert.config.hidden_size, n_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = bert_output[0][:, 0]
        logits = self.classifier(pooled_output)
        return logits