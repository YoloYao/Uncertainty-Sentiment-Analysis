import torch.nn as nn
from torch.nn.utils import spectral_norm
from transformers import DistilBertModel
from .. import config


class SNGPClassifier(nn.Module):
    """
    Sentiment classifier model based on SNGP
    Add a classification layer to the pre trained DistilBERT model
    """

    def __init__(self, n_classes, model_name=config.MODEL_NAME):
        super(SNGPClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)

        # Add Spectral Normalization layer
        self.classifier = spectral_norm(
            nn.Linear(self.bert.config.hidden_size, n_classes)
        )

    def forward(self, input_ids, attention_mask):
        """
        Define the forward propagation of the model
        :param input_ids: Input ID tensor
        :param attention_mask: Attention Mask Tensor
        :return: Model outputs (logits)
        """
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = bert_output[0][:, 0]
        logits = self.classifier(pooled_output)
        return logits
