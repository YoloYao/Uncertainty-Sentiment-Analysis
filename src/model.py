import torch.nn as nn
# During the model training phase, use the following line to import config
# import config
# When using a prediction application, use the following line to import config
from . import config
from transformers import DistilBertModel


class SentimentClassifier(nn.Module):
    """
    Sentiment classifier model
    Add a classification layer to the pre trained DistilBERT model
    """

    def __init__(self, n_classes, model_name=config.MODEL_NAME):
        super(SentimentClassifier, self).__init__()
        # Load pre trained DistilBERT model
        self.bert = DistilBertModel.from_pretrained(model_name)
        # Add a Dropout layer for regularization to prevent overfitting
        # Randomly discard 30% of features to prevent overfitting
        self.drop = nn.Dropout(p=0.3)
        # Add a fully connected layer as a classifier
        # The output dimension of DistilBERT is 768
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        Define the forward propagation of the model
        :param input_ids: Input ID tensor
        :param attention_mask: Attention Mask Tensor
        :return: Model outputs (logits)
        """
        # The output of DistillelBERT is a tuple, requiring only the first element (last_ midden_state)
        # The shape of bert_output [0] is [batch_size, sequence_length, hidden_size]
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # The hidden state of the last layer
        # shape: [batch_size, sequence_length, hidden_size]
        hidden_state = bert_output[0]
        # shape: [batch_size, hidden_size]
        pooled_output = hidden_state[:, 0]
        # Send to Dropout layer
        output = self.drop(pooled_output)
        # Sent to the fully connected layer
        return self.out(output)
