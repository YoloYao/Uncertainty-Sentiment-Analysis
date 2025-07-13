# src/model.py

import torch.nn as nn
from transformers import DistilBertModel

class SentimentClassifier(nn.Module):
    """
    情感分类器模型。
    该模型加载一个预训练的DistilBERT模型，并在其上添加一个自定义的分类头。
    """
    def __init__(self, n_classes, model_name='distilbert-base-uncased'):
        super(SentimentClassifier, self).__init__()
        # 加载预训练的DistilBERT模型
        self.bert = DistilBertModel.from_pretrained(model_name)
        # 添加一个Dropout层，用于正则化，防止过拟合
        self.drop = nn.Dropout(p=0.3)
        # 添加一个全连接层（线性层）作为分类器
        # DistilBERT的输出维度是768
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        定义模型的前向传播。
        :param input_ids: 输入的ID张量
        :param attention_mask: 注意力掩码张量
        :return: 模型的输出（logits）
        """
        # DistilBERT的输出是一个元组，我们只需要第一个元素（last_hidden_state）
        # bert_output[0] 的形状是 [batch_size, sequence_length, hidden_size]
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 我们只取 [CLS] token对应的输出用于分类，它在序列的第一个位置
        # hidden_state 的形状是 [batch_size, sequence_length, hidden_size]
        hidden_state = bert_output[0] 
        # pooled_output 的形状是 [batch_size, hidden_size]
        pooled_output = hidden_state[:, 0]
        # 将 pooled_output 通过Dropout层和最后的线性层
        output = self.drop(pooled_output)
        return self.out(output)