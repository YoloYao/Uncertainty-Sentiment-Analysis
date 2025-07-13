import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import DistilBertTokenizer

# A class encapsulates the logic for loading and processing datasets
class SentimentDataset(Dataset):
    """
    用于情感分析的PyTorch数据集类。
    负责加载数据、使用tokenizer进行处理，并返回模型所需的格式。
    """
    def __init__(self, file_path, tokenizer, max_len):
        """
        初始化方法。
        :param file_path: str, 数据文件路径 (例如 'data/processed/train.csv')
        :param tokenizer: transformers.Tokenizer, 用于文本处理的tokenizer
        :param max_len: int, 文本序列的最大长度
        """
        self.df = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = self.df.text.values
        self.labels = self.df.sentiment.values

    def __len__(self):
        """返回数据集的总长度"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        根据索引获取单个数据样本。
        这是模型训练时调用此类的核心方法。
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 创建一个辅助函数来生成 DataLoader
def create_data_loader(file_path, tokenizer, max_len, batch_size):
    ds = SentimentDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2
    )