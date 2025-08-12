import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class SentimentDataset(Dataset):
    """
    PyTorch dataset class for sentiment analysis.
    Load data, use tokenizer for processing, and return the format required by the model.
    """

    def __init__(self, file_path, tokenizer, max_len):
        """
        Initialization method
        :param file_path: str, Data file path (e.g. 'data/processed/train.csv')
        :param tokenizer: transformers.Tokenizer
        :param max_len: int, Maximum length of text sequence
        """
        self.df = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = self.df.text.values
        self.labels = self.df.sentiment.values

    def __len__(self):
        # Return the total length of the dataset
        return len(self.df)

    def __getitem__(self, idx):
        """
        Obtain a single data sample based on the index.
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


def create_data_loader(file_path, tokenizer, max_len, batch_size):
    """
    Method for generating DataLoader
    """
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
