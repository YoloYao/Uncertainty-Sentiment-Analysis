# src/train.py

import torch
import torch.nn as nn
import pandas as pd
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from collections import defaultdict
import os

# 从我们自己写的文件中导入
import config
from dataset import create_data_loader
from model import SentimentClassifier
from engine import train_epoch, eval_model

def run():
    # --- 1. 加载数据 ---
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "validation.csv"))

    # --- 2. 初始化分词器和数据加载器 ---
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    
    train_data_loader = create_data_loader(config.PROCESSED_DATA_PATH + "/train.csv", tokenizer, config.MAX_LEN, config.BATCH_SIZE)
    val_data_loader = create_data_loader(config.PROCESSED_DATA_PATH + "/validation.csv", tokenizer, config.MAX_LEN, config.BATCH_SIZE)

    # --- 3. 初始化模型、优化器、损失函数等 ---
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    model = SentimentClassifier(n_classes=config.N_CLASSES, model_name=config.MODEL_NAME)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_data_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    # --- 4. 训练循环 ---
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            len(train_df)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(val_df)
        )
        print(f'Val loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        # 保存表现最好的模型
        if val_acc > best_accuracy:
            model_path = os.path.join(config.SAVED_MODELS_PATH, 'best_model_state.bin')
            torch.save(model.state_dict(), model_path)
            best_accuracy = val_acc

if __name__ == '__main__':
    run()