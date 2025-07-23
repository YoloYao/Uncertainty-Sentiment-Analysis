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
    test_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "test.csv"))

    # --- 2. 初始化分词器和数据加载器 ---
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    
    train_data_loader = create_data_loader(config.PROCESSED_DATA_PATH + "/train.csv", tokenizer, config.MAX_LEN, config.BATCH_SIZE)
    val_data_loader = create_data_loader(config.PROCESSED_DATA_PATH + "/validation.csv", tokenizer, config.MAX_LEN, config.BATCH_SIZE)
    test_data_loader = create_data_loader(config.PROCESSED_DATA_PATH + "/test.csv", tokenizer, config.MAX_LEN, config.BATCH_SIZE)

    # --- 3. 初始化模型、优化器、损失函数等 ---
    device = torch.device(config.DEVICE)
    print("-" * 54)
    print("Model training begins")
    print(f"Using device: {device}")
    print("-" * 54)
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
    best_f1_score = 0

    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        print('-' * 10)
        
        train_results = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            len(train_df)
        )
        
        # 从字典中获取所有需要的指标
        train_acc = train_results['accuracy']
        train_loss = train_results['loss']
        train_f1 = train_results['weighted_f1'] # 获取F1分数
        
        print(f'\nTrain Accuracy: {train_acc:.4f}   Train Loss: {train_loss:.4f}')
        print("\n" + "-" * 8 + " Training Set Classification Report " + "-" * 9)
        print(train_results['classification_report'])
        print("\n" + "-" * 10 + " Training Set Confusion Matrix " + "-" * 11)
        print(train_results['confusion_matrix'])
        print("-" * 54)

        # --- 修改评估结果的接收和打印 ---
        eval_results = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(val_df)
        )
        # 从字典中获取所有需要的指标
        val_acc = eval_results['accuracy']
        val_loss = eval_results['loss']
        val_f1 = eval_results['weighted_f1'] # 获取F1分数
        
        print(f'\nVal Accuracy: {val_acc:.4f}   Val Loss: {val_loss:.4f}')
        print("\n" + "-" * 8 + " Validation Set Classification Report " + "-" * 7)
        print(eval_results['classification_report'])
        print("\n" + "-" * 10 + " Validation Set Confusion Matrix " + "-" * 9)
        print(eval_results['confusion_matrix'])
        print("-" * 54)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc) # val_acc现在来自报告字典
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1) # （可选）也可以记录F1分数

        # 保存表现最好的模型
        if val_f1 > best_f1_score:
            model_path = os.path.join(config.SAVED_MODELS_PATH, 'best_model_state.bin')
            torch.save(model.state_dict(), model_path)
            best_f1_score = val_f1 # 更新最佳F1分数
            print("*" * 54)
            print(f" * New best model has been saved: (F1-score: {best_f1_score:.4f}) *")
            print("*" * 54)

if __name__ == '__main__':
    run()