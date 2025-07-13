# src/engine.py

import torch
import torch.nn as nn
from tqdm import tqdm # tqdm是一个强大的进度条库

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    """
    执行一个训练周期的函数。
    """
    model = model.train() # 将模型设置为训练模式
    losses = []
    correct_predictions = 0

    # 使用tqdm来显示进度条
    for d in tqdm(data_loader, desc="Training"):
        # 将数据移动到指定设备
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 计算损失
        loss = loss_fn(outputs, labels)

        # 计算准确率
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        # 反向传播和优化
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.float() / n_examples, sum(losses) / len(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    """
    执行模型评估的函数。
    """
    model = model.eval() # 将模型设置为评估模式
    losses = []
    correct_predictions = 0

    with torch.no_grad(): # 在评估时，不计算梯度，以节省内存和计算资源
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.float() / n_examples, sum(losses) / len(losses)