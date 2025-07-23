# src/engine.py

import torch
import torch.nn as nn
from tqdm import tqdm # tqdm是一个强大的进度条库
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    """
    执行一个训练周期的函数。
    """
    model = model.train() # 将模型设置为训练模式
    losses = []
    correct_predictions = 0

    # 用于存储所有标签和预测
    all_labels = []
    all_preds = []
    
    # 使用tqdm来显示进度条
    for d in tqdm(data_loader, desc="Training"):
    # for i, d in enumerate(tqdm(data_loader, desc="Training")):
        # --- 添加这两行代码 ---
        # if i > 10:  # 只运行10个步骤就跳出循环
            # break
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

        # --- 将当前批次的标签和预测结果收集起来 ---
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
        # 反向传播和优化
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = sum(losses) / len(losses)

    # --- 使用sklearn计算详细指标 ---
    # 1. 让 classification_report 输出一个字典
    report_dict = classification_report(
        all_labels, 
        all_preds, 
        target_names=['negative', 'neutral', 'positive'], 
        labels=[0, 1, 2],
        zero_division=0,
        output_dict=True  # <-- 设置为True，返回字典
    )
    # 2. 从报告字典中提取我们需要的指标
    accuracy = report_dict['accuracy']
    weighted_f1 = report_dict['weighted avg']['f1-score'] # 提取加权平均F1分数
    report_str = classification_report(
        all_labels, 
        all_preds, 
        target_names=['negative', 'neutral', 'positive'], 
        labels=[0, 1, 2],
        zero_division=0
    ) # 用于打印的字符串版本报告
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    # --------------------------------

    # 返回一个包含所有指标的字典
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'classification_report': report_str, # 返回字符串报告用于打印
        'confusion_matrix': cm,
        'weighted_f1': weighted_f1 # <-- 新增返回加权F1分数
    }

def eval_model(model, data_loader, loss_fn, device, n_examples):
    """
    执行模型评估的函数。
    """
    model = model.eval() # 将模型设置为评估模式
    losses = []
    correct_predictions = 0

    # 用于存储所有标签和预测
    all_labels = []
    all_preds = []
    
    with torch.no_grad(): # 在评估时，不计算梯度，以节省内存和计算资源
        for d in tqdm(data_loader, desc="Evaluating"):
        # for i, d in enumerate(tqdm(data_loader, desc="Evaluating")):
            # --- 添加这两行代码 ---
            # if i > 10: # 只运行10个步骤就跳出循环
                # break
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
            
            # --- 将当前批次的标签和预测结果收集起来 ---
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = sum(losses) / len(losses)

    # --- 使用sklearn计算详细指标 ---
    # 生成分类报告
    # 1. 让 classification_report 输出一个字典
    report_dict = classification_report(
        all_labels, 
        all_preds, 
        target_names=['negative', 'neutral', 'positive'], 
        labels=[0, 1, 2],
        zero_division=0,
        output_dict=True  # <-- 设置为True，返回字典
    )
    # 2. 从报告字典中提取我们需要的指标
    accuracy = report_dict['accuracy']
    weighted_f1 = report_dict['weighted avg']['f1-score'] # 提取加权平均F1分数
    report_str = classification_report(
        all_labels, 
        all_preds, 
        target_names=['negative', 'neutral', 'positive'], 
        labels=[0, 1, 2],
        zero_division=0
    ) # 用于打印的字符串版本报告
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    # --------------------------------

    # 返回一个包含所有指标的字典
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'classification_report': report_str, # 返回字符串报告用于打印
        'confusion_matrix': cm,
        'weighted_f1': weighted_f1 # <-- 新增返回加权F1分数
    }