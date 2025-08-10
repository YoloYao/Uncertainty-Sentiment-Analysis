import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src import config
import numpy as np

def enable_dropout(model):
    """ 在评估模式下，选择性地开启所有Dropout层 """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
def get_mc_dropout_predictions(model, data_loader, n_samples, device):
    """
    使用MC Dropout进行预测。
    
    :param model: 训练好的模型
    :param data_loader: 测试集的数据加载器
    :param n_samples: 前向传播的采样次数
    :param device: 计算设备
    :return: 包含所有结果的字典
    """
    model.eval()      # 首先设置为评估模式
    enable_dropout(model) # 然后只开启Dropout层
    
    all_probs_samples = [] # 修改变量名以反映其内容
    all_labels = []
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc=f"MC Dropout (N={n_samples})"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            # --- 核心：进行N次采样 ---
            batch_probs_samples = []
            for _ in range(n_samples):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs, dim=1)
                batch_probs_samples.append(probs.unsqueeze(0)) # 增加一个维度用于堆叠
            
            # 形状: [n_samples, batch_size, n_classes]
            batch_probs_samples = torch.cat(batch_probs_samples, dim=0)
            all_probs_samples.append(batch_probs_samples)
            all_labels.append(labels)
          
    # 将所有批次的结果拼接起来
    # 形状: [n_samples, total_samples, n_classes]
    all_probs_samples = torch.cat(all_probs_samples, dim=1)
    all_labels = torch.cat(all_labels, dim=0)
    
    # --- 计算均值和方差 ---
    # 平均概率 (作为最终的预测概率)
    mean_probs = all_probs_samples.mean(dim=0)
    # 预测的方差 (作为不确定性度量)
    variance = all_probs_samples.var(dim=0)
    # 从平均概率中得到最终的预测类别和置信度
    confidences, predictions = torch.max(mean_probs, 1)
    
    return {
        "predictions": predictions.cpu().numpy(),
        "confidences": confidences.cpu().numpy(),
        "mean_probs": mean_probs.cpu().numpy(),
        "variance": variance.cpu().numpy(),
        "labels": all_labels.cpu().numpy(),
        "raw_probs": all_probs_samples.cpu().numpy()
    }
    
# --- 用于预测单句话的函数 ---
def predict_single_with_mc_dropout(text, model, tokenizer, device, n_samples):
    """
    使用MC Dropout预测单条文本，并返回置信度和不确定性。
    """
    model.eval()
    enable_dropout(model)
    
    # 1. 文本编码
    encoded_text = tokenizer.encode_plus(
        text, max_length=config.MAX_LEN, add_special_tokens=True,
        return_token_type_ids=False, padding='max_length',
        return_attention_mask=True, return_tensors='pt',
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    # 2. 多次采样
    with torch.no_grad():
        probs_samples = []
        for _ in range(n_samples):
            outputs = model(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)
            probs_samples.append(probs)
        
        # 形状: [n_samples, 1, n_classes] -> [n_samples, n_classes]
        probs_samples = torch.cat(probs_samples, dim=0)

    # 3. 计算统计量
    mean_probs = probs_samples.mean(dim=0)
    variance = probs_samples.var(dim=0)
    
    mean_confidence, prediction_idx = torch.max(mean_probs, dim=0)
    
    # 提取预测类别对应的不确定性(方差)
    uncertainty_score = variance[prediction_idx].item()
    
    return mean_confidence.item(), uncertainty_score