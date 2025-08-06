# src/uncertainty/temperature_scaling.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def find_optimal_temperature(model, val_loader, device):
    """
    在验证集上寻找最优温度T。
    
    :param model: 训练好的模型。
    :param val_loader: 验证集的数据加载器。
    :param device: 计算设备 (e.g., 'cuda' or 'cpu')。
    :return: 最优温度值 (float)。
    """
    model.eval()
    
    # --- 在验证集上获取模型的原始输出 (logits) ---
    all_logits = []
    all_labels = []

    print("正在验证集上获取Logits...")
    with torch.no_grad():
        for d in tqdm(val_loader, desc="Finding Optimal T"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            all_logits.append(logits)
            all_labels.append(labels)

    # 将列表拼接成一个大的张量
    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device)

    # --- 定义并优化温度参数 T ---
    # 将温度T定义为一个可学习的参数，并初始化为1.5
    temperature = nn.Parameter(torch.ones(1).to(device) * 1.5)
    
    # 定义损失函数 (NLL Loss, 等价于交叉熵)
    nll_criterion = nn.CrossEntropyLoss().to(device)
    
    # 使用LBFGS优化器，它很适合这种单参数的简单优化问题
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval_temp():
        # 在优化步骤中，我们需要计算带有温度的logits的损失
        loss = nll_criterion(all_logits / temperature, all_labels)
        loss.backward()
        return loss

    # 开始优化温度参数
    optimizer.step(eval_temp)
    optimal_temperature = temperature.item()
    
    return optimal_temperature


# --- 用于计算校准后置信度的函数 ---
def get_temp_scaled_confidence(logits, temperature):
    """
    将学习到的温度应用到logits上，并返回校准后的置信度。
    
    :param logits: 模型的原始输出logits。
    :param temperature: 优化得到的最优温度。
    :return: 校准后的置信度 (float)。
    """
    # 应用温度缩放
    scaled_logits = logits / temperature
    # 计算校准后的概率
    calibrated_probs = F.softmax(scaled_logits, dim=1)
    # 提取最大概率作为置信度
    calibrated_confidence, _ = torch.max(calibrated_probs, dim=1)
    
    return calibrated_confidence.item()