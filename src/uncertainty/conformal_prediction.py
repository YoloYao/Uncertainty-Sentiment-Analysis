# src/uncertainty/conformal_prediction.py

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

def find_conformal_threshold(model, calib_loader, device, alpha):
    """
    使用校准集来找到保形预测的阈值 q_hat。
    
    :param model: 训练好的模型
    :param calib_loader: 校准集(验证集)的数据加载器
    :param device: 计算设备
    :param alpha: 允许的错误率
    :return: 阈值 q_hat (float)
    """
    model.eval()
    all_calib_probs = []
    all_calib_labels = []

    with torch.no_grad():
        for d in tqdm(calib_loader, desc="Calibrating Conformal Threshold"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs, dim=1)
            
            all_calib_probs.append(probs)
            all_calib_labels.append(labels)
            
    calib_probs = torch.cat(all_calib_probs).cpu().numpy()
    calib_labels = torch.cat(all_calib_labels).cpu().numpy()
    
    # 计算非符合性分数
    n_calib = len(calib_labels)
    scores = 1 - calib_probs[np.arange(n_calib), calib_labels]
    
    # 计算阈值 q_hat
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    q_hat = np.quantile(scores, q_level, method="higher")
    
    return q_hat

def get_conformal_set(probs, q_hat):
    """
    根据给定的概率和阈值q_hat，生成预测集。
    
    :param probs: 模型的softmax概率输出 (torch.Tensor, 形状 [1, n_classes])
    :param q_hat: 校准得到的阈值
    :return: 预测集 (set), 预测集大小 (int)
    """
    # 预测集包含所有概率大于 (1 - q_hat) 的类别
    pred_set_indices = np.where(probs.cpu().numpy() > (1 - q_hat))[1]
    
    if len(pred_set_indices) == 0:
        pred_set_indices = np.array([torch.argmax(probs).item()])
        
    return set(pred_set_indices), len(pred_set_indices)

def get_softmax_outputs(data_loader, model, device):
    """获取模型在给定数据集上的Softmax概率输出"""
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Getting softmax outputs"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs)
            all_labels.append(labels)
            
    return torch.cat(all_probs).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def conformal_predict(calib_probs, calib_labels, test_probs, alpha):
    """
    手动实现保形预测。
    
    :param calib_probs: 校准集的softmax概率输出 (numpy array)
    :param calib_labels: 校准集的真实标签 (numpy array)
    :param test_probs: 测试集的softmax概率输出 (numpy array)
    :param alpha: 允许的错误率 (e.g., 0.05 for 95% confidence)
    :return: 预测集 (list of sets), 预测集大小 (numpy array)
    """
    
    # a. 在校准集上计算“非符合性分数”(non-conformity scores)
    #    这里我们使用一个简单的分数: 1 - 正确类别的概率
    n_calib = len(calib_labels)
    scores = 1 - calib_probs[np.arange(n_calib), calib_labels]
    
    # b. 计算阈值 q_hat
    #    它是非符合性分数的 (1-alpha) 分位数，并经过有限样本修正
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    q_hat = np.quantile(scores, q_level, method="higher")
    
    # c. 在测试集上生成预测集
    prediction_sets = []
    for probs in tqdm(test_probs, desc="Generating Prediction Sets"):
        # 预测集包含所有概率大于 (1 - q_hat) 的类别
        pred_set = np.where(probs > (1 - q_hat))[0]
        # 如果预测集为空 (因为所有概率都太低)，则包含概率最高的那个类别
        if len(pred_set) == 0:
            pred_set = np.array([np.argmax(probs)])
        prediction_sets.append(set(pred_set))
        
    return prediction_sets

def evaluate_conformal(true_labels, pred_sets):
    """评估保形预测的覆盖率和预测集大小"""
    # a. 计算覆盖率
    is_covered = [true_labels[i] in pred_sets[i] for i in range(len(true_labels))]
    coverage = np.mean(is_covered)
    
    # b. 计算平均预测集大小
    set_sizes = [len(s) for s in pred_sets]
    avg_set_size = np.mean(set_sizes)
    
    return coverage, avg_set_size