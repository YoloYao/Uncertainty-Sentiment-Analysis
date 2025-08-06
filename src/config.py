# src/config.py

import torch
import os
import json

# 使用 os.path.abspath 和 __file__ 来获取当前文件的绝对路径，然后推导出项目根目录
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 这一行在某些IDE中可能不准确
# BASE_PATH = ".." # 使用相对路径更简单，假设我们总是在src目录或根目录运行脚本

DATA_PATH = os.path.join(BASE_PATH, "data")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")
SAVED_MODELS_PATH = os.path.join(BASE_PATH, "saved_models")
SAVED_MODELS_NAME = 'best_model_state.bin'

SRC_PATH = os.path.join(BASE_PATH, "src")
UQ_PARAMS_PATH = os.path.join(SRC_PATH, 'uq_params.json')
# 确保保存模型的目录存在
os.makedirs(SAVED_MODELS_PATH, exist_ok=True)

# --- 模型与分词器 ---
MODEL_NAME = 'distilbert-base-uncased'
TOKENIZER = None # 将在训练脚本中初始化

# --- 训练参数 ---
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5

# --- 数据集信息 ---
# 标签映射，与之前预处理时一致
# 0: 负面, 1: 中性, 2: 正面
CLASS_NAMES = ['negative', 'neutral', 'positive']
N_CLASSES = len(CLASS_NAMES)

# 量化方法参数
_UQ_PARAMS = {}
def _load_uq_params():
    """(内部函数) 加载uq_params.json文件内容到_UQ_PARAMS变量中。"""
    global _UQ_PARAMS
    try:
        with open(UQ_PARAMS_PATH, 'r') as f:
            _UQ_PARAMS = json.load(f)
        print(f"Uncertainty quantification method parameters have been successfully loaded.")
    except FileNotFoundError:
        print(f"Warning: UQ parameter file not found: {UQ_PARAMS_PATH}. Some functions may not be available.")
        _UQ_PARAMS = {}

def get_uq_param(param_name):
    """
    从已加载的参数中获取指定参数的值。
    
    :param param_name: (str) 参数名称。
    :return: 参数的值，如果找不到则返回 None。
    """
    return _UQ_PARAMS.get(param_name)

def update_uq_params(**kwargs):
    """
    更新并保存UQ参数文件 (uq_params.json)。
    (修正版：增加了自动类型转换功能)
    """
    global _UQ_PARAMS
    params_path = UQ_PARAMS_PATH # 确保 UQ_PARAMS_PATH 在文件上方已定义
    
    # 1. 读取现有参数
    try:
        with open(params_path, 'r') as f:
            existing_params = json.load(f)
    except FileNotFoundError:
        existing_params = {}
    
    # 2. 用新传入的参数更新字典
    existing_params.update(kwargs)
    
    # --- 核心修改：在保存前，遍历字典并转换所有非标准数字类型 ---
    import numpy as np
    
    params_to_save = {}
    for key, value in existing_params.items():
        if isinstance(value, (np.float32, np.float64)):
            params_to_save[key] = float(value) # 将Numpy浮点数转为Python浮点数
        elif isinstance(value, (np.int32, np.int64)):
            params_to_save[key] = int(value) # 将Numpy整数转为Python整数
        else:
            params_to_save[key] = value # 其他类型保持不变
    # -----------------------------------------------------------------

    # 3. 将清理过的字典写入文件
    with open(params_path, 'w') as f:
        json.dump(params_to_save, f, indent=4)
        
    print(f"UQ parameters have been successfully updated.")

_load_uq_params()