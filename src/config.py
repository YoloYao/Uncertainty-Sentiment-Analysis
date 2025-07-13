# src/config.py

import torch

# --- 项目路径 ---
# 使用 os.path.abspath 和 __file__ 来获取当前文件的绝对路径，然后推导出项目根目录
import os
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 这一行在某些IDE中可能不准确
# BASE_PATH = ".." # 使用相对路径更简单，假设我们总是在src目录或根目录运行脚本

DATA_PATH = os.path.join(BASE_PATH, "data")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")
SAVED_MODELS_PATH = os.path.join(BASE_PATH, "saved_models")

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
EPOCHS = 5 # 初始可以设置少一点，比如3-5个epoch
LEARNING_RATE = 2e-5 # 对于Transformer微调，一个较小的学习率通常效果更好

# --- 数据集信息 ---
# 标签映射，与之前预处理时一致
# 0: 负面, 1: 中性, 2: 正面
CLASS_NAMES = ['negative', 'neutral', 'positive']
N_CLASSES = len(CLASS_NAMES)