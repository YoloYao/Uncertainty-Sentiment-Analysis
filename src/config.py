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
        # print(f"Uncertainty quantification method parameters have been successfully loaded.")
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

def get_all_uq_params():
    """
    从已加载的参数中获取指定参数的值。
    
    :param param_name: (str) 参数名称。
    :return: 参数的值，如果找不到则返回 None。
    """
    return _UQ_PARAMS

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

# --- 3. 新增！可复用的特征分析函数 ---

# 将特征计算所需的词典也定义在这里
POSITIVE_WORDS = {'love', 'great', 'amazing', 'fantastic', 'good', 'best', 'happy', 'delicious'}
NEGATIVE_WORDS = {'bad', 'worst', 'terrible', 'boring', 'slow', 'hate', 'mess', 'disease'}
CONTRAST_WORDS = {'but', 'however', 'although', 'though', 'despite', 'not'}

def _count_sentiment_words(text, sentiment_words):
    count = 0
    for word in str(text).split():
        if word in sentiment_words:
            count += 1
    return count

def _contains_contrast_words(text):
    for word in str(text).split():
        if word in CONTRAST_WORDS:
            return 1
    return 0

def _get_text_length(text):
    return len(str(text).split())

def analyze_features_by_group(df, uncertainty_col, group_name=""):
    """
    接收一个DataFrame，将其分为四个组进行特征分析并打印报告。
    
    :param df: 必须包含 'text', 'predicted_label', 和指定的uncertainty_col列
    :param uncertainty_col: (str) DataFrame中不确定性分数列的名称
    :param group_name: (str) 在报告标题中显示的分析组名称
    """
    print("\n" + "="*20 + f" {group_name} 特征对比分析 " + "="*20)
    
    # 1. 定义阈值和标签
    uncertainty_median = df[uncertainty_col].median()
    POSITIVE_LABEL = CLASS_NAMES.index('positive')
    NEGATIVE_LABEL = CLASS_NAMES.index('negative')

    # 2. 筛选四个子组
    certain_mask = df[uncertainty_col] <= uncertainty_median
    uncertain_mask = df[uncertainty_col] > uncertainty_median
    
    positive_mask = df['predicted_label'] == POSITIVE_LABEL
    negative_mask = df['predicted_label'] == NEGATIVE_LABEL

    certain_correct_positive = df[certain_mask & positive_mask]
    certain_correct_negative = df[certain_mask & negative_mask]
    uncertain_correct_positive = df[uncertain_mask & positive_mask]
    uncertain_correct_negative = df[uncertain_mask & negative_mask]

    all_groups = {
        f"Certain Positive ({group_name})": certain_correct_positive,
        f"Certain Negative ({group_name})": certain_correct_negative,
        f"Uncertain Positive ({group_name})": uncertain_correct_positive,
        f"Uncertain Negative ({group_name})": uncertain_correct_negative
    }
    
    # 3. 计算特征并打印
    for name, group_df in all_groups.items():
        if not group_df.empty:
            pos_words = group_df['text'].apply(lambda x: _count_sentiment_words(x, POSITIVE_WORDS)).mean()
            neg_words = group_df['text'].apply(lambda x: _count_sentiment_words(x, NEGATIVE_WORDS)).mean()
            contrast_ratio = group_df['text'].apply(_contains_contrast_words).mean()
            text_len = group_df['text'].apply(_get_text_length).mean()
            
            print(f"\n--- 分析组: {name} (共 {len(group_df)} 个样本) ---")
            print(f"  - 平均正面词数: {pos_words:.2f}")
            print(f"  - 平均负面词数: {neg_words:.2f}")
            print(f"  - 包含转折词的比例: {contrast_ratio:.2%}")
            print(f"  - 平均文本长度: {text_len:.2f} 词")
        else:
            print(f"\n--- 分析组: {name} (共 0 个样本) ---")
            
    print("="*(54 + len(group_name)))