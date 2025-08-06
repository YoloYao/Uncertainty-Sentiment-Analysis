from transformers import DistilBertTokenizer
from src.model import SentimentClassifier
from src.dataset import create_data_loader
import src.uncertainty.temperature_scaling as temperature_scaling
from src.uncertainty.temperature_scaling import get_temp_scaled_confidence
from src.uncertainty.mc_dropout import predict_single_with_mc_dropout
from src.uncertainty.conformal_prediction import find_conformal_threshold, get_conformal_set
from src import config
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# --- 路径设置，确保可以导入src目录下的模块 ---
# 将项目根目录添加到系统路径中
# __file__ 代表当前脚本(predict.py)的路径
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)
# ------------------------------------------

# ==============================================================================
# 1. 加载模型和分词器 (程序启动时执行一次)
# ==============================================================================

def load_model_and_tokenizer():
    """加载训练好的基线模型和分词器"""
    print("Loading model and tokenizer ...")
    device = torch.device(config.DEVICE)
    model = SentimentClassifier(n_classes=config.N_CLASSES)

    # 加载已保存的最佳模型权重
    model_path = os.path.join(config.SAVED_MODELS_PATH, 'best_model_state.bin')
    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()  # 切换到评估模式

    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    print("Loading completed!")
    return model, tokenizer, device

# ==============================================================================
# 2. 定义各种预测和不确定性量化(UQ)方法
# ==============================================================================


def get_baseline_prediction(text, model, tokenizer, device):
    """获取基线模型的预测结果、置信度、Logits和Probs"""
    encoded_text = tokenizer.encode_plus(
        text, max_length=config.MAX_LEN, add_special_tokens=True,
        return_token_type_ids=False, padding='max_length',
        return_attention_mask=True, return_tensors='pt',
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        # 计算softmax概率
        probs = F.softmax(outputs, dim=1)
    
    # 从概率中获取置信度和预测类别索引
    confidence, prediction_idx = torch.max(probs, dim=1)
    # 将类别索引转换为类别名称
    prediction_class = config.CLASS_NAMES[prediction_idx.item()]
    
    # 返回所有需要的值
    return prediction_class, confidence.item(), outputs, probs

# ==============================================================================
# 3. 主函数
# ==============================================================================
def main():
    """主执行函数"""
    model, tokenizer, device = load_model_and_tokenizer()

    # 读取参数
    OPTIMAL_TEMPERATURE = config.get_uq_param('temperature')
    Q_HAT = config.get_uq_param('conformal_q_hat')
    ALPHA = config.get_uq_param('conformal_alpha')

    # 检查参数是否加载成功
    if OPTIMAL_TEMPERATURE is None or Q_HAT is None or ALPHA is None:
        print("Error: The necessary UQ parameters could not be loaded from uq-params.json, please check the file.")
        return
    
    print("\nThe sentiment analysis prediction system has been launched.")
    print("Enter a sentence for analysis, enter 'exit' or 'exit' to end the program.")
    print("="*54)
    
    while True:
        user_input = input("\nPlease enter a sentence:")
        if user_input.lower() in ['exit', '退出']:
            print("The program has exited.")
            break

        # --- 1. 基线模型预测 ---
        pred_class, base_confidence, logits, probs = get_baseline_prediction(user_input, model, tokenizer, device)

        print("\n" + "="*20 + " Analysis Result " + "="*20)
        print(f"\nEmotion Prediction Results: 【{pred_class}】\n")
        print("-" * 50)

        # --- 2. 各种UQ方法的结果 ---
        # 基线结果
        print(f"【Baseline Model】")
        print(f"  - Confidence level: {base_confidence:.4f}")

        # Temperature Scaling
        calibrated_conf = get_temp_scaled_confidence(logits, OPTIMAL_TEMPERATURE)
        print(f"\n【Temperature Scaling (T={OPTIMAL_TEMPERATURE:.2f})】")
        print(f"  - 校准后置信度: {calibrated_conf:.4f}")
        # MC Dropout
        mc_confidence, mc_uncertainty = predict_single_with_mc_dropout(user_input, model, tokenizer, device)
        print(f"\n【MC Dropout】")
        print(f"  - 平均置信度: {mc_confidence:.4f}")
        print(f"  - 不确定性 (方差): {mc_uncertainty:.6f}") # 方差通常很小，多显示几位小数

        # Conformal Prediction
        pred_set_indices, set_size = get_conformal_set(probs, Q_HAT)
        pred_set_names = {config.CLASS_NAMES[i] for i in pred_set_indices}
        print(f"\n【Conformal Prediction (置信度 {1-ALPHA:.0%})】")
        print(f"  - 预测集: {pred_set_names}")
        print(f"  - 预测集大小: {set_size}")

        # SNGP
        # sngp_pred, sngp_uncertainty = get_sngp_prediction(user_input)
        # print(f"\n【SNGP】")
        # print(f"  - Confidence level: {sngp_pred}")
        # print(f"  - 不确定性: {sngp_uncertainty}")

        print("="*54)


# 当该脚本被直接运行时，执行main函数
if __name__ == '__main__':
    main()
