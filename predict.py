from transformers import DistilBertTokenizer
from src.model import SentimentClassifier
from src.dataset import create_data_loader
import src.uncertainty.temperature_scaling as temperature_scaling
from src.uncertainty.temperature_scaling import get_temp_scaled_confidence
from src.uncertainty.mc_dropout import predict_single_with_mc_dropout
from src.uncertainty.conformal_prediction import find_conformal_threshold, get_conformal_set
from src.uncertainty.sngp_model import SNGPClassifier
from src import config
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)
# ------------------------------------------

# ==============================================================================
# 1. 加载模型和分词器 (程序启动时执行一次)
# ==============================================================================
def load_all_resources():
    """在程序启动时，加载所有需要的模型、分词器和参数。"""
    print("="*21 + " Loading all resources " + "="*21)
    device = torch.device(config.DEVICE)
    
    # 1. 加载分词器
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    print("Tokenizer loaded successfully!")

    # 2. 加载基线模型
    baseline_model = SentimentClassifier(n_classes=config.N_CLASSES)
    baseline_model_path = os.path.join(config.SAVED_MODELS_PATH, 'best_model_state.bin')
    baseline_model.load_state_dict(torch.load(baseline_model_path, map_location=device))
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    print("Baseline model loaded successfully!")

    # 3. 加载SNGP模型
    sngp_model = SNGPClassifier(n_classes=config.N_CLASSES)
    sngp_model_path = os.path.join(config.SAVED_MODELS_PATH, 'sngp_best_model_state.bin')
    sngp_model.load_state_dict(torch.load(sngp_model_path, map_location=device))
    sngp_model = sngp_model.to(device)
    sngp_model.eval()
    print("SNGP model loaded successfully!")
    
    # 4. 加载UQ参数
    uq_params = config.get_all_uq_params()
    print("UQ method parameters loaded successfully!")
    print("="*65)
    return baseline_model, sngp_model, tokenizer, device, uq_params
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

def get_sngp_prediction(text, sngp_model, tokenizer, device):
    """使用SNGP模型进行预测，并返回类别和置信度"""
    # SNGP的预测流程与基线模型完全一样，只是传入的模型不同
    pred_class, confidence, _, _ = get_baseline_prediction(text, sngp_model, tokenizer, device)
    # 对于我们简化的SNGP模型，其不确定性也通过置信度来体现
    return pred_class, confidence

# ==============================================================================
# 3. 主函数
# ==============================================================================
def main():
    """主执行函数"""
    baseline_model, sngp_model, tokenizer, device, uq_params = load_all_resources()
    # 检查参数是否加载成功
    if uq_params is None:
        print("Error: The necessary UQ parameters could not be loaded from uq-params.json, please check the file.")
        return
    
    # 读取参数
    OPTIMAL_TEMPERATURE = uq_params.get('temperature')
    Q_HAT = uq_params.get('conformal_q_hat')
    ALPHA = uq_params.get('conformal_alpha')

    # 检查参数是否加载成功
    if OPTIMAL_TEMPERATURE is None or Q_HAT is None or ALPHA is None:
        print("Error: The necessary UQ parameters could not be loaded from uq-params.json, please check the file.")
        return
    
    print("\nThe sentiment analysis prediction system has been launched.")
    print("Enter a sentence for analysis, enter 'exit' or 'exit' to end the program.")
    print("="*65)
    
    while True:
        user_input = input("\nPlease enter a sentence:")
        if user_input.lower() in ['exit', '退出']:
            print("The program has exited.")
            break

        # --- 1. 基线模型预测 ---
        pred_class, base_confidence, logits, probs = get_baseline_prediction(user_input, baseline_model, tokenizer, device)

        print("\n" + "="*24 + " Analysis Result " + "="*24)
        print(f"\nEmotion Prediction Results: 【  {pred_class}  】\n")
        print("-" * 65)

        # --- 2. 各种UQ方法的结果 ---
        # 基线结果
        print(f"【Baseline Model】")
        print(f"  - Confidence level: {base_confidence:.4f}")

        # Temperature Scaling
        calibrated_conf = get_temp_scaled_confidence(logits, OPTIMAL_TEMPERATURE)
        print(f"\n【Temperature Scaling (T={OPTIMAL_TEMPERATURE:.2f})】")
        print(f"  - Post calibration confidence: {calibrated_conf:.4f}")
        
        # MC Dropout
        mc_confidence, mc_uncertainty = predict_single_with_mc_dropout(user_input, baseline_model, tokenizer, device)
        print(f"\n【MC Dropout】")
        print(f"  - Average Confidence: {mc_confidence:.4f}")
        print(f"  - Uncertainty (variance): {mc_uncertainty:.6f}") # 方差通常很小，多显示几位小数

        # Conformal Prediction
        pred_set_indices, set_size = get_conformal_set(probs, Q_HAT)
        pred_set_names = {config.CLASS_NAMES[i] for i in pred_set_indices}
        print(f"\n【Conformal Prediction (Confidence {1-ALPHA:.0%})】")
        print(f"  - Prediction set: {pred_set_names}")
        print(f"  - Prediction set size: {set_size}")

        # SNGP
        sngp_class, sngp_confidence = get_sngp_prediction(user_input, sngp_model, tokenizer, device)
        print(f"\n【SNGP Model】")
        # 注意：SNGP的预测结果(sngp_class)可能与基线模型(pred_class)不同
        print(f"  - Prediction result: {sngp_class}")
        print(f"  - Confidence: {sngp_confidence:.4f}")

        print("="*65)


# 当该脚本被直接运行时，执行main函数
if __name__ == '__main__':
    main()
