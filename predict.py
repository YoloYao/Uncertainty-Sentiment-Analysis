from transformers import DistilBertTokenizer
from src.model import SentimentClassifier
import src.uncertainty.temperature_scaling as temperature_scaling
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
    """获取基线模型的预测结果和原始置信度"""
    encoded_text = tokenizer.encode_plus(
        text,
        max_length=config.MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = F.softmax(outputs, dim=1)

    confidence, prediction = torch.max(probs, dim=1)
    prediction_class = config.CLASS_NAMES[prediction]

    return prediction_class, confidence.item(), outputs


def get_temp_scaled_prediction(logits, temperature):
    """应用温度缩放来获取校准后的置信度"""
    # T值是我们在验证集上找到的最优值
    scaled_logits = logits / temperature
    calibrated_probs = F.softmax(scaled_logits, dim=1)
    calibrated_confidence, _ = torch.max(calibrated_probs, dim=1)

    return calibrated_confidence.item()


def get_mc_dropout_prediction(text, model, tokenizer, device, n_samples=30):
    """
    (占位函数) 应用MC Dropout获取预测和不确定性
    注意：这需要您后续去实现
    """
    # 真正实现时，需要在这里开启模型的dropout层 (model.train())，
    # 然后循环n_samples次进行预测，最后统计预测的均值和方差。
    # print("  [提示] MC Dropout方法尚未实现。")
    return "N/A", "N/A"


def get_conformal_prediction(logits):
    """
    (占位函数) 应用保形预测获取预测集
    注意：这需要您后续去实现
    """
    # 真正实现时，需要在这里实现保形预测的逻辑，计算预测集
    # print("  [提示] Conformal Prediction方法尚未实现。")
    return "{...}", "N/A"


def get_sngp_prediction(text):
    """
    (占位函数) 使用SNGP模型进行预测
    注意：这需要您后续去实现
    """
    # 真正实现时，需要在这里加载并运行SNGP模型
    # print("  [提示] SNGP方法尚未实现。")
    return "N/A", "N/A"


# ==============================================================================
# 3. 主函数
# ==============================================================================
def main():
    """主执行函数"""
    model, tokenizer, device = load_model_and_tokenizer()

    print("\nThe sentiment analysis prediction system has been launched.")
    print("Enter a sentence for analysis, enter 'exit' or 'exit' to end the program.")

    while True:
        user_input = input("\nPlease enter a sentence:")
        if user_input.lower() in ['exit', '退出']:
            print("The program has exited.")
            break

        # --- 1. 基线模型预测 ---
        pred_class, base_confidence, logits = get_baseline_prediction(
            user_input, model, tokenizer, device)

        print("\n" + "="*20 + " Analysis Result " + "="*20)
        print(f"Emotion Prediction Results: {pred_class}")
        print("-" * 50)

        # --- 2. 各种UQ方法的结果 ---
        # 基线结果
        print(f"【Baseline Model】")
        print(f"  - Confidence level: {base_confidence:.4f}")

        # Temperature Scaling
        calibrated_conf = get_temp_scaled_prediction(
            logits, temperature_scaling.OPTIMAL_TEMPERATURE)
        print(f"\n【Temperature Scaling (T={temperature_scaling.OPTIMAL_TEMPERATURE:.2f})】")
        print(f"  - Post calibration reliability: {calibrated_conf:.4f}")

        # MC Dropout
        mc_pred, mc_uncertainty = get_mc_dropout_prediction(user_input, model, tokenizer, device)
        print(f"\n【MC Dropout】")
        print(f"  - 平均置信度: {mc_pred}")
        print(f"  - 不确定性 (方差): {mc_uncertainty}")

        # Conformal Prediction
        pred_set, set_size = get_conformal_prediction(logits)
        print(f"\n【Conformal Prediction (Confidence level at 95%)】")
        print(f"  - 预测集: {pred_set}")
        print(f"  - 预测集大小: {set_size}")

        # SNGP
        sngp_pred, sngp_uncertainty = get_sngp_prediction(user_input)
        print(f"\n【SNGP】")
        print(f"  - Confidence level: {sngp_pred}")
        # print(f"  - 不确定性: {sngp_uncertainty}")

        print("="*54)


# 当该脚本被直接运行时，执行main函数
if __name__ == '__main__':
    main()
