from transformers import DistilBertTokenizer
from src.model import SentimentClassifier
from src.uncertainty.temperature_scaling import get_temp_scaled_confidence
from src.uncertainty.mc_dropout import predict_single_with_mc_dropout
from src.uncertainty.conformal_prediction import get_conformal_set
from src.uncertainty.sngp_model import SNGPClassifier
from src import config
import torch
import torch.nn.functional as F
import sys
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)
# ==============================================================================
# 1. Load model and tokenizer (executed once at program startup)
# ==============================================================================


def load_all_resources():
    """
    At program startup, load all required models, tokenizers, and parameters
    """
    print("="*21 + " Loading all resources " + "="*21)
    device = torch.device(config.DEVICE)

    # 1. Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    print("Tokenizer loaded successfully!")

    # 2. Load baseline model
    baseline_model = SentimentClassifier(n_classes=config.N_CLASSES)
    baseline_model_path = os.path.join(
        config.SAVED_MODELS_PATH, config.SAVED_MODELS_NAME)
    baseline_model.load_state_dict(torch.load(
        baseline_model_path, map_location=device))
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    print("Baseline model loaded successfully!")

    # 3. Loading the SNGP model
    sngp_model = SNGPClassifier(n_classes=config.N_CLASSES)
    sngp_model_path = os.path.join(
        config.SAVED_MODELS_PATH, config.SAVED_SNGP_MODELS_NAME)
    sngp_model.load_state_dict(torch.load(
        sngp_model_path, map_location=device))
    sngp_model = sngp_model.to(device)
    sngp_model.eval()
    print("SNGP model loaded successfully!")

    # 4. Load UQ parameters
    uq_params = config.get_all_uq_params()
    print("UQ method parameters loaded successfully!")
    print("="*65)
    return baseline_model, sngp_model, tokenizer, device, uq_params

# ==============================================================================
# 2. Define the method for obtaining prediction results
# ==============================================================================


def get_baseline_prediction(text, model, tokenizer, device):
    """
    Get the prediction results, confidence level, Logits of the baseline model
    """
    encoded_text = tokenizer.encode_plus(
        text, max_length=config.MAX_LEN, add_special_tokens=True,
        return_token_type_ids=False, padding='max_length',
        return_attention_mask=True, return_tensors='pt',
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        # Calculate softmax probability
        probs = F.softmax(outputs, dim=1)

    # Obtain confidence and predicted category index from probability
    confidence, prediction_idx = torch.max(probs, dim=1)
    # Convert category index to category name
    prediction_class = config.CLASS_NAMES[prediction_idx.item()]

    return prediction_class, confidence.item(), outputs, probs


def get_sngp_prediction(text, sngp_model, tokenizer, device):
    """
    Get the prediction results, confidence level, Logits of the SNGP model
    """
    pred_class, confidence, _, _ = get_baseline_prediction(
        text, sngp_model, tokenizer, device)
    return pred_class, confidence

# ==============================================================================
# 3. Main Function
# ==============================================================================


def main():
    """
    Main Function
    """
    baseline_model, sngp_model, tokenizer, device, uq_params = load_all_resources()
    # Check if the parameters have been loaded successfully
    if uq_params is None:
        print("Error: The necessary UQ parameters could not be loaded from uq-params.json, please check the file.")
        return

    # Read parameters
    OPTIMAL_TEMPERATURE = uq_params.get('temperature')
    Q_HAT = uq_params.get('conformal_q_hat')
    ALPHA = uq_params.get('conformal_alpha')

    # Check if the parameters have been loaded successfully
    if OPTIMAL_TEMPERATURE is None or Q_HAT is None or ALPHA is None:
        print("Error: The necessary UQ parameters could not be loaded from uq-params.json, please check the file.")
        return

    print("\nThe sentiment analysis prediction system has been launched.")
    print("Enter a sentence for analysis, enter 'exit' or 'exit' to end the program.")
    print("="*65)

    while True:
        user_input = input("\nPlease enter a sentence:")
        if user_input.lower() in ['exit', 'quit', 'close']:
            print("The program has exited.")
            break

        # --- 1. Baseline model prediction ---
        pred_class, base_confidence, logits, probs = get_baseline_prediction(
            user_input, baseline_model, tokenizer, device)

        print("\n" + "="*24 + " Analysis Result " + "="*24)
        print(f"\nSentiment Prediction Results: 【  {pred_class}  】\n")
        print("-" * 65)

        # --- 2. The results of various UQ methods ---
        # Baseline model results
        print(f"【Baseline Model】")
        print(f"  - Confidence level: {base_confidence:.4f}")

        # Temperature Scaling
        calibrated_conf = get_temp_scaled_confidence(
            logits, OPTIMAL_TEMPERATURE)
        print(f"\n【Temperature Scaling (T={OPTIMAL_TEMPERATURE:.2f})】")
        print(f"  - Post calibration confidence: {calibrated_conf:.4f}")

        # MC Dropout
        mc_confidence, mc_uncertainty = predict_single_with_mc_dropout(
            user_input, baseline_model, tokenizer, device, config.get_uq_param('mc_sampling_times'))
        print(f"\n【MC Dropout】")
        print(f"  - Average Confidence: {mc_confidence:.4f}")
        # The variance is usually small, display a few more decimal places
        print(f"  - Uncertainty (variance): {mc_uncertainty:.6f}")

        # Conformal Prediction
        pred_set_indices, set_size, set_probs = get_conformal_set(probs, Q_HAT)
        pred_set_names = {config.CLASS_NAMES[i] for i in pred_set_indices}

        print(f"\n【Conformal Prediction (Confidence {1-ALPHA:.0%})】")

        if set_size == 1:
            # If the set size = 1, this is a definite prediction
            single_pred_index = list(pred_set_indices)[0]
            single_pred_confidence = set_probs[single_pred_index]
            print(f"  - Predicting Status: Certain")
            print(f"  - Prediction set: {pred_set_names}")
            print(f"  - Confidence: {single_pred_confidence:.4f}")
        else:
            # If the set size > 1, this is a fuzzy prediction
            print(f"  - Predicting Status: Ambiguous")
            print(f"  - Prediction set: {pred_set_names}")
            print("  - Probability of each category:")
            for idx, prob in set_probs.items():
                print(f"    - {config.CLASS_NAMES[idx]}: {prob:.4f}")

        # SNGP
        sngp_class, sngp_confidence = get_sngp_prediction(
            user_input, sngp_model, tokenizer, device)
        print(f"\n【SNGP Model】")
        # The prediction results of SNGP may differ from the baseline model
        print(f"  - Prediction result: {sngp_class}")
        print(f"  - Confidence: {sngp_confidence:.4f}")

        print("="*65)


if __name__ == '__main__':
    main()
