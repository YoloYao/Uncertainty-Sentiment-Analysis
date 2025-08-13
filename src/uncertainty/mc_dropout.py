import torch
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from tqdm import tqdm
from src import config


def enable_dropout(model):
    """ In evaluation mode, selectively enable all Dropout layers """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_mc_dropout_predictions(model, data_loader, n_samples, device):
    """
    Using MC Dropout for prediction

    :param model: Trained model
    :param data_loader: Test set data loader
    :param n_samples: Sampling frequency of forward propagation
    :param device: Computing device
    :return: Dictionary containing all results
    """
    model.eval()

    # Enable Dropout layer
    enable_dropout(model)

    all_probs_samples = []
    all_labels = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc=f"MC Dropout (N={n_samples})"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            # --- Perform N samplings ---
            batch_probs_samples = []
            for _ in range(n_samples):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                probs = F.softmax(outputs, dim=1)
                batch_probs_samples.append(probs.unsqueeze(0))

            batch_probs_samples = torch.cat(batch_probs_samples, dim=0)
            all_probs_samples.append(batch_probs_samples)
            all_labels.append(labels)

    # Combine the results of all batches together
    all_probs_samples = torch.cat(all_probs_samples, dim=1)
    all_labels = torch.cat(all_labels, dim=0)

    # --- Calculate mean and variance ---
    # Average probability (as the final predicted probability)
    mean_probs = all_probs_samples.mean(dim=0)
    # Predicted variance (as a measure of uncertainty)
    variance = all_probs_samples.var(dim=0)
    # Obtain the final predicted category and confidence level from the average probability
    confidences, predictions = torch.max(mean_probs, 1)

    return {
        "predictions": predictions.cpu().numpy(),
        "confidences": confidences.cpu().numpy(),
        "mean_probs": mean_probs.cpu().numpy(),
        "variance": variance.cpu().numpy(),
        "labels": all_labels.cpu().numpy(),
        "raw_probs": all_probs_samples.cpu().numpy()
    }


def predict_single_with_mc_dropout(text, model, tokenizer, device, n_samples):
    """
    Use MC Dropout to predict a single text and return confidence and uncertainty
    """
    model.eval()
    enable_dropout(model)

    # 1. Text encoding
    encoded_text = tokenizer.encode_plus(
        text, max_length=config.MAX_LEN, add_special_tokens=True,
        return_token_type_ids=False, padding='max_length',
        return_attention_mask=True, return_tensors='pt',
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    # 2. Multiple sampling
    with torch.no_grad():
        probs_samples = []
        for _ in range(n_samples):
            outputs = model(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)
            probs_samples.append(probs)

        probs_samples = torch.cat(probs_samples, dim=0)

    # 3. Calculate statistical measures
    mean_probs = probs_samples.mean(dim=0)
    variance = probs_samples.var(dim=0)

    mean_confidence, prediction_idx = torch.max(mean_probs, dim=0)

    # Extract the uncertainty (variance) corresponding to the predicted category
    uncertainty_score = variance[prediction_idx].item()

    return mean_confidence.item(), uncertainty_score


def plot_uncertainty_distribution(ax, scores, correct_mask, title):
    """
    在指定的matplotlib轴上绘制“不确定性 vs 错误”的KDE图。
    :param ax: Matplotlib subplot axis
    :param scores: 该方法的不确定性分数数组
    :param correct_mask: 标记预测是否正确的布尔数组
    :param title: 子图的标题
    """
    # 绘制正确预测的分布 (蓝色)
    sns.kdeplot(scores[correct_mask], ax=ax, label='Correct Predictions',
                fill=True, alpha=0.6, linewidth=2)
    # 绘制错误预测的分布 (橙色)
    sns.kdeplot(scores[~correct_mask], ax=ax, label='Incorrect Predictions',
                fill=True, alpha=0.6, linewidth=2)

    # 计算并绘制两条分布的平均值垂直线
    mean_correct = np.mean(scores[correct_mask])
    mean_incorrect = np.mean(scores[~correct_mask])
    ax.axvline(mean_correct, color='blue', linestyle='--',
               label=f'Mean (Correct) = {mean_correct:.3f}')
    ax.axvline(mean_incorrect, color='darkorange', linestyle='--',
               label=f'Mean (Incorrect) = {mean_incorrect:.3f}')

    # 美化子图
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Uncertainty (Predictive Entropy)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle='dotted')
