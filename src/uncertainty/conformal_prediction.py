import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.patches as mpatches


def get_softmax_outputs(data_loader, model, device):
    """
    Get the Softmax probability output and true labels of the model on a given dataset
    """
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


def find_conformal_threshold(model, calib_loader, device, alpha):
    """
    Using the calibration set to find the threshold q-hat for conformal prediction

    :param model: Trained model
    :param calib_loader: Validation set data loader
    :param device: Computing device
    :param alpha: Allowable error rate
    :return: Threshold q-hat (float)
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

    # Calculate non conformity score
    n_calib = len(calib_labels)
    scores = 1 - calib_probs[np.arange(n_calib), calib_labels]

    # Calculate threshold q-hat
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    q_hat = np.quantile(scores, q_level, method="higher")

    return q_hat


def get_conformal_set(probs, q_hat):
    """
    Generate a prediction set based on probability and threshold q'hat
    """
    probs_np = probs.cpu().numpy().flatten()
    # Filter eligible data
    pred_set_indices = np.where(probs_np > (1 - q_hat))[0]

    # When all probabilities are low, return the highest probability term
    if len(pred_set_indices) == 0:
        pred_set_indices = np.array([np.argmax(probs_np)])

    # Store the probability of each category in the prediction set
    set_probs = {idx: probs_np[idx] for idx in pred_set_indices}

    return set(pred_set_indices), len(pred_set_indices), set_probs


def conformal_predict(calib_probs, calib_labels, test_probs, alpha):
    """
    手动实现保形预测。

    :param calib_probs: 校准集的softmax概率输出 (numpy array)
    :param calib_labels: 校准集的真实标签 (numpy array)
    :param test_probs: 测试集的softmax概率输出 (numpy array)
    :param alpha: 允许的错误率 (e.g., 0.05 for 95% confidence)
    :return: 预测集 (list of sets), 预测集大小 (numpy array)
    """

    # a. Calculate the  non conformance score on the validation set
    n_calib = len(calib_labels)
    scores = 1 - calib_probs[np.arange(n_calib), calib_labels]

    # b. Calculate threshold q-hat
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    q_hat = np.quantile(scores, q_level, method="higher")

    # c. Generate a prediction set on the test set
    prediction_sets = []
    for probs in tqdm(test_probs, desc="Generating Prediction Sets"):
        pred_set = np.where(probs > (1 - q_hat))[0]
        # When all probabilities are low, return the highest probability term
        if len(pred_set) == 0:
            pred_set = np.array([np.argmax(probs)])
        prediction_sets.append(set(pred_set))

    return prediction_sets


def evaluate_conformal(true_labels, pred_sets):
    """
    Evaluate the coverage and prediction set size of conformal prediction
    """
    # 1. Calculate coverage rate
    is_covered = [true_labels[i] in pred_sets[i]
                  for i in range(len(true_labels))]
    coverage = np.mean(is_covered)

    # 2. Calculate the average prediction set size
    set_sizes = [len(s) for s in pred_sets]
    avg_set_size = np.mean(set_sizes)

    return coverage, avg_set_size


def plot_coverage_guarantee_binned(ax, desired_confidences, actual_coverages, title):
    """
    以柱状图和Gap的形式绘制覆盖率保证图。
    """
    n_bins = 10
    # 我们只关注我们感兴趣的置信度范围，例如 0.5 到 1.0
    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)

    ax.plot([0.5, 1], [0.5, 1], 'k--', zorder=2)  # 绘制对角虚线

    for i in range(n_bins):
        # 找到落入当前箱内的数据点
        in_bin_mask = (desired_confidences >= bin_boundaries[i]) & (
            desired_confidences < bin_boundaries[i+1])

        if np.sum(in_bin_mask) > 0:
            # 计算箱内的平均期望置信度和平均实际覆盖率
            avg_desired_conf = np.mean(desired_confidences[in_bin_mask])
            avg_actual_coverage = np.mean(actual_coverages[in_bin_mask])

            bin_width = bin_boundaries[i+1] - bin_boundaries[i]
            bin_center = bin_boundaries[i] + bin_width / 2

            # 绘制蓝色柱子 (实际覆盖率)
            ax.bar(
                bin_center,
                avg_actual_coverage,
                width=bin_width,
                color='blue',
                edgecolor='black',
                alpha=0.9,
                zorder=1
            )

            # 绘制红色误差区域 (Gap)
            # Gap = 期望值 - 实际值。在这里我们只展示 Under-coverage 的情况
            gap = avg_desired_conf - avg_actual_coverage
            if gap > 0:  # 只在实际覆盖率低于期望时显示Gap (Under-coverage)
                ax.bar(
                    bin_center,
                    gap,
                    width=bin_width,
                    bottom=avg_actual_coverage,
                    color='red',
                    edgecolor='red',
                    alpha=0.3,
                    hatch='//',
                    zorder=1
                )

    ax.set_xlabel('Desired Confidence Level (1 - alpha)', fontsize=14)
    ax.set_ylabel('Empirical Coverage', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.5, 1)
    ax.grid(True, linestyle='dotted')

    # 添加图例
    output_patch = mpatches.Patch(
        facecolor='blue', edgecolor='black', label='Empirical Coverage')
    gap_patch = mpatches.Patch(facecolor='red', edgecolor='red',
                               alpha=0.3, hatch='//', label='Gap (Under-coverage)')
    ax.legend(handles=[output_patch, gap_patch], loc='upper left', fontsize=12)
