import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from tqdm import tqdm


def get_confidence_level(val_data_loader, device, model):
    """
    Get the confidence level of the current model (overconfidence or lack of confidence)

    :param val_loader: Validation set data loader
    :param device: Computing device
    :param model: Trained model
    :return: Average Confidence and Accuracy (float)
    """
    all_confidences = []
    correct_predictions = 0
    total_samples = 0

    print("Calibrating the model before validation set evaluation...")
    with torch.no_grad():
        for d in tqdm(val_data_loader, desc="Diagnosing Calibration"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            # 1. Get model output
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 2. Calculate probability and confidence
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)

            # 3. Collect all confidence levels
            all_confidences.extend(confidences.cpu().numpy())

            # 4. Accumulate the number of correct predictions and the total sample size
            correct_predictions += torch.sum(predictions == labels).item()
            total_samples += len(labels)

    # 5. Calculate the final indicator
    average_confidence = np.mean(all_confidences)
    accuracy = correct_predictions / total_samples

    return average_confidence, accuracy


def find_optimal_temperature(model, val_loader, device):
    """
    Find the optimal temperature T on the validation set

    :param model: Trained model
    :param val_loader: Validation set data loader
    :param device: Computing device
    :return: Optimal temperature value (float)
    """
    model.eval()

    # --- 1. Obtain the raw output of the model on the validation set (logits) ---
    all_logits = []
    all_labels = []

    print("Getting Logits on Validation set...")
    with torch.no_grad():
        for d in tqdm(val_loader, desc="Finding Optimal T"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            all_logits.append(logits)
            all_labels.append(labels)

    # Concatenate the list into a large tensor
    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device)

    # --- 2. Define and optimize temperature parameters T ---
    # Define temperature T as a learnable parameter and initialize it to 1.5
    temperature = nn.Parameter(torch.ones(1).to(device) * 1.5)

    # Define loss function
    nll_criterion = nn.CrossEntropyLoss().to(device)

    # Using LBFGS optimizer for single parameter optimization
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval_temp():
        # In the optimization step, calculate the loss of logits with temperature
        loss = nll_criterion(all_logits / temperature, all_labels)
        # Calculate the gradient of loss with respect to temperature T
        loss.backward()
        return loss

    # Start optimizing temperature parameters
    optimizer.step(eval_temp)
    optimal_temperature = temperature.item()

    return optimal_temperature


def get_temp_scaled_confidence(logits, temperature):
    """
    Apply the learned temperature to logits and return the calibrated confidence level

    :param logits: The original output logits of the model
    :param temperature: Optimal temperature obtained through optimization
    :return: Confidence level after calibration (float)
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Calculate the probability after calibration
    calibrated_probs = F.softmax(scaled_logits, dim=1)
    # Extract the maximum probability as the confidence level
    calibrated_confidence, _ = torch.max(calibrated_probs, dim=1)

    return calibrated_confidence.item()


def get_temp_logits(test_data_loader, device, model):
    """
    Retrieve Logits on the test set

    :param test_data_loader: Test set data loader
    :param device: Computing device
    :param model: Trained model
    :return: test_logits and Labels
    """
    test_logits = []
    test_labels = []
    with torch.no_grad():
        for d in tqdm(test_data_loader, desc="Testing"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            test_logits.append(logits)
            test_labels.append(labels)

    test_logits = torch.cat(test_logits)
    test_labels = torch.cat(test_labels)

    return test_logits, test_labels


def get_adaptive_bins(probs, labels, n_bins):
    """
    Get the confidence and accuracy of adaptive binning
    """
    confidences = np.max(probs, axis=1)
    accuracies = (np.argmax(probs, axis=1) == labels)
    sorted_indices = np.argsort(confidences)
    confidences, accuracies = confidences[sorted_indices], accuracies[sorted_indices]
    samples_per_bin = len(confidences) / n_bins
    bin_avg_conf, bin_accuracy = [], []
    for i in range(n_bins):
        start_idx = int(i * samples_per_bin)
        end_idx = int((i + 1) * samples_per_bin)
        if start_idx == end_idx:
            continue
        bin_avg_conf.append(np.mean(confidences[start_idx:end_idx]))
        bin_accuracy.append(np.mean(accuracies[start_idx:end_idx]))
    return bin_avg_conf, bin_accuracy


def calculate_ece_adaptive(probs, labels, n_bins):
    """
    Calculate the Expected Calibration Error (ECE)
    """
    # Find the maximum probability value of each sample as confidence level
    confidences = np.max(probs, axis=1)
    # Record accuracies
    accuracies = (np.argmax(probs, axis=1) == labels)

    # Sort all samples from low to high according to their confidence scores
    sorted_indices = np.argsort(confidences)
    confidences = confidences[sorted_indices]
    accuracies = accuracies[sorted_indices]

    ece = 0.0
    # Calculate the average sample size for each sub box
    samples_per_bin = len(confidences) / n_bins

    for i in range(n_bins):
        start_idx = int(i * samples_per_bin)
        end_idx = int((i + 1) * samples_per_bin)

        bin_confidences = confidences[start_idx:end_idx]
        bin_accuracies = accuracies[start_idx:end_idx]

        if len(bin_confidences) > 0:
            avg_confidence = np.mean(bin_confidences)
            accuracy = np.mean(bin_accuracies)
            ece += np.abs(avg_confidence - accuracy)

    return (ece / n_bins) * 100


def plot_reliability_diagram_final(ax, probs, labels, title):
    """
    Draw a reliability chart (bar chart)
    """
    confidences = np.max(probs, axis=1)
    accuracies = (np.argmax(probs, axis=1) == labels)
    n_bins = 10

    # Ensure that each box contains the same number of samples (10% of data)
    bin_boundaries = np.percentile(
        confidences, np.linspace(0, 100, n_bins + 1))

    ece = calculate_ece_adaptive(probs, labels, n_bins)

    # Draw diagonal dashed lines
    ax.plot([0, 1], [0, 1], 'k--', zorder=2)

    for i in range(n_bins):
        # Filter out all samples with confidence levels falling within the current box boundary
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i+1])
        # For the first box, it contains a confidence level equal to 0
        if i == 0:
            in_bin = (confidences >= bin_boundaries[i]) & (
                confidences <= bin_boundaries[i+1])

        prop_in_bin = np.mean(in_bin)

        # Calculate the average accuracy and average confidence within the box
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            # Draw unequal width bar charts
            bin_width = bin_boundaries[i+1] - bin_boundaries[i]
            bin_center = bin_boundaries[i] + bin_width / 2

            # Draw a blue column, height represents the true accuracy
            ax.bar(
                bin_center,
                accuracy_in_bin,
                width=bin_width,
                color='blue',
                edgecolor='black',
                alpha=0.9,
                zorder=1
            )

            gap = avg_confidence_in_bin - accuracy_in_bin
            if gap > 0:
                ax.bar(
                    bin_center,
                    gap,
                    width=bin_width,
                    bottom=accuracy_in_bin,
                    color='red',
                    edgecolor='red',
                    alpha=0.3,
                    hatch='//',
                    zorder=1
                )

    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='dotted')
    ax.text(0.5, 0.1, f'ECE = {ece:.1f}%',
            ha='center', va='center', fontsize=20,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=2))
    output_patch = mpatches.Patch(
        facecolor='blue', edgecolor='black', label='Outputs (Accuracy)')
    gap_patch = mpatches.Patch(facecolor='red', edgecolor='red',
                               alpha=0.3, hatch='//', label='Gap (Overconfidence)')
    ax.legend(handles=[output_patch, gap_patch], loc='upper left', fontsize=12)


def plot_reliability_diagram_adaptive(ax, probs, labels, title):
    """
    Draw a reliability chart (line chart)
    """
    confidences = np.max(probs, axis=1)
    accuracies = (np.argmax(probs, axis=1) == labels)
    n_bins = 10

    # Sort by confidence level
    sorted_indices = np.argsort(confidences)
    confidences = confidences[sorted_indices]
    accuracies = accuracies[sorted_indices]

    samples_per_bin = len(confidences) / n_bins

    bin_avg_conf = []
    bin_accuracy = []

    for i in range(n_bins):
        start_idx = int(i * samples_per_bin)
        end_idx = int((i + 1) * samples_per_bin)

        bin_confidences = confidences[start_idx:end_idx]
        bin_accuracies = accuracies[start_idx:end_idx]

        if len(bin_confidences) > 0:
            bin_avg_conf.append(np.mean(bin_confidences))
            bin_accuracy.append(np.mean(bin_accuracies))

    ece = calculate_ece_adaptive(probs, labels, n_bins)

    ax.plot([0, 1], [0, 1], 'k--', zorder=1)

    # Draw a line chart, as the width of the boxes varies, it is more appropriate to use a line chart
    ax.plot(bin_avg_conf, bin_accuracy, 's-', color='blue',
            label='Outputs (Accuracy)', zorder=2)

    # Fill Gap
    y_upper = np.maximum(bin_accuracy, bin_avg_conf)
    y_lower = np.minimum(bin_accuracy, bin_avg_conf)
    ax.fill_between(bin_avg_conf, y_lower, y_upper, color='red',
                    alpha=0.3, label='Gap (Miscalibration)', zorder=1)

    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='dotted')

    ax.text(0.5, 0.1, f'ECE = {ece:.1f}%',
            ha='center', va='center', fontsize=20,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=2))

    ax.legend(loc='upper left', fontsize=12)


def plot_reliability_diagram_line_style(ax, probs, labels, title, color):
    """
    Draw a reliability chart (line chart with color)
    """
    n_bins = 10
    ece = calculate_ece_adaptive(probs, labels, n_bins)
    avg_conf, accuracy_in_bin = get_adaptive_bins(probs, labels, n_bins)

    avg_conf_np = np.array(avg_conf)
    accuracy_in_bin_np = np.array(accuracy_in_bin)

    ax.plot([0, 1], [0, 1], 'k--', zorder=2)
    ax.plot(avg_conf_np, accuracy_in_bin_np, 'o-',
            color=color, label='Outputs (Accuracy)', zorder=3)

    ax.fill_between(
        avg_conf_np,
        accuracy_in_bin_np,
        avg_conf_np,
        where=avg_conf_np > accuracy_in_bin_np,
        color='red',
        alpha=0.3,
        interpolate=True,
        label='Gap (Miscalibration)',
        zorder=1
    )

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='dotted')
    ax.text(0.5, 0.1, f'ECE = {ece:.1f}%', ha='center', va='center', fontsize=18,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=2))

    output_patch = mlines.Line2D(
        [], [], color=color, marker='o', label='Outputs (Accuracy)')
    gap_patch = mpatches.Patch(
        color='red', alpha=0.3, label='Gap (Miscalibration)')
    ax.legend(handles=[output_patch, gap_patch], loc='upper left', fontsize=10)
