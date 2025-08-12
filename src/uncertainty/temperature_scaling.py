import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


def find_optimal_temperature(model, val_loader, device):
    """
    Find the optimal temperature T on the validation set

    :param model: Trained model
    :param val_loader: Validation set data loader
    :param device: Computing device
    :return: Optimal temperature value (float)。
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
