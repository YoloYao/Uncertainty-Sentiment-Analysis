import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    """
    Execute for one training Epoch.
    """
    # Set the model to training mode, Activate some model layers
    model = model.train()
    losses = []
    correct_predictions = 0

    # Store all labels and predictions
    all_labels = []
    all_preds = []

    # Use tqdm to display progress bar
    for d in tqdm(data_loader, desc="Training"):
        # for i, d in enumerate(tqdm(data_loader, desc="Training")):
        # if i > 10:  # Jumping out of the loop after only running 10 steps
        # break

        # Move data to the designated device
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        # Forward Propagation
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Calculate Losses
        loss = loss_fn(outputs, labels)

        # Calculate Accuracy
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        # Collect labels and predicted results for the current batch
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        # Backward propagation and optimization
        loss.backward()
        # Gradient clipping to prevent excessive training deviation
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # Parameter update
        optimizer.zero_grad()  # Clean up calculation records

    # Average loss
    avg_loss = sum(losses) / len(losses)

    # --- Calculate detailed metrics using sklearn ---
    # 1. Generate evaluation dictionary
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=['negative', 'neutral', 'positive'],
        labels=[0, 1, 2],
        zero_division=0,
        output_dict=True
    )
    # 2. Extract key indicators from the report dictionary
    accuracy = report_dict['accuracy']
    weighted_f1 = report_dict['weighted avg']['f1-score']
    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=['negative', 'neutral', 'positive'],
        labels=[0, 1, 2],
        zero_division=0
    )

    # 3. Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    # --------------------------------

    # Return a dictionary containing all indicators
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'classification_report': report_str,  # String report for printing
        'confusion_matrix': cm,
        'weighted_f1': weighted_f1
    }


def eval_model(model, data_loader, loss_fn, device, n_examples):
    """
    Execute the function of model evaluation
    """
    # Set the model to evaluation mode
    model = model.eval()
    losses = []
    correct_predictions = 0

    # Store all labels and predictions
    all_labels = []
    all_preds = []

    # When evaluating, do not calculate gradients to save memory and computing resources
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            # for i, d in enumerate(tqdm(data_loader, desc="Evaluating")):
            # if i > 10: # Jumping out of the loop after only running 10 steps
            # break

            # Move data to the designated device
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            # Forward Propagation
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Calculate Losses
            loss = loss_fn(outputs, labels)

            # Calculate Accuracy
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            # Collect labels and predicted results for the current batch
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Average loss
    avg_loss = sum(losses) / len(losses)

    # --- Calculate detailed metrics using sklearn ---
    # 1. Generate evaluation dictionary
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=['negative', 'neutral', 'positive'],
        labels=[0, 1, 2],
        zero_division=0,
        output_dict=True
    )
    # 2. Extract key indicators from the report dictionary
    accuracy = report_dict['accuracy']
    weighted_f1 = report_dict['weighted avg']['f1-score']
    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=['negative', 'neutral', 'positive'],
        labels=[0, 1, 2],
        zero_division=0
    )
    # 3. Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    # --------------------------------

    # Return a dictionary containing all indicators
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'classification_report': report_str,  # String report for printing
        'confusion_matrix': cm,
        'weighted_f1': weighted_f1
    }
