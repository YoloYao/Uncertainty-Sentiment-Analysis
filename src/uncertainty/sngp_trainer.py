import torch
import torch.nn as nn
import pandas as pd
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from collections import defaultdict
import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from src.uncertainty.sngp_model import SNGPClassifier
from src.engine import train_epoch, eval_model
from src.dataset import create_data_loader
from src import config

def run_sngp_training():
    # 1. Load data
    train_df = pd.read_csv(os.path.join(
        config.PROCESSED_DATA_PATH, config.TRAIN_FILE_NAME))
    val_df = pd.read_csv(os.path.join(
        config.PROCESSED_DATA_PATH, config.VALIDATION_FILE_NAME))

    # --- 2. Initialize tokenizer and data loader ---
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    train_data_loader = create_data_loader(
        config.PROCESSED_DATA_PATH + "/" + config.TRAIN_FILE_NAME, tokenizer, config.MAX_LEN, config.BATCH_SIZE)
    val_data_loader = create_data_loader(
        config.PROCESSED_DATA_PATH + "/" + config.VALIDATION_FILE_NAME, tokenizer, config.MAX_LEN, config.BATCH_SIZE)

    # --- 3. Initialize models, optimizers, loss functions, etc ---
    device = torch.device(config.DEVICE)
    print("-" * 54)
    print("SNGP Model training begins")
    print(f"Using device: {device}")
    print("-" * 54)
    model = SNGPClassifier(n_classes=config.N_CLASSES,
                           model_name=config.MODEL_NAME)
    model = model.to(device)

    # Create AdamW optimizer
    # (one of the most commonly used optimizers for fine-tuning Transformer models)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Create a cross entropy loss function
    # (the most standard loss function for classification tasks)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # --- 4. Training cycle ---
    # Create a dictionary to record the training records for each cycle
    history = defaultdict(list)

    # Record the best F1 score that has appeared on the validation set
    best_f1_score = 0

    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        print('-' * 10)

        train_results = train_epoch(
            model, train_data_loader, loss_fn, optimizer, device, len(train_df))

        train_acc = train_results['accuracy']
        train_loss = train_results['loss']
        train_f1 = train_results['weighted_f1']

        print(
            f'\nTrain Accuracy: {train_acc:.4f}   Train Loss: {train_loss:.4f}')
        print("\n" + "-" * 8 + " Training Set Classification Report " + "-" * 9)
        print(train_results['classification_report'])
        print("\n" + "-" * 10 + " Training Set Confusion Matrix " + "-" * 11)
        print(train_results['confusion_matrix'])
        print("-" * 54)

        # --- Evaluation results ---
        eval_results = eval_model(
            model, val_data_loader, loss_fn, device, len(val_df))

        val_acc = eval_results['accuracy']
        val_loss = eval_results['loss']
        val_f1 = eval_results['weighted_f1']

        print(f'\nVal Accuracy: {val_acc:.4f}   Val Loss: {val_loss:.4f}')
        print("\n" + "-" * 8 + " Validation Set Classification Report " + "-" * 7)
        print(eval_results['classification_report'])
        print("\n" + "-" * 10 + " Validation Set Confusion Matrix " + "-" * 9)
        print(eval_results['confusion_matrix'])
        print("-" * 54)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)

        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        # Save the best performing model
        if val_f1 > best_f1_score:
            model_path = os.path.join(
                config.SAVED_MODELS_PATH, config.SAVED_SNGP_MODELS_NAME)
            torch.save(model.state_dict(), model_path)
            best_f1_score = val_f1
            print("*" * 54)
            print(
                f" * New best SNGP model has been saved: (F1-score: {best_f1_score:.4f}) *")
            print("*" * 54)


if __name__ == '__main__':
    run_sngp_training()
