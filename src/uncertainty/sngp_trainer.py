import torch
import torch.nn as nn
import pandas as pd
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW # <-- 从 torch.optim 导入 AdamW
from collections import defaultdict
import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from src import config
from src.dataset import create_data_loader
from src.engine import train_epoch, eval_model
from src.uncertainty.sngp_model import SNGPClassifier # <-- 从同一目录导入SNGP模型
# -------------------------------------

def run_sngp_training():
    print("="*20 + " SNGP Model Training " + "="*20)
    
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, "validation.csv"))

    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    train_data_loader = create_data_loader(config.PROCESSED_DATA_PATH + "/train.csv", tokenizer, config.MAX_LEN, config.BATCH_SIZE)
    val_data_loader = create_data_loader(config.PROCESSED_DATA_PATH + "/validation.csv", tokenizer, config.MAX_LEN, config.BATCH_SIZE)

    device = torch.device(config.DEVICE)
    model = SNGPClassifier(n_classes=config.N_CLASSES, model_name=config.MODEL_NAME)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_data_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_f1_score = 0

    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        print('-' * 10)

        train_results = train_epoch(model, train_data_loader, loss_fn, optimizer, device, len(train_df))
        eval_results = eval_model(model, val_data_loader, loss_fn, device, len(val_df))
        
        val_f1 = eval_results['weighted_f1']
        
        if val_f1 > best_f1_score:
            model_path = os.path.join(config.SAVED_MODELS_PATH, 'sngp_best_model_state.bin')
            torch.save(model.state_dict(), model_path)
            best_f1_score = val_f1
            print(f"*** New best SNGP model saved (F1-score: {best_f1_score:.4f}) ***")

if __name__ == '__main__':
    run_sngp_training()