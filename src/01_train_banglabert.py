import os
import random
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from torch.utils.data import DataLoader

# --- 1. SETUP ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# --- 2. LOAD DATA ---
# NOTE: Update these paths to where your CSVs actually live!
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

# --- 3. TOKENIZATION ---
tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")

def tokenize(data):
    return tokenizer(data['Content'], padding='max_length', truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df[['Content', 'Label']]).rename_column("Label", "label")
val_dataset = Dataset.from_pandas(val_df[['Content', 'Label']]).rename_column("Label", "label")
test_dataset = Dataset.from_pandas(test_df[['Content', 'Label']]).rename_column("Label", "label")

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# --- 4. MODEL SETUP & TRAINING ---
model = AutoModelForSequenceClassification.from_pretrained("sagorsarker/bangla-bert-base", num_labels=2)
model.config.hidden_dropout_prob = 0.3
model.config.attention_probs_dropout_prob = 0.3

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    lr_scheduler_type="linear",
    warmup_ratio=0.1
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# --- 5. EMBEDDING EXTRACTION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def create_dataloader(dataset, batch_size=16):
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return DataLoader(dataset, batch_size=batch_size)

def get_embeddings(dataloader):
    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :] 
            all_embeddings.append(cls_embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

train_loader = create_dataloader(train_dataset)
val_loader = create_dataloader(val_dataset)
test_loader = create_dataloader(test_dataset)

train_embeddings = get_embeddings(train_loader)
val_embeddings = get_embeddings(val_loader)
test_embeddings = get_embeddings(test_loader)

# --- 6. SAVE OUTPUTS ---
torch.save(train_embeddings, "train_embeddings.pt")
torch.save(val_embeddings, "val_embeddings.pt")
torch.save(test_embeddings, "test_embeddings.pt")
torch.save(torch.tensor(train_dataset['label']), "train_labels.pt")
torch.save(torch.tensor(val_dataset['label']), "val_labels.pt")
torch.save(torch.tensor(test_dataset['label']), "test_labels.pt")
print("Phase 1 Complete: Embeddings saved!")