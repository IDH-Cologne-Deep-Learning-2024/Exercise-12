import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset


df = pd.read_csv("gmb.csv", sep=",", encoding='unicode_escape')
df = df.head(100_000)
df = df.dropna()


label_list = df.Tag.unique().tolist()
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}


dataset = Dataset.from_pandas(df)


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["Word"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["Tag"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)


train_dataset, test_dataset = tokenized_dataset.train_test_split(test_size=0.1)


model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)


trainer.train()


predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=2)


true_labels = [[label_list[l] for l in label if l != -100] for label in predictions.label_ids]
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(preds, predictions.label_ids)
]


print(classification_report(true_labels, true_predictions))
