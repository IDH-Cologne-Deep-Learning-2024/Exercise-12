import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
from transformers import pipeline
import numpy as np

# Load data
df = pd.read_csv("gmb.csv", sep=",", encoding='unicode_escape')
df = df.head(8000)
df = df.dropna()

X = df.Word.astype(str).tolist()
y_ner = df.Tag.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y_ner, test_size=0.2, random_state=42)

unique_tags = list(set(y_ner))
label_map = {tag: idx for idx, tag in enumerate(unique_tags)}
num_labels = len(unique_tags)

# Initialize tokenizer and model
# Using the BERT-model on the test-data
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", num_labels=num_labels, ignore_mismatched_sizes=True)
classifier = pipeline("ner", model=model, tokenizer=tokenizer)

ner_results = classifier(X_test)
print(ner_results)

# Preprocessing the data for training
# Without this i got a ValueError: Expected input batch_size (112) to match target batch_size (16).
def encode_labels(words, labels):
    encoded_labels = []
    for word, label in zip(words, labels):
        word_tokens = tokenizer.tokenize(word)
        encoded_labels.extend([label_map[label]] + [-100] * (len(word_tokens) - 1))
    return encoded_labels

train_encodings = tokenizer(X_train, truncation=True, padding=True, is_split_into_words=False)
train_labels = [encode_labels([word], [label]) for word, label in zip(X_train, y_train)]
train_labels = [label + [-100] * (len(train_encodings["input_ids"][i]) - len(label)) for i, label in enumerate(train_labels)]

test_encodings = tokenizer(X_test, truncation=True, padding=True, is_split_into_words=False)
test_labels = [encode_labels([word], [label]) for word, label in zip(X_test, y_test)]
test_labels = [label + [-100] * (len(test_encodings["input_ids"][i]) - len(label)) for i, label in enumerate(test_labels)]


train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
})

# Train arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Start training
trainer.train()
results = trainer.evaluate()
print(results)
