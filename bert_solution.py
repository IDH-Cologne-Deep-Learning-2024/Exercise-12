import os
os.environ["WANDB_DISABLED"] = "true"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import evaluate

df = pd.read_csv("gmb.csv", sep=",", encoding="unicode_escape")
df = df.head(10_000)
df = df.dropna()
X_train, X_test, y_train, y_test = train_test_split(df.Word, df.Tag, test_size=0.1, random_state=42)

ner_bert_model_name = "dslim/bert-base-NER"
classifier = pipeline("token-classification", model=ner_bert_model_name)
results = classifier(X_test.tolist())
print(results)

# Fine Tuning


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


model_name = "bert-base-cased"
df = df[["Word", "Tag"]]
df = df.rename(columns={"Word": "text", "Tag": "labels"})
label_dict = {v: k for v, k in enumerate(list(df["labels"].unique()))}
rev_label_dict = {v: k for k, v in label_dict.items()}
df_train, df_test = train_test_split(df, test_size=0.1, stratify=df.labels, random_state=42)
df_train["labels"].replace(rev_label_dict, inplace=True)
df_test["labels"].replace(rev_label_dict, inplace=True)
dataset_train = Dataset.from_pandas(df_train)
dataset_test = Dataset.from_pandas(df_test)
dataset = DatasetDict()
dataset["train"] = dataset_train
dataset["test"] = dataset_test
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_dict), torch_dtype="auto")
training_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate(tokenized_dataset["test"])
