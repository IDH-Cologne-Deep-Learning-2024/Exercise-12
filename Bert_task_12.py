import numpy as np
import pandas as pd
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import pipeline

df = pd.read_csv("gmb.csv", sep=",", encoding='unicode_escape')
df = df.dropna()

X = df.Word.astype(str).tolist()
y_ner = df.Tag.tolist() 

X_train, X_test, y_train, y_test = train_test_split(X, y_ner, test_size=0.2, random_state=404)

train_data = {'tokens': X_train, 'labels': y_train}
test_data = {'tokens': X_test, 'labels': y_test}

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

tokenizer = BertTokenizerFast.from_pretrained("dslim/bert-base-NER")

def tokenize_function(examples):
    return tokenizer(examples['tokens'], padding=True, truncation=True, is_split_into_words=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER", num_labels=len(set(y_ner)))

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label] for label in labels]
    true_preds = [[prediction] for prediction in predictions]

    true_labels = np.array(true_labels).flatten()
    true_preds = np.array(true_preds).flatten()

    return classification_report(true_labels, true_preds, output_dict=True)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)


trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    compute_metrics=compute_metrics,     # function to compute metrics
)