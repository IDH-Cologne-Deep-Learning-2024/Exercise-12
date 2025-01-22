import numpy as np
import pandas as pd
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import pipeline

# Load data
df = pd.read_csv("gmb.csv", sep=",", encoding='unicode_escape')
df = df.head(100_000)  # Limit for testing
df = df.dropna()

# Prepare your input data
X = df.Word.astype(str).tolist()
y_ner = df.Tag.tolist()  # Use NER tags for labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_ner, test_size=0.1, random_state=42)

# Convert to Hugging Face Dataset format
train_data = {'tokens': X_train, 'labels': y_train}
test_data = {'tokens': X_test, 'labels': y_test}

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("dslim/bert-base-NER")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['tokens'], padding=True, truncation=True, is_split_into_words=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load the pre-trained BERT model with a token classification head
model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER", num_labels=len(set(y_ner)))

# Define the metrics function (to compute precision, recall, F1-score)
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label] for label in labels]
    true_preds = [[prediction] for prediction in predictions]

    # Flatten and compute the classification report
    true_labels = np.array(true_labels).flatten()
    true_preds = np.array(true_preds).flatten()

    return classification_report(true_labels, true_preds, output_dict=True)

# Define TrainingArguments
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

# Initialize Trainer
trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    compute_metrics=compute_metrics,     # function to compute metrics
)

# Fine-tune the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Print the evaluation results
print("Evaluation results:")
print(results)
