import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import classification_report

# Step 1: Load the Dataset
print("Loading the dataset...")
data = pd.read_csv("path_to_dataset.csv")  # Replace with the actual file path to your dataset
print(data.head())

# Step 2: Pre-trained BERT for NER
print("Initializing pre-trained BERT pipeline...")
model_name = "dslim/bert-base-NER"
classifier = pipeline("token-classification", model=model_name)

# Perform NER with pre-trained BERT
print("Performing NER with pre-trained BERT...")
results = []
for sentence in data["text"]:  # Assuming 'text' column contains the sentences
    classification = classifier(sentence)
    results.append(classification)

# Add results to the dataset
data["bert_results"] = results

# Save pre-trained BERT results
data.to_csv("pretrained_bert_results.csv", index=False)
print("Pre-trained BERT results saved to 'pretrained_bert_results.csv'.")

# Step 3: Fine-tune BERT for NER
print("Preparing for fine-tuning BERT...")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization and alignment function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],  # Assuming 'tokens' column contains tokenized words
        truncation=True,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_labels"]):  # Assuming 'ner_labels' column has labels
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Prepare dataset for HuggingFace
dataset = Dataset.from_pandas(data)
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Load pre-trained BERT model for token classification
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(set(data["ner_labels"].explode())))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="epoch"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)

# Fine-tune the model
print("Fine-tuning BERT...")
trainer.train()

# Save the fine-tuned model
print("Saving fine-tuned BERT model...")
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")

# Step 4: Evaluate Fine-tuned BERT
print("Evaluating fine-tuned BERT...")
predictions, labels, _ = trainer.predict(tokenized_dataset["test"])
predictions = predictions.argmax(axis=2)

# Align predictions and labels
true_labels = []
pred_labels = []
for i, label in enumerate(labels):
    true_labels.extend([l for l, p in zip(label, predictions[i]) if l != -100])
    pred_labels.extend([p for l, p in zip(label, predictions[i]) if l != -100])

# Classification report
report = classification_report(true_labels, pred_labels, target_names=tokenizer.get_vocab())
print(report)

# Save evaluation results
with open("fine_tuned_bert_evaluation.txt", "w") as f:
    f.write(report)
print("Evaluation results saved to 'fine_tuned_bert_evaluation.txt'.")

