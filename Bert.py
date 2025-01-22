import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

df = pd.read_csv("gmb.csv", sep=",", encoding='unicode_escape')
df = df.head(100_000)
df = df.dropna()

X = df.Word.astype(str).tolist()
y_ner = df.Tag.tolist()

labels = sorted(list(set(y_ner)))
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

data = {"words": [], "labels": []}
for word, label in zip(X, y_ner):
    data["words"].append([word])
    data["labels"].append([label])

dataset = Dataset.from_dict(data)
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda p: classification_report(
        p.label_ids.flatten(), 
        p.predictions.argmax(-1).flatten(), 
        target_names=labels, 
        output_dict=True
    ),
)

trainer.train()

results = trainer.evaluate()
print(results)

predictions, labels, _ = trainer.predict(test_dataset)
predicted_labels = predictions.argmax(-1)

true_labels = [label for seq in labels for label in seq if label != -100]
predicted_labels = [pred for seq in predicted_labels for pred in seq if pred != -100]

print(classification_report(true_labels, predicted_labels, target_names=labels))