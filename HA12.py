import numpy as np
from datasets import load_metric
from transformers import BertForTokenClassification, Trainer, TrainingArguments, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import classification_report
import torch

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label[word_id])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

num_labels = len(set(y_ner))
model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER", num_labels=num_labels)

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [pred for pred, label in zip(prediction, label) if label != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label for pred, label in zip(prediction, label) if label != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()

print("Evaluation results:")
print(results)

predictions, labels, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=2)

true_predictions = [
    [pred for pred, label in zip(prediction, label) if label != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label for pred, label in zip(prediction, label) if label != -100]
    for prediction, label in zip(predictions, labels)
]

print(classification_report(
    [label for labels in true_labels for label in labels],
    [pred for preds in true_predictions for pred in preds]
))

