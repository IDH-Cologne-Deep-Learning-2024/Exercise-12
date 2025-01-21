
import pandas as pd
df = pd.read_csv("gmb.csv", sep=",", encoding='unicode_escape')
df.head()

from datasets import Dataset
def process_data(df):
    tokens = []
    ner_tags = []
    sentence = []
    tags = []
    for _, row in df.iterrows():
        if pd.isna(row['Word']):
            if sentence:
                tokens.append(sentence)
                ner_tags.append(tags)
                sentence = []
                tags = []
        else:
            sentence.append(row['Word'])
            tags.append(row['Tag'])

    return Dataset.from_dict({"tokens": tokens, "ner_tags": ner_tags})

dataset = process_data(df)

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
bert_model = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(bert_model)
model = AutoModelForTokenClassification.from_pretrained(bert_model, num_labels=len(set(df['Tag'])))
def tokenizer_labels(examples):
    inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []

    for i, label in enumerate(examples["ner_tags"]):
        word_ids = inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(label_ids)

    inputs["labels"] = labels
    return inputs
tokenized_dataset = dataset.map(tokenizer_labels, batched=True)

args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)
trainer.train()

results = trainer.evaluate()
print(results)
