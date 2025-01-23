import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, TrainingArguments, TFTrainer
from datasets import Dataset

# Data Preprocessing
df = pd.read_csv("gmb.csv", sep=",", encoding='unicode_escape')
df = df.head(1000)
df = df.ffill()
df = df.dropna()


tokens = df.Word.astype(str).tolist()
ner_tags = df.Tag.astype(str).tolist()
# Need to convert to numerical labels
map_dict = {}
count = 0
for label in ner_tags:
    if label not in map_dict.keys():
        map_dict[label] = count
        count += 1
ner_tags = [map_dict[x] for x in ner_tags]

sentence_numbers = df["Sentence #"].astype(str).tolist()
sentences = []
ner_tags_sentences = []

# Not very good way to get the tokens into a list for each sentence
num_sentences = 2
for j in range(num_sentences):
    sentence = []
    ner_list = []
    j += 1
    for i in range(int(len(tokens)/4)):
        if sentence_numbers[i] == "Sentence: " + str(j):
            sentence.append(tokens[i])
            ner_list.append(ner_tags[i])
    sentences.append(sentence)
    ner_tags_sentences.append(ner_list)

sentences = list(filter(None, sentences))
ner_tags_sentences = list(filter(None, ner_tags_sentences))
# print(tuple(zip(sentences, ner_tags_sentences)))
# for s, n in zip(sentences, ner_tags_sentences):
#     print(f"{s}: {n}")

X_train, X_test, y_train, y_test = train_test_split(sentences, ner_tags_sentences, test_size=0.1, random_state=1337)


train_data = {"tokens": X_train, "ner_tags": y_train}
test_data = {"tokens": X_test, "ner_tags": y_test}

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)
# print(train_dataset)


tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    tokenized_inputs.tokens()
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

unique_labels = set(label for sentence in ner_tags_sentences for label in sentence)
num_labels = len(unique_labels)
print(f"Unique labels: {unique_labels}, Number of labels: {num_labels}")

model = TFAutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
model.classifier = tf.keras.layers.Dense(num_labels, activation="softmax")
model.classifier.build((None, model.config.hidden_size))

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()

results = trainer.evaluate()
print(results)