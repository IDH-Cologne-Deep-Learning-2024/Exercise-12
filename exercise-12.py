import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
from datasets import Datasset, DatasetDict

# mixed prev and rev up so absolutely useless

#Sentence #,Word,POS,Tag
df = pd.read_csv("gmb.csv", sep=",", encoding='unicode_escape')
df = df.head(100_000)
df = df.dropna()

X = df.Word.astype(str).tolist()
y_ner = df.labels.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y_ner, test_size=0.1, random_state=42)

model_type = ["bert-base-cased"]
tokenizer = AutoTokenizer.from_pretrained(model_type)

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    #allgning labels tbc
    return tokenized_inputs

tokenized_datasets = df.map(tokenize_and_align_labels, batched=True)
model = BertForTokenClassification.from_pretrained(model_type, num_labels=len(set(df["labels"])))

# training tbc 
# metrics tbc Trainer only when compute_metrics() defined
# follow up defining model w mapping ID to label, label to ID
