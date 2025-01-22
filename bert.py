import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import pipeline, AutoTokenizer

def to_number(labels):
    map_dict = {}
    count = 0
    for label in labels:
        if label not in map_dict.keys():
            map_dict[label] = count
            count += 1
    return [map_dict[x] for x in labels]

model_name = "dslim/bert-base-NER"

# loading the csv file
df = pd.read_csv("gmb.csv", sep=",", encoding='unicode_escape')
df = df.head(100_000)
df = df.dropna()

# extracting the words and the ne out of the dataset
X = df.Word.astype(str).tolist()
y = to_number(df.Tag.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_data = Dataset.from_dict({ 'tokens': X_train,'labels': y_train })
test_data = Dataset.from_dict({ 'tokens': X_test,'labels': y_test })

tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer(train_data['tokens'], is_split_into_words=True)
labels = train_data['labels']
word_ids = inputs.word_ids()
print(labels)

# Comparison could not be made as I was somehow not able to do the full exercise
