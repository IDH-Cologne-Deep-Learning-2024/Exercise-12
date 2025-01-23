import pandas as pd
import numpy as np
import tensorflow as tf
# I need to call it via tf_keras now unlike before (running in new environement (linux in WSL))
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from keras._tf_keras.keras.optimizers import SGD, Adam
from keras._tf_keras.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import pipeline
from datasets import Dataset


# Data Preprocessing
df = pd.read_csv("gmb.csv", sep=",", encoding='unicode_escape')
df = df.head(100000)
df = df.ffill()
df = df.dropna()


tokens = df.Word.astype(str).tolist()
ner_tags = df.Tag.astype(str).tolist()
pred_ner_tags =[]

X_train, X_test, y_train, y_test = train_test_split(tokens, ner_tags, test_size=0.1, random_state=1337)

# BERT
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

for token in X_test:
    ner_result = ner_pipeline(token)
    if ner_result:
        pred_ner_tags.append(f"{ner_result[0]['entity_group']}")
    else:
        pred_ner_tags.append("O")

print(classification_report(y_test, pred_ner_tags))
# print(tuple(zip(y_test,pred_ner_tags)))

# LSTM/RNN
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(tokens)

X_train = word_tokenizer.texts_to_sequences(X_train)
X_test = word_tokenizer.texts_to_sequences(X_test)
vocab_size = len(word_tokenizer.word_index)+1

MAX_LENGTH = 1
X_train = pad_sequences(X_train, maxlen=MAX_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH)

labels = y_train + y_test
map_dict = {}
count = 0
for label in labels:
    if label not in map_dict.keys():
        map_dict[label] = count
        count += 1
y_train = [map_dict[x] for x in y_train]
y_test = [map_dict[x] for x in y_test]
y_train = np.array(y_train)
y_test = np.array(y_test)

number_classes = len(set(labels))

# for i, x in enumerate(X_train):
#     print(f"{x}: {y_train[i]}")

evaluation_scores = {}
model_types = ["LSTM", "RNN"]
for model_type in model_types:
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=300, input_length=MAX_LENGTH))
        if model_type == "LSTM":
            model.add(Bidirectional(LSTM(64)))
        elif model_type == "RNN":
            model.add(Bidirectional(SimpleRNN(64)))
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss='crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=20, verbose=0)
        y_pred = model.predict(X_test)
        y_pred = y_pred.argmax(axis=1)
        evaluation_scores[f"{model_type}"] = classification_report(y_test, y_pred)

for k in evaluation_scores:
    print(k)
    print(evaluation_scores[k])


# Comparision Values for BERT vs. LSTM, RNN with overall tokens set to 100,000 => testtokens = 10,000
# Apparantly LSTM and RNN are performing better than the BERT model with this data
# BERT
#               precision    recall  f1-score   support

#        B-art       0.00      0.00      0.00        10
#        B-eve       0.00      0.00      0.00         7
#        B-geo       0.00      0.00      0.00       321
#        B-gpe       0.00      0.00      0.00       170
#        B-nat       0.00      0.00      0.00         5
#        B-org       0.00      0.00      0.00       169
#        B-per       0.00      0.00      0.00       166
#        B-tim       0.00      0.00      0.00       186
#        I-art       0.00      0.00      0.00         4
#        I-eve       0.00      0.00      0.00         7
#        I-geo       0.00      0.00      0.00        72
#        I-gpe       0.00      0.00      0.00        12
#        I-nat       0.00      0.00      0.00         1
#        I-org       0.00      0.00      0.00       146
#        I-per       0.00      0.00      0.00       186
#        I-tim       0.00      0.00      0.00        49
#          LOC       0.00      0.00      0.00         0
#         MISC       0.00      0.00      0.00         0
#            O       0.95      0.99      0.97      8489
#          ORG       0.00      0.00      0.00         0
#          PER       0.00      0.00      0.00         0

#     accuracy                           0.84     10000
#    macro avg       0.05      0.05      0.05     10000
# weighted avg       0.81      0.84      0.82     10000



# LSTM
#               precision    recall  f1-score   support

#            0       0.96      0.99      0.97      8489
#            1       0.45      0.24      0.31       146
#            2       0.27      0.06      0.10        49
#            3       0.89      0.76      0.82       186
#            4       0.72      0.77      0.75       321
#            5       0.77      0.44      0.56       186
#            6       0.61      0.64      0.63       166
#            7       0.53      0.33      0.41        72
#            8       0.66      0.40      0.50       169
#            9       0.86      0.81      0.83       170
#           10       1.00      0.50      0.67        12
#           11       0.50      0.14      0.22         7
#           12       0.00      0.00      0.00        10
#           13       0.00      0.00      0.00         4
#           14       1.00      0.14      0.25         7
#           15       0.00      0.00      0.00         1
#           16       0.50      0.20      0.29         5

#     accuracy                           0.93     10000
#    macro avg       0.57      0.38      0.43     10000
# weighted avg       0.92      0.93      0.92     10000

# RNN
#               precision    recall  f1-score   support

#            0       0.95      0.99      0.97      8489
#            1       0.52      0.18      0.27       146
#            2       0.12      0.02      0.04        49
#            3       0.84      0.80      0.82       186
#            4       0.72      0.76      0.74       321
#            5       0.71      0.59      0.64       186
#            6       0.71      0.54      0.62       166
#            7       0.55      0.40      0.46        72
#            8       0.72      0.38      0.50       169
#            9       0.85      0.81      0.83       170
#           10       0.67      0.67      0.67        12
#           11       0.00      0.00      0.00         7
#           12       0.00      0.00      0.00        10
#           13       0.00      0.00      0.00         4
#           14       1.00      0.14      0.25         7
#           15       0.33      1.00      0.50         1
#           16       0.00      0.00      0.00         5

#     accuracy                           0.93     10000
#    macro avg       0.51      0.43      0.43     10000
# weighted avg       0.91      0.93      0.92     10000