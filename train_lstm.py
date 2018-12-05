import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.preprocessing.text as kt
from keras.backend.tensorflow_backend import set_session
from keras.layers import Embedding, LSTM, Dropout, TimeDistributed, Activation, Input, Dense, Flatten
from keras.models import Sequential

# Prevent allocation of all gpu-memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                   # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)


# Load data from csv
train_df = pd.DataFrame.from_csv('data/train.csv')
test_df = pd.DataFrame.from_csv('data/test.csv')

print("Dimensions of training data")
print(train_df.shape)

print("Dimensions of testing data")
print(test_df.shape)

# Vocabulary Creation
words_frequency = {}
max_length = 0


for i in range(train_df.shape[0]):
    
    # Create list of words
    ws = kt.text_to_word_sequence(train_df['text'][i], filters = '"?!-')
    
    # Determine max number of words in a sequence
    max_length = max(max_length, len(ws))
    for w in ws:
        if w in words_frequency.keys():
            words_frequency[w] += 1
        else:
            words_frequency[w] = 1
            
# Replace all http
words = {}
words["http"] = 0

# Remove stuffs
for key in words_frequency.keys():
    if "http" in key:
        words["http"] += words_frequency[key]
    else:
        words[key] = words_frequency[key]

# Create dictionaries for lookups
idx2word = {}
word2idx = {}
word2idx['END TOKEN'] = 0
idx2word[0] = 'END TOKEN'

counter = 0
for i in words.keys():
    counter += 1
    word2idx[i] = counter
    idx2word[counter] = i        
        
vocab_size = len(word2idx.keys())


# Creating training dataset with correct integer values
x_data = np.zeros([train_df.shape[0], max_length + 1])
for i in range(train_df.shape[0]):
    
    ws = kt.text_to_word_sequence(train_df['text'][i], filters = '"?!-')
    
    for j in range(len(ws)):
        if "http" in ws[j]:
            x_data[i, j] = word2idx["http"]
        elif words[ws[j]] > 10:
            x_data[i, j] = word2idx[ws[j]]

y_data = train_df['label']


# Create model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vecor_length, input_length=32))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
print(model.summary())


model.fit(x_data, y_data, batch_size=16, epochs=25, shuffle=True, verbose = 1)

model.save("trump.h5")