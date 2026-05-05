import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("bbc_news_data.csv")

# Take small dataset
texts = data['Text'][:1000].astype(str)
summaries = data['Summary'][:1000].astype(str)

# =========================
# TOKENIZER
# =========================

vocab_size = 5000

tokenizer = Tokenizer(num_words=vocab_size)

# Fit tokenizer on text
tokenizer.fit_on_texts(texts)

# Convert text into sequences
X = tokenizer.texts_to_sequences(texts)

# Convert summaries into sequences
y = tokenizer.texts_to_sequences(summaries)

# =========================
# PADDING
# =========================

X = pad_sequences(X, maxlen=100)

y = pad_sequences(y, maxlen=20)

# Use first word of summary as target
y = y[:, 0]

# =========================
# BUILD MODEL
# =========================

model = Sequential()

# Embedding Layer
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=64,
        input_length=100
    )
)

# LSTM Layer
model.add(LSTM(128))

# Output Layer
model.add(
    Dense(
        vocab_size,
        activation='softmax'
    )
)

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    X,
    y,
    epochs=5,
    batch_size=32
)

# =========================
# SAVE MODEL
# =========================

model.save("summarizer_model.h5")

print("\nModel Trained Successfully!")