import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model(
    "summarizer_model.h5"
)

# Sample text
text = """
Artificial Intelligence is transforming the world.
It is used in healthcare, education, robotics,
and automation industries.
"""

# Tokenizer
tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts([text])

# Convert text
sequence = tokenizer.texts_to_sequences([text])

# Padding
sequence = pad_sequences(sequence, maxlen=100)

# Predict
prediction = model.predict(sequence)

print("\nSummary Generated Successfully!")
print("Predicted Output:", np.argmax(prediction))