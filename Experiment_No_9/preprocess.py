import pandas as pd
import numpy as np
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv("bbc_news_data.csv")

print(data.head())

# Clean text function
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    return text

# Clean article and summary
data['Text'] = data['Text'].apply(clean_text)
data['Summary'] = data['Summary'].apply(clean_text)

print("\nPreprocessing completed!")