import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import re
import pandas as pd

# Load the dataset from a CSV file
file_path = 'email_spam.csv'  # Replace with your file path
data = pd.read_csv(file_path)


# Download necessary NLTK data and corpora
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to remove punctuation and emojis
def remove_punctuation_and_emojis(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove emojis and other non-word characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to tokenize text into words and sentences
def tokenize_text(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return sentences, words

# Function to remove stopwords
def remove_stopwords(words):
    return [word for word in words if word.lower() not in stop_words and word.isalpha()]

# Function to perform stemming
def stem_words(words):
    return [stemmer.stem(word) for word in words]

# Function to perform lemmatization
def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and emojis
    text = remove_punctuation_and_emojis(text)
    # Tokenize text
    sentences, words = tokenize_text(text)
    # Remove stopwords
    words = remove_stopwords(words)
    # Stem words
    stemmed_words = stem_words(words)
    # Lemmatize words
    lemmatized_words = lemmatize_words(stemmed_words)
    return sentences, words, stemmed_words, lemmatized_words

# Apply preprocessing to the text column and store the results in new columns
data['sentences'], data['words'], data['stemmed_words'], data['lemmatized_words'] = zip(*data['text'].apply(preprocess_text))

# Display the first few rows of the processed dataset
for i in range(10):
    print(f"Original Text:\n{data['text'].iloc[i]}\n")
    print(f"Sentences:\n{data['sentences'].iloc[i]}\n")
    print(f"Words:\n{data['words'].iloc[i]}\n")
    print(f"Stemmed Words:\n{data['stemmed_words'].iloc[i]}\n")
    print(f"Lemmatized Words:\n{data['lemmatized_words'].iloc[i]}\n")
    print("="*80)