# Text Preprocessing Module
import pandas as pd
import numpy as np
import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import logging
from config import Config

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        # Add custom stopwords for Reddit and medical context
        custom_stopwords = {
            'reddit', 'r', 'u', 'amp', 'gt', 'lt', 'http', 'https', 'www',
            'com', 'edit', 'update', 'tldr', 'tl', 'dr', 'pm', 'delete',
            'removed', 'moderator', 'mod', 'automod'
        }
        self.stop_words.update(custom_stopwords)

        # Try to load spaCy model for advanced processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Advanced processing disabled.")
            self.nlp = None

    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+|/r/\w+', '', text)  # Remove user/subreddit mentions
        text = re.sub(r'\[deleted\]|\[removed\]', '', text)
        text = re.sub(r'>.*?\n', '', text)  # Remove quoted text

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def remove_medical_identifiers(self, text):
        """Remove potential medical identifiers while preserving medical terms"""
        # Remove dates in various formats
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
        text = re.sub(r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{2,4}\b', 
                     '', text, flags=re.IGNORECASE)

        # Remove specific measurements that could be identifying
        text = re.sub(r'\b\d+\.\d+\s*(lbs?|pounds?)\b', 'weight_measurement', text, flags=re.IGNORECASE)
        text = re.sub(r"\b\d+'\d+\"\b", 'height_measurement', text)

        # Remove specific ages/weeks
        text = re.sub(r'\b\d+\s*weeks?\s*(and|\+)\s*\d+\s*days?\b', 'gestational_age', text, flags=re.IGNORECASE)

        return text

    def tokenize_and_filter(self, text):
        """Tokenize text and filter tokens"""
        if not text:
            return []

        # Tokenize
        tokens = word_tokenize(text)

        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip if too short or too long
            if len(token) < Config.MIN_WORD_LENGTH or len(token) > Config.MAX_WORD_LENGTH:
                continue

            # Skip if it's a stopword
            if Config.REMOVE_STOPWORDS and token.lower() in self.stop_words:
                continue

            # Skip if it's just numbers
            if token.isdigit():
                continue

            # Skip if it's just punctuation
            if token in string.punctuation:
                continue

            filtered_tokens.append(token.lower())

        return filtered_tokens

    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens"""
        if self.nlp:
            # Use spaCy for better lemmatization
            doc = self.nlp(' '.join(tokens))
            return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        else:
            # Fallback to NLTK
            return [self.lemmatizer.lemmatize(token) for token in tokens]

    def extract_medical_terms(self, text):
        """Extract pre-eclampsia related medical terms"""
        medical_terms = {
            'symptoms': ['headache', 'swelling', 'edema', 'protein', 'proteinuria', 'vision', 'blurred',
                        'pain', 'nausea', 'vomiting', 'dizzy', 'dizziness', 'shortness', 'breath'],
            'measurements': ['blood pressure', 'bp', 'systolic', 'diastolic', 'mmhg', 'protein urine'],
            'treatments': ['medication', 'medicine', 'labetalol', 'nifedipine', 'methyldopa', 'aspirin',
                          'magnesium', 'sulfate', 'delivery', 'cesarean', 'c-section', 'induction'],
            'complications': ['hellp', 'syndrome', 'seizure', 'eclampsia', 'stroke', 'liver', 'kidney',
                            'placenta', 'abruption', 'preterm', 'premature'],
            'emotions': ['scared', 'worried', 'anxious', 'fear', 'stressed', 'concerned', 'relieved',
                        'grateful', 'frustrated', 'angry', 'sad', 'happy', 'hopeful']
        }

        found_terms = {}
        text_lower = text.lower()

        for category, terms in medical_terms.items():
            found_terms[category] = []
            for term in terms:
                if term in text_lower:
                    found_terms[category].append(term)

        return found_terms

    def preprocess_dataframe(self, df):
        """Preprocess entire dataframe"""
        logger.info("Starting text preprocessing...")

        # Create processed dataframe
        processed_df = df.copy()

        # Clean text fields
        processed_df['title_clean'] = processed_df['title'].apply(self.clean_text)
        processed_df['selftext_clean'] = processed_df['selftext'].apply(self.clean_text)
        processed_df['full_text_clean'] = processed_df['full_text'].apply(self.clean_text)

        # Remove medical identifiers
        processed_df['full_text_clean'] = processed_df['full_text_clean'].apply(self.remove_medical_identifiers)

        # Filter out posts that are too short
        min_length = Config.MIN_POST_LENGTH
        processed_df = processed_df[processed_df['full_text_clean'].str.split().str.len() >= min_length]

        # Tokenize and lemmatize
        logger.info("Tokenizing text...")
        processed_df['tokens'] = processed_df['full_text_clean'].apply(self.tokenize_and_filter)
        processed_df['tokens_lemmatized'] = processed_df['tokens'].apply(self.lemmatize_tokens)

        # Extract medical terms
        logger.info("Extracting medical terms...")
        processed_df['medical_terms'] = processed_df['full_text_clean'].apply(self.extract_medical_terms)

        # Create processed text for analysis
        processed_df['text_for_analysis'] = processed_df['tokens_lemmatized'].apply(lambda x: ' '.join(x))

        # Add text statistics
        processed_df['word_count'] = processed_df['tokens_lemmatized'].str.len()
        processed_df['char_count'] = processed_df['full_text_clean'].str.len()
        processed_df['avg_word_length'] = processed_df['tokens_lemmatized'].apply(
            lambda tokens: np.mean([len(token) for token in tokens]) if tokens else 0
        )

        logger.info(f"Preprocessing complete. {len(processed_df)} posts processed.")

        # Save processed data
        processed_df.to_csv(Config.PROCESSED_DATA_FILE, index=False)

        return processed_df

if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv(Config.RAW_DATA_FILE)

    # Preprocess
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.preprocess_dataframe(df)

    print(f"Processed {len(processed_df)} posts")
    print(f"Average word count: {processed_df['word_count'].mean():.1f}")
