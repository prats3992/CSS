"""
Data Cleaning and Preprocessing for Pre-eclampsia Reddit Data
Uses NLTK, spaCy, and other NLP tools
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import emoji
import contractions
from firebase_manager import FirebaseManager
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self):
        """Initialize the data cleaner with necessary NLTK resources"""
        self.firebase = FirebaseManager()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Download required NLTK data
        self._download_nltk_resources()
        
        # Medical terms to preserve (don't remove these as stopwords)
        self.medical_preserve_terms = {
            'preeclampsia', 'pre-eclampsia', 'eclampsia', 'hellp', 'toxemia',
            'gestational', 'hypertension', 'proteinuria', 'edema', 'seizure',
            'magnesium', 'sulfate', 'delivery', 'nicu', 'postpartum', 'prenatal',
            'trimester', 'bp', 'blood pressure', 'protein', 'urine', 'swelling',
            'headache', 'vision', 'liver', 'enzymes', 'platelets', 'induced',
            'emergency', 'c-section', 'premature', 'preterm', 'maternal'
        }
        
        # Get stopwords but preserve medical terms
        self.stop_words = set(stopwords.words('english'))
        self.stop_words = self.stop_words - self.medical_preserve_terms
        
    def _download_nltk_resources(self):
        """Download necessary NLTK resources"""
        resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
        
        print("Downloading NLTK resources...")
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except:
                print(f"  Warning: Could not download {resource}")
        print("✓ NLTK resources ready\n")
    
    def load_data_from_firebase(self):
        """Load posts and comments from Firebase into DataFrames"""
        print("\n" + "="*60)
        print("LOADING DATA FROM FIREBASE")
        print("="*60 + "\n")
        
        # Get posts
        posts_ref = self.firebase.db.child('reddit_posts').get()
        
        if not posts_ref:
            print("No posts found in Firebase")
            return None, None
        
        # Convert to DataFrame
        posts_list = []
        for post_id, post_data in posts_ref.items():
            posts_list.append(post_data)
        
        posts_df = pd.DataFrame(posts_list)
        print(f"✓ Loaded {len(posts_df)} posts")
        
        # Get comments if available
        comments_ref = self.firebase.db.child('reddit_comments').get()
        comments_df = None
        
        if comments_ref:
            comments_list = []
            for comment_id, comment_data in comments_ref.items():
                comments_list.append(comment_data)
            comments_df = pd.DataFrame(comments_list)
            print(f"✓ Loaded {len(comments_df)} comments")
        else:
            print("No comments found in Firebase")
        
        return posts_df, comments_df
    
    def clean_text(self, text, remove_urls=True, remove_emojis=True, 
                   expand_contractions=True, lowercase=True):
        """
        Clean and normalize text
        
        Args:
            text: Input text
            remove_urls: Remove URLs
            remove_emojis: Remove emoji characters
            expand_contractions: Expand contractions (don't -> do not)
            lowercase: Convert to lowercase
        """
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # Remove [deleted], [removed] markers
        text = re.sub(r'\[deleted\]|\[removed\]', '', text)
        
        # Expand contractions (don't -> do not)
        if expand_contractions:
            try:
                text = contractions.fix(text)
            except:
                pass
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)
            text = re.sub(r'r/\w+', '', text)  # Remove subreddit mentions
            text = re.sub(r'u/\w+', '', text)  # Remove user mentions
        
        # Handle emojis
        if remove_emojis:
            text = emoji.replace_emoji(text, replace='')
        else:
            # Convert emojis to text descriptions
            text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Convert to lowercase (preserving medical terms if needed)
        if lowercase:
            text = text.lower()
        
        return text
    
    def remove_stopwords(self, text, preserve_medical=True):
        """
        Remove stopwords while preserving medical terms
        
        Args:
            text: Input text
            preserve_medical: Keep medical terms even if they're stopwords
        """
        if not text:
            return ''
        
        tokens = word_tokenize(text.lower())
        
        if preserve_medical:
            filtered_tokens = [
                word for word in tokens 
                if word not in self.stop_words or word in self.medical_preserve_terms
            ]
        else:
            filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        return ' '.join(filtered_tokens)
    
    def lemmatize_text(self, text):
        """Lemmatize text to base forms"""
        if not text:
            return ''
        
        tokens = word_tokenize(text.lower())
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized)
    
    def stem_text(self, text):
        """Apply stemming to text"""
        if not text:
            return ''
        
        tokens = word_tokenize(text.lower())
        stemmed = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed)
    
    def extract_medical_terms(self, text):
        """Extract medical terms from text"""
        if not text:
            return []
        
        text_lower = text.lower()
        found_terms = []
        
        for term in self.medical_preserve_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def calculate_text_statistics(self, text):
        """Calculate various text statistics"""
        if pd.isna(text) or text == '':
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'question_mark_count': 0,
                'exclamation_count': 0
            }
        
        text = str(text)
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'question_mark_count': text.count('?'),
            'exclamation_count': text.count('!')
        }
    
    def process_posts(self, posts_df):
        """
        Clean and preprocess posts data
        
        Returns:
            DataFrame with cleaned data and new features
        """
        print("\n" + "="*60)
        print("PROCESSING POSTS")
        print("="*60 + "\n")
        
        df = posts_df.copy()
        
        # Combine title and selftext
        print("Combining title and selftext...")
        df['full_text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
        
        # Clean text - multiple versions for different use cases
        print("Cleaning text...")
        tqdm.pandas(desc="Basic cleaning")
        df['text_cleaned'] = df['full_text'].progress_apply(
            lambda x: self.clean_text(x, remove_urls=True, expand_contractions=True)
        )
        
        # Version without stopwords
        print("\nRemoving stopwords...")
        tqdm.pandas(desc="Stopword removal")
        df['text_no_stopwords'] = df['text_cleaned'].progress_apply(
            lambda x: self.remove_stopwords(x, preserve_medical=True)
        )
        
        # Lemmatized version
        print("\nLemmatizing...")
        tqdm.pandas(desc="Lemmatization")
        df['text_lemmatized'] = df['text_cleaned'].progress_apply(self.lemmatize_text)
        
        # Extract medical terms
        print("\nExtracting medical terms...")
        tqdm.pandas(desc="Medical term extraction")
        df['medical_terms'] = df['full_text'].progress_apply(self.extract_medical_terms)
        df['medical_term_count'] = df['medical_terms'].apply(len)
        
        # Calculate text statistics
        print("\nCalculating text statistics...")
        tqdm.pandas(desc="Text statistics")
        text_stats = df['full_text'].progress_apply(self.calculate_text_statistics)
        stats_df = pd.DataFrame(text_stats.tolist())
        df = pd.concat([df, stats_df], axis=1)
        
        # Add temporal features
        print("\nAdding temporal features...")
        df['created_datetime'] = pd.to_datetime(df['created_utc'], unit='s')
        df['year'] = df['created_datetime'].dt.year
        df['month'] = df['created_datetime'].dt.month
        df['day_of_week'] = df['created_datetime'].dt.dayofweek
        df['hour'] = df['created_datetime'].dt.hour
        
        # Engagement metrics
        df['engagement_ratio'] = df['num_comments'] / (df['score'] + 1)
        
        print(f"\n✓ Processed {len(df)} posts")
        return df
    
    def save_cleaned_data(self, posts_df, comments_df=None, output_format='csv'):
        """
        Save cleaned data to files
        
        Args:
            posts_df: Cleaned posts DataFrame
            comments_df: Cleaned comments DataFrame (optional)
            output_format: 'csv', 'json', or 'both'
        """
        print("\n" + "="*60)
        print("SAVING CLEANED DATA")
        print("="*60 + "\n")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_format in ['csv', 'both']:
            posts_file = f'cleaned_posts_{timestamp}.csv'
            posts_df.to_csv(posts_file, index=False, encoding='utf-8')
            print(f"✓ Saved posts to {posts_file}")
            
            if comments_df is not None:
                comments_file = f'cleaned_comments_{timestamp}.csv'
                comments_df.to_csv(comments_file, index=False, encoding='utf-8')
                print(f"✓ Saved comments to {comments_file}")
        
        if output_format in ['json', 'both']:
            posts_file = f'cleaned_posts_{timestamp}.json'
            posts_df.to_json(posts_file, orient='records', indent=2)
            print(f"✓ Saved posts to {posts_file}")
            
            if comments_df is not None:
                comments_file = f'cleaned_comments_{timestamp}.json'
                comments_df.to_json(comments_file, orient='records', indent=2)
                print(f"✓ Saved comments to {comments_file}")
        
        # Save summary statistics
        summary_file = f'cleaning_summary_{timestamp}.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("DATA CLEANING SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total Posts: {len(posts_df)}\n")
            if comments_df is not None:
                f.write(f"Total Comments: {len(comments_df)}\n")
            
            f.write(f"\nDate Range: {posts_df['created_datetime'].min()} to {posts_df['created_datetime'].max()}\n")
            
            f.write(f"\nPosts by Subreddit:\n")
            subreddit_counts = posts_df['subreddit'].value_counts()
            for subreddit, count in subreddit_counts.items():
                f.write(f"  r/{subreddit}: {count}\n")
            
            f.write(f"\nText Statistics (Posts):\n")
            f.write(f"  Avg word count: {posts_df['word_count'].mean():.2f}\n")
            f.write(f"  Avg char count: {posts_df['char_count'].mean():.2f}\n")
            f.write(f"  Avg medical terms: {posts_df['medical_term_count'].mean():.2f}\n")
            
            f.write(f"\nEngagement Statistics:\n")
            f.write(f"  Avg score: {posts_df['score'].mean():.2f}\n")
            f.write(f"  Avg comments: {posts_df['num_comments'].mean():.2f}\n")
            
        print(f"✓ Saved summary to {summary_file}")
        print("\n" + "="*60 + "\n")
    
    def run_full_pipeline(self, save_format='csv'):
        """
        Run the complete cleaning and preprocessing pipeline
        
        Args:
            save_format: Output format ('csv', 'json', or 'both')
        """
        print("\n" + "="*70)
        print(" "*15 + "DATA CLEANING PIPELINE")
        print("="*70)
        
        # Load data
        posts_df, comments_df = self.load_data_from_firebase()
        
        if posts_df is None or len(posts_df) == 0:
            print("No data to process!")
            return None, None
        
        # Process posts
        cleaned_posts = self.process_posts(posts_df)
        
        # Process comments if available
        cleaned_comments = None
        if comments_df is not None and len(comments_df) > 0:
            print("\n" + "="*60)
            print("PROCESSING COMMENTS")
            print("="*60 + "\n")
            # Similar processing for comments
            # (Simplified for now)
            cleaned_comments = comments_df.copy()
        
        # Save cleaned data
        self.save_cleaned_data(cleaned_posts, cleaned_comments, output_format=save_format)
        
        print("✓ Data cleaning pipeline complete!")
        print("="*70 + "\n")
        
        return cleaned_posts, cleaned_comments


def main():
    """Run the cleaning pipeline"""
    cleaner = DataCleaner()
    cleaned_posts, cleaned_comments = cleaner.run_full_pipeline(save_format='csv')
    
    # Display sample
    if cleaned_posts is not None:
        print("\n" + "="*60)
        print("SAMPLE CLEANED DATA")
        print("="*60 + "\n")
        
        print("Columns available:")
        for col in cleaned_posts.columns:
            print(f"  - {col}")
        
        print(f"\nFirst post sample:")
        sample = cleaned_posts.iloc[0]
        print(f"  Subreddit: r/{sample['subreddit']}")
        print(f"  Title: {sample['title'][:100]}...")
        print(f"  Word count: {sample['word_count']}")
        print(f"  Medical terms: {sample['medical_terms']}")
        print(f"  Relevance score: {sample['relevance_score']}")


if __name__ == "__main__":
    main()
