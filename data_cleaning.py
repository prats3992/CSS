"""
Data Cleaning and Preprocessing Module
Loads data from Firebase, cleans text, handles missing values, removes duplicates
"""

import pandas as pd
import numpy as np
import re
import emoji
import contractions
from firebase_manager import FirebaseManager
from datetime import datetime
import json

class DataCleaner:
    def __init__(self):
        """Initialize the data cleaner with Firebase connection"""
        self.firebase = FirebaseManager()
        
    def load_data_from_firebase(self):
        """
        Load posts and comments from Firebase
        
        Returns:
            tuple: (posts_df, comments_df)
        """
        print("Loading data from Firebase...")
        
        # Get posts
        posts_ref = self.firebase.db.child('reddit_posts').get()
        posts_list = []
        
        if posts_ref:
            for post_id, post_data in posts_ref.items():
                post_data['id'] = post_id
                posts_list.append(post_data)
        
        # Get comments
        comments_ref = self.firebase.db.child('reddit_comments').get()
        comments_list = []
        
        if comments_ref:
            for comment_id, comment_data in comments_ref.items():
                comment_data['id'] = comment_id
                comments_list.append(comment_data)
        
        posts_df = pd.DataFrame(posts_list) if posts_list else pd.DataFrame()
        comments_df = pd.DataFrame(comments_list) if comments_list else pd.DataFrame()
        
        print(f"Loaded {len(posts_df)} posts and {len(comments_df)} comments")
        
        return posts_df, comments_df
    
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Expand contractions
        text = contractions.fix(text)
        
        # Convert emojis to text descriptions
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'\[deleted\]|\[removed\]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        return text
    
    def parse_timestamp(self, timestamp):
        """
        Parse various timestamp formats to datetime
        
        Args:
            timestamp: Unix timestamp or ISO string
            
        Returns:
            datetime: Parsed datetime object
        """
        if pd.isna(timestamp):
            return None
        
        try:
            # Try Unix timestamp
            if isinstance(timestamp, (int, float)):
                return datetime.fromtimestamp(timestamp)
            
            # Try ISO string
            if isinstance(timestamp, str):
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
        except Exception as e:
            print(f"Error parsing timestamp {timestamp}: {e}")
            return None
    
    def clean_posts(self, posts_df):
        """
        Clean posts dataframe
        
        Args:
            posts_df (DataFrame): Raw posts data
            
        Returns:
            DataFrame: Cleaned posts data
        """
        if posts_df.empty:
            print("No posts to clean")
            return posts_df
        
        print("\nCleaning posts data...")
        print(f"Initial posts: {len(posts_df)}")
        
        # Remove duplicates
        posts_df = posts_df.drop_duplicates(subset=['id'], keep='first')
        print(f"After removing duplicates: {len(posts_df)}")
        
        # Clean text fields
        if 'title' in posts_df.columns:
            posts_df['title_clean'] = posts_df['title'].apply(self.clean_text)
        
        if 'selftext' in posts_df.columns:
            posts_df['selftext_clean'] = posts_df['selftext'].apply(self.clean_text)
        
        # Combine title and selftext for full content
        if 'title_clean' in posts_df.columns and 'selftext_clean' in posts_df.columns:
            posts_df['full_text'] = posts_df['title_clean'] + ' ' + posts_df['selftext_clean']
            posts_df['full_text'] = posts_df['full_text'].str.strip()
        
        # Parse timestamps
        if 'created_utc' in posts_df.columns:
            posts_df['created_datetime'] = posts_df['created_utc'].apply(self.parse_timestamp)
            posts_df['year'] = posts_df['created_datetime'].dt.year
            posts_df['month'] = posts_df['created_datetime'].dt.month
            posts_df['year_month'] = posts_df['created_datetime'].dt.to_period('M')
        
        # Handle missing scores
        if 'score' in posts_df.columns:
            posts_df['score'] = posts_df['score'].fillna(0).astype(int)
        
        # Handle missing num_comments
        if 'num_comments' in posts_df.columns:
            posts_df['num_comments'] = posts_df['num_comments'].fillna(0).astype(int)
        
        # Add COVID period flag
        if 'created_datetime' in posts_df.columns:
            covid_start = pd.Timestamp('2020-03-01')
            posts_df['is_post_covid'] = posts_df['created_datetime'] >= covid_start
            posts_df['covid_period'] = posts_df['is_post_covid'].map({
                True: 'Post-COVID', 
                False: 'Pre-COVID'
            })
        
        # Remove posts with no content
        if 'full_text' in posts_df.columns:
            posts_df = posts_df[posts_df['full_text'].str.len() > 0]
            print(f"After removing empty posts: {len(posts_df)}")
        
        # Sort by date
        if 'created_datetime' in posts_df.columns:
            posts_df = posts_df.sort_values('created_datetime')
        
        print(f"Final cleaned posts: {len(posts_df)}")
        
        return posts_df
    
    def clean_comments(self, comments_df):
        """
        Clean comments dataframe
        
        Args:
            comments_df (DataFrame): Raw comments data
            
        Returns:
            DataFrame: Cleaned comments data
        """
        if comments_df.empty:
            print("No comments to clean")
            return comments_df
        
        print("\nCleaning comments data...")
        print(f"Initial comments: {len(comments_df)}")
        
        # Remove duplicates
        comments_df = comments_df.drop_duplicates(subset=['id'], keep='first')
        print(f"After removing duplicates: {len(comments_df)}")
        
        # Clean text fields
        if 'body' in comments_df.columns:
            comments_df['body_clean'] = comments_df['body'].apply(self.clean_text)
        
        # Parse timestamps
        if 'created_utc' in comments_df.columns:
            comments_df['created_datetime'] = comments_df['created_utc'].apply(self.parse_timestamp)
            comments_df['year'] = comments_df['created_datetime'].dt.year
            comments_df['month'] = comments_df['created_datetime'].dt.month
            comments_df['year_month'] = comments_df['created_datetime'].dt.to_period('M')
        
        # Handle missing scores
        if 'score' in comments_df.columns:
            comments_df['score'] = comments_df['score'].fillna(0).astype(int)
        
        # Add COVID period flag
        if 'created_datetime' in comments_df.columns:
            covid_start = pd.Timestamp('2020-03-01')
            comments_df['is_post_covid'] = comments_df['created_datetime'] >= covid_start
            comments_df['covid_period'] = comments_df['is_post_covid'].map({
                True: 'Post-COVID', 
                False: 'Pre-COVID'
            })
        
        # Remove comments with no content
        if 'body_clean' in comments_df.columns:
            comments_df = comments_df[comments_df['body_clean'].str.len() > 0]
            print(f"After removing empty comments: {len(comments_df)}")
        
        # Sort by date
        if 'created_datetime' in comments_df.columns:
            comments_df = comments_df.sort_values('created_datetime')
        
        print(f"Final cleaned comments: {len(comments_df)}")
        
        return comments_df
    
    def get_data_summary(self, posts_df, comments_df):
        """
        Generate summary statistics for the cleaned data
        
        Args:
            posts_df (DataFrame): Cleaned posts
            comments_df (DataFrame): Cleaned comments
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_posts': len(posts_df),
            'total_comments': len(comments_df),
        }
        
        if not posts_df.empty:
            if 'subreddit' in posts_df.columns:
                summary['unique_subreddits'] = posts_df['subreddit'].nunique()
                summary['posts_by_subreddit'] = posts_df['subreddit'].value_counts().to_dict()
            
            if 'created_datetime' in posts_df.columns:
                summary['date_range'] = {
                    'start': str(posts_df['created_datetime'].min()),
                    'end': str(posts_df['created_datetime'].max())
                }
                summary['posts_by_year'] = posts_df['year'].value_counts().sort_index().to_dict()
            
            if 'covid_period' in posts_df.columns:
                summary['posts_by_covid_period'] = posts_df['covid_period'].value_counts().to_dict()
        
        return summary
    
    def save_cleaned_data(self, posts_df, comments_df, output_dir='cleaned_data'):
        """
        Save cleaned data to CSV files
        
        Args:
            posts_df (DataFrame): Cleaned posts
            comments_df (DataFrame): Cleaned comments
            output_dir (str): Output directory path
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save posts
        if not posts_df.empty:
            posts_file = os.path.join(output_dir, 'cleaned_posts.csv')
            posts_df.to_csv(posts_file, index=False)
            print(f"\nSaved cleaned posts to {posts_file}")
        
        # Save comments
        if not comments_df.empty:
            comments_file = os.path.join(output_dir, 'cleaned_comments.csv')
            comments_df.to_csv(comments_file, index=False)
            print(f"Saved cleaned comments to {comments_file}")
        
        # Save summary
        summary = self.get_data_summary(posts_df, comments_df)
        summary_file = os.path.join(output_dir, 'data_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved data summary to {summary_file}")
    
    def run_full_cleaning(self, save_output=True):
        """
        Run complete data cleaning pipeline
        
        Args:
            save_output (bool): Whether to save cleaned data
            
        Returns:
            tuple: (cleaned_posts_df, cleaned_comments_df)
        """
        # Load data
        posts_df, comments_df = self.load_data_from_firebase()
        
        # Clean data
        posts_df = self.clean_posts(posts_df)
        comments_df = self.clean_comments(comments_df)
        
        # Print summary
        summary = self.get_data_summary(posts_df, comments_df)
        print("\n" + "="*60)
        print("DATA CLEANING SUMMARY")
        print("="*60)
        print(json.dumps(summary, indent=2))
        
        # Save if requested
        if save_output:
            self.save_cleaned_data(posts_df, comments_df)
        
        return posts_df, comments_df


if __name__ == "__main__":
    cleaner = DataCleaner()
    posts_df, comments_df = cleaner.run_full_cleaning(save_output=True)
    print("\nData cleaning completed successfully!")
