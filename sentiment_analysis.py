"""
Sentiment Analysis Module
Uses VADER sentiment analyzer to score posts and comments
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import os

class SentimentAnalyzer:
    def __init__(self):
        """Initialize VADER sentiment analyzer"""
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using VADER
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores (neg, neu, pos, compound)
        """
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
            return {
                'neg': 0.0,
                'neu': 1.0,
                'pos': 0.0,
                'compound': 0.0
            }
        
        return self.analyzer.polarity_scores(text)
    
    def categorize_sentiment(self, compound_score):
        """
        Categorize sentiment based on compound score
        
        Args:
            compound_score (float): VADER compound score
            
        Returns:
            str: Sentiment category (positive, negative, neutral)
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_posts(self, posts_df):
        """
        Analyze sentiment for all posts
        
        Args:
            posts_df (DataFrame): Cleaned posts dataframe
            
        Returns:
            DataFrame: Posts with sentiment scores added
        """
        if posts_df.empty:
            print("No posts to analyze")
            return posts_df
        
        print("\nAnalyzing sentiment for posts...")
        
        # Determine which text field to use
        text_field = 'full_text' if 'full_text' in posts_df.columns else 'title_clean'
        
        if text_field not in posts_df.columns:
            print(f"Warning: {text_field} column not found")
            return posts_df
        
        # Analyze sentiment for each post
        sentiment_scores = []
        for text in tqdm(posts_df[text_field], desc="Analyzing posts"):
            sentiment_scores.append(self.analyze_sentiment(text))
        
        # Add sentiment columns
        sentiment_df = pd.DataFrame(sentiment_scores)
        posts_df['sentiment_neg'] = sentiment_df['neg']
        posts_df['sentiment_neu'] = sentiment_df['neu']
        posts_df['sentiment_pos'] = sentiment_df['pos']
        posts_df['sentiment_compound'] = sentiment_df['compound']
        
        # Add sentiment category
        posts_df['sentiment_category'] = posts_df['sentiment_compound'].apply(
            self.categorize_sentiment
        )
        
        print(f"Completed sentiment analysis for {len(posts_df)} posts")
        print(f"Sentiment distribution:")
        print(posts_df['sentiment_category'].value_counts())
        
        return posts_df
    
    def analyze_comments(self, comments_df):
        """
        Analyze sentiment for all comments
        
        Args:
            comments_df (DataFrame): Cleaned comments dataframe
            
        Returns:
            DataFrame: Comments with sentiment scores added
        """
        if comments_df.empty:
            print("No comments to analyze")
            return comments_df
        
        print("\nAnalyzing sentiment for comments...")
        
        text_field = 'body_clean'
        
        if text_field not in comments_df.columns:
            print(f"Warning: {text_field} column not found")
            return comments_df
        
        # Analyze sentiment for each comment
        sentiment_scores = []
        for text in tqdm(comments_df[text_field], desc="Analyzing comments"):
            sentiment_scores.append(self.analyze_sentiment(text))
        
        # Add sentiment columns
        sentiment_df = pd.DataFrame(sentiment_scores)
        comments_df['sentiment_neg'] = sentiment_df['neg']
        comments_df['sentiment_neu'] = sentiment_df['neu']
        comments_df['sentiment_pos'] = sentiment_df['pos']
        comments_df['sentiment_compound'] = sentiment_df['compound']
        
        # Add sentiment category
        comments_df['sentiment_category'] = comments_df['sentiment_compound'].apply(
            self.categorize_sentiment
        )
        
        print(f"Completed sentiment analysis for {len(comments_df)} comments")
        print(f"Sentiment distribution:")
        print(comments_df['sentiment_category'].value_counts())
        
        return comments_df
    
    def get_sentiment_summary(self, posts_df, comments_df):
        """
        Generate summary statistics for sentiment analysis
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
            comments_df (DataFrame): Comments with sentiment scores
            
        Returns:
            dict: Sentiment summary statistics
        """
        summary = {}
        
        if not posts_df.empty and 'sentiment_compound' in posts_df.columns:
            summary['posts'] = {
                'total': len(posts_df),
                'avg_compound': float(posts_df['sentiment_compound'].mean()),
                'median_compound': float(posts_df['sentiment_compound'].median()),
                'std_compound': float(posts_df['sentiment_compound'].std()),
                'sentiment_distribution': posts_df['sentiment_category'].value_counts().to_dict(),
                'sentiment_percentages': (posts_df['sentiment_category'].value_counts(normalize=True) * 100).round(2).to_dict()
            }
        
        if not comments_df.empty and 'sentiment_compound' in comments_df.columns:
            summary['comments'] = {
                'total': len(comments_df),
                'avg_compound': float(comments_df['sentiment_compound'].mean()),
                'median_compound': float(comments_df['sentiment_compound'].median()),
                'std_compound': float(comments_df['sentiment_compound'].std()),
                'sentiment_distribution': comments_df['sentiment_category'].value_counts().to_dict(),
                'sentiment_percentages': (comments_df['sentiment_category'].value_counts(normalize=True) * 100).round(2).to_dict()
            }
        
        return summary
    
    def save_sentiment_data(self, posts_df, comments_df, output_dir='cleaned_data'):
        """
        Save sentiment-analyzed data
        
        Args:
            posts_df (DataFrame): Posts with sentiment
            comments_df (DataFrame): Comments with sentiment
            output_dir (str): Output directory
        """
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save posts with sentiment
        if not posts_df.empty:
            posts_file = os.path.join(output_dir, 'posts_with_sentiment.csv')
            posts_df.to_csv(posts_file, index=False)
            print(f"\nSaved posts with sentiment to {posts_file}")
        
        # Save comments with sentiment
        if not comments_df.empty:
            comments_file = os.path.join(output_dir, 'comments_with_sentiment.csv')
            comments_df.to_csv(comments_file, index=False)
            print(f"Saved comments with sentiment to {comments_file}")
        
        # Save sentiment summary
        summary = self.get_sentiment_summary(posts_df, comments_df)
        summary_file = os.path.join(output_dir, 'sentiment_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved sentiment summary to {summary_file}")
    
    def run_full_analysis(self, posts_df, comments_df, save_output=True):
        """
        Run complete sentiment analysis pipeline
        
        Args:
            posts_df (DataFrame): Cleaned posts
            comments_df (DataFrame): Cleaned comments
            save_output (bool): Whether to save results
            
        Returns:
            tuple: (posts_df, comments_df) with sentiment scores
        """
        # Analyze sentiment
        posts_df = self.analyze_posts(posts_df)
        comments_df = self.analyze_comments(comments_df)
        
        # Print summary
        summary = self.get_sentiment_summary(posts_df, comments_df)
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("="*60)
        import json
        print(json.dumps(summary, indent=2))
        
        # Save if requested
        if save_output:
            self.save_sentiment_data(posts_df, comments_df)
        
        return posts_df, comments_df


if __name__ == "__main__":
    from data_cleaning import DataCleaner
    
    # Load and clean data
    cleaner = DataCleaner()
    posts_df, comments_df = cleaner.run_full_cleaning(save_output=False)
    
    # Run sentiment analysis
    analyzer = SentimentAnalyzer()
    posts_df, comments_df = analyzer.run_full_analysis(
        posts_df, comments_df, save_output=True
    )
    
    print("\nSentiment analysis completed successfully!")
