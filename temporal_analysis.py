"""
Temporal Trend Analysis Module
Analyzes sentiment and post volume trends over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from collections import Counter

class TemporalAnalyzer:
    def __init__(self, output_dir='analysis_output/temporal'):
        """Initialize temporal analyzer"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
    
    def analyze_sentiment_trend_over_years(self, posts_df):
        """
        Analyze sentiment trend over years
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
            
        Returns:
            DataFrame: Yearly sentiment statistics
        """
        print("\nAnalyzing sentiment trends over years...")
        
        if 'year' not in posts_df.columns or 'sentiment_compound' not in posts_df.columns:
            print("Missing required columns")
            return None
        
        # Group by year
        yearly_sentiment = posts_df.groupby('year').agg({
            'sentiment_compound': ['mean', 'median', 'std', 'count'],
            'sentiment_pos': 'mean',
            'sentiment_neg': 'mean',
            'sentiment_neu': 'mean'
        }).round(3)
        
        yearly_sentiment.columns = ['_'.join(col).strip() for col in yearly_sentiment.columns.values]
        yearly_sentiment = yearly_sentiment.reset_index()
        
        # Calculate sentiment category percentages by year
        sentiment_by_year = posts_df.groupby(['year', 'sentiment_category']).size().unstack(fill_value=0)
        sentiment_pct_by_year = sentiment_by_year.div(sentiment_by_year.sum(axis=1), axis=0) * 100
        
        # Plot sentiment trend (excluding stacked area chart)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Average compound score over years
        axes[0].plot(yearly_sentiment['year'], yearly_sentiment['sentiment_compound_mean'], 
                       marker='o', linewidth=2, markersize=8, color='#2E86AB')
        axes[0].fill_between(yearly_sentiment['year'], 
                                yearly_sentiment['sentiment_compound_mean'] - yearly_sentiment['sentiment_compound_std'],
                                yearly_sentiment['sentiment_compound_mean'] + yearly_sentiment['sentiment_compound_std'],
                                alpha=0.3, color='#2E86AB')
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Average Sentiment (Compound)', fontsize=12)
        axes[0].set_title('Sentiment Trend Over Years', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Sentiment components (pos, neg, neu) over years
        axes[1].plot(yearly_sentiment['year'], yearly_sentiment['sentiment_pos_mean'], 
                       marker='o', label='Positive', linewidth=2, color='#06D6A0')
        axes[1].plot(yearly_sentiment['year'], yearly_sentiment['sentiment_neg_mean'], 
                       marker='s', label='Negative', linewidth=2, color='#EF476F')
        axes[1].plot(yearly_sentiment['year'], yearly_sentiment['sentiment_neu_mean'], 
                       marker='^', label='Neutral', linewidth=2, color='#FFD166')
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Average Score', fontsize=12)
        axes[1].set_title('Sentiment Components Over Years', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Post count per year
        axes[2].bar(yearly_sentiment['year'], yearly_sentiment['sentiment_compound_count'],
                      color='#118AB2', alpha=0.7)
        axes[2].set_xlabel('Year', fontsize=12)
        axes[2].set_ylabel('Number of Posts', fontsize=12)
        axes[2].set_title('Post Volume Over Years', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment_trend_over_years.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved sentiment trend visualization")
        
        return yearly_sentiment
    
    def analyze_post_volume_by_sentiment(self, posts_df):
        """
        Analyze post volume over years colored by sentiment
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
        """
        print("\nAnalyzing post volume by sentiment...")
        
        if 'year' not in posts_df.columns or 'sentiment_category' not in posts_df.columns:
            print("Missing required columns")
            return
        
        # Group by year and sentiment category
        volume_by_sentiment = posts_df.groupby(['year', 'sentiment_category']).size().unstack(fill_value=0)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        volume_by_sentiment.plot(kind='bar', stacked=True, ax=ax,
                                 color=['#EF476F', '#FFD166', '#06D6A0'],
                                 width=0.8)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Posts', fontsize=12)
        ax.set_title('Post Volume Over Years (Colored by Sentiment)', fontsize=14, fontweight='bold')
        ax.legend(title='Sentiment', loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'post_volume_by_sentiment.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved post volume by sentiment visualization")
    
    def analyze_positive_percentage_by_year(self, posts_df):
        """
        Analyze percentage of positive posts by year
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
            
        Returns:
            DataFrame: Yearly positive percentage
        """
        print("\nAnalyzing positive post percentage by year...")
        
        if 'year' not in posts_df.columns or 'sentiment_category' not in posts_df.columns:
            print("Missing required columns")
            return None
        
        # Calculate positive percentage by year
        sentiment_by_year = posts_df.groupby(['year', 'sentiment_category']).size().unstack(fill_value=0)
        positive_pct = (sentiment_by_year['positive'] / sentiment_by_year.sum(axis=1) * 100).round(2)
        
        # Create DataFrame
        positive_pct_df = pd.DataFrame({
            'year': positive_pct.index,
            'positive_percentage': positive_pct.values,
            'total_posts': sentiment_by_year.sum(axis=1).values
        })
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(positive_pct_df['year'], positive_pct_df['positive_percentage'], 
               marker='o', linewidth=3, markersize=10, color='#06D6A0')
        ax.fill_between(positive_pct_df['year'], positive_pct_df['positive_percentage'],
                       alpha=0.3, color='#06D6A0')
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Positive Posts (%)', fontsize=12)
        ax.set_title('Percentage of Positive Posts by Year', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(positive_pct_df['year'], positive_pct_df['positive_percentage']):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'positive_percentage_by_year.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved positive percentage visualization")
        
        return positive_pct_df
    
    def analyze_yoy_sentiment_change(self, posts_df):
        """
        Analyze year-over-year sentiment change
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
            
        Returns:
            DataFrame: YoY sentiment changes
        """
        print("\nAnalyzing year-over-year sentiment changes...")
        
        if 'year' not in posts_df.columns or 'sentiment_compound' not in posts_df.columns:
            print("Missing required columns")
            return None
        
        # Calculate yearly average sentiment
        yearly_avg = posts_df.groupby('year')['sentiment_compound'].mean()
        
        # Calculate YoY change
        yoy_change = yearly_avg.diff()
        yoy_pct_change = yearly_avg.pct_change() * 100
        
        # Create DataFrame
        yoy_df = pd.DataFrame({
            'year': yearly_avg.index,
            'avg_sentiment': yearly_avg.values.round(3),
            'yoy_change': yoy_change.values.round(3),
            'yoy_pct_change': yoy_pct_change.values.round(2)
        })
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Absolute change only
        colors = ['#06D6A0' if x >= 0 else '#EF476F' for x in yoy_df['yoy_change'][1:]]
        ax.bar(yoy_df['year'][1:], yoy_df['yoy_change'][1:], color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('YoY Sentiment Change', fontsize=12)
        ax.set_title('Year-over-Year Sentiment Change', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'yoy_sentiment_change.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved YoY sentiment change visualization")
        
        return yoy_df
    
    def generate_wordcloud_by_year(self, posts_df, text_field='full_text', max_years=None):
        """
        Generate word clouds for different years
        
        Args:
            posts_df (DataFrame): Posts with text
            text_field (str): Text field to use
            max_years (int): Maximum number of years to display
        """
        print("\nGenerating word clouds by year...")
        
        if 'year' not in posts_df.columns or text_field not in posts_df.columns:
            print("Missing required columns")
            return
        
        years = sorted(posts_df['year'].unique())
        
        if max_years:
            # Select evenly distributed years
            step = max(1, len(years) // max_years)
            years = years[::step][:max_years]
        
        # Generate wordclouds
        n_years = len(years)
        n_cols = 3
        n_rows = (n_years + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, year in enumerate(years):
            year_posts = posts_df[posts_df['year'] == year]
            text = ' '.join(year_posts[text_field].dropna().astype(str))
            
            if len(text.strip()) > 0:
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=100,
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate(text)
                
                axes[idx].imshow(wordcloud, interpolation='bilinear')
                axes[idx].set_title(f'Year {year} (n={len(year_posts)})', 
                                   fontsize=12, fontweight='bold')
                axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[idx].set_title(f'Year {year}', fontsize=12, fontweight='bold')
                axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(years), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'wordcloud_by_year.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved wordcloud by year visualization")
    
    def run_full_analysis(self, posts_df):
        """
        Run complete temporal analysis
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
            
        Returns:
            dict: Analysis results
        """
        print("\n" + "="*60)
        print("TEMPORAL TREND ANALYSIS")
        print("="*60)
        
        results = {}
        
        # Run all analyses
        results['yearly_sentiment'] = self.analyze_sentiment_trend_over_years(posts_df)
        self.analyze_post_volume_by_sentiment(posts_df)
        results['positive_percentage'] = self.analyze_positive_percentage_by_year(posts_df)
        results['yoy_change'] = self.analyze_yoy_sentiment_change(posts_df)
        self.generate_wordcloud_by_year(posts_df, max_years=6)
        
        print(f"\nAll temporal analyses saved to: {self.output_dir}")
        
        return results


if __name__ == "__main__":
    from data_cleaning import DataCleaner
    from sentiment_analysis import SentimentAnalyzer
    
    # Load, clean, and analyze sentiment
    cleaner = DataCleaner()
    posts_df, comments_df = cleaner.run_full_cleaning(save_output=False)
    
    analyzer = SentimentAnalyzer()
    posts_df, comments_df = analyzer.run_full_analysis(
        posts_df, comments_df, save_output=False
    )
    
    # Run temporal analysis
    temporal = TemporalAnalyzer()
    results = temporal.run_full_analysis(posts_df)
    
    print("\nTemporal analysis completed successfully!")
