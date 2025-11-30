"""
COVID Comparison Analysis Module
Compares pre-COVID vs post-COVID periods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy import stats
import os

class CovidComparison:
    def __init__(self, output_dir='analysis_output/covid_comparison'):
        """Initialize COVID comparison analyzer"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        
        self.covid_start = pd.Timestamp('2020-03-01')
    
    def split_by_covid_period(self, df):
        """
        Split dataframe into pre and post COVID periods
        
        Args:
            df (DataFrame): Dataframe with created_datetime column
            
        Returns:
            tuple: (pre_covid_df, post_covid_df)
        """
        if 'created_datetime' not in df.columns:
            print("Warning: created_datetime column not found")
            return None, None
        
        pre_covid = df[df['created_datetime'] < self.covid_start]
        post_covid = df[df['created_datetime'] >= self.covid_start]
        
        return pre_covid, post_covid
    
    def compare_sentiment_distribution(self, posts_df):
        """
        Compare sentiment distribution pre vs post COVID
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
        """
        print("\nComparing sentiment distribution pre vs post COVID...")
        
        if 'sentiment_category' not in posts_df.columns or 'covid_period' not in posts_df.columns:
            print("Missing required columns")
            return
        
        # Get sentiment distribution by COVID period
        sentiment_dist = posts_df.groupby(['covid_period', 'sentiment_category']).size().unstack(fill_value=0)
        sentiment_pct = sentiment_dist.div(sentiment_dist.sum(axis=1), axis=0) * 100
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute counts
        sentiment_dist.plot(kind='bar', ax=axes[0], color=['#EF476F', '#FFD166', '#06D6A0'], 
                           width=0.7, alpha=0.8)
        axes[0].set_xlabel('COVID Period', fontsize=12)
        axes[0].set_ylabel('Number of Posts', fontsize=12)
        axes[0].set_title('Sentiment Distribution: Absolute Counts', fontsize=14, fontweight='bold')
        axes[0].legend(title='Sentiment')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
        
        # Percentages
        sentiment_pct.plot(kind='bar', ax=axes[1], color=['#EF476F', '#FFD166', '#06D6A0'], 
                          width=0.7, alpha=0.8)
        axes[1].set_xlabel('COVID Period', fontsize=12)
        axes[1].set_ylabel('Percentage (%)', fontsize=12)
        axes[1].set_title('Sentiment Distribution: Percentages', fontsize=14, fontweight='bold')
        axes[1].legend(title='Sentiment')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment_distribution_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved sentiment distribution comparison")
        
        # Print statistics
        print("\nSentiment Distribution:")
        print(sentiment_pct.round(2))
    
    def compare_vader_scores(self, posts_df):
        """
        Compare average VADER scores pre vs post COVID
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
            
        Returns:
            DataFrame: Comparison statistics
        """
        print("\nComparing VADER scores pre vs post COVID...")
        
        if 'sentiment_compound' not in posts_df.columns or 'covid_period' not in posts_df.columns:
            print("Missing required columns")
            return None
        
        pre_covid, post_covid = self.split_by_covid_period(posts_df)
        
        if pre_covid is None or post_covid is None:
            return None
        
        # Calculate statistics
        stats_df = posts_df.groupby('covid_period').agg({
            'sentiment_compound': ['mean', 'median', 'std', 'count'],
            'sentiment_pos': 'mean',
            'sentiment_neg': 'mean',
            'sentiment_neu': 'mean'
        }).round(3)
        
        stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
        
        # Statistical test
        pre_scores = pre_covid['sentiment_compound'].dropna()
        post_scores = post_covid['sentiment_compound'].dropna()
        t_stat, p_value = stats.ttest_ind(pre_scores, post_scores)
        
        print(f"\nT-test results:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Visualization - only bar chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Bar chart of averages
        avg_scores = posts_df.groupby('covid_period')['sentiment_compound'].mean()
        ax.bar(avg_scores.index, avg_scores.values, color=['#2E86AB', '#A23B72'], alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('COVID Period', fontsize=12)
        ax.set_ylabel('Average Sentiment Score', fontsize=12)
        ax.set_title('Average VADER Score Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (label, value) in enumerate(avg_scores.items()):
            ax.text(i, value, f'{value:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'vader_score_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved VADER score comparison")
        print("\nStatistics by COVID Period:")
        print(stats_df)
        
        return stats_df
    
    def compare_post_volume(self, posts_df):
        """
        Compare post volume pre vs post COVID
        
        Args:
            posts_df (DataFrame): Posts dataframe
        """
        print("\nComparing post volume pre vs post COVID...")
        
        if 'covid_period' not in posts_df.columns or 'year_month' not in posts_df.columns:
            print("Missing required columns")
            return
        
        # Volume by period
        volume_by_period = posts_df['covid_period'].value_counts()
        
        # Volume over time
        monthly_volume = posts_df.groupby(['year_month', 'covid_period']).size().unstack(fill_value=0)
        
        # Visualization
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # Bar chart of total volume
        axes[0].bar(volume_by_period.index, volume_by_period.values, 
                   color=['#2E86AB', '#A23B72'], alpha=0.7, width=0.6)
        axes[0].set_xlabel('COVID Period', fontsize=12)
        axes[0].set_ylabel('Number of Posts', fontsize=12)
        axes[0].set_title('Total Post Volume by COVID Period', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (label, value) in enumerate(volume_by_period.items()):
            axes[0].text(i, value, f'{value:,}', ha='center', va='bottom', fontsize=11)
        
        # Time series
        monthly_volume.plot(ax=axes[1], color=['#2E86AB', '#A23B72'], linewidth=2, marker='o')
        axes[1].axvline(x=pd.Period('2020-03', 'M'), color='red', linestyle='--', 
                       linewidth=2, label='COVID-19 Start')
        axes[1].set_xlabel('Year-Month', fontsize=12)
        axes[1].set_ylabel('Number of Posts', fontsize=12)
        axes[1].set_title('Monthly Post Volume Over Time', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'post_volume_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved post volume comparison")
        print(f"\nPre-COVID posts: {volume_by_period.get('Pre-COVID', 0):,}")
        print(f"Post-COVID posts: {volume_by_period.get('Post-COVID', 0):,}")
    
    def compare_sentiment_score_distributions(self, posts_df):
        """
        Compare detailed sentiment score distributions
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
        """
        print("\nComparing sentiment score distributions...")
        
        if 'sentiment_compound' not in posts_df.columns or 'covid_period' not in posts_df.columns:
            print("Missing required columns")
            return
        
        pre_covid, post_covid = self.split_by_covid_period(posts_df)
        
        if pre_covid is None or post_covid is None:
            return
        
        # Simplified distribution plot - only histogram and sentiment components
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram comparison
        axes[0].hist([pre_covid['sentiment_compound'], post_covid['sentiment_compound']], 
                       bins=30, label=['Pre-COVID', 'Post-COVID'], 
                       color=['#2E86AB', '#A23B72'], alpha=0.6)
        axes[0].set_xlabel('Sentiment Compound Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Sentiment Score Distribution (Histogram)', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Sentiment components comparison
        components = ['sentiment_pos', 'sentiment_neg', 'sentiment_neu']
        pre_means = [pre_covid[col].mean() for col in components]
        post_means = [post_covid[col].mean() for col in components]
        
        x = np.arange(len(components))
        width = 0.35
        
        axes[1].bar(x - width/2, pre_means, width, label='Pre-COVID', 
                      color='#2E86AB', alpha=0.7)
        axes[1].bar(x + width/2, post_means, width, label='Post-COVID', 
                      color='#A23B72', alpha=0.7)
        axes[1].set_xlabel('Sentiment Component', fontsize=12)
        axes[1].set_ylabel('Average Score', fontsize=12)
        axes[1].set_title('Sentiment Components Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['Positive', 'Negative', 'Neutral'])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment_score_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved sentiment score distributions")
    
    def compare_subreddit_distribution(self, posts_df):
        """
        Compare post distribution across subreddits
        
        Args:
            posts_df (DataFrame): Posts dataframe
        """
        print("\nComparing post distribution across subreddits...")
        
        if 'subreddit' not in posts_df.columns or 'covid_period' not in posts_df.columns:
            print("Missing required columns")
            return
        
        # Get top subreddits
        top_subreddits = posts_df['subreddit'].value_counts().head(15).index
        
        # Filter for top subreddits
        filtered_df = posts_df[posts_df['subreddit'].isin(top_subreddits)]
        
        # Create crosstab
        subreddit_covid = pd.crosstab(filtered_df['subreddit'], 
                                      filtered_df['covid_period'], normalize='columns') * 100
        
        # Visualization
        fig, ax = plt.subplots(figsize=(14, 10))
        subreddit_covid.plot(kind='barh', ax=ax, color=['#2E86AB', '#A23B72'], alpha=0.7)
        ax.set_xlabel('Percentage (%)', fontsize=12)
        ax.set_ylabel('Subreddit', fontsize=12)
        ax.set_title('Post Distribution Across Top Subreddits (by COVID Period)', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='COVID Period', loc='best')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'subreddit_distribution_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved subreddit distribution comparison")
    
    def compare_wordclouds(self, posts_df, text_field='full_text'):
        """
        Compare word clouds pre vs post COVID
        
        Args:
            posts_df (DataFrame): Posts with text
            text_field (str): Text field to use
        """
        print("\nComparing word clouds pre vs post COVID...")
        
        if text_field not in posts_df.columns or 'covid_period' not in posts_df.columns:
            print("Missing required columns")
            return
        
        pre_covid, post_covid = self.split_by_covid_period(posts_df)
        
        if pre_covid is None or post_covid is None:
            return
        
        # Generate text corpora
        pre_text = ' '.join(pre_covid[text_field].dropna().astype(str))
        post_text = ' '.join(post_covid[text_field].dropna().astype(str))
        
        # Create word clouds
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        if len(pre_text.strip()) > 0:
            wc_pre = WordCloud(width=800, height=400, background_color='white',
                              colormap='Blues', max_words=100, 
                              relative_scaling=0.5, min_font_size=10).generate(pre_text)
            axes[0].imshow(wc_pre, interpolation='bilinear')
            axes[0].set_title(f'Pre-COVID Word Cloud (n={len(pre_covid):,})', 
                            fontsize=14, fontweight='bold')
            axes[0].axis('off')
        
        if len(post_text.strip()) > 0:
            wc_post = WordCloud(width=800, height=400, background_color='white',
                               colormap='Reds', max_words=100,
                               relative_scaling=0.5, min_font_size=10).generate(post_text)
            axes[1].imshow(wc_post, interpolation='bilinear')
            axes[1].set_title(f'Post-COVID Word Cloud (n={len(post_covid):,})', 
                            fontsize=14, fontweight='bold')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'wordcloud_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved word cloud comparison")
    
    def run_full_analysis(self, posts_df):
        """
        Run complete COVID comparison analysis
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
            
        Returns:
            dict: Analysis results
        """
        print("\n" + "="*60)
        print("PRE vs POST COVID COMPARISON")
        print("="*60)
        
        results = {}
        
        # Run all analyses
        self.compare_sentiment_distribution(posts_df)
        results['vader_comparison'] = self.compare_vader_scores(posts_df)
        self.compare_post_volume(posts_df)
        self.compare_sentiment_score_distributions(posts_df)
        self.compare_subreddit_distribution(posts_df)
        self.compare_wordclouds(posts_df)
        
        print(f"\nAll COVID comparison analyses saved to: {self.output_dir}")
        
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
    
    # Run COVID comparison
    covid_comp = CovidComparison()
    results = covid_comp.run_full_analysis(posts_df)
    
    print("\nCOVID comparison analysis completed successfully!")
