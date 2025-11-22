"""
Exploratory Data Analysis (EDA) for Pre-eclampsia Reddit Data
Includes temporal trends, TF-IDF analysis, and bigram/unigram word clouds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)


class EDAAnalyzer:
    def __init__(self, csv_file=None):
        """
        Initialize EDA analyzer
        
        Args:
            csv_file: Path to cleaned CSV file (if None, will look for latest)
        """
        # Load data
        if csv_file:
            self.df = pd.read_csv(csv_file)
        else:
            # Find latest cleaned file
            import glob
            files = glob.glob('cleaned_posts_*.csv')
            if files:
                latest = max(files)
                print(f"Loading latest cleaned file: {latest}")
                self.df = pd.read_csv(latest)
            else:
                raise FileNotFoundError("No cleaned CSV file found!")
        
        print(f"Loaded {len(self.df)} posts for EDA\n")
        
        # Ensure datetime column
        if 'created_datetime' not in self.df.columns:
            self.df['created_datetime'] = pd.to_datetime(self.df['created_utc'], unit='s')
        
        self.df['year'] = self.df['created_datetime'].dt.year
        self.df['month'] = self.df['created_datetime'].dt.to_period('M')
    
    def create_temporal_line_plots(self, output_dir='eda_output'):
        """Create line plots for posts and comments per year"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating temporal line plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. Posts per year
        posts_per_year = self.df.groupby('year').size()
        axes[0, 0].plot(posts_per_year.index, posts_per_year.values, 
                       marker='o', linewidth=2.5, markersize=8, color='#3498db')
        axes[0, 0].fill_between(posts_per_year.index, posts_per_year.values, 
                               alpha=0.3, color='#3498db')
        axes[0, 0].set_xlabel('Year', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Posts', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Posts Per Year', fontsize=14, fontweight='bold', pad=15)
        axes[0, 0].grid(True, alpha=0.4)
        
        # Add COVID marker
        covid_year = 2020
        if covid_year in posts_per_year.index:
            axes[0, 0].axvline(x=covid_year, color='red', linestyle='--', 
                             linewidth=2, alpha=0.7, label='COVID-19')
            axes[0, 0].legend(fontsize=10)
        
        # Add value labels
        for x, y in zip(posts_per_year.index, posts_per_year.values):
            axes[0, 0].text(x, y, f'{y}', ha='center', va='bottom', 
                          fontsize=9, fontweight='bold')
        
        # 2. Comments per year (if available)
        if 'num_comments' in self.df.columns:
            comments_per_year = self.df.groupby('year')['num_comments'].sum()
            axes[0, 1].plot(comments_per_year.index, comments_per_year.values,
                           marker='s', linewidth=2.5, markersize=8, color='#e67e22')
            axes[0, 1].fill_between(comments_per_year.index, comments_per_year.values,
                                   alpha=0.3, color='#e67e22')
            axes[0, 1].set_xlabel('Year', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Total Comments', fontsize=12, fontweight='bold')
            axes[0, 1].set_title('Comments Per Year', fontsize=14, fontweight='bold', pad=15)
            axes[0, 1].grid(True, alpha=0.4)
            
            if covid_year in comments_per_year.index:
                axes[0, 1].axvline(x=covid_year, color='red', linestyle='--',
                                 linewidth=2, alpha=0.7, label='COVID-19')
                axes[0, 1].legend(fontsize=10)
            
            # Add value labels
            for x, y in zip(comments_per_year.index, comments_per_year.values):
                axes[0, 1].text(x, y, f'{int(y):,}', ha='center', va='bottom',
                              fontsize=9, fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'Comment data not available',
                          ha='center', va='center', fontsize=14, transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Comments Per Year', fontsize=14, fontweight='bold', pad=15)
        
        # 3. Average comments per post per year
        if 'num_comments' in self.df.columns:
            avg_comments = self.df.groupby('year')['num_comments'].mean()
            axes[1, 0].plot(avg_comments.index, avg_comments.values,
                           marker='D', linewidth=2.5, markersize=8, color='#2ecc71')
            axes[1, 0].fill_between(avg_comments.index, avg_comments.values,
                                   alpha=0.3, color='#2ecc71')
            axes[1, 0].set_xlabel('Year', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Average Comments Per Post', fontsize=12, fontweight='bold')
            axes[1, 0].set_title('Engagement Trend (Avg Comments)', 
                                fontsize=14, fontweight='bold', pad=15)
            axes[1, 0].grid(True, alpha=0.4)
            
            if covid_year in avg_comments.index:
                axes[1, 0].axvline(x=covid_year, color='red', linestyle='--',
                                 linewidth=2, alpha=0.7, label='COVID-19')
                axes[1, 0].legend(fontsize=10)
            
            # Add value labels
            for x, y in zip(avg_comments.index, avg_comments.values):
                axes[1, 0].text(x, y, f'{y:.1f}', ha='center', va='bottom',
                              fontsize=9, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'Comment data not available',
                          ha='center', va='center', fontsize=14, transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Engagement Trend', fontsize=14, fontweight='bold', pad=15)
        
        # 4. Monthly post volume (last 24 months or all data if less)
        monthly_posts = self.df.groupby('month').size()
        if len(monthly_posts) > 24:
            monthly_posts = monthly_posts.iloc[-24:]
        
        monthly_posts.index = monthly_posts.index.astype(str)
        axes[1, 1].plot(range(len(monthly_posts)), monthly_posts.values,
                       marker='o', linewidth=2, markersize=6, color='#9b59b6')
        axes[1, 1].fill_between(range(len(monthly_posts)), monthly_posts.values,
                               alpha=0.3, color='#9b59b6')
        axes[1, 1].set_xlabel('Month', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Posts', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Monthly Post Volume (Recent)', 
                            fontsize=14, fontweight='bold', pad=15)
        axes[1, 1].grid(True, alpha=0.4)
        
        # Set x-axis labels (show every 3rd month to avoid crowding)
        tick_positions = range(0, len(monthly_posts), max(1, len(monthly_posts)//8))
        tick_labels = [monthly_posts.index[i] for i in tick_positions]
        axes[1, 1].set_xticks(tick_positions)
        axes[1, 1].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
        
        plt.suptitle('Temporal Trends: Posts and Comments Analysis',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = f'{output_dir}/temporal_line_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
        
        # Create statistics summary
        self._create_temporal_statistics(output_dir)
    
    def _create_temporal_statistics(self, output_dir):
        """Create detailed statistics about temporal trends"""
        stats_file = f'{output_dir}/temporal_statistics.txt'
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TEMPORAL TRENDS STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            # Posts per year stats
            posts_per_year = self.df.groupby('year').size()
            f.write("POSTS PER YEAR:\n")
            f.write("-" * 80 + "\n")
            for year, count in posts_per_year.items():
                pct = (count / len(self.df)) * 100
                f.write(f"  {int(year)}: {count:,} posts ({pct:.1f}% of total)\n")
            
            f.write(f"\n  Total years covered: {len(posts_per_year)}\n")
            f.write(f"  Average posts per year: {posts_per_year.mean():.0f}\n")
            f.write(f"  Peak year: {posts_per_year.idxmax()} ({posts_per_year.max()} posts)\n")
            f.write(f"  Lowest year: {posts_per_year.idxmin()} ({posts_per_year.min()} posts)\n")
            
            # Growth rate
            if len(posts_per_year) > 1:
                first_year_posts = posts_per_year.iloc[0]
                last_year_posts = posts_per_year.iloc[-1]
                growth_rate = ((last_year_posts / first_year_posts) - 1) * 100
                f.write(f"  Overall growth: {growth_rate:+.1f}%\n")
            
            # Comments stats
            if 'num_comments' in self.df.columns:
                f.write("\n\nCOMMENTS ANALYSIS:\n")
                f.write("-" * 80 + "\n")
                
                comments_per_year = self.df.groupby('year')['num_comments'].agg(['sum', 'mean', 'median'])
                for year in comments_per_year.index:
                    row = comments_per_year.loc[year]
                    f.write(f"  {int(year)}:\n")
                    f.write(f"    Total: {int(row['sum']):,} comments\n")
                    f.write(f"    Avg per post: {row['mean']:.1f}\n")
                    f.write(f"    Median per post: {row['median']:.0f}\n")
                
                total_comments = self.df['num_comments'].sum()
                avg_comments = self.df['num_comments'].mean()
                f.write(f"\n  Total comments (all time): {int(total_comments):,}\n")
                f.write(f"  Average comments per post: {avg_comments:.1f}\n")
            
            # COVID impact
            f.write("\n\nCOVID-19 IMPACT:\n")
            f.write("-" * 80 + "\n")
            
            covid_date = pd.Timestamp('2020-03-01')
            pre_covid = self.df[self.df['created_datetime'] < covid_date]
            post_covid = self.df[self.df['created_datetime'] >= covid_date]
            
            if len(pre_covid) > 0 and len(post_covid) > 0:
                pre_years = (covid_date - pre_covid['created_datetime'].min()).days / 365.25
                post_years = (self.df['created_datetime'].max() - covid_date).days / 365.25
                
                pre_rate = len(pre_covid) / pre_years
                post_rate = len(post_covid) / post_years
                
                f.write(f"  Pre-COVID posts: {len(pre_covid):,} ({pre_rate:.0f} per year)\n")
                f.write(f"  Post-COVID posts: {len(post_covid):,} ({post_rate:.0f} per year)\n")
                f.write(f"  Growth rate: {((post_rate/pre_rate - 1) * 100):+.1f}%\n")
        
        print(f"✓ Saved statistics to: {stats_file}")
    
    def perform_tfidf_analysis(self, output_dir='eda_output', max_features=200):
        """
        Perform TF-IDF analysis and create unigram and bigram word clouds
        
        Args:
            output_dir: Directory to save outputs
            max_features: Maximum number of features for TF-IDF
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nPerforming TF-IDF analysis...")
        
        # Prepare text data
        if 'text_no_stopwords' in self.df.columns:
            texts = self.df['text_no_stopwords'].fillna('').astype(str).tolist()
        else:
            texts = self.df['text_cleaned'].fillna('').astype(str).tolist()
        
        # Remove empty texts
        texts = [t for t in texts if len(t.strip()) > 0]
        
        # 1. UNIGRAM TF-IDF
        print("  Computing unigram TF-IDF...")
        vectorizer_unigram = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix_unigram = vectorizer_unigram.fit_transform(texts)
        feature_names_unigram = vectorizer_unigram.get_feature_names_out()
        
        # Get average TF-IDF scores
        avg_tfidf_unigram = tfidf_matrix_unigram.mean(axis=0).A1
        unigram_scores = dict(zip(feature_names_unigram, avg_tfidf_unigram))
        
        # Sort by score
        top_unigrams = dict(sorted(unigram_scores.items(), 
                                   key=lambda x: x[1], reverse=True)[:100])
        
        # 2. BIGRAM TF-IDF
        print("  Computing bigram TF-IDF...")
        vectorizer_bigram = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(2, 2),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix_bigram = vectorizer_bigram.fit_transform(texts)
        feature_names_bigram = vectorizer_bigram.get_feature_names_out()
        
        # Get average TF-IDF scores
        avg_tfidf_bigram = tfidf_matrix_bigram.mean(axis=0).A1
        bigram_scores = dict(zip(feature_names_bigram, avg_tfidf_bigram))
        
        # Sort by score
        top_bigrams = dict(sorted(bigram_scores.items(),
                                 key=lambda x: x[1], reverse=True)[:100])
        
        # 3. Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 3.1 Unigram Word Cloud
        wordcloud_unigram = WordCloud(
            width=800,
            height=600,
            background_color='white',
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(top_unigrams)
        
        axes[0, 0].imshow(wordcloud_unigram, interpolation='bilinear')
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Unigram Word Cloud (TF-IDF Weighted)',
                            fontsize=16, fontweight='bold', pad=15)
        
        # 3.2 Bigram Word Cloud
        wordcloud_bigram = WordCloud(
            width=800,
            height=600,
            background_color='white',
            colormap='plasma',
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(top_bigrams)
        
        axes[0, 1].imshow(wordcloud_bigram, interpolation='bilinear')
        axes[0, 1].axis('off')
        axes[0, 1].set_title('Bigram Word Cloud (TF-IDF Weighted)',
                            fontsize=16, fontweight='bold', pad=15)
        
        # 3.3 Top 20 Unigrams Bar Chart
        top_20_unigrams = dict(list(top_unigrams.items())[:20])
        words = list(top_20_unigrams.keys())
        scores = list(top_20_unigrams.values())
        
        bars = axes[1, 0].barh(range(len(words)), scores, color='#3498db', alpha=0.8)
        axes[1, 0].set_yticks(range(len(words)))
        axes[1, 0].set_yticklabels(words, fontsize=10)
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlabel('TF-IDF Score', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Top 20 Unigrams by TF-IDF',
                            fontsize=14, fontweight='bold', pad=10)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            axes[1, 0].text(score, i, f' {score:.4f}', va='center', fontsize=9)
        
        # 3.4 Top 20 Bigrams Bar Chart
        top_20_bigrams = dict(list(top_bigrams.items())[:20])
        words = list(top_20_bigrams.keys())
        scores = list(top_20_bigrams.values())
        
        bars = axes[1, 1].barh(range(len(words)), scores, color='#e67e22', alpha=0.8)
        axes[1, 1].set_yticks(range(len(words)))
        axes[1, 1].set_yticklabels(words, fontsize=10)
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_xlabel('TF-IDF Score', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Top 20 Bigrams by TF-IDF',
                            fontsize=14, fontweight='bold', pad=10)
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            axes[1, 1].text(score, i, f' {score:.4f}', va='center', fontsize=9)
        
        plt.suptitle('TF-IDF Analysis: Unigrams and Bigrams',
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = f'{output_dir}/tfidf_wordclouds.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
        
        # Save TF-IDF results to file
        self._save_tfidf_results(top_unigrams, top_bigrams, output_dir)
        
        return top_unigrams, top_bigrams
    
    def _save_tfidf_results(self, unigrams, bigrams, output_dir):
        """Save TF-IDF results to text file"""
        results_file = f'{output_dir}/tfidf_results.txt'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TF-IDF ANALYSIS RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write("WHAT IS TF-IDF?\n")
            f.write("-" * 80 + "\n")
            f.write("TF-IDF (Term Frequency-Inverse Document Frequency) identifies words that are:\n")
            f.write("  • Frequently used in specific posts/documents\n")
            f.write("  • Relatively rare across all posts/documents\n")
            f.write("  • More distinctive and meaningful than common words\n\n")
            f.write("Higher TF-IDF scores = More important/distinctive terms\n\n")
            
            f.write("TOP 50 UNIGRAMS (Single Words):\n")
            f.write("-" * 80 + "\n")
            for rank, (word, score) in enumerate(list(unigrams.items())[:50], 1):
                f.write(f"{rank:2d}. {word:25s} - {score:.6f}\n")
            
            f.write("\n\nTOP 50 BIGRAMS (Two-Word Phrases):\n")
            f.write("-" * 80 + "\n")
            for rank, (phrase, score) in enumerate(list(bigrams.items())[:50], 1):
                f.write(f"{rank:2d}. {phrase:35s} - {score:.6f}\n")
            
            f.write("\n\n" + "="*80 + "\n")
            f.write("KEY INSIGHTS:\n")
            f.write("="*80 + "\n\n")
            
            f.write("UNIGRAM INSIGHTS:\n")
            f.write("  • Most distinctive single words in the corpus\n")
            f.write("  • Reveals core medical terminology and concepts\n")
            f.write("  • Shows what makes these discussions unique\n\n")
            
            f.write("BIGRAM INSIGHTS:\n")
            f.write("  • Common two-word medical phrases\n")
            f.write("  • Reveals symptom combinations and medical conditions\n")
            f.write("  • Shows natural language patterns in discussions\n\n")
            
            # Analyze medical vs emotional terms
            medical_unigrams = [w for w in list(unigrams.keys())[:50] 
                              if any(term in w for term in ['blood', 'pressure', 'protein', 
                                                           'hospital', 'doctor', 'preeclampsia'])]
            
            f.write(f"Medical Terms in Top 50 Unigrams: {len(medical_unigrams)}\n")
            f.write(f"Examples: {', '.join(medical_unigrams[:10])}\n")
        
        print(f"✓ Saved TF-IDF results to: {results_file}")
    
    def run_complete_eda(self):
        """Run all EDA analyses"""
        print("\n" + "="*70)
        print(" "*20 + "EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        # Create temporal line plots
        self.create_temporal_line_plots()
        
        # Perform TF-IDF analysis with word clouds
        self.perform_tfidf_analysis()
        
        print("\n" + "="*70)
        print("✓ EDA COMPLETE!")
        print("="*70 + "\n")


def main():
    """Run EDA pipeline"""
    analyzer = EDAAnalyzer()
    analyzer.run_complete_eda()


if __name__ == "__main__":
    main()
