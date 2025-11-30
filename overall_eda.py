"""
Overall EDA Module
Comprehensive exploratory data analysis including sentiment distribution,
TF-IDF analysis, topic modeling, and medical terms analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import os
from collections import Counter
import re
from config import KEYWORDS_BY_LLM

class OverallEDA:
    def __init__(self, output_dir='analysis_output/overall'):
        """Initialize overall EDA analyzer"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        
        # Medical terms from config
        self.medical_terms = self._extract_medical_terms()
    
    def _extract_medical_terms(self):
        """Extract all medical terms from config"""
        terms = set()
        for llm, categories in KEYWORDS_BY_LLM.items():
            for category, keywords in categories.items():
                if category in ['core_terms', 'symptoms', 'monitoring', 'diagnostic']:
                    terms.update([k.lower() for k in keywords])
        return terms
    
    def analyze_overall_sentiment_distribution(self, posts_df, comments_df):
        """
        Analyze overall sentiment distribution for posts and comments
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
            comments_df (DataFrame): Comments with sentiment scores
        """
        print("\nAnalyzing overall sentiment distribution...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = {'positive': '#06D6A0', 'negative': '#EF476F', 'neutral': '#FFD166'}
        
        # Posts - sentiment categories
        if 'sentiment_category' in posts_df.columns and not posts_df.empty:
            posts_sentiment = posts_df['sentiment_category'].value_counts()
            plot_colors = [colors.get(cat, '#999999') for cat in posts_sentiment.index]
            
            axes[0, 0].pie(posts_sentiment.values, labels=posts_sentiment.index, autopct='%1.1f%%',
                          colors=plot_colors, startangle=90)
            axes[0, 0].set_title(f'Posts Sentiment Distribution\n(n={len(posts_df):,})', 
                               fontsize=12, fontweight='bold')
        
        # Comments - sentiment categories
        if 'sentiment_category' in comments_df.columns and not comments_df.empty:
            comments_sentiment = comments_df['sentiment_category'].value_counts()
            plot_colors = [colors.get(cat, '#999999') for cat in comments_sentiment.index]
            
            axes[0, 1].pie(comments_sentiment.values, labels=comments_sentiment.index, autopct='%1.1f%%',
                          colors=plot_colors, startangle=90)
            axes[0, 1].set_title(f'Comments Sentiment Distribution\n(n={len(comments_df):,})', 
                               fontsize=12, fontweight='bold')
        
        # Posts - compound score distribution
        if 'sentiment_compound' in posts_df.columns and not posts_df.empty:
            axes[1, 0].hist(posts_df['sentiment_compound'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_xlabel('Compound Score', fontsize=11)
            axes[1, 0].set_ylabel('Frequency', fontsize=11)
            axes[1, 0].set_title('Posts: Compound Score Distribution', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Comments - compound score distribution
        if 'sentiment_compound' in comments_df.columns and not comments_df.empty:
            axes[1, 1].hist(comments_df['sentiment_compound'], bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[1, 1].set_xlabel('Compound Score', fontsize=11)
            axes[1, 1].set_ylabel('Frequency', fontsize=11)
            axes[1, 1].set_title('Comments: Compound Score Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'overall_sentiment_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved overall sentiment distribution")
    
    def analyze_sentiment_by_subreddit(self, posts_df):
        """
        Analyze average sentiment by subreddit
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
            
        Returns:
            DataFrame: Sentiment statistics by subreddit
        """
        print("\nAnalyzing sentiment by subreddit...")
        
        if 'subreddit' not in posts_df.columns or 'sentiment_compound' not in posts_df.columns:
            print("Missing required columns")
            return None
        
        # Calculate stats by subreddit
        subreddit_sentiment = posts_df.groupby('subreddit').agg({
            'sentiment_compound': ['mean', 'median', 'std', 'count'],
            'sentiment_pos': 'mean',
            'sentiment_neg': 'mean',
            'id': 'count'
        }).round(3)
        
        subreddit_sentiment.columns = ['avg_sentiment', 'median_sentiment', 'std_sentiment', 
                                       'sentiment_count', 'avg_pos', 'avg_neg', 'post_count']
        subreddit_sentiment = subreddit_sentiment.reset_index()
        subreddit_sentiment = subreddit_sentiment.sort_values('avg_sentiment', ascending=False)
        
        # Visualizations
        fig, axes = plt.subplots(2, 1, figsize=(16, 14))
        
        # Top 20 subreddits by post count
        top_20 = subreddit_sentiment.nlargest(20, 'post_count')
        
        # Average sentiment
        colors = ['#06D6A0' if x >= 0 else '#EF476F' for x in top_20['avg_sentiment']]
        axes[0].barh(top_20['subreddit'], top_20['avg_sentiment'], color=colors, alpha=0.7)
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[0].set_xlabel('Average Sentiment Score', fontsize=12)
        axes[0].set_ylabel('Subreddit', fontsize=12)
        axes[0].set_title('Average Sentiment by Subreddit (Top 20 by Volume)', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Add post count labels
        for i, (idx, row) in enumerate(top_20.iterrows()):
            axes[0].text(row['avg_sentiment'], i, f"  n={row['post_count']}", 
                        va='center', fontsize=8)
        
        # Sentiment components by subreddit
        top_10 = subreddit_sentiment.nlargest(10, 'post_count')
        x = np.arange(len(top_10))
        width = 0.25
        
        axes[1].bar(x - width, top_10['avg_pos'], width, label='Positive', 
                   color='#06D6A0', alpha=0.7)
        axes[1].bar(x, top_10['avg_neg'], width, label='Negative', 
                   color='#EF476F', alpha=0.7)
        axes[1].bar(x + width, top_10['avg_sentiment'], width, label='Compound', 
                   color='#2E86AB', alpha=0.7)
        
        axes[1].set_xlabel('Subreddit', fontsize=12)
        axes[1].set_ylabel('Average Score', fontsize=12)
        axes[1].set_title('Sentiment Components by Subreddit (Top 10 by Volume)', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(top_10['subreddit'], rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment_by_subreddit.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved sentiment by subreddit analysis")
        
        # Save to CSV
        subreddit_sentiment.to_csv(
            os.path.join(self.output_dir, 'sentiment_by_subreddit.csv'), 
            index=False
        )
        
        return subreddit_sentiment
    
    def perform_tfidf_analysis(self, posts_df, text_field='full_text', max_features=100):
        """
        Perform TF-IDF analysis
        
        Args:
            posts_df (DataFrame): Posts with text
            text_field (str): Text field to analyze
            max_features (int): Maximum number of features
            
        Returns:
            DataFrame: Top TF-IDF terms
        """
        print("\nPerforming TF-IDF analysis...")
        
        if text_field not in posts_df.columns:
            print(f"Missing {text_field} column")
            return None
        
        # Prepare corpus
        corpus = posts_df[text_field].dropna().astype(str).tolist()
        
        if len(corpus) == 0:
            print("No text data available")
            return None
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores
        avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        tfidf_df = pd.DataFrame({
            'term': feature_names,
            'tfidf_score': avg_tfidf
        }).sort_values('tfidf_score', ascending=False)
        
        # Visualization
        top_30 = tfidf_df.head(30)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.barh(range(len(top_30)), top_30['tfidf_score'], color='#2E86AB', alpha=0.7)
        ax.set_yticks(range(len(top_30)))
        ax.set_yticklabels(top_30['term'])
        ax.invert_yaxis()
        ax.set_xlabel('Average TF-IDF Score', fontsize=12)
        ax.set_ylabel('Term', fontsize=12)
        ax.set_title('Top 30 Terms by TF-IDF Score', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tfidf_top_terms.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved TF-IDF analysis")
        
        # Save to CSV
        tfidf_df.to_csv(os.path.join(self.output_dir, 'tfidf_scores.csv'), index=False)
        
        return tfidf_df
    
    def perform_topic_modeling(self, posts_df, text_field='full_text', n_topics=5, n_words=10):
        """
        Perform topic modeling using LDA
        
        Args:
            posts_df (DataFrame): Posts with text
            text_field (str): Text field to analyze
            n_topics (int): Number of topics
            n_words (int): Number of words per topic
            
        Returns:
            dict: Topic modeling results
        """
        print(f"\nPerforming topic modeling ({n_topics} topics)...")
        
        if text_field not in posts_df.columns:
            print(f"Missing {text_field} column")
            return None
        
        # Prepare corpus
        corpus = posts_df[text_field].dropna().astype(str).tolist()
        
        if len(corpus) < 10:
            print("Insufficient text data for topic modeling")
            return None
        
        # Vectorization
        vectorizer = CountVectorizer(
            max_features=1000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        doc_term_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        
        # LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=50
        )
        
        lda.fit(doc_term_matrix)
        
        # Extract topics
        topics = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[f'Topic {topic_idx + 1}'] = top_words
        
        # Visualization
        fig, ax = plt.subplots(figsize=(14, n_topics * 1.5))
        
        y_pos = np.arange(n_topics)
        topic_labels = []
        
        for i, (topic_name, words) in enumerate(topics.items()):
            # Create topic label with top 5 words
            label = f"{topic_name}: {', '.join(words[:5])}"
            topic_labels.append(label)
        
        # Get document-topic distribution
        doc_topics = lda.transform(doc_term_matrix)
        topic_weights = doc_topics.mean(axis=0)
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_topics))
        ax.barh(y_pos, topic_weights, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(topic_labels, fontsize=10)
        ax.set_xlabel('Average Topic Weight', fontsize=12)
        ax.set_title('Topic Modeling Results (LDA)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'topic_modeling.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved topic modeling results")
        
        # Print topics
        print("\nDiscovered Topics:")
        for topic_name, words in topics.items():
            print(f"{topic_name}: {', '.join(words)}")
        
        return {
            'topics': topics,
            'model': lda,
            'vectorizer': vectorizer
        }
    
    def compare_subreddits_tfidf(self, posts_df, subreddits=None, text_field='full_text', n_terms=20):
        """
        Compare TF-IDF terms across different subreddits
        
        Args:
            posts_df (DataFrame): Posts with text
            subreddits (list): List of subreddits to compare (None for top 5)
            text_field (str): Text field to analyze
            n_terms (int): Number of top terms per subreddit
        """
        print("\nComparing subreddits using TF-IDF...")
        
        if 'subreddit' not in posts_df.columns or text_field not in posts_df.columns:
            print("Missing required columns")
            return
        
        # Select subreddits
        if subreddits is None:
            subreddits = posts_df['subreddit'].value_counts().head(5).index.tolist()
        
        # Analyze each subreddit
        fig, axes = plt.subplots(len(subreddits), 1, figsize=(14, 5*len(subreddits)))
        if len(subreddits) == 1:
            axes = [axes]
        
        for idx, subreddit in enumerate(subreddits):
            sub_posts = posts_df[posts_df['subreddit'] == subreddit]
            corpus = sub_posts[text_field].dropna().astype(str).tolist()
            
            if len(corpus) == 0:
                continue
            
            vectorizer = TfidfVectorizer(
                max_features=100,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            
            tfidf_df = pd.DataFrame({
                'term': feature_names,
                'score': avg_tfidf
            }).sort_values('score', ascending=False).head(n_terms)
            
            axes[idx].barh(range(len(tfidf_df)), tfidf_df['score'], 
                          color=plt.cm.Set3(idx), alpha=0.7)
            axes[idx].set_yticks(range(len(tfidf_df)))
            axes[idx].set_yticklabels(tfidf_df['term'])
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('TF-IDF Score', fontsize=11)
            axes[idx].set_title(f'r/{subreddit} - Top {n_terms} Terms (n={len(corpus):,})', 
                              fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'subreddit_tfidf_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved subreddit TF-IDF comparison")
    
    def analyze_medical_terms(self, posts_df, text_field='full_text'):
        """
        Analyze usage of medical terms from config
        
        Args:
            posts_df (DataFrame): Posts with text
            text_field (str): Text field to analyze
        """
        print("\nAnalyzing medical terms usage...")
        
        if text_field not in posts_df.columns:
            print(f"Missing {text_field} column")
            return
        
        # Count medical terms
        term_counts = Counter()
        
        for text in posts_df[text_field].dropna():
            text_lower = str(text).lower()
            for term in self.medical_terms:
                if term in text_lower:
                    term_counts[term] += 1
        
        if not term_counts:
            print("No medical terms found")
            return
        
        # Create DataFrame
        medical_df = pd.DataFrame(
            term_counts.most_common(50),
            columns=['term', 'count']
        )
        
        # Visualizations
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Bar chart of top 30 terms
        top_30 = medical_df.head(30)
        axes[0].barh(range(len(top_30)), top_30['count'], color='#118AB2', alpha=0.7)
        axes[0].set_yticks(range(len(top_30)))
        axes[0].set_yticklabels(top_30['term'])
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Frequency', fontsize=12)
        axes[0].set_title('Top 30 Medical Terms by Frequency', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Word cloud of medical terms
        medical_text = ' '.join([f"{term} " * count for term, count in term_counts.items()])
        
        if len(medical_text.strip()) > 0:
            wordcloud = WordCloud(
                width=800, height=600,
                background_color='white',
                colormap='RdPu',
                max_words=100,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(medical_text)
            
            axes[1].imshow(wordcloud, interpolation='bilinear')
            axes[1].set_title('Medical Terms Word Cloud', fontsize=14, fontweight='bold')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'medical_terms_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved medical terms analysis")
        
        # Save to CSV
        medical_df.to_csv(os.path.join(self.output_dir, 'medical_terms_frequency.csv'), 
                         index=False)
    
    def run_full_analysis(self, posts_df, comments_df):
        """
        Run complete overall EDA
        
        Args:
            posts_df (DataFrame): Posts with sentiment scores
            comments_df (DataFrame): Comments with sentiment scores
            
        Returns:
            dict: Analysis results
        """
        print("\n" + "="*60)
        print("OVERALL EDA")
        print("="*60)
        
        results = {}
        
        # Run all analyses
        self.analyze_overall_sentiment_distribution(posts_df, comments_df)
        results['subreddit_sentiment'] = self.analyze_sentiment_by_subreddit(posts_df)
        results['tfidf'] = self.perform_tfidf_analysis(posts_df)
        results['topics'] = self.perform_topic_modeling(posts_df, n_topics=5)
        self.compare_subreddits_tfidf(posts_df)
        self.analyze_medical_terms(posts_df)
        
        print(f"\nAll overall EDA analyses saved to: {self.output_dir}")
        
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
    
    # Run overall EDA
    eda = OverallEDA()
    results = eda.run_full_analysis(posts_df, comments_df)
    
    print("\nOverall EDA completed successfully!")
