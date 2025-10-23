# Sentiment Analysis using VADER
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        """Analyze sentiment of a single text"""
        if pd.isna(text) or not isinstance(text, str):
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'sentiment_label': 'neutral'
            }

        # Get VADER scores
        scores = self.analyzer.polarity_scores(text)

        # Determine sentiment label
        compound = scores['compound']
        if compound >= Config.VADER_THRESHOLD_POSITIVE:
            sentiment_label = 'positive'
        elif compound <= Config.VADER_THRESHOLD_NEGATIVE:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'

        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'sentiment_label': sentiment_label
        }

    def analyze_dataframe(self, df, text_column='full_text_clean'):
        """Analyze sentiment for entire dataframe"""
        logger.info("Starting sentiment analysis...")

        # Create copy
        result_df = df.copy()

        # Analyze sentiment for each text
        sentiment_results = df[text_column].apply(self.analyze_sentiment)

        # Extract sentiment components
        result_df['sentiment_compound'] = sentiment_results.apply(lambda x: x['compound'])
        result_df['sentiment_positive'] = sentiment_results.apply(lambda x: x['positive'])
        result_df['sentiment_negative'] = sentiment_results.apply(lambda x: x['negative'])
        result_df['sentiment_neutral'] = sentiment_results.apply(lambda x: x['neutral'])
        result_df['sentiment_label'] = sentiment_results.apply(lambda x: x['sentiment_label'])

        # Add sentiment intensity categories
        result_df['sentiment_intensity'] = pd.cut(
            result_df['sentiment_compound'].abs(),
            bins=[0, 0.1, 0.5, 1.0],
            labels=['low', 'medium', 'high']
        )

        logger.info(f"Sentiment analysis complete. Results:")
        sentiment_counts = result_df['sentiment_label'].value_counts()
        for sentiment, count in sentiment_counts.items():
            logger.info(f"{sentiment}: {count} ({count/len(result_df)*100:.1f}%)")

        return result_df

    def create_sentiment_visualizations(self, df):
        """Create various sentiment visualizations"""
        Config.create_directories()

        # 1. Sentiment distribution pie chart
        plt.figure(figsize=(15, 12))

        # Pie chart
        plt.subplot(2, 3, 1)
        sentiment_counts = df['sentiment_label'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Overall Sentiment Distribution')

        # 2. Sentiment by subreddit
        plt.subplot(2, 3, 2)
        sentiment_subreddit = pd.crosstab(df['subreddit'], df['sentiment_label'], normalize='index') * 100
        sentiment_subreddit.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Sentiment Distribution by Subreddit')
        plt.xlabel('Subreddit')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        plt.legend(title='Sentiment')

        # 3. Compound score distribution
        plt.subplot(2, 3, 3)
        plt.hist(df['sentiment_compound'], bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(Config.VADER_THRESHOLD_POSITIVE, color='green', linestyle='--', 
                   label=f'Positive threshold ({Config.VADER_THRESHOLD_POSITIVE})')
        plt.axvline(Config.VADER_THRESHOLD_NEGATIVE, color='red', linestyle='--', 
                   label=f'Negative threshold ({Config.VADER_THRESHOLD_NEGATIVE})')
        plt.xlabel('Compound Sentiment Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Compound Sentiment Scores')
        plt.legend()

        # 4. Sentiment over time
        if 'created_utc' in df.columns:
            plt.subplot(2, 3, 4)
            df['date'] = pd.to_datetime(df['created_utc']).dt.date
            daily_sentiment = df.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0)
            daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100

            if len(daily_sentiment_pct) > 1:
                daily_sentiment_pct.plot(ax=plt.gca())
                plt.title('Sentiment Trends Over Time')
                plt.xlabel('Date')
                plt.ylabel('Percentage')
                plt.xticks(rotation=45)

        # 5. Sentiment intensity by label
        plt.subplot(2, 3, 5)
        intensity_counts = pd.crosstab(df['sentiment_label'], df['sentiment_intensity'])
        intensity_counts.plot(kind='bar', ax=plt.gca())
        plt.title('Sentiment Intensity Distribution')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Count')
        plt.xticks(rotation=0)

        # 6. Score vs upvote ratio
        plt.subplot(2, 3, 6)
        if 'score' in df.columns and 'upvote_ratio' in df.columns:
            scatter_colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
            for sentiment in df['sentiment_label'].unique():
                sentiment_data = df[df['sentiment_label'] == sentiment]
                plt.scatter(sentiment_data['sentiment_compound'], 
                           sentiment_data['score'],
                           c=scatter_colors.get(sentiment, 'gray'),
                           alpha=0.6, label=sentiment)
            plt.xlabel('Sentiment Score')
            plt.ylabel('Reddit Score')
            plt.title('Sentiment vs Reddit Score')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'{Config.VISUALIZATIONS_DIR}/sentiment_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def create_word_clouds(self, df):
        """Create word clouds for different sentiments"""
        sentiments = df['sentiment_label'].unique()

        plt.figure(figsize=(15, 5))

        for i, sentiment in enumerate(sentiments, 1):
            plt.subplot(1, len(sentiments), i)

            # Get text for this sentiment
            sentiment_text = ' '.join(df[df['sentiment_label'] == sentiment]['text_for_analysis'].dropna())

            if sentiment_text:
                # Create word cloud
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    max_words=100,
                    relative_scaling=0.5,
                    colormap='viridis'
                ).generate(sentiment_text)

                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'{sentiment.title()} Sentiment Words')
                plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'{Config.VISUALIZATIONS_DIR}/sentiment_wordclouds.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_sentiment_by_topic(self, df):
        """Analyze sentiment distribution across topics"""
        if 'dominant_topic' not in df.columns:
            logger.warning("No topic information found. Run topic modeling first.")
            return

        # Create sentiment-topic crosstab
        topic_sentiment = pd.crosstab(df['dominant_topic'], df['sentiment_label'], normalize='index') * 100

        plt.figure(figsize=(12, 8))
        sns.heatmap(topic_sentiment, annot=True, fmt='.1f', cmap='RdYlBu_r')
        plt.title('Sentiment Distribution by Topic (%)')
        plt.xlabel('Sentiment')
        plt.ylabel('Topic')
        plt.tight_layout()
        plt.savefig(f'{Config.VISUALIZATIONS_DIR}/sentiment_by_topic.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

        return topic_sentiment

    def get_extreme_sentiments(self, df, n=10):
        """Get posts with most extreme positive and negative sentiments"""
        # Most positive
        most_positive = df.nlargest(n, 'sentiment_compound')[['title', 'sentiment_compound', 'sentiment_label']]

        # Most negative  
        most_negative = df.nsmallest(n, 'sentiment_compound')[['title', 'sentiment_compound', 'sentiment_label']]

        print("\n=== MOST POSITIVE POSTS ===")
        for i, row in most_positive.iterrows():
            print(f"Score: {row['sentiment_compound']:.3f} - {row['title'][:100]}...")

        print("\n=== MOST NEGATIVE POSTS ===")
        for i, row in most_negative.iterrows():
            print(f"Score: {row['sentiment_compound']:.3f} - {row['title'][:100]}...")

        return most_positive, most_negative

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv(Config.PROCESSED_DATA_FILE)

    # Initialize analyzer
    analyzer = SentimentAnalyzer()

    # Analyze sentiment
    df_with_sentiment = analyzer.analyze_sentiment(df)

    # Create visualizations
    analyzer.create_sentiment_visualizations(df_with_sentiment)
    analyzer.create_word_clouds(df_with_sentiment)

    # Show extreme sentiments
    analyzer.get_extreme_sentiments(df_with_sentiment)

    # Save results
    df_with_sentiment.to_csv(Config.FINAL_DATA_FILE, index=False)
