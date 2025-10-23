# Main analysis script that combines all components with Firebase
import pandas as pd
import numpy as np
import warnings
import datetime
warnings.filterwarnings('ignore')

from config import Config
from data_collection import RedditDataCollector  
from text_preprocessing import TextPreprocessor
from topic_modeling import TopicModeler
from sentiment_analysis import SentimentAnalyzer
from firebase_config import FirebaseConfig, FirebaseDataManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreeclampsiaAnalysis:
    def __init__(self):
        self.collector = RedditDataCollector()
        self.preprocessor = TextPreprocessor()
        self.topic_modeler = TopicModeler()
        self.sentiment_analyzer = SentimentAnalyzer()

        # Initialize Firebase
        try:
            self.firebase_config = FirebaseConfig()
            self.firebase_config.initialize_firebase(
                service_account_path=Config.FIREBASE_SERVICE_ACCOUNT_PATH,
                database_url=Config.FIREBASE_DATABASE_URL
            )
            self.firebase_manager = FirebaseDataManager(self.firebase_config)
            logger.info("Firebase initialized successfully")
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            self.firebase_manager = None

        # Create local directories for backup
        Config.create_directories()

    def run_complete_analysis(self, skip_collection=False, use_firebase_data=True):
        """Run the complete analysis pipeline with Firebase integration"""
        logger.info("=== STARTING PRE-ECLAMPSIA REDDIT ANALYSIS ===")

        # Step 1: Data Collection/Loading
        if not skip_collection:
            logger.info("\n1. COLLECTING DATA FROM REDDIT...")
            raw_df = self.collector.collect_data()
        else:
            if use_firebase_data and self.firebase_manager:
                logger.info("\n1. LOADING DATA FROM FIREBASE...")
                raw_df = self.collector.load_data_from_firebase()
                if raw_df is None or raw_df.empty:
                    logger.warning("No data in Firebase, trying local backup...")
                    raw_df = pd.read_csv(Config.RAW_DATA_FILE)
            else:
                logger.info("\n1. LOADING DATA FROM LOCAL FILE...")
                raw_df = pd.read_csv(Config.RAW_DATA_FILE)

        if raw_df.empty:
            logger.error("No data available for analysis!")
            return None, None

        logger.info(f"Raw data: {len(raw_df)} posts")

        # Step 2: Text Preprocessing  
        logger.info("\n2. PREPROCESSING TEXT DATA...")

        # Check if processed data exists in Firebase
        processed_df = None
        if self.firebase_manager:
            try:
                processed_df = self.firebase_manager.load_processed_data('processed')
                if not processed_df.empty:
                    logger.info(f"Loaded existing processed data from Firebase: {len(processed_df)} posts")
            except Exception as e:
                logger.warning(f"Could not load processed data from Firebase: {e}")

        if processed_df is None or processed_df.empty:
            # Process the data
            processed_df = self.preprocessor.preprocess_dataframe(raw_df)

            # Save to Firebase
            if self.firebase_manager:
                try:
                    self.firebase_manager.save_processed_data(processed_df, 'processed')
                    logger.info("Processed data saved to Firebase")
                except Exception as e:
                    logger.error(f"Failed to save processed data to Firebase: {e}")

        logger.info(f"Processed data: {len(processed_df)} posts")

        # Step 3: Topic Modeling
        logger.info("\n3. PERFORMING TOPIC MODELING...")

        # Check if topic analysis exists
        topic_results = None
        if self.firebase_manager:
            try:
                topic_results = self.firebase_manager.load_analysis_results('topic_modeling')
            except Exception as e:
                logger.warning(f"Could not load topic results from Firebase: {e}")

        if not topic_results:
            # Prepare text data
            if isinstance(processed_df['tokens_lemmatized'].iloc[0], str):
                texts = processed_df['tokens_lemmatized'].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                ).tolist()
            else:
                texts = processed_df['tokens_lemmatized'].tolist()

            # Train LDA model
            self.topic_modeler.train_lda_model(texts, num_topics=Config.NUM_TOPICS)

            # Get topics for each document
            doc_topics = self.topic_modeler.get_document_topics(texts)
            topic_df = pd.DataFrame(doc_topics)

            # Merge with processed data
            processed_df = pd.concat([processed_df.reset_index(drop=True), 
                                    topic_df.reset_index(drop=True)], axis=1)

            # Save topic results to Firebase
            if self.firebase_manager:
                try:
                    topic_labels = self.topic_modeler.print_topics()
                    coherence_score = self.topic_modeler.coherence_model.get_coherence() if self.topic_modeler.coherence_model else None

                    topic_results = {
                        'topic_labels': topic_labels,
                        'coherence_score': coherence_score,
                        'num_topics': Config.NUM_TOPICS,
                        'model_parameters': {
                            'passes': Config.LDA_PASSES,
                            'iterations': Config.LDA_ITERATIONS,
                            'alpha': Config.LDA_ALPHA,
                            'beta': Config.LDA_BETA
                        }
                    }

                    self.firebase_manager.save_analysis_results(topic_results, 'topic_modeling')
                    logger.info("Topic modeling results saved to Firebase")
                except Exception as e:
                    logger.error(f"Failed to save topic results to Firebase: {e}")

        # Print discovered topics
        if topic_results and 'topic_labels' in topic_results:
            logger.info("\nDiscovered Topics:")
            for i, label in enumerate(topic_results['topic_labels']):
                logger.info(f"Topic {i}: {label}")

        # Create topic visualization
        try:
            self.topic_modeler.visualize_topics()
        except Exception as e:
            logger.warning(f"Could not create topic visualization: {e}")

        # Step 4: Sentiment Analysis
        logger.info("\n4. PERFORMING SENTIMENT ANALYSIS...")

        # Check if sentiment analysis exists
        sentiment_results = None
        if self.firebase_manager:
            try:
                sentiment_results = self.firebase_manager.load_analysis_results('sentiment_analysis')
            except Exception as e:
                logger.warning(f"Could not load sentiment results from Firebase: {e}")

        if not sentiment_results:
            # Perform sentiment analysis
            final_df = self.sentiment_analyzer.analyze_dataframe(processed_df)

            # Save sentiment results to Firebase
            if self.firebase_manager:
                try:
                    sentiment_stats = {
                        'total_posts': len(final_df),
                        'sentiment_distribution': final_df['sentiment_label'].value_counts().to_dict(),
                        'average_sentiment': float(final_df['sentiment_compound'].mean()),
                        'sentiment_by_subreddit': final_df.groupby('subreddit')['sentiment_compound'].mean().to_dict()
                    }

                    self.firebase_manager.save_analysis_results(sentiment_stats, 'sentiment_analysis')
                    self.firebase_manager.save_processed_data(final_df, 'analyzed')
                    logger.info("Sentiment analysis results saved to Firebase")
                except Exception as e:
                    logger.error(f"Failed to save sentiment results to Firebase: {e}")
        else:
            # Load analyzed data
            final_df = self.firebase_manager.load_processed_data('analyzed')
            if final_df.empty:
                final_df = self.sentiment_analyzer.analyze_dataframe(processed_df)

        # Create sentiment visualizations
        try:
            self.sentiment_analyzer.create_sentiment_visualizations(final_df)
            self.sentiment_analyzer.create_word_clouds(final_df)
            self.sentiment_analyzer.analyze_sentiment_by_topic(final_df)
        except Exception as e:
            logger.warning(f"Could not create some visualizations: {e}")

        # Step 5: Generate insights
        logger.info("\n5. GENERATING INSIGHTS...")
        insights = self.generate_insights(final_df)

        # Save insights to Firebase
        if self.firebase_manager:
            try:
                self.firebase_manager.save_analysis_results(insights, 'insights')
                logger.info("Insights saved to Firebase")
            except Exception as e:
                logger.error(f"Failed to save insights to Firebase: {e}")

        # Save final results locally as backup
        if Config.USE_LOCAL_BACKUP:
            final_df.to_csv(Config.FINAL_DATA_FILE, index=False)
            logger.info(f"Local backup saved to {Config.FINAL_DATA_FILE}")

        logger.info("\nAnalysis complete!")
        return final_df, insights

    def generate_insights(self, df):
        """Generate key insights from the analysis"""
        insights = {}

        # Basic statistics
        insights['total_posts'] = len(df)
        insights['unique_subreddits'] = df['subreddit'].nunique()
        insights['date_range'] = {
            'start': str(df['created_utc'].min()) if 'created_utc' in df.columns else None,
            'end': str(df['created_utc'].max()) if 'created_utc' in df.columns else None
        }

        # Topic insights
        topic_cols = [col for col in df.columns if col.startswith('topic_')]
        if topic_cols:
            insights['num_topics'] = len(topic_cols)
            if 'dominant_topic' in df.columns:
                insights['dominant_topics'] = df['dominant_topic'].value_counts().head().to_dict()

        # Sentiment insights
        if 'sentiment_label' in df.columns:
            insights['sentiment_distribution'] = df['sentiment_label'].value_counts().to_dict()
            insights['average_sentiment'] = float(df['sentiment_compound'].mean()) if 'sentiment_compound' in df.columns else None

            if 'subreddit' in df.columns:
                insights['sentiment_by_subreddit'] = df.groupby('subreddit')['sentiment_compound'].mean().to_dict()

        # Medical term insights
        if 'medical_terms' in df.columns:
            all_symptoms = []
            all_treatments = []
            all_emotions = []

            for terms in df['medical_terms'].dropna():
                if isinstance(terms, str):
                    try:
                        terms = eval(terms)
                    except:
                        continue
                if isinstance(terms, dict):
                    all_symptoms.extend(terms.get('symptoms', []))
                    all_treatments.extend(terms.get('treatments', []))
                    all_emotions.extend(terms.get('emotions', []))

            if all_symptoms:
                insights['common_symptoms'] = pd.Series(all_symptoms).value_counts().head().to_dict()
            if all_treatments:
                insights['common_treatments'] = pd.Series(all_treatments).value_counts().head().to_dict()
            if all_emotions:
                insights['common_emotions'] = pd.Series(all_emotions).value_counts().head().to_dict()

        # Print insights
        self.print_insights(insights)

        return insights

    def print_insights(self, insights):
        """Print key insights to console"""
        print("\n=== KEY INSIGHTS ===")
        print(f"Total posts analyzed: {insights['total_posts']}")
        print(f"Subreddits covered: {insights['unique_subreddits']}")
        print(f"Topics discovered: {insights.get('num_topics', 'N/A')}")

        if 'average_sentiment' in insights and insights['average_sentiment'] is not None:
            print(f"Overall sentiment: {insights['average_sentiment']:.3f}")

        if 'sentiment_distribution' in insights:
            print("\nSentiment distribution:")
            for sentiment, count in insights['sentiment_distribution'].items():
                pct = count / insights['total_posts'] * 100
                print(f"  {sentiment}: {count} ({pct:.1f}%)")

        if 'common_symptoms' in insights:
            print("\nMost mentioned symptoms:")
            for symptom, count in list(insights['common_symptoms'].items())[:5]:
                print(f"  {symptom}: {count}")

        if 'common_emotions' in insights:
            print("\nMost expressed emotions:")
            for emotion, count in list(insights['common_emotions'].items())[:5]:
                print(f"  {emotion}: {count}")

    def get_team_dashboard_data(self):
        """Get data for team dashboard from Firebase"""
        if not self.firebase_manager:
            logger.error("Firebase not available")
            return None

        try:
            # Get database stats
            stats = self.firebase_manager.get_database_stats()

            # Get latest insights
            insights = self.firebase_manager.load_analysis_results('insights')

            # Get topic modeling results
            topic_results = self.firebase_manager.load_analysis_results('topic_modeling')

            # Get sentiment analysis results
            sentiment_results = self.firebase_manager.load_analysis_results('sentiment_analysis')

            dashboard_data = {
                'database_stats': stats,
                'insights': insights,
                'topic_results': topic_results,
                'sentiment_results': sentiment_results,
                'last_updated': datetime.now().isoformat()
            }

            return dashboard_data

        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return None

if __name__ == "__main__":
    # Initialize analysis
    analysis = PreeclampsiaAnalysis()

    # Run complete analysis
    # Set skip_collection=True if you already have data
    # Set use_firebase_data=True to load existing data from Firebase
    final_df, insights = analysis.run_complete_analysis(
        skip_collection=False,
        use_firebase_data=True
    )

    if final_df is not None:
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print("\nData is stored in Firebase and available to your team!")
        print("\nCheck the following for outputs:")
        print(f"- Local visualizations: {Config.VISUALIZATIONS_DIR}")
        print(f"- Local models: {Config.MODELS_DIR}")  
        print(f"- Local reports: {Config.REPORTS_DIR}")
        print("- Firebase Console: https://console.firebase.google.com/")

        # Show database stats
        stats = analysis.firebase_manager.get_database_stats() if analysis.firebase_manager else {}
        if stats:
            print(f"\nFirebase Database Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    else:
        print("Analysis failed - please check the logs for errors")
