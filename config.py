# Configuration file for Reddit API credentials and Firebase settings

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Reddit API Credentials - Load from environment variables
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = 'css-preeclampsia-research/1.0 by /u/researcher_css'

    # Firebase Configuration
    FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH', 'firebase-service-account.json')
    FIREBASE_DATABASE_URL = os.getenv('FIREBASE_DATABASE_URL')

    # Firebase Web Config (for Pyrebase)
    FIREBASE_API_KEY = os.getenv('FIREBASE_API_KEY')
    FIREBASE_AUTH_DOMAIN = os.getenv('FIREBASE_AUTH_DOMAIN')
    FIREBASE_PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID')
    FIREBASE_STORAGE_BUCKET = os.getenv('FIREBASE_STORAGE_BUCKET')
    FIREBASE_MESSAGING_SENDER_ID = os.getenv('FIREBASE_MESSAGING_SENDER_ID')
    FIREBASE_APP_ID = os.getenv('FIREBASE_APP_ID')

    # Data Collection Settings
    SUBREDDITS = ['preeclampsia', 'BabyBumps', 'pregnant', 'PregnancyUK', 'pregnant_life', 
                  'Mommit', 'beyondthebump', 'pregnancy', 'pregnancysupport']
    KEYWORDS = ['pre-eclampsia', 'preeclampsia', 'pre eclampsia', 'PE', 'high blood pressure pregnancy',
                'pregnancy induced hypertension', 'PIH', 'toxemia', 'gestational hypertension']

    # Search parameters
    MAX_POSTS_PER_SUBREDDIT = 1000
    MAX_COMMENTS_PER_POST = 500
    TIME_FILTER = 'all'  # 'hour', 'day', 'week', 'month', 'year', 'all'

    # Text processing settings
    MIN_WORD_LENGTH = 3
    MAX_WORD_LENGTH = 50
    MIN_POST_LENGTH = 10  # minimum words in a post
    REMOVE_STOPWORDS = True

    # Topic modeling settings
    NUM_TOPICS = 8
    LDA_PASSES = 10
    LDA_ITERATIONS = 400
    LDA_ALPHA = 'auto'
    LDA_BETA = 'auto'
    RANDOM_STATE = 42

    # Sentiment analysis settings
    VADER_THRESHOLD_POSITIVE = 0.05
    VADER_THRESHOLD_NEGATIVE = -0.05

    # File paths (for local backup/cache)
    DATA_DIR = 'data'
    RAW_DATA_FILE = os.path.join(DATA_DIR, 'reddit_posts_raw.csv')
    PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'reddit_posts_processed.csv')
    FINAL_DATA_FILE = os.path.join(DATA_DIR, 'reddit_posts_analyzed.csv')

    # Output directories
    VISUALIZATIONS_DIR = 'visualizations'
    MODELS_DIR = 'models'
    REPORTS_DIR = 'reports'

    # Firebase batch settings
    FIREBASE_BATCH_SIZE = 500  # Number of records to save in each batch
    USE_LOCAL_BACKUP = True   # Whether to also save data locally as backup

    @classmethod
    def create_directories(cls):
        """Create necessary project directories"""
        directories = [cls.DATA_DIR, cls.VISUALIZATIONS_DIR, cls.MODELS_DIR, cls.REPORTS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

    @classmethod
    def get_firebase_config(cls):
        """Get Firebase configuration dictionary"""
        return {
            "apiKey": cls.FIREBASE_API_KEY,
            "authDomain": cls.FIREBASE_AUTH_DOMAIN,
            "databaseURL": cls.FIREBASE_DATABASE_URL,
            "projectId": cls.FIREBASE_PROJECT_ID,
            "storageBucket": cls.FIREBASE_STORAGE_BUCKET,
            "messagingSenderId": cls.FIREBASE_MESSAGING_SENDER_ID,
            "appId": cls.FIREBASE_APP_ID
        }
