# Firebase Configuration and Database Management
import firebase_admin
from firebase_admin import credentials, db
import pyrebase
import json
import os
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirebaseConfig:
    def __init__(self):
        self.app = None
        self.pyrebase_app = None
        self.db_ref = None

    def initialize_firebase(self, service_account_path=None, database_url=None):
        """
        Initialize Firebase Admin SDK and Pyrebase

        Args:
            service_account_path (str): Path to Firebase service account JSON file
            database_url (str): Firebase Realtime Database URL
        """
        try:
            # Initialize Firebase Admin SDK
            if service_account_path and os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
                self.app = firebase_admin.initialize_app(cred, {
                    'databaseURL': database_url
                })
            else:
                # Use environment variables or default credentials
                cred = credentials.ApplicationDefault()
                self.app = firebase_admin.initialize_app(cred, {
                    'databaseURL': database_url
                })

            # Initialize Pyrebase for easier data operations
            config = {
                "apiKey": os.getenv('FIREBASE_API_KEY'),
                "authDomain": os.getenv('FIREBASE_AUTH_DOMAIN'),
                "databaseURL": database_url,
                "projectId": os.getenv('FIREBASE_PROJECT_ID'),
                "storageBucket": os.getenv('FIREBASE_STORAGE_BUCKET'),
                "messagingSenderId": os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
                "appId": os.getenv('FIREBASE_APP_ID')
            }

            # Only initialize Pyrebase if we have the required config
            if all(config.values()):
                self.pyrebase_app = pyrebase.initialize_app(config)

            self.db_ref = db.reference()
            logger.info("Firebase initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise

    def test_connection(self):
        """Test Firebase connection"""
        try:
            # Try to write and read a test value
            test_ref = self.db_ref.child('test')
            test_ref.set({'timestamp': datetime.now().isoformat()})
            result = test_ref.get()
            test_ref.delete()  # Clean up
            logger.info("Firebase connection test successful")
            return True
        except Exception as e:
            logger.error(f"Firebase connection test failed: {e}")
            return False

class FirebaseDataManager:
    def __init__(self, firebase_config):
        self.config = firebase_config
        self.db_ref = firebase_config.db_ref

    def save_posts(self, posts_df, batch_size=500):
        """
        Save posts to Firebase in batches

        Args:
            posts_df (pd.DataFrame): DataFrame containing posts
            batch_size (int): Number of posts to save in each batch
        """
        logger.info(f"Saving {len(posts_df)} posts to Firebase...")

        try:
            posts_ref = self.db_ref.child('posts')

            # Convert DataFrame to dictionary format
            posts_dict = {}

            for idx, row in posts_df.iterrows():
                post_id = row['post_id']

                # Convert timestamps to string format
                post_data = row.to_dict()
                if 'created_utc' in post_data and pd.notnull(post_data['created_utc']):
                    if isinstance(post_data['created_utc'], datetime):
                        post_data['created_utc'] = post_data['created_utc'].isoformat()

                # Handle list/dict fields
                if 'comments' in post_data and isinstance(post_data['comments'], list):
                    # Process comments separately
                    comments_data = {}
                    for i, comment in enumerate(post_data['comments']):
                        if isinstance(comment, dict):
                            comment_copy = comment.copy()
                            if 'comment_created_utc' in comment_copy:
                                if isinstance(comment_copy['comment_created_utc'], datetime):
                                    comment_copy['comment_created_utc'] = comment_copy['comment_created_utc'].isoformat()
                            comments_data[f"comment_{i}"] = comment_copy
                    post_data['comments'] = comments_data

                # Clean NaN values
                post_data = self._clean_data(post_data)
                posts_dict[post_id] = post_data

                # Save in batches to avoid memory issues
                if len(posts_dict) >= batch_size:
                    posts_ref.update(posts_dict)
                    logger.info(f"Saved batch of {len(posts_dict)} posts")
                    posts_dict = {}

            # Save remaining posts
            if posts_dict:
                posts_ref.update(posts_dict)
                logger.info(f"Saved final batch of {len(posts_dict)} posts")

            logger.info("All posts saved successfully to Firebase")

        except Exception as e:
            logger.error(f"Error saving posts to Firebase: {e}")
            raise

    def save_processed_data(self, df, data_type='processed'):
        """
        Save processed analysis data

        Args:
            df (pd.DataFrame): DataFrame to save
            data_type (str): Type of data (processed, analyzed, etc.)
        """
        logger.info(f"Saving {data_type} data to Firebase...")

        try:
            data_ref = self.db_ref.child(f'{data_type}_data')

            # Convert DataFrame to dictionary
            data_dict = {}
            for idx, row in df.iterrows():
                row_data = row.to_dict()

                # Handle datetime objects
                for key, value in row_data.items():
                    if isinstance(value, datetime):
                        row_data[key] = value.isoformat()
                    elif pd.isna(value):
                        row_data[key] = None

                # Use post_id as key if available, otherwise use index
                key = row_data.get('post_id', f'record_{idx}')
                data_dict[key] = self._clean_data(row_data)

            # Save to Firebase
            data_ref.set(data_dict)
            logger.info(f"Saved {len(data_dict)} records of {data_type} data")

        except Exception as e:
            logger.error(f"Error saving {data_type} data: {e}")
            raise

    def save_analysis_results(self, results_dict, analysis_type='sentiment'):
        """
        Save analysis results (topic modeling, sentiment analysis, etc.)

        Args:
            results_dict (dict): Dictionary containing analysis results
            analysis_type (str): Type of analysis
        """
        try:
            results_ref = self.db_ref.child('analysis_results').child(analysis_type)

            # Add timestamp
            results_dict['timestamp'] = datetime.now().isoformat()
            results_dict['analysis_type'] = analysis_type

            # Clean the data
            clean_results = self._clean_data(results_dict)

            results_ref.set(clean_results)
            logger.info(f"Saved {analysis_type} analysis results to Firebase")

        except Exception as e:
            logger.error(f"Error saving {analysis_type} results: {e}")
            raise

    def load_posts(self, limit=None):
        """
        Load posts from Firebase

        Args:
            limit (int): Maximum number of posts to load

        Returns:
            pd.DataFrame: DataFrame containing posts
        """
        try:
            logger.info("Loading posts from Firebase...")
            posts_ref = self.db_ref.child('posts')

            if limit:
                posts_data = posts_ref.limit_to_first(limit).get()
            else:
                posts_data = posts_ref.get()

            if not posts_data:
                logger.warning("No posts found in Firebase")
                return pd.DataFrame()

            # Convert to DataFrame
            posts_list = []
            for post_id, post_data in posts_data.items():
                post_data['post_id'] = post_id

                # Convert timestamp back to datetime
                if 'created_utc' in post_data and post_data['created_utc']:
                    try:
                        post_data['created_utc'] = pd.to_datetime(post_data['created_utc'])
                    except:
                        pass

                # Handle comments
                if 'comments' in post_data and isinstance(post_data['comments'], dict):
                    comments_list = []
                    for comment_key, comment_data in post_data['comments'].items():
                        if 'comment_created_utc' in comment_data:
                            try:
                                comment_data['comment_created_utc'] = pd.to_datetime(comment_data['comment_created_utc'])
                            except:
                                pass
                        comments_list.append(comment_data)
                    post_data['comments'] = comments_list

                posts_list.append(post_data)

            df = pd.DataFrame(posts_list)
            logger.info(f"Loaded {len(df)} posts from Firebase")
            return df

        except Exception as e:
            logger.error(f"Error loading posts from Firebase: {e}")
            return pd.DataFrame()

    def load_processed_data(self, data_type='processed'):
        """Load processed data from Firebase"""
        try:
            logger.info(f"Loading {data_type} data from Firebase...")
            data_ref = self.db_ref.child(f'{data_type}_data')
            data = data_ref.get()

            if not data:
                logger.warning(f"No {data_type} data found in Firebase")
                return pd.DataFrame()

            # Convert to DataFrame
            data_list = []
            for key, row_data in data.items():
                # Convert timestamps back
                for field_key, value in row_data.items():
                    if isinstance(value, str) and ('utc' in field_key or 'date' in field_key):
                        try:
                            row_data[field_key] = pd.to_datetime(value)
                        except:
                            pass

                data_list.append(row_data)

            df = pd.DataFrame(data_list)
            logger.info(f"Loaded {len(df)} records of {data_type} data")
            return df

        except Exception as e:
            logger.error(f"Error loading {data_type} data: {e}")
            return pd.DataFrame()

    def load_analysis_results(self, analysis_type='sentiment'):
        """Load analysis results from Firebase"""
        try:
            results_ref = self.db_ref.child('analysis_results').child(analysis_type)
            results = results_ref.get()

            if results:
                logger.info(f"Loaded {analysis_type} analysis results from Firebase")
                return results
            else:
                logger.warning(f"No {analysis_type} analysis results found")
                return {}

        except Exception as e:
            logger.error(f"Error loading {analysis_type} results: {e}")
            return {}

    def get_database_stats(self):
        """Get statistics about data in Firebase"""
        try:
            stats = {}

            # Count posts
            posts_ref = self.db_ref.child('posts')
            posts_data = posts_ref.shallow().get()
            stats['total_posts'] = len(posts_data) if posts_data else 0

            # Count processed data
            processed_ref = self.db_ref.child('processed_data')
            processed_data = processed_ref.shallow().get()
            stats['processed_records'] = len(processed_data) if processed_data else 0

            # Count analysis results
            results_ref = self.db_ref.child('analysis_results')
            results_data = results_ref.shallow().get()
            stats['analysis_types'] = list(results_data.keys()) if results_data else []

            logger.info(f"Database stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

    def clear_data(self, data_type=None):
        """
        Clear data from Firebase (use with caution!)

        Args:
            data_type (str): Specific data type to clear, or None to clear all
        """
        try:
            if data_type:
                self.db_ref.child(data_type).delete()
                logger.info(f"Cleared {data_type} data from Firebase")
            else:
                # Clear all project data
                self.db_ref.delete()
                logger.info("Cleared all data from Firebase")

        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            raise

    def _clean_data(self, data):
        """Clean data for Firebase storage"""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if pd.isna(value):
                    cleaned[key] = None
                elif isinstance(value, (int, float, str, bool)):
                    cleaned[key] = value
                elif isinstance(value, list):
                    cleaned[key] = [self._clean_data(item) for item in value]
                elif isinstance(value, dict):
                    cleaned[key] = self._clean_data(value)
                else:
                    cleaned[key] = str(value)
            return cleaned
        elif isinstance(data, list):
            return [self._clean_data(item) for item in data]
        elif pd.isna(data):
            return None
        else:
            return data

# Example Firebase configuration
FIREBASE_CONFIG_TEMPLATE = {
    "service_account_path": "path/to/your/firebase-service-account.json",
    "database_url": "https://your-project-default-rtdb.firebaseio.com/",
    "api_key": "your-api-key",
    "auth_domain": "your-project.firebaseapp.com",
    "project_id": "your-project-id",
    "storage_bucket": "your-project.appspot.com",
    "messaging_sender_id": "123456789",
    "app_id": "1:123456789:web:abcdef123456"
}
