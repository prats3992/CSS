# Reddit Data Collection Module with Firebase Integration
import praw
import pandas as pd
import time
from datetime import datetime, timedelta
import re
import logging
from config import Config
from firebase_config import FirebaseConfig, FirebaseDataManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedditDataCollector:
    def __init__(self):
        """Initialize Reddit API connection and Firebase"""
        # Initialize Reddit API
        self.reddit = praw.Reddit(
            client_id=Config.REDDIT_CLIENT_ID,
            client_secret=Config.REDDIT_CLIENT_SECRET,
            user_agent=Config.REDDIT_USER_AGENT
        )

        # Test Reddit connection
        try:
            logger.info(f"Connected to Reddit as: {self.reddit.user.me() if self.reddit.read_only else 'Read-only mode'}")
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {e}")
            raise

        # Initialize Firebase
        self.firebase_config = FirebaseConfig()
        try:
            self.firebase_config.initialize_firebase(
                service_account_path=Config.FIREBASE_SERVICE_ACCOUNT_PATH,
                database_url=Config.FIREBASE_DATABASE_URL
            )
            self.firebase_manager = FirebaseDataManager(self.firebase_config)

            # Test Firebase connection
            if not self.firebase_config.test_connection():
                raise Exception("Firebase connection test failed")

        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            logger.info("Proceeding without Firebase - data will only be saved locally")
            self.firebase_manager = None

    def search_subreddit_posts(self, subreddit_name, keywords, limit=100):
        """
        Search for posts in a subreddit using keywords

        Args:
            subreddit_name (str): Name of the subreddit
            keywords (list): List of keywords to search for
            limit (int): Maximum number of posts to retrieve

        Returns:
            list: List of post dictionaries
        """
        posts_data = []

        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Search by keywords
            for keyword in keywords:
                logger.info(f"Searching for '{keyword}' in r/{subreddit_name}")

                search_results = subreddit.search(keyword, limit=limit//len(keywords), 
                                                time_filter=Config.TIME_FILTER)

                for post in search_results:
                    post_data = self.extract_post_data(post)
                    if post_data:
                        posts_data.append(post_data)

                time.sleep(1)  # Rate limiting

            logger.info(f"Collected {len(posts_data)} posts from r/{subreddit_name}")

        except Exception as e:
            logger.error(f"Error searching r/{subreddit_name}: {e}")

        return posts_data

    def extract_post_data(self, post):
        """Extract relevant data from a Reddit post"""
        try:
            # Get comments (limited to avoid API limits)
            post.comments.replace_more(limit=0)
            comments = []

            for comment in post.comments.list()[:Config.MAX_COMMENTS_PER_POST]:
                if hasattr(comment, 'body') and comment.body != '[deleted]':
                    comments.append({
                        'comment_id': comment.id,
                        'comment_body': comment.body,
                        'comment_score': comment.score,
                        'comment_created_utc': datetime.fromtimestamp(comment.created_utc)
                    })

            return {
                'post_id': post.id,
                'title': post.title,
                'selftext': post.selftext,
                'subreddit': str(post.subreddit),
                'author': str(post.author) if post.author else '[deleted]',
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'url': post.url,
                'is_self': post.is_self,
                'comments': comments,
                'full_text': f"{post.title} {post.selftext}".strip(),
                'collected_at': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error extracting post data: {e}")
            return None

    def collect_data(self):
        """Main data collection function"""
        all_posts = []

        for subreddit in Config.SUBREDDITS:
            posts = self.search_subreddit_posts(subreddit, Config.KEYWORDS, 
                                              Config.MAX_POSTS_PER_SUBREDDIT)
            all_posts.extend(posts)

            # Add delay between subreddits
            time.sleep(2)

        # Convert to DataFrame
        df = pd.DataFrame(all_posts)

        if len(df) == 0:
            logger.warning("No posts collected!")
            return df

        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['post_id'])
        logger.info(f"Removed {initial_count - len(df)} duplicate posts")

        # Save to Firebase
        if self.firebase_manager:
            try:
                self.firebase_manager.save_posts(df, batch_size=Config.FIREBASE_BATCH_SIZE)
                logger.info("Data successfully saved to Firebase")
            except Exception as e:
                logger.error(f"Failed to save to Firebase: {e}")

        # Save local backup if enabled
        if Config.USE_LOCAL_BACKUP:
            Config.create_directories()
            df.to_csv(Config.RAW_DATA_FILE, index=False)
            logger.info(f"Local backup saved to {Config.RAW_DATA_FILE}")

        return df

    def load_data_from_firebase(self, limit=None):
        """Load data from Firebase"""
        if not self.firebase_manager:
            logger.error("Firebase not initialized - cannot load data")
            return None

        try:
            df = self.firebase_manager.load_posts(limit=limit)

            # Also save local backup
            if not df.empty and Config.USE_LOCAL_BACKUP:
                Config.create_directories()
                df.to_csv(Config.RAW_DATA_FILE, index=False)
                logger.info("Data loaded from Firebase and saved as local backup")

            return df

        except Exception as e:
            logger.error(f"Failed to load data from Firebase: {e}")
            return None

    def get_database_stats(self):
        """Get statistics about data in Firebase"""
        if not self.firebase_manager:
            return {}

        return self.firebase_manager.get_database_stats()

    def sync_with_firebase(self, df):
        """Sync local DataFrame with Firebase"""
        if not self.firebase_manager:
            logger.warning("Firebase not available - skipping sync")
            return

        try:
            # Check existing data in Firebase
            existing_df = self.firebase_manager.load_posts()

            if existing_df.empty:
                # No existing data, save all
                logger.info("No existing data in Firebase, saving all posts")
                self.firebase_manager.save_posts(df)
            else:
                # Find new posts
                existing_ids = set(existing_df['post_id'].values)
                new_posts = df[~df['post_id'].isin(existing_ids)]

                if not new_posts.empty:
                    logger.info(f"Saving {len(new_posts)} new posts to Firebase")
                    self.firebase_manager.save_posts(new_posts)
                else:
                    logger.info("No new posts to sync")

        except Exception as e:
            logger.error(f"Error syncing with Firebase: {e}")

if __name__ == "__main__":
    collector = RedditDataCollector()

    # Collect new data
    data = collector.collect_data()
    print(f"Collected {len(data)} posts")

    # Get database stats
    stats = collector.get_database_stats()
    print(f"Database stats: {stats}")
