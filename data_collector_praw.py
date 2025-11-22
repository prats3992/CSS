"""
Reddit Data Collector using PRAW (Reddit API)
Alternative to Pushshift when it's unavailable
"""

import praw
import time
from datetime import datetime
from tqdm import tqdm
import os
from dotenv import load_dotenv
from config import (
    SUBREDDIT_WEIGHTS, 
    KEYWORDS_BY_LLM, 
    KEYWORD_WEIGHTS,
    DATA_COLLECTION_CONFIG,
    get_post_limit_for_subreddit
)
from firebase_manager import FirebaseManager
import re

load_dotenv()

class RedditDataCollectorPRAW:
    def __init__(self):
        """Initialize the data collector with PRAW"""
        # Initialize PRAW using credentials from .env file
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID').strip("'\""),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET').strip("'\""),
            user_agent=os.getenv('REDDIT_USER_AGENT').strip("'\"")
        )
        self.firebase = FirebaseManager()
        self.collected_post_ids = set()
        
    def calculate_relevance_score(self, text, subreddit, matched_keywords):
        """Calculate relevance score based on subreddit weight and keyword matches"""
        subreddit_weight = SUBREDDIT_WEIGHTS.get(subreddit, {}).get('weight', 0.3)
        
        keyword_score = 0
        keyword_count = 0
        
        for category, keywords in matched_keywords.items():
            if keywords:
                category_weight = KEYWORD_WEIGHTS.get(category, 0.5)
                keyword_score += len(keywords) * category_weight
                keyword_count += len(keywords)
        
        if keyword_count > 0:
            keyword_score = min(keyword_score / (keyword_count * 2), 1.0)
        else:
            keyword_score = 0
        
        final_score = (subreddit_weight * 0.4) + (keyword_score * 0.6)
        return round(final_score, 3)
    
    def match_keywords(self, text, llm_name=None):
        """Match keywords in text and categorize them"""
        text_lower = text.lower()
        matched = {}
        
        llms_to_check = [llm_name] if llm_name else ['claude', 'gemini', 'gpt5']
        
        for llm in llms_to_check:
            matched[llm] = {}
            keywords_dict = KEYWORDS_BY_LLM.get(llm, {})
            
            for category, keywords in keywords_dict.items():
                matched[llm][category] = []
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        matched[llm][category].append(keyword)
        
        return matched
    
    def has_relevant_keywords(self, text, focus_level):
        """Check if text contains relevant keywords based on focus level"""
        text_lower = text.lower()
        
        # Core terms that must be present for broad subreddits
        core_terms = ['preeclampsia', 'pre-eclampsia', 'pre eclampsia', 'eclampsia', 
                      'hellp', 'toxemia', 'gestational hypertension']
        
        if focus_level == 'dedicated':
            return True  # Accept all posts from dedicated subreddits
        
        # Check for core terms
        for term in core_terms:
            if term.lower() in text_lower:
                return True
        
        return False
    
    def collect_posts_from_subreddit(self, subreddit_name, max_posts=None):
        """
        Collect posts from a specific subreddit using PRAW
        """
        print(f"\n{'='*60}")
        print(f"Collecting from r/{subreddit_name}")
        print(f"{'='*60}")
        
        subreddit_info = SUBREDDIT_WEIGHTS.get(subreddit_name, {})
        subreddit_weight = subreddit_info.get('weight', 0.3)
        llm_suggested = subreddit_info.get('llm', [])
        focus_level = subreddit_info.get('focus', 'broad')
        
        if max_posts is None:
            max_posts = get_post_limit_for_subreddit(subreddit_name)
        
        print(f"Weight: {subreddit_weight} → Collecting up to {max_posts} posts")
        
        existing_ids = self.firebase.get_existing_post_ids(subreddit_name)
        print(f"Found {len(existing_ids)} existing posts in Firebase")
        
        collected_posts = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Determine search strategy
            if focus_level == 'dedicated':
                print(f"Strategy: Collect recent posts (dedicated subreddit)")
                # For dedicated subreddits, get top and new posts
                posts_iter = list(subreddit.top(time_filter='all', limit=max_posts))
                posts_iter.extend(list(subreddit.new(limit=max_posts)))
            else:
                print(f"Strategy: Search with keywords (focus: {focus_level})")
                # Build search query from core keywords
                core_keywords = []
                for llm in llm_suggested:
                    core_keywords.extend(KEYWORDS_BY_LLM[llm].get('core_terms', []))
                
                # Use top 3 unique keywords
                unique_keywords = list(set(core_keywords))[:3]
                query = ' OR '.join(unique_keywords)
                print(f"Query: {query}")
                
                # Search the subreddit
                posts_iter = list(subreddit.search(query, limit=max_posts, time_filter='all'))
            
            print(f"Retrieved {len(posts_iter)} posts from Reddit API")
            
            # Remove duplicates
            seen_ids = set()
            unique_posts = []
            for post in posts_iter:
                if post.id not in seen_ids:
                    seen_ids.add(post.id)
                    unique_posts.append(post)
            
            # Process posts
            for post in tqdm(unique_posts, desc="Processing posts"):
                post_id = post.id
                
                if post_id in existing_ids or post_id in self.collected_post_ids:
                    continue
                
                # Extract text
                title = post.title
                selftext = post.selftext if hasattr(post, 'selftext') else ''
                full_text = f"{title} {selftext}"
                
                # Check relevance for non-dedicated subreddits
                if focus_level != 'dedicated' and not self.has_relevant_keywords(full_text, focus_level):
                    continue
                
                # Match keywords
                matched_keywords = self.match_keywords(full_text)
                
                # Check if post has relevant keywords
                total_matches = sum(
                    len(kws) 
                    for llm_kws in matched_keywords.values() 
                    for kws in llm_kws.values()
                )
                
                if total_matches == 0 and focus_level not in ['dedicated', 'high']:
                    continue
                
                # Calculate relevance score
                relevance_score = self.calculate_relevance_score(
                    full_text, subreddit_name, 
                    matched_keywords.get(llm_suggested[0] if llm_suggested else 'claude', {})
                )
                
                # Prepare post data
                post_data = {
                    'id': post_id,
                    'subreddit': subreddit_name,
                    'title': title,
                    'selftext': selftext,
                    'author': str(post.author) if post.author else '[deleted]',
                    'created_utc': int(post.created_utc),
                    'created_date': datetime.fromtimestamp(post.created_utc).isoformat(),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'url': post.url,
                    'permalink': f"https://reddit.com{post.permalink}",
                    'subreddit_weight': subreddit_weight,
                    'subreddit_focus': focus_level,
                    'llm_suggested_by': llm_suggested,
                    'matched_keywords': matched_keywords,
                    'relevance_score': relevance_score,
                }
                
                collected_posts.append(post_data)
                self.collected_post_ids.add(post_id)
                
                if len(collected_posts) >= max_posts:
                    break
            
            print(f"✓ Collected {len(collected_posts)} new posts")
            
        except Exception as e:
            print(f"✗ Error collecting from r/{subreddit_name}: {str(e)}")
        
        return collected_posts
    
    def run_collection(self, subreddits=None, collect_comments=False):
        """Run the full data collection pipeline"""
        print("\n" + "="*60)
        print("REDDIT DATA COLLECTION PIPELINE (PRAW)")
        print("Pre-eclampsia Research Project")
        print("="*60)
        
        if not subreddits:
            subreddits = list(SUBREDDIT_WEIGHTS.keys())
        
        total_expected = sum(get_post_limit_for_subreddit(s) for s in subreddits)
        
        print(f"\nConfiguration:")
        print(f"  Subreddits: {len(subreddits)}")
        print(f"  Expected total posts: ~{total_expected}")
        print(f"  Collect comments: {collect_comments} (not implemented yet)")
        
        all_posts = []
        
        # Collect from each subreddit (sorted by weight, highest first)
        sorted_subreddits = sorted(subreddits, 
                                   key=lambda s: SUBREDDIT_WEIGHTS[s]['weight'], 
                                   reverse=True)
        
        for subreddit in sorted_subreddits:
            posts = self.collect_posts_from_subreddit(subreddit, max_posts=None)
            
            if posts:
                print(f"Storing {len(posts)} posts to Firebase...")
                self.firebase.store_batch_posts(posts)
                all_posts.extend(posts)
            
            # Rate limiting
            time.sleep(2)
        
        # Update metadata
        print("\nUpdating collection metadata...")
        stats = self.firebase.get_collection_stats()
        print(f"\nCollection Summary:")
        print(f"  Total posts collected this run: {len(all_posts)}")
        print(f"  Total in database: {stats['total_posts']} posts")
        
        print("\n✓ Data collection complete!")
        return all_posts


if __name__ == "__main__":
    collector = RedditDataCollectorPRAW()
    posts = collector.run_collection(subreddits=None, collect_comments=False)
