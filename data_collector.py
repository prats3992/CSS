"""
Reddit Data Collector using Arctic Shift (Pushshift) API
Collects posts and comments related to pre-eclampsia with weighted relevance
"""

import requests
import time
from datetime import datetime, timedelta
from pmaw import PushshiftAPI
import pandas as pd
from tqdm import tqdm
from config import (
    SUBREDDIT_WEIGHTS, 
    KEYWORDS_BY_LLM, 
    KEYWORD_WEIGHTS,
    DATA_COLLECTION_CONFIG,
    get_post_limit_for_subreddit
)
from firebase_manager import FirebaseManager
import re

class RedditDataCollector:
    def __init__(self):
        """Initialize the data collector"""
        self.api = PushshiftAPI()
        self.firebase = FirebaseManager()
        self.collected_post_ids = set()
        
    def calculate_relevance_score(self, text, subreddit, matched_keywords):
        """
        Calculate relevance score based on subreddit weight and keyword matches
        
        Args:
            text (str): Post/comment text
            subreddit (str): Subreddit name
            matched_keywords (dict): Dictionary of matched keywords by category
            
        Returns:
            float: Relevance score between 0 and 1
        """
        # Base score from subreddit weight
        subreddit_weight = SUBREDDIT_WEIGHTS.get(subreddit, {}).get('weight', 0.3)
        
        # Calculate keyword score
        keyword_score = 0
        keyword_count = 0
        
        for category, keywords in matched_keywords.items():
            if keywords:
                category_weight = KEYWORD_WEIGHTS.get(category, 0.5)
                keyword_score += len(keywords) * category_weight
                keyword_count += len(keywords)
        
        # Normalize keyword score
        if keyword_count > 0:
            keyword_score = min(keyword_score / (keyword_count * 2), 1.0)
        else:
            keyword_score = 0
        
        # Combined score (weighted average)
        final_score = (subreddit_weight * 0.4) + (keyword_score * 0.6)
        
        return round(final_score, 3)
    
    def match_keywords(self, text, llm_name=None):
        """
        Match keywords in text and categorize them
        
        Args:
            text (str): Text to search
            llm_name (str): Specific LLM keywords to use, or None for all
            
        Returns:
            dict: Matched keywords by category and LLM
        """
        text_lower = text.lower()
        matched = {}
        
        llms_to_check = [llm_name] if llm_name else ['claude', 'gemini', 'gpt5']
        
        for llm in llms_to_check:
            matched[llm] = {}
            keywords_dict = KEYWORDS_BY_LLM.get(llm, {})
            
            for category, keywords in keywords_dict.items():
                matched[llm][category] = []
                for keyword in keywords:
                    # Case-insensitive search
                    if keyword.lower() in text_lower:
                        matched[llm][category].append(keyword)
        
        return matched
    
    def collect_posts_from_subreddit(self, subreddit, start_date, end_date, max_posts=None):
        """
        Collect posts from a specific subreddit
        
        Args:
            subreddit (str): Subreddit name
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            max_posts (int): Maximum number of posts to collect (None = use weight-based limit)
            
        Returns:
            list: Collected posts with metadata
        """
        print(f"\n{'='*60}")
        print(f"Collecting from r/{subreddit}")
        print(f"{'='*60}")
        
        # Get subreddit metadata
        subreddit_info = SUBREDDIT_WEIGHTS.get(subreddit, {})
        subreddit_weight = subreddit_info.get('weight', 0.3)
        llm_suggested = subreddit_info.get('llm', [])
        focus_level = subreddit_info.get('focus', 'broad')
        
        # Calculate post limit based on weight if not specified
        if max_posts is None:
            max_posts = get_post_limit_for_subreddit(subreddit)
        
        print(f"Weight: {subreddit_weight} → Collecting up to {max_posts} posts")
        
        # Check existing posts
        existing_ids = self.firebase.get_existing_post_ids(subreddit)
        print(f"Found {len(existing_ids)} existing posts in Firebase")
        
        # Convert dates to timestamps
        start_epoch = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_epoch = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        collected_posts = []
        
        # Determine search strategy based on focus level
        if focus_level == 'dedicated':
            # For dedicated subreddits, collect all posts
            print(f"Strategy: Collect all posts (dedicated subreddit)")
            query = None
        else:
            # For broader subreddits, use keyword filtering
            print(f"Strategy: Keyword-filtered collection (focus: {focus_level})")
            # Build keyword query from core terms
            core_keywords = []
            for llm in llm_suggested:
                core_keywords.extend(KEYWORDS_BY_LLM[llm].get('core_terms', []))
            
            # Use top keywords
            query = ' OR '.join(list(set(core_keywords))[:5])
            print(f"Query: {query}")
        
        try:
            # Search submissions
            if query:
                posts = self.api.search_submissions(
                    subreddit=subreddit,
                    q=query,
                    after=start_epoch,
                    before=end_epoch,
                    limit=max_posts,
                    filter=['id', 'title', 'selftext', 'author', 'created_utc', 
                           'score', 'num_comments', 'url', 'permalink']
                )
            else:
                posts = self.api.search_submissions(
                    subreddit=subreddit,
                    after=start_epoch,
                    before=end_epoch,
                    limit=max_posts,
                    filter=['id', 'title', 'selftext', 'author', 'created_utc', 
                           'score', 'num_comments', 'url', 'permalink']
                )
            
            posts_list = list(posts)
            print(f"Retrieved {len(posts_list)} posts from API")
            
            # Process posts
            for post in tqdm(posts_list, desc="Processing posts"):
                post_id = post.get('id')
                
                # Skip if already collected
                if post_id in existing_ids or post_id in self.collected_post_ids:
                    continue
                
                # Extract text
                title = post.get('title', '')
                selftext = post.get('selftext', '')
                full_text = f"{title} {selftext}"
                
                # Match keywords
                matched_keywords = self.match_keywords(full_text)
                
                # Check if post has relevant keywords
                total_matches = sum(
                    len(kws) 
                    for llm_kws in matched_keywords.values() 
                    for kws in llm_kws.values()
                )
                
                if total_matches == 0 and focus_level != 'dedicated':
                    continue  # Skip irrelevant posts from broad subreddits
                
                # Calculate relevance score
                relevance_score = self.calculate_relevance_score(
                    full_text, subreddit, 
                    matched_keywords.get(llm_suggested[0] if llm_suggested else 'claude', {})
                )
                
                # Prepare post data
                post_data = {
                    'id': post_id,
                    'subreddit': subreddit,
                    'title': title,
                    'selftext': selftext,
                    'author': str(post.get('author', '[deleted]')),
                    'created_utc': post.get('created_utc'),
                    'created_date': datetime.fromtimestamp(post.get('created_utc', 0)).isoformat(),
                    'score': post.get('score', 0),
                    'num_comments': post.get('num_comments', 0),
                    'url': post.get('url', ''),
                    'permalink': post.get('permalink', ''),
                    'subreddit_weight': subreddit_weight,
                    'subreddit_focus': focus_level,
                    'llm_suggested_by': llm_suggested,
                    'matched_keywords': matched_keywords,
                    'relevance_score': relevance_score,
                }
                
                collected_posts.append(post_data)
                self.collected_post_ids.add(post_id)
            
            print(f"✓ Collected {len(collected_posts)} new posts")
            
        except Exception as e:
            print(f"✗ Error collecting from r/{subreddit}: {str(e)}")
        
        return collected_posts
    
    def collect_comments_for_post(self, post_id, post_subreddit, max_comments=50):
        """
        Collect comments for a specific post
        
        Args:
            post_id (str): Reddit post ID
            post_subreddit (str): Subreddit name
            max_comments (int): Maximum comments to collect
            
        Returns:
            list: Collected comments with metadata
        """
        collected_comments = []
        
        try:
            comments = self.api.search_comments(
                link_id=post_id,
                limit=max_comments,
                filter=['id', 'body', 'author', 'created_utc', 'score', 'parent_id']
            )
            
            comments_list = list(comments)
            
            for comment in comments_list:
                comment_id = comment.get('id')
                body = comment.get('body', '')
                
                # Skip short comments or deleted
                if len(body) < DATA_COLLECTION_CONFIG['min_comment_length']:
                    continue
                
                # Match keywords
                matched_keywords = self.match_keywords(body)
                
                # Get subreddit info
                subreddit_info = SUBREDDIT_WEIGHTS.get(post_subreddit, {})
                
                # Calculate relevance
                relevance_score = self.calculate_relevance_score(
                    body, post_subreddit,
                    matched_keywords.get(subreddit_info.get('llm', ['claude'])[0], {})
                )
                
                comment_data = {
                    'id': comment_id,
                    'post_id': post_id,
                    'subreddit': post_subreddit,
                    'body': body,
                    'author': str(comment.get('author', '[deleted]')),
                    'created_utc': comment.get('created_utc'),
                    'created_date': datetime.fromtimestamp(comment.get('created_utc', 0)).isoformat(),
                    'score': comment.get('score', 0),
                    'parent_id': comment.get('parent_id', ''),
                    'matched_keywords': matched_keywords,
                    'relevance_score': relevance_score,
                }
                
                collected_comments.append(comment_data)
            
        except Exception as e:
            print(f"Error collecting comments for post {post_id}: {str(e)}")
        
        return collected_comments
    
    def run_collection(self, subreddits=None, collect_comments=True):
        """
        Run the full data collection pipeline
        
        Args:
            subreddits (list): List of subreddits to collect from (None = all)
            collect_comments (bool): Whether to collect comments
        """
        print("\n" + "="*60)
        print("REDDIT DATA COLLECTION PIPELINE")
        print("Pre-eclampsia Research Project")
        print("="*60)
        
        # Use config
        start_date = DATA_COLLECTION_CONFIG['start_date']
        end_date = DATA_COLLECTION_CONFIG['end_date']
        base_posts = DATA_COLLECTION_CONFIG['base_posts_per_subreddit']
        
        # Determine subreddits
        if not subreddits:
            subreddits = list(SUBREDDIT_WEIGHTS.keys())
        
        # Calculate total expected posts
        total_expected = sum(get_post_limit_for_subreddit(s) for s in subreddits)
        
        print(f"\nConfiguration:")
        print(f"  Date range: {start_date} to {end_date}")
        print(f"  Subreddits: {len(subreddits)}")
        print(f"  Base posts per subreddit: {base_posts} (scaled by weight)")
        print(f"  Expected total posts: ~{total_expected}")
        print(f"  Collect comments: {collect_comments}")
        
        print(f"\nPost Limits by Weight:")
        for sub in sorted(subreddits, key=lambda s: SUBREDDIT_WEIGHTS[s]['weight'], reverse=True):
            weight = SUBREDDIT_WEIGHTS[sub]['weight']
            limit = get_post_limit_for_subreddit(sub)
            print(f"  r/{sub:25s} (weight: {weight:.1f}) → {limit:4d} posts")
        
        all_posts = []
        all_comments = []
        
        # Collect from each subreddit (sorted by weight, highest first)
        sorted_subreddits = sorted(subreddits, 
                                   key=lambda s: SUBREDDIT_WEIGHTS[s]['weight'], 
                                   reverse=True)
        
        for subreddit in sorted_subreddits:
            posts = self.collect_posts_from_subreddit(
                subreddit, start_date, end_date, max_posts=None  # Use weight-based limit
            )
            
            if posts:
                # Store posts in Firebase
                print(f"Storing {len(posts)} posts to Firebase...")
                self.firebase.store_batch_posts(posts)
                all_posts.extend(posts)
                
                # Collect comments if requested
                if collect_comments:
                    print(f"Collecting comments for posts...")
                    for post in tqdm(posts[:50], desc="Collecting comments"):  # Limit comment collection
                        comments = self.collect_comments_for_post(
                            post['id'], 
                            post['subreddit'],
                            max_comments=DATA_COLLECTION_CONFIG['max_comments_per_post']
                        )
                        if comments:
                            all_comments.extend(comments)
                    
                    if all_comments:
                        print(f"Storing {len(all_comments)} comments to Firebase...")
                        self.firebase.store_batch_comments(all_comments)
            
            # Rate limiting
            time.sleep(2)
        
        # Update metadata
        print("\nUpdating collection metadata...")
        stats = self.firebase.get_collection_stats()
        print(f"\nCollection Summary:")
        print(f"  Total posts collected: {len(all_posts)}")
        print(f"  Total comments collected: {len(all_comments)}")
        print(f"  Total in database: {stats['total_posts']} posts, {stats['total_comments']} comments")
        
        # Store LLM comparison metadata
        llm_comparison = {
            'collection_run': datetime.utcnow().isoformat(),
            'subreddits_by_llm': {},
            'keyword_matches_by_llm': {'claude': 0, 'gemini': 0, 'gpt5': 0}
        }
        
        for llm in ['claude', 'gemini', 'gpt5']:
            llm_subreddits = [s for s, info in SUBREDDIT_WEIGHTS.items() if llm in info.get('llm', [])]
            llm_comparison['subreddits_by_llm'][llm] = llm_subreddits
        
        self.firebase.store_llm_comparison_data(llm_comparison)
        
        print("\n✓ Data collection complete!")
        return all_posts, all_comments


if __name__ == "__main__":
    collector = RedditDataCollector()
    
    # Run collection
    posts, comments = collector.run_collection(
        subreddits=None,  # Collect from all subreddits
        collect_comments=True
    )
