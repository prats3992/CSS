"""
Firebase Manager for storing and retrieving Reddit data
"""

import firebase_admin
from firebase_admin import credentials, db
import os
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

class FirebaseManager:
    def __init__(self):
        """Initialize Firebase connection"""
        if not firebase_admin._apps:
            # Use the service account JSON file
            cred = credentials.Certificate('pre-eclampsia-analysis-firebase-adminsdk-fbsvc-b293a2d5eb.json')
            
            firebase_admin.initialize_app(cred, {
                'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
            })
        
        self.db = db.reference()
    
    def store_post(self, post_data):
        """
        Store a Reddit post in Firebase
        
        Args:
            post_data (dict): Post information including metadata
        """
        post_id = post_data['id']
        ref = self.db.child('reddit_posts').child(post_id)
        
        # Add timestamp
        post_data['stored_at'] = datetime.utcnow().isoformat()
        
        ref.set(post_data)
        return post_id
    
    def store_comment(self, comment_data):
        """
        Store a Reddit comment in Firebase
        
        Args:
            comment_data (dict): Comment information including metadata
        """
        comment_id = comment_data['id']
        ref = self.db.child('reddit_comments').child(comment_id)
        
        # Add timestamp
        comment_data['stored_at'] = datetime.utcnow().isoformat()
        
        ref.set(comment_data)
        return comment_id
    
    def store_batch_posts(self, posts_list):
        """Store multiple posts efficiently"""
        updates = {}
        for post in posts_list:
            post['stored_at'] = datetime.utcnow().isoformat()
            updates[f"reddit_posts/{post['id']}"] = post
        
        self.db.update(updates)
        return len(posts_list)
    
    def store_batch_comments(self, comments_list):
        """Store multiple comments efficiently"""
        updates = {}
        for comment in comments_list:
            comment['stored_at'] = datetime.utcnow().isoformat()
            updates[f"reddit_comments/{comment['id']}"] = comment
        
        self.db.update(updates)
        return len(comments_list)
    
    def update_metadata(self, collection_name, metadata):
        """
        Update collection metadata (stats, timestamps, etc.)
        
        Args:
            collection_name (str): Name of the collection
            metadata (dict): Metadata to store
        """
        ref = self.db.child('collection_metadata').child(collection_name)
        metadata['last_updated'] = datetime.utcnow().isoformat()
        ref.update(metadata)
    
    def get_existing_post_ids(self, subreddit=None):
        """Get set of already collected post IDs to avoid duplicates"""
        ref = self.db.child('reddit_posts')
        
        # Get all posts (indexing not required)
        posts = ref.get()
        
        if posts:
            # Filter by subreddit if specified
            if subreddit:
                filtered_ids = {post_id for post_id, post_data in posts.items() 
                               if post_data.get('subreddit') == subreddit}
                return filtered_ids
            return set(posts.keys())
        return set()
    
    def get_collection_stats(self):
        """Get statistics about the collected data"""
        posts_ref = self.db.child('reddit_posts').get()
        comments_ref = self.db.child('reddit_comments').get()
        
        stats = {
            'total_posts': len(posts_ref) if posts_ref else 0,
            'total_comments': len(comments_ref) if comments_ref else 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Count by subreddit
        if posts_ref:
            subreddit_counts = {}
            llm_counts = {'claude': 0, 'gemini': 0, 'gpt5': 0}
            
            for post_id, post in posts_ref.items():
                subreddit = post.get('subreddit', 'unknown')
                subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
                
                # Count LLM attribution
                for llm in post.get('llm_suggested_by', []):
                    llm_counts[llm] = llm_counts.get(llm, 0) + 1
            
            stats['posts_by_subreddit'] = subreddit_counts
            stats['posts_by_llm'] = llm_counts
        
        return stats
    
    def store_llm_comparison_data(self, comparison_data):
        """
        Store data specifically for LLM comparison analysis
        
        Args:
            comparison_data (dict): Comparison metrics and findings
        """
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        ref = self.db.child('llm_comparison_analysis').child(timestamp)
        comparison_data['timestamp'] = datetime.utcnow().isoformat()
        ref.set(comparison_data)
    
    def export_to_json(self, output_file):
        """Export all data to JSON file for backup/analysis"""
        all_data = {
            'posts': self.db.child('reddit_posts').get(),
            'comments': self.db.child('reddit_comments').get(),
            'metadata': self.db.child('collection_metadata').get(),
            'exported_at': datetime.utcnow().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        return output_file
