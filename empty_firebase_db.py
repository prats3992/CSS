"""
Script to empty all Reddit posts and comments from Firebase DB
"""
from firebase_manager import FirebaseManager

if __name__ == "__main__":
    fb = FirebaseManager()
    print("Deleting all posts...")
    fb.db.child('reddit_posts').delete()
    print("Deleting all comments...")
    fb.db.child('reddit_comments').delete()
    print("Deleting all collection metadata...")
    fb.db.child('collection_metadata').delete()
    print("Deleting all LLM comparison analysis...")
    fb.db.child('llm_comparison_analysis').delete()
    print("✓ Database emptied.")
