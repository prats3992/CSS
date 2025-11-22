"""
Quick script to check Firebase collection statistics
"""

from firebase_manager import FirebaseManager
import json

def main():
    print("\n" + "="*60)
    print("FIREBASE COLLECTION STATISTICS")
    print("="*60 + "\n")
    
    firebase = FirebaseManager()
    stats = firebase.get_collection_stats()
    
    print(f"Total Posts: {stats['total_posts']}")
    print(f"Total Comments: {stats['total_comments']}")
    print(f"Last Updated: {stats['timestamp']}")
    
    if 'posts_by_subreddit' in stats and stats['posts_by_subreddit']:
        print("\n" + "-"*60)
        print("Posts by Subreddit:")
        print("-"*60)
        
        # Sort by count
        sorted_subs = sorted(stats['posts_by_subreddit'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        for subreddit, count in sorted_subs:
            print(f"  r/{subreddit:<30} {count:>5} posts")
    
    if 'posts_by_llm' in stats and stats['posts_by_llm']:
        print("\n" + "-"*60)
        print("Posts by LLM Suggestion:")
        print("-"*60)
        
        for llm, count in stats['posts_by_llm'].items():
            print(f"  {llm.upper():<10} {count:>5} posts")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
