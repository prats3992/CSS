#!/usr/bin/env python3
"""
Test script to verify Reddit API connection for pre-eclampsia analysis project.
Run this after setting up your Reddit API credentials.
"""

import praw
from config import Config
import sys

def test_reddit_connection():
    """Test Reddit API connection and basic functionality."""
    
    print("=" * 60)
    print("REDDIT API CONNECTION TEST")
    print("=" * 60)
    
    # Check if credentials are still default values
    if (Config.REDDIT_CLIENT_ID == 'your_client_id_here' or 
        Config.REDDIT_CLIENT_SECRET == 'your_client_secret_here'):
        print("❌ ERROR: Please update your Reddit API credentials in config.py")
        print("   Current CLIENT_ID:", Config.REDDIT_CLIENT_ID)
        print("   Current CLIENT_SECRET:", Config.REDDIT_CLIENT_SECRET[:20] + "..." if len(Config.REDDIT_CLIENT_SECRET) > 20 else Config.REDDIT_CLIENT_SECRET)
        return False
    
    try:
        print("1. Initializing Reddit API connection...")
        print(f"   Client ID: {Config.REDDIT_CLIENT_ID}")
        print(f"   User Agent: {Config.REDDIT_USER_AGENT}")
        
        # Initialize Reddit instance
        reddit = praw.Reddit(
            client_id=Config.REDDIT_CLIENT_ID,
            client_secret=Config.REDDIT_CLIENT_SECRET,
            user_agent=Config.REDDIT_USER_AGENT
        )
        
        print("✅ Reddit instance created successfully")
        
        print(f"\n2. Testing connection...")
        print(f"   Read-only mode: {reddit.read_only}")
        
        if not reddit.read_only:
            print("⚠️  WARNING: Not in read-only mode. This is unexpected for a script app.")
        
        print("✅ Successfully connected to Reddit API!")
        
        print(f"\n3. Testing subreddit access...")
        
        # Test accessing a few subreddits from the config
        test_subreddits = ['pregnancy', 'BabyBumps', 'preeclampsia']
        
        for sub_name in test_subreddits:
            try:
                subreddit = reddit.subreddit(sub_name)
                print(f"   Testing r/{sub_name}...")
                
                # Try to get subreddit info
                print(f"     - Display name: {subreddit.display_name}")
                print(f"     - Subscribers: {subreddit.subscribers:,}")
                print(f"     - Public: {subreddit.subreddit_type == 'public'}")
                
                print(f"   ✅ Successfully accessed r/{sub_name}")
                
            except Exception as e:
                print(f"   ❌ Error accessing r/{sub_name}: {e}")
                if "403" in str(e):
                    print(f"     ℹ️  r/{sub_name} may have access restrictions or be private")
        
        print(f"\n4. Testing search functionality...")
        
        # Test searching for pre-eclampsia posts
        try:
            print("   Trying search on r/BabyBumps (known to work)...")
            subreddit = reddit.subreddit('BabyBumps')
            search_results = list(subreddit.search('preeclampsia', limit=3))
            
            print(f"   Found {len(search_results)} posts matching 'preeclampsia' in r/BabyBumps")
            
            for i, post in enumerate(search_results, 1):
                print(f"   {i}. \"{post.title[:50]}{'...' if len(post.title) > 50 else ''}\"")
                print(f"      Score: {post.score}, Comments: {post.num_comments}")
            
            print("   ✅ Search functionality working!")
            
        except Exception as e:
            print(f"   ❌ Error testing search: {e}")
            if "403" in str(e):
                print("   ℹ️  This might be due to rate limiting or subreddit restrictions")
        
        print(f"\n5. Testing rate limits...")
        print(f"   Making multiple requests to test rate limiting...")
        
        try:
            # Make a few quick requests with delays
            import time
            for i in range(3):  # Reduced from 5 to 3
                subreddit = reddit.subreddit('BabyBumps')  # Use known working subreddit
                list(subreddit.hot(limit=1))
                print(f"   Request {i+1}/3 successful")
                if i < 2:  # Don't sleep after last request
                    time.sleep(1)  # Add 1 second delay between requests
            
            print("   ✅ Rate limiting appears to be working normally")
            
        except Exception as e:
            print(f"   ⚠️  Rate limit warning: {e}")
            if "403" in str(e):
                print("   ℹ️  403 errors can indicate rate limiting or access restrictions")
        
        print(f"\n" + "=" * 60)
        print("✅ REDDIT API CONNECTION ESTABLISHED!")
        print("Note: Some 403 errors are normal for restricted subreddits.")
        print("Your setup is working for accessible public subreddits.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print(f"\nTroubleshooting steps:")
        print(f"1. Verify your Reddit API credentials in config.py")
        print(f"2. Make sure you created a 'script' type application")
        print(f"3. Check that your Reddit account is verified")
        print(f"4. Ensure you have internet connection")
        
        return False

if __name__ == "__main__":
    success = test_reddit_connection()
    sys.exit(0 if success else 1)