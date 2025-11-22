"""
Display the data collection plan showing how weights affect post collection
"""

from config import SUBREDDIT_WEIGHTS, get_post_limit_for_subreddit, DATA_COLLECTION_CONFIG

def show_collection_plan():
    """Display the collection plan with post limits"""
    
    print("\n" + "="*80)
    print("PRE-ECLAMPSIA DATA COLLECTION PLAN")
    print("="*80)
    
    print(f"\nBase Posts per Subreddit: {DATA_COLLECTION_CONFIG['base_posts_per_subreddit']}")
    print("Actual posts collected = Base Posts × Weight")
    
    # Group by focus level
    focus_groups = {
        'dedicated': [],
        'high': [],
        'medium': [],
        'broad': []
    }
    
    for subreddit, info in SUBREDDIT_WEIGHTS.items():
        focus = info['focus']
        focus_groups[focus].append(subreddit)
    
    total_posts = 0
    
    # Display each group
    for focus in ['dedicated', 'high', 'medium', 'broad']:
        if not focus_groups[focus]:
            continue
            
        print(f"\n{'='*80}")
        print(f"FOCUS LEVEL: {focus.upper()}")
        print("="*80)
        print(f"{'Subreddit':<30} {'Weight':<10} {'Posts':<10} {'LLMs'}")
        print("-"*80)
        
        # Sort by weight within group
        sorted_subs = sorted(focus_groups[focus], 
                            key=lambda s: SUBREDDIT_WEIGHTS[s]['weight'], 
                            reverse=True)
        
        group_total = 0
        for subreddit in sorted_subs:
            info = SUBREDDIT_WEIGHTS[subreddit]
            weight = info['weight']
            post_limit = get_post_limit_for_subreddit(subreddit)
            llms = ', '.join(info['llm'])
            
            print(f"r/{subreddit:<28} {weight:<10.1f} {post_limit:<10d} {llms}")
            
            group_total += post_limit
            total_posts += post_limit
        
        print("-"*80)
        print(f"{'Subtotal:':<30} {'':<10} {group_total:<10d}")
    
    print("\n" + "="*80)
    print(f"{'TOTAL EXPECTED POSTS:':<30} {'':<10} {total_posts:<10d}")
    print("="*80)
    
    # LLM breakdown
    print("\n" + "="*80)
    print("POSTS BY LLM SUGGESTION")
    print("="*80)
    
    llm_posts = {'claude': 0, 'gemini': 0, 'gpt5': 0}
    llm_subreddits = {'claude': [], 'gemini': [], 'gpt5': []}
    
    for subreddit, info in SUBREDDIT_WEIGHTS.items():
        post_limit = get_post_limit_for_subreddit(subreddit)
        for llm in info['llm']:
            llm_posts[llm] += post_limit
            llm_subreddits[llm].append(subreddit)
    
    for llm in ['claude', 'gemini', 'gpt5']:
        print(f"\n{llm.upper()}:")
        print(f"  Subreddits suggested: {len(llm_subreddits[llm])}")
        print(f"  Total posts: {llm_posts[llm]}")
        print(f"  Subreddits: {', '.join(['r/' + s for s in llm_subreddits[llm][:5]])}...")
    
    # Weight distribution
    print("\n" + "="*80)
    print("WEIGHT DISTRIBUTION SUMMARY")
    print("="*80)
    
    weight_ranges = {
        '1.0 (Dedicated)': [s for s, i in SUBREDDIT_WEIGHTS.items() if i['weight'] == 1.0],
        '0.7-0.9 (High)': [s for s, i in SUBREDDIT_WEIGHTS.items() if 0.7 <= i['weight'] < 1.0],
        '0.5-0.6 (Medium)': [s for s, i in SUBREDDIT_WEIGHTS.items() if 0.5 <= i['weight'] < 0.7],
        '0.3-0.4 (Broad)': [s for s, i in SUBREDDIT_WEIGHTS.items() if i['weight'] < 0.5],
    }
    
    for range_name, subs in weight_ranges.items():
        if subs:
            posts_in_range = sum(get_post_limit_for_subreddit(s) for s in subs)
            percentage = (posts_in_range / total_posts) * 100
            print(f"\n{range_name}:")
            print(f"  Subreddits: {len(subs)}")
            print(f"  Posts: {posts_in_range} ({percentage:.1f}%)")
    
    print("\n" + "="*80)
    print("Data Collection Strategy:")
    print("  • Dedicated/High (weight ≥0.7): Collect most/all posts")
    print("  • Medium (weight 0.5-0.6): Use keyword filtering")
    print("  • Broad (weight <0.5): Strict keyword filtering")
    print("="*80 + "\n")


if __name__ == "__main__":
    show_collection_plan()
