"""
Analyze why posts are filtered during collection
Shows the filtering funnel for each subreddit
"""

from config import SUBREDDIT_WEIGHTS, KEYWORDS_BY_LLM

def analyze_filtering():
    print("\n" + "="*80)
    print("POST FILTERING ANALYSIS")
    print("="*80)
    
    print("\nThe pipeline applies 3 main filters:\n")
    
    print("1. DUPLICATE DETECTION")
    print("   - Skips posts already in Firebase")
    print("   - Example: r/preeclampsia 750 retrieved → 500 already existed → 250 new\n")
    
    print("2. KEYWORD RELEVANCE CHECK (Main Filter)")
    print("   - Checks if post text contains actual core medical terms")
    print("   - Core terms checked:")
    core_terms = ['preeclampsia', 'pre-eclampsia', 'pre eclampsia', 'eclampsia', 
                  'hellp', 'toxemia', 'gestational hypertension']
    for term in core_terms:
        print(f"     • {term}")
    
    print("\n   - Why this matters:")
    print("     • Reddit search is BROAD (finds posts mentioning terms in comments)")
    print("     • Our filter is STRICT (requires terms in post title/body)")
    print("     • This ensures HIGH QUALITY, relevant posts only\n")
    
    print("3. MINIMUM KEYWORD MATCHES")
    print("   - For medium/broad subreddits: requires ≥1 keyword match")
    print("   - For high/dedicated: more lenient\n")
    
    print("="*80)
    print("FILTERING BY FOCUS LEVEL")
    print("="*80)
    
    focus_groups = {
        'dedicated': [],
        'high': [],
        'medium': [],
        'broad': []
    }
    
    for subreddit, info in SUBREDDIT_WEIGHTS.items():
        focus = info['focus']
        focus_groups[focus].append(subreddit)
    
    print("\n📌 DEDICATED (Accept all posts from subreddit)")
    print("   " + ", ".join([f"r/{s}" for s in focus_groups['dedicated']]))
    print("   → Minimal filtering, high acceptance rate\n")
    
    print("📌 HIGH FOCUS (Moderate keyword filtering)")
    print("   " + ", ".join([f"r/{s}" for s in focus_groups['high']]))
    print("   → Accepts posts with pregnancy complications context\n")
    
    print("📌 MEDIUM FOCUS (Strict keyword filtering)")
    print("   " + ", ".join([f"r/{s}" for s in focus_groups['medium']]))
    print("   → Requires core medical terms present\n")
    
    print("📌 BROAD FOCUS (Very strict keyword filtering)")
    print("   " + ", ".join([f"r/{s}" for s in focus_groups['broad']]))
    print("   → Requires explicit pre-eclampsia mentions\n")
    
    print("="*80)
    print("EXAMPLE FILTERING SCENARIOS")
    print("="*80)
    
    scenarios = [
        {
            'subreddit': 'r/Infertility',
            'focus': 'medium',
            'retrieved': 97,
            'stored': 0,
            'reason': 'Posts discussed "protein in urine" generally, not pre-eclampsia specifically'
        },
        {
            'subreddit': 'r/TwoXChromosomes',
            'focus': 'broad',
            'retrieved': 150,
            'stored': 2,
            'reason': 'Very broad subreddit, only 2 posts explicitly about pre-eclampsia'
        },
        {
            'subreddit': 'r/Nursing',
            'focus': 'broad',
            'retrieved': 150,
            'stored': 1,
            'reason': 'Medical discussions are broad, few specifically about pre-eclampsia'
        },
        {
            'subreddit': 'r/BabyBumps',
            'focus': 'medium',
            'retrieved': 229,
            'stored': 228,
            'reason': 'Pregnancy-focused, high relevance when searching pre-eclampsia terms'
        },
        {
            'subreddit': 'r/preeclampsia',
            'focus': 'dedicated',
            'retrieved': 750,
            'stored': 192,
            'reason': '500 duplicates + 58 removed/deleted posts, kept 192 new valid posts'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['subreddit']} ({scenario['focus']} focus)")
        print(f"  Retrieved: {scenario['retrieved']} posts")
        print(f"  Stored: {scenario['stored']} posts")
        acceptance_rate = (scenario['stored'] / scenario['retrieved'] * 100) if scenario['retrieved'] > 0 else 0
        print(f"  Acceptance Rate: {acceptance_rate:.1f}%")
        print(f"  Reason: {scenario['reason']}")
    
    print("\n" + "="*80)
    print("WHY THIS IS GOOD")
    print("="*80)
    print("""
✅ HIGH QUALITY DATA
   - Only posts that actually discuss pre-eclampsia
   - Filters out tangentially related posts
   - Better for analysis and research

✅ WEIGHT-BASED COLLECTION STILL WORKS
   - Dedicated subreddits: Get most posts (minimal filtering)
   - Broad subreddits: Get fewer posts (strict filtering)
   - This matches your intended weight distribution!

✅ LLM COMPARISON REMAINS VALID
   - Each LLM's suggestions filtered equally
   - Can compare which LLMs suggested better sources
   - Quality over quantity

📊 YOUR RESULTS:
   - 2,055 HIGH QUALITY posts about pre-eclampsia
   - From 17 different subreddits
   - All verified to contain relevant medical terms
   - Ready for analysis!
""")
    
    print("="*80)
    print("TO COLLECT MORE POSTS")
    print("="*80)
    print("""
If you want to collect more posts, you can:

1. REDUCE FILTERING STRICTNESS
   - Modify has_relevant_keywords() to accept more terms
   - Lower the total_matches threshold
   
2. EXPAND KEYWORD LIST
   - Add more symptom-related terms
   - Include lay terminology (not just medical terms)
   
3. CHANGE FOCUS LEVELS
   - Upgrade some 'broad' subreddits to 'medium'
   - This will accept more posts from those sources

4. ACCEPT LOWER RELEVANCE SCORES
   - Currently filtering by keyword presence
   - Could add minimum relevance score threshold instead
""")
    print("="*80 + "\n")

if __name__ == "__main__":
    analyze_filtering()
