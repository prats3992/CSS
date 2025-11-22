"""
Master Analysis Pipeline - Run All New Analyses
Executes EDA, VADER validation, engagement analysis, user analysis, and temporal shifts
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {title:^78} ")
    print("="*80 + "\n")


def run_complete_analysis():
    """Run all new analysis modules"""
    start_time = datetime.now()
    
    print_header("COMPREHENSIVE ANALYSIS PIPELINE")
    print("This script will run:")
    print("  1. Exploratory Data Analysis (EDA)")
    print("  2. Sentiment Analysis with VADER validation")
    print("  3. Engagement Analysis")
    print("  4. User Behavior Analysis")
    print("\nStarted at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("\n" + "-"*80)
    
    results_summary = []
    
    # 1. Run EDA
    try:
        print_header("1/4: EXPLORATORY DATA ANALYSIS")
        from eda_analysis import EDAAnalyzer
        
        eda = EDAAnalyzer()
        eda.run_complete_eda()
        
        results_summary.append("✓ EDA: Temporal line plots + TF-IDF word clouds")
    except Exception as e:
        print(f"\n❌ ERROR in EDA: {e}")
        results_summary.append(f"❌ EDA: Failed - {e}")
    
    # 2. Run Sentiment Analysis (with VADER validation and temporal shifts)
    try:
        print_header("2/4: SENTIMENT ANALYSIS")
        from sentiment_analysis import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        
        # Check if sentiment analysis already done
        if 'vader_compound' not in analyzer.df.columns:
            print("Running complete sentiment analysis...")
            analyzer.run_sentiment_analysis()
            analyzer.create_visualizations()
            analyzer.create_precovid_postcovid_analysis()
        else:
            print("Sentiment scores already exist, skipping basic analysis...")
        
        # Run new analyses
        print("\nRunning VADER validation...")
        analyzer.validate_vader_with_examples()
        
        print("\nRunning temporal sentiment shift analysis...")
        analyzer.analyze_temporal_sentiment_shifts()
        
        analyzer.save_results()
        
        results_summary.append("✓ Sentiment: VADER validation + Temporal shifts")
    except Exception as e:
        print(f"\n❌ ERROR in Sentiment Analysis: {e}")
        results_summary.append(f"❌ Sentiment Analysis: Failed - {e}")
    
    # 3. Run Engagement Analysis
    try:
        print_header("3/4: ENGAGEMENT ANALYSIS")
        from engagement_analysis import EngagementAnalyzer
        
        engagement = EngagementAnalyzer()
        engagement.analyze_sentiment_engagement()
        
        results_summary.append("✓ Engagement: Sentiment vs upvotes/comments")
    except Exception as e:
        print(f"\n❌ ERROR in Engagement Analysis: {e}")
        results_summary.append(f"❌ Engagement Analysis: Failed - {e}")
    
    # 4. Run User Analysis
    try:
        print_header("4/4: USER BEHAVIOR ANALYSIS")
        from user_analysis import UserAnalyzer
        
        user = UserAnalyzer()
        user.analyze_user_behavior()
        
        results_summary.append("✓ User Analysis: Overlap + Frequency + Topics")
    except Exception as e:
        print(f"\n❌ ERROR in User Analysis: {e}")
        results_summary.append(f"❌ User Analysis: Failed - {e}")
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_header("ANALYSIS COMPLETE!")
    print("RESULTS SUMMARY:")
    print("-" * 80)
    for result in results_summary:
        print(f"  {result}")
    
    print("\n" + "-" * 80)
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {int(duration // 60)}m {int(duration % 60)}s")
    
    print("\n" + "="*80)
    print(" "*25 + "CHECK OUTPUT FOLDERS:")
    print("="*80)
    print("  • eda_output/")
    print("  • sentiment_analysis_output/")
    print("  • engagement_output/")
    print("  • user_analysis_output/")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    run_complete_analysis()
