"""
Quick test script to verify EDA modules are working
"""

import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn")
    except ImportError as e:
        print(f"✗ seaborn: {e}")
        return False
    
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("✓ vaderSentiment")
    except ImportError as e:
        print(f"✗ vaderSentiment: {e}")
        return False
    
    try:
        from wordcloud import WordCloud
        print("✓ wordcloud")
    except ImportError as e:
        print(f"✗ wordcloud: {e}")
        return False
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        import contractions
        print("✓ contractions")
    except ImportError as e:
        print(f"✗ contractions: {e}")
        return False
    
    try:
        import emoji
        print("✓ emoji")
    except ImportError as e:
        print(f"✗ emoji: {e}")
        return False
    
    return True

def test_modules():
    """Test if custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from data_cleaning import DataCleaner
        print("✓ data_cleaning.py")
    except ImportError as e:
        print(f"✗ data_cleaning.py: {e}")
        return False
    
    try:
        from sentiment_analysis import SentimentAnalyzer
        print("✓ sentiment_analysis.py")
    except ImportError as e:
        print(f"✗ sentiment_analysis.py: {e}")
        return False
    
    try:
        from temporal_analysis import TemporalAnalyzer
        print("✓ temporal_analysis.py")
    except ImportError as e:
        print(f"✗ temporal_analysis.py: {e}")
        return False
    
    try:
        from covid_comparison import CovidComparison
        print("✓ covid_comparison.py")
    except ImportError as e:
        print(f"✗ covid_comparison.py: {e}")
        return False
    
    try:
        from overall_eda import OverallEDA
        print("✓ overall_eda.py")
    except ImportError as e:
        print(f"✗ overall_eda.py: {e}")
        return False
    
    return True

def test_vader():
    """Test VADER sentiment analyzer"""
    print("\nTesting VADER sentiment analysis...")
    
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # Test positive sentence
        pos_test = "I am so happy and excited about my healthy pregnancy!"
        pos_scores = analyzer.polarity_scores(pos_test)
        print(f"  Positive test: compound={pos_scores['compound']:.3f}")
        
        # Test negative sentence
        neg_test = "I am scared and worried about preeclampsia complications."
        neg_scores = analyzer.polarity_scores(neg_test)
        print(f"  Negative test: compound={neg_scores['compound']:.3f}")
        
        if pos_scores['compound'] > 0 and neg_scores['compound'] < 0:
            print("✓ VADER working correctly")
            return True
        else:
            print("✗ VADER not classifying correctly")
            return False
    except Exception as e:
        print(f"✗ VADER test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("EDA PIPELINE - MODULE VERIFICATION TEST")
    print("="*60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n❌ Some required packages are missing!")
        print("Run: pip install -r requirements.txt")
    
    # Test custom modules
    if not test_modules():
        all_passed = False
        print("\n❌ Some custom modules have import errors!")
    
    # Test VADER
    if not test_vader():
        all_passed = False
        print("\n❌ VADER sentiment analysis not working correctly!")
    
    # Final result
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready to run EDA pipeline!")
        print("\nRun: python run_eda.py")
    else:
        print("❌ SOME TESTS FAILED - Please fix errors above")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
