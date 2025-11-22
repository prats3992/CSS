# New Analysis Modules - Quick Start Guide

## Overview
This document covers the 5 new analysis modules added to the pre-eclampsia Reddit analysis project.

## What's New?

### 1. **Exploratory Data Analysis (EDA)** - `eda_analysis.py`
**Purpose:** Understand temporal trends and identify key terms using TF-IDF

**Features:**
- Line plots for posts and comments per year
- Monthly post volume trends
- TF-IDF analysis with unigram and bigram extraction
- Word clouds weighted by TF-IDF scores
- Top 20 terms visualization

**Outputs:**
- `eda_output/temporal_line_plots.png` - 4-panel visualization
- `eda_output/temporal_statistics.txt` - Detailed statistics
- `eda_output/tfidf_wordclouds.png` - Unigram & bigram word clouds
- `eda_output/tfidf_results.txt` - Top 50 unigrams and bigrams

**Run:**
```python
from eda_analysis import EDAAnalyzer
analyzer = EDAAnalyzer()
analyzer.run_complete_eda()
```

---

### 2. **VADER Validation** - Enhanced `sentiment_analysis.py`
**Purpose:** Validate VADER accuracy with examples and compare to BERT

**Features:**
- Random sampling of positive, negative, and neutral posts
- Manual validation assessment for each example
- VADER component scores breakdown
- VADER vs TextBlob correlation analysis
- VADER vs BERT comparison discussion

**Outputs:**
- `sentiment_analysis_output/vader_validation_examples.txt`
  - 10 examples per sentiment category
  - Detailed scoring breakdown
  - Manual validation notes
  - BERT comparison analysis

**Run:**
```python
from sentiment_analysis import SentimentAnalyzer
analyzer = SentimentAnalyzer()
analyzer.run_sentiment_analysis()  # If not done already
analyzer.validate_vader_with_examples()
```

**Key Finding:** VADER is appropriate for this analysis because:
- Optimized for social media text
- Fast processing for large datasets
- Enhanced with medical lexicon adjustments
- Provides explainable component scores

---

### 3. **Engagement Analysis** - `engagement_analysis.py`
**Purpose:** Determine which sentiment gets more engagement (upvotes, comments)

**Features:**
- Average engagement by sentiment (positive/neutral/negative)
- Statistical significance testing (One-Way ANOVA)
- Correlation analysis (sentiment vs engagement)
- Box plots for engagement distribution
- Scatter plots: sentiment vs upvotes/comments

**Outputs:**
- `engagement_output/sentiment_engagement_analysis.png` - 9-panel visualization
- `engagement_output/engagement_analysis_report.txt` - Statistical results

**Key Questions Answered:**
- Which sentiment gets more upvotes?
- Which sentiment gets more comments?
- Is the difference statistically significant?
- How strong is the correlation?

**Run:**
```python
from engagement_analysis import EngagementAnalyzer
analyzer = EngagementAnalyzer()
analyzer.analyze_sentiment_engagement()
```

---

### 4. **User Behavior Analysis** - `user_analysis.py`
**Purpose:** Analyze user overlap, posting frequency, and what active users discuss

**Features:**
- Posting frequency histogram
- User categorization (one-time, occasional, regular, power users)
- User overlap heatmap across subreddits
- Top 20 most active users
- Topic analysis: active users vs one-time posters
- Word clouds for different user types

**Outputs:**
- `user_analysis_output/user_behavior_analysis.png` - 6-panel visualization
- `user_analysis_output/active_users_topics.png` - Topic comparison
- `user_analysis_output/user_behavior_report.txt` - Detailed statistics
- `user_analysis_output/user_topics_comparison.txt` - Word frequency lists

**Key Insights:**
- How many one-time vs repeat posters?
- Who are the most active users?
- Do users post in multiple subreddits?
- What do active users discuss differently?

**Run:**
```python
from user_analysis import UserAnalyzer
analyzer = UserAnalyzer()
analyzer.analyze_user_behavior()
```

---

### 5. **Temporal Sentiment Shifts** - Enhanced `sentiment_analysis.py`
**Purpose:** Analyze how sentiment changes year-over-year

**Features:**
- Year-by-year sentiment trend analysis
- Linear regression for overall trend
- Year-over-year change visualization
- Statistical significance testing
- COVID-19 impact comparison
- Sentiment distribution heatmap by year

**Outputs:**
- `sentiment_analysis_output/temporal_sentiment_shifts.png` - 6-panel visualization
- `sentiment_analysis_output/temporal_shift_report.txt` - Detailed year-by-year analysis

**Key Questions Answered:**
- Is sentiment improving or declining over time?
- Are changes statistically significant?
- What was the COVID-19 impact?
- Which years were most positive/negative?

**Run:**
```python
from sentiment_analysis import SentimentAnalyzer
analyzer = SentimentAnalyzer()
analyzer.run_sentiment_analysis()  # If not done already
analyzer.analyze_temporal_sentiment_shifts()
```

---

## Quick Start: Run Everything

### Option 1: Master Script (Recommended)
Run all 5 new analyses in one command:
```bash
python run_all_analyses.py
```

This will:
1. Run EDA with TF-IDF analysis
2. Run sentiment analysis with VADER validation
3. Analyze engagement patterns
4. Analyze user behavior
5. Analyze temporal sentiment shifts

**Estimated Time:** 5-15 minutes depending on dataset size

### Option 2: Individual Modules
Run analyses separately:

```python
# 1. EDA
from eda_analysis import EDAAnalyzer
EDAAnalyzer().run_complete_eda()

# 2. Sentiment Analysis (complete)
from sentiment_analysis import SentimentAnalyzer
analyzer = SentimentAnalyzer()
analyzer.run_sentiment_analysis()
analyzer.create_visualizations()
analyzer.validate_vader_with_examples()
analyzer.analyze_temporal_sentiment_shifts()
analyzer.create_precovid_postcovid_analysis()
analyzer.save_results()

# 3. Engagement Analysis
from engagement_analysis import EngagementAnalyzer
EngagementAnalyzer().analyze_sentiment_engagement()

# 4. User Behavior Analysis
from user_analysis import UserAnalyzer
UserAnalyzer().analyze_user_behavior()
```

---

## Requirements

All modules use the same dependencies as the main project:
- pandas
- numpy
- matplotlib
- seaborn
- vaderSentiment
- textblob
- wordcloud
- scikit-learn (for TF-IDF)
- scipy (for statistical tests)

No additional installations needed!

---

## Data Requirements

### Required CSV Columns:
- `author` - Username (for user analysis)
- `created_utc` - Unix timestamp
- `text_cleaned` or `text_no_stopwords` - Cleaned text
- `score` - Upvotes (for engagement analysis)
- `num_comments` - Comment count (for engagement analysis)
- `subreddit` - Subreddit name (for overlap analysis)

### For Sentiment Modules:
After running `sentiment_analysis.py`, these columns are added:
- `vader_compound`, `vader_pos`, `vader_neg`, `vader_neu`
- `textblob_polarity`, `textblob_subjectivity`
- `sentiment_class` (positive/negative/neutral)

---

## Output Structure

```
project_root/
├── eda_output/
│   ├── temporal_line_plots.png
│   ├── temporal_statistics.txt
│   ├── tfidf_wordclouds.png
│   └── tfidf_results.txt
│
├── sentiment_analysis_output/
│   ├── vader_validation_examples.txt
│   ├── temporal_sentiment_shifts.png
│   ├── temporal_shift_report.txt
│   ├── [existing sentiment outputs...]
│
├── engagement_output/
│   ├── sentiment_engagement_analysis.png
│   └── engagement_analysis_report.txt
│
└── user_analysis_output/
    ├── user_behavior_analysis.png
    ├── active_users_topics.png
    ├── user_behavior_report.txt
    └── user_topics_comparison.txt
```

---

## Interpretation Guide

### 1. EDA Results
- **Temporal plots:** Look for growth trends, seasonal patterns, COVID-19 impact
- **TF-IDF:** High scores = distinctive terms for this corpus
- **Bigrams:** Common two-word medical phrases

### 2. VADER Validation
- **Strong sentiment** (|compound| > 0.3): High confidence
- **Borderline** (0.05 < |compound| < 0.15): May vary with context
- **Correlation > 0.7:** Strong agreement with TextBlob

### 3. Engagement Analysis
- **P-value < 0.05:** Statistically significant difference
- **Higher comments:** Often indicates support-seeking posts
- **Higher upvotes:** Often indicates positive/success stories

### 4. User Behavior
- **One-time posters > 50%:** Typical for health communities
- **Top 10% contribution:** Shows core community engagement
- **Overlap heatmap:** Identifies cross-posting users

### 5. Temporal Shifts
- **Positive slope:** Sentiment improving over time
- **P-value < 0.05:** Trend is statistically significant
- **COVID-19 impact:** Compare pre/post 2020 averages

---

## Troubleshooting

### Error: "No cleaned CSV file found"
**Solution:** Run `data_cleaning.py` first to create cleaned data

### Error: "Missing required columns"
**Solution:** Ensure your CSV has `author`, `created_utc`, `score`, `num_comments`

### Error: "No sentiment analysis results found"
**Solution:** Run `sentiment_analysis.py` first before engagement analysis

### Plots look crowded
**Solution:** The code automatically adjusts for data volume, but you can:
- Increase `figsize` in module code
- Increase `dpi` parameter (default 300)
- Filter to specific time periods

---

## Customization

### Change Sample Size for VADER Validation:
```python
analyzer.validate_vader_with_examples(n_samples=20)  # Default: 10
```

### Change TF-IDF Parameters:
```python
analyzer.perform_tfidf_analysis(max_features=300)  # Default: 200
```

### Change User Category Thresholds:
Edit line in `user_analysis.py`:
```python
bins=[0, 1, 5, 20, float('inf')]  # Adjust these numbers
```

---

## Research Questions Answered

✅ **Which sentiment gets more engagement?** → `engagement_analysis.py`

✅ **How accurate is VADER?** → `validate_vader_with_examples()`

✅ **What are the key terms?** → TF-IDF analysis in `eda_analysis.py`

✅ **How many active vs one-time users?** → `user_analysis.py`

✅ **Is sentiment changing over time?** → `analyze_temporal_sentiment_shifts()`

✅ **Do users overlap across subreddits?** → User overlap heatmap

✅ **What do active users post about?** → Active user topic analysis

---

## Tips for Analysis

1. **Run in order:** EDA → Sentiment → Engagement → User → Temporal
2. **Check statistics files** for numerical details beyond visualizations
3. **Compare word clouds** across different groups to find unique themes
4. **Look for COVID-19 patterns** in temporal analyses
5. **Cross-reference findings** across modules for deeper insights

---

## Next Steps

After running all analyses, you can:
1. Extract specific insights for your research paper
2. Create custom visualizations by modifying module code
3. Filter data to specific time periods or subreddits
4. Perform additional statistical tests
5. Compare with other health communities

---

## Questions?

Refer to the main `README.md` for:
- Original project setup
- Data collection details
- Existing analysis modules
- Full feature list

All new modules follow the same patterns as existing code for consistency!

---

## Module File Summary

| File | Purpose | Key Output |
|------|---------|------------|
| `eda_analysis.py` | Temporal trends + TF-IDF | Line plots, word clouds |
| `engagement_analysis.py` | Sentiment vs engagement | Statistical tests, correlations |
| `user_analysis.py` | User behavior patterns | Frequency, overlap, topics |
| `run_all_analyses.py` | Master pipeline | Runs everything |

**Updates to existing files:**
- `sentiment_analysis.py`: Added VADER validation + temporal shifts
- Both new methods integrate seamlessly with existing pipeline

---

**Happy Analyzing! 📊**
