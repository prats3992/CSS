# Pre-eclampsia Reddit Data Analysis - EDA Pipeline

Comprehensive exploratory data analysis (EDA) and data cleaning pipeline for pre-eclampsia Reddit discussion data.

## Overview

This pipeline analyzes Reddit posts and comments related to pre-eclampsia across 23 pregnancy and health-related subreddits, performing sentiment analysis, temporal trends analysis, COVID-19 impact comparison, and comprehensive text analytics.

## Project Structure

```
CSS/
├── config.py                      # Configuration (subreddits, keywords, weights)
├── firebase_manager.py            # Firebase database connection
├── data_collector_praw.py         # Reddit data collection (PRAW)
├── data_collector.py              # Reddit data collection (Pushshift)
│
├── data_cleaning.py               # Data cleaning and preprocessing
├── sentiment_analysis.py          # VADER sentiment analysis
├── temporal_analysis.py           # Temporal trend analysis
├── covid_comparison.py            # Pre vs Post COVID comparison
├── overall_eda.py                 # Overall EDA with TF-IDF and topic modeling
├── run_eda.py                     # Main orchestrator script
│
├── cleaned_data/                  # Output: cleaned datasets
├── analysis_output/               # Output: visualizations and reports
│   ├── temporal/
│   ├── covid_comparison/
│   ├── overall/
│   └── analysis_report.html
├── final_report.tex               # LaTeX source for final paper
├── final_report.pdf               # Compiled final paper
└── requirements.txt
```

## Quick Start

### 1. Run Complete EDA Pipeline

```powershell
python run_eda.py
```

This will execute all analyses in sequence and generate:
- Cleaned datasets
- 20+ visualizations
- Comprehensive HTML report

### 2. Run Individual Modules

```powershell
# Data cleaning only
python data_cleaning.py

# Sentiment analysis only
python sentiment_analysis.py

# Temporal analysis only
python temporal_analysis.py

# COVID comparison only
python covid_comparison.py

# Overall EDA only
python overall_eda.py
```

## Analysis Components

### 1. Data Cleaning (`data_cleaning.py`)
- Loads data from Firebase
- Removes duplicates
- Cleans text (URLs, emojis, contractions)
- Parses timestamps
- Adds COVID period flags
- Handles missing values

**Outputs:**
- `cleaned_data/cleaned_posts.csv`
- `cleaned_data/cleaned_comments.csv`
- `cleaned_data/data_summary.json`

### 2. Sentiment Analysis (`sentiment_analysis.py`)
- VADER sentiment scoring
- Compound, positive, negative, neutral scores
- Sentiment categorization (positive/negative/neutral)

**Outputs:**
- `cleaned_data/posts_with_sentiment.csv`
- `cleaned_data/comments_with_sentiment.csv`
- `cleaned_data/sentiment_summary.json`

### 3. Temporal Trend Analysis (`temporal_analysis.py`)
- Sentiment trends over years
- Post volume by year (colored by sentiment)
- Percentage of positive posts by year
- Year-over-year sentiment change
- Year-over-year wordcloud comparison

**Outputs:**
- `analysis_output/temporal/sentiment_trend_over_years.png`
- `analysis_output/temporal/post_volume_by_sentiment.png`
- `analysis_output/temporal/positive_percentage_by_year.png`
- `analysis_output/temporal/yoy_sentiment_change.png`
- `analysis_output/temporal/wordcloud_by_year.png`

### 4. COVID Comparison Analysis (`covid_comparison.py`)
- Sentiment distribution (pre vs post COVID)
- Average VADER score comparison
- Post volume difference
- Sentiment score distributions (histograms, KDE, CDF)
- Post distribution across subreddits
- Wordcloud difference

**Outputs:**
- `analysis_output/covid_comparison/sentiment_distribution_comparison.png`
- `analysis_output/covid_comparison/vader_score_comparison.png`
- `analysis_output/covid_comparison/post_volume_comparison.png`
- `analysis_output/covid_comparison/sentiment_score_distributions.png`
- `analysis_output/covid_comparison/subreddit_distribution_comparison.png`
- `analysis_output/covid_comparison/wordcloud_comparison.png`

### 5. Overall EDA (`overall_eda.py`)
- Overall sentiment distribution
- Average sentiment by subreddit
- TF-IDF analysis (top terms extraction)
- Topic modeling (LDA with 5 topics)
- Subreddit comparison using TF-IDF
- Medical terms wordcloud and frequency analysis

**Outputs:**
- `analysis_output/overall/overall_sentiment_distribution.png`
- `analysis_output/overall/sentiment_by_subreddit.png`
- `analysis_output/overall/tfidf_top_terms.png`
- `analysis_output/overall/topic_modeling.png`
- `analysis_output/overall/subreddit_tfidf_comparison.png`
- `analysis_output/overall/medical_terms_analysis.png`
- `analysis_output/overall/sentiment_by_subreddit.csv`
- `analysis_output/overall/tfidf_scores.csv`
- `analysis_output/overall/medical_terms_frequency.csv`

### 6. Comprehensive Report (`run_eda.py`)
Generates an interactive HTML report with:
- Executive summary with key metrics
- All visualizations embedded
- Statistical summaries
- Key findings
- Methodology documentation

**Output:**
- `analysis_output/analysis_report.html`

## Analyses Performed

### Temporal Trends
1. **Sentiment over Time**: Average compound scores, pos/neg/neu components
2. **Post Volume**: Stacked bar charts by sentiment category
3. **Positive Percentage**: Yearly trends in positive sentiment
4. **YoY Changes**: Absolute and percentage changes in sentiment
5. **Wordclouds**: Evolution of terminology across years

### COVID-19 Impact
1. **Distribution Shift**: How sentiment categories changed
2. **Statistical Testing**: T-tests for significance
3. **Volume Analysis**: Monthly trends with COVID marker
4. **Score Distributions**: Histograms, KDE, CDF comparisons
5. **Subreddit Patterns**: How communities responded differently
6. **Language Changes**: Wordcloud comparison

### Overall Patterns
1. **Sentiment Overview**: Posts vs comments comparison
2. **Subreddit Rankings**: Which communities are most positive/negative
3. **Key Terms**: TF-IDF identifies most important words
4. **Topics**: LDA discovers main discussion themes
5. **Community Differences**: TF-IDF comparison across subreddits
6. **Medical Language**: Frequency of medical terminology

## Technical Details

### Dependencies
- pandas, numpy: Data manipulation
- matplotlib, seaborn: Visualizations
- wordcloud: Word cloud generation
- vaderSentiment: Sentiment analysis
- sklearn: TF-IDF, topic modeling (LDA)
- nltk, spacy: NLP utilities
- contractions, emoji: Text cleaning
- firebase-admin: Database connection

### Key Parameters
- **COVID Start Date**: March 1, 2020
- **Sentiment Thresholds**: 
  - Positive: compound >= 0.05
  - Negative: compound <= -0.05
  - Neutral: -0.05 < compound < 0.05
- **TF-IDF**: max_features=100, min_df=5, max_df=0.8
- **Topic Modeling**: 5 topics, LDA algorithm
- **Wordclouds**: max_words=100

## Output Files Summary

### Data Files
| File | Description |
|------|-------------|
| `cleaned_posts.csv` | Cleaned posts with metadata |
| `cleaned_comments.csv` | Cleaned comments with metadata |
| `posts_with_sentiment.csv` | Posts with sentiment scores |
| `comments_with_sentiment.csv` | Comments with sentiment scores |
| `sentiment_by_subreddit.csv` | Sentiment statistics by subreddit |
| `tfidf_scores.csv` | TF-IDF term scores |
| `medical_terms_frequency.csv` | Medical terminology frequency |

### Visualizations
- **5** temporal trend visualizations
- **6** COVID comparison visualizations
- **6** overall EDA visualizations
- **1** comprehensive HTML report

Total: **18+ visualizations**

## Usage Examples

### Example 1: Quick Analysis
```python
from run_eda import main
main()
```

### Example 2: Custom Analysis
```python
from data_cleaning import DataCleaner
from sentiment_analysis import SentimentAnalyzer

# Load and clean data
cleaner = DataCleaner()
posts_df, comments_df = cleaner.run_full_cleaning()

# Analyze sentiment
analyzer = SentimentAnalyzer()
posts_df, comments_df = analyzer.run_full_analysis(posts_df, comments_df)

# Your custom analysis here
print(posts_df[['title', 'sentiment_category', 'sentiment_compound']].head())
```

### Example 3: Specific Time Period
```python
from temporal_analysis import TemporalAnalyzer

temporal = TemporalAnalyzer()
# Filter data to specific years
filtered_posts = posts_df[posts_df['year'].between(2020, 2023)]
results = temporal.run_full_analysis(filtered_posts)
```

## Key Features

- **Complete Automation**: Single command runs entire pipeline  
- **Modular Design**: Each analysis can run independently  
- **Rich Visualizations**: 18+ publication-ready plots  
- **Statistical Rigor**: T-tests, distributions, confidence intervals  
- **COVID Analysis**: Dedicated pre/post pandemic comparison  
- **Text Analytics**: TF-IDF, LDA topic modeling  
- **Medical Focus**: Custom medical terminology extraction  
- **HTML Report**: Professional, shareable analysis report  

## Notes

- Ensure Firebase connection is configured before running
- First run may take several minutes depending on data size
- All visualizations are saved at 300 DPI for publication quality
- HTML report includes all visualizations and can be shared standalone

## Next Steps

After running the EDA, you can:
1. Review the HTML report for insights
2. Use the cleaned CSVs for further analysis
3. Fine-tune parameters in individual modules
4. Export specific visualizations for presentations
5. Use topic modeling results for content categorization

## Support

For issues or questions about the EDA pipeline, check:
- Individual module docstrings for detailed parameter info
- Config file for subreddit weights and keyword lists
- Output logs for processing statistics

---

**Analysis Date**: November 2025  
**Data Period**: 2013-2025 (Pre-eclampsia subreddit inception to present)  
**Framework**: Python 3.x with pandas, scikit-learn, VADER
