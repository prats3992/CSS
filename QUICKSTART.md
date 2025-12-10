# Quick Start Guide - EDA Pipeline

## Getting Started (5 minutes)

### Step 1: Run Complete Analysis
```powershell
python run_eda.py
```
This will:
- Load data from Firebase
- Clean and preprocess
- Perform sentiment analysis
- Generate 18+ visualizations
- Create HTML report

**Estimated time**: 5-15 minutes (depends on data size)

### Step 2: Compile Final Report
```powershell
pdflatex final_report.tex
```
This will generate the final PDF report (`final_report.pdf`) incorporating the latest analysis results.

### Step 3: View Results
Open the generated report:
```powershell
# Windows
start analysis_output/analysis_report.html

# Or manually navigate to:
# analysis_output/analysis_report.html
```

---

## What You'll Get

### Cleaned Data
- `cleaned_data/cleaned_posts.csv` - Ready-to-analyze posts
- `cleaned_data/posts_with_sentiment.csv` - Posts with sentiment scores

### Visualizations (18 total)

#### Temporal Trends (5 charts)
1. Sentiment trends over years
2. Post volume by sentiment category
3. Positive posts percentage by year
4. Year-over-year sentiment changes
5. Word clouds by year

#### COVID Comparison (6 charts)
1. Sentiment distribution comparison
2. VADER score comparison (box + violin + bar)
3. Post volume over time
4. Detailed score distributions (histogram + KDE + CDF)
5. Subreddit distribution changes
6. Word cloud comparison

#### Overall EDA (6 charts)
1. Overall sentiment distribution (posts vs comments)
2. Sentiment by subreddit (top 20)
3. Top TF-IDF terms
4. Topic modeling (5 topics)
5. Subreddit TF-IDF comparison
6. Medical terms analysis + word cloud

### Report
- `analysis_output/analysis_report.html` - Interactive HTML report with all visualizations
- `final_report.pdf` - Final academic paper

---

## Common Commands

### Run Individual Analyses

```powershell
# Just clean the data
python data_cleaning.py

# Just sentiment analysis
python sentiment_analysis.py

# Just temporal trends
python temporal_analysis.py

# Just COVID comparison
python covid_comparison.py

# Just overall EDA
python overall_eda.py
```

### Custom Analysis Example

```python
# custom_analysis.py
from data_cleaning import DataCleaner
from sentiment_analysis import SentimentAnalyzer

# Load and prepare data
cleaner = DataCleaner()
posts_df, comments_df = cleaner.run_full_cleaning(save_output=False)

analyzer = SentimentAnalyzer()
posts_df, _ = analyzer.run_full_analysis(posts_df, comments_df, save_output=False)

# Your custom code here
# Filter posts from specific subreddit
preeclampsia_posts = posts_df[posts_df['subreddit'] == 'preeclampsia']

# Analyze sentiment over time
yearly_sentiment = preeclampsia_posts.groupby('year')['sentiment_compound'].mean()
print(yearly_sentiment)
```

---

## Analyses Checklist

### Temporal Trends
- [x] Sentiment trend over the years
- [x] Post volume over the years (colored by sentiment)
- [x] % positive posts by year
- [x] Year over year sentiment change
- [x] Year over year change in wordcloud

### Pre vs Post COVID
- [x] Sentiment distribution
- [x] Avg VADER score
- [x] Post volume difference
- [x] Sentiment score distributions
- [x] Post distribution across subreddits
- [x] Wordcloud difference

### Overall
- [x] Sentiment distribution
- [x] Avg sentiment by subreddit
- [x] TF-IDF inclusion -> topic modeling
- [x] Subreddit comparison
- [x] Wordcloud of medical terms being used

---

## Output Structure

```
analysis_output/
├── analysis_report.html          # Main report
├── temporal/                     # 5 visualizations
│   ├── sentiment_trend_over_years.png
│   ├── post_volume_by_sentiment.png
│   ├── positive_percentage_by_year.png
│   ├── yoy_sentiment_change.png
│   └── wordcloud_by_year.png
├── covid_comparison/             # 6 visualizations
│   ├── sentiment_distribution_comparison.png
│   ├── vader_score_comparison.png
│   ├── post_volume_comparison.png
│   ├── sentiment_score_distributions.png
│   ├── subreddit_distribution_comparison.png
│   └── wordcloud_comparison.png
└── overall/                      # 6 visualizations + 3 CSVs
    ├── overall_sentiment_distribution.png
    ├── sentiment_by_subreddit.png
    ├── tfidf_top_terms.png
    ├── topic_modeling.png
    ├── subreddit_tfidf_comparison.png
    ├── medical_terms_analysis.png
    ├── sentiment_by_subreddit.csv
    ├── tfidf_scores.csv
    └── medical_terms_frequency.csv

cleaned_data/
├── cleaned_posts.csv
├── cleaned_comments.csv
├── posts_with_sentiment.csv
├── comments_with_sentiment.csv
├── data_summary.json
└── sentiment_summary.json
```

---

## Troubleshooting

### Error: "No posts data available"
**Solution**: Run data collection first:
```powershell
python data_collector_praw.py
```

### Error: "Missing required columns"
**Solution**: Ensure data cleaning ran successfully. Check `cleaned_data/` folder exists.

### Error: ImportError for packages
**Solution**: Install dependencies:
```powershell
pip install -r requirements.txt
```

### Visualizations not generating
**Solution**: Check if matplotlib backend is configured:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

### Firebase connection error
**Solution**: 
1. Check `.env` file has correct `FIREBASE_DATABASE_URL`
2. Verify `pre-eclampsia-analysis-firebase-adminsdk-*.json` exists

---

## Tips

1. **First Run**: May take 10-15 minutes depending on data size
2. **Subsequent Runs**: Use cached cleaned data for faster analysis
3. **Memory**: Large datasets may require 4GB+ RAM
4. **Visualizations**: All saved at 300 DPI for publication quality
5. **Report**: HTML report can be shared - all images are embedded/linked

---

## Key Metrics in Report

- **Total Posts**: Count of analyzed posts
- **Total Comments**: Count of analyzed comments
- **Subreddits**: Number of unique communities
- **Average Sentiment**: Overall compound score
- **Date Range**: Start to end of data collection
- **COVID Impact**: Pre vs post pandemic comparison

---

## Next Steps After EDA

1. **Review HTML Report**: Get overview of all findings
2. **Check CSVs**: Use cleaned data for custom analyses
3. **Examine Topics**: Review LDA topics for themes
4. **Medical Terms**: Check frequency of clinical terminology
5. **Subreddit Insights**: Identify most positive/negative communities
6. **Time Patterns**: Note sentiment changes over years

---

## Verification Checklist

Before running analysis, ensure:
- [ ] Firebase connection configured
- [ ] `.env` file exists with correct values
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Data has been collected (posts exist in Firebase)
- [ ] Sufficient disk space (500MB+ recommended)

After analysis, verify:
- [ ] `cleaned_data/` folder created
- [ ] `analysis_output/` folder with 3 subfolders
- [ ] 18+ PNG files generated
- [ ] `analysis_report.html` opens in browser
- [ ] No error messages in console

---

**Ready to analyze?** Run: `python run_eda.py`
