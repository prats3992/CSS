# Pre-eclampsia Reddit Data Analysis Pipeline

This project collects and analyzes Reddit discussions about pre-eclampsia from June 2013 to present. The pipeline includes data collection, sentiment analysis, word cloud generation, comprehensive temporal analysis, engagement patterns, and user behavior analysis.

## 🆕 Latest Updates (December 2025)

### New Analysis Modules Added:
- ✅ **EDA with TF-IDF**: Temporal line plots + TF-IDF word clouds (unigrams & bigrams)
- ✅ **VADER Validation**: Accuracy assessment with examples + VADER vs BERT comparison
- ✅ **Engagement Analysis**: Which sentiment gets more upvotes/comments (statistical tests)
- ✅ **User Behavior**: Posting frequency, user overlap, active user topics
- ✅ **Temporal Sentiment Shifts**: Year-over-year sentiment changes with statistical significance

📄 See `NEW_ANALYSES_GUIDE.md` for complete guide to new features!

### Previous Updates (November 2025):
- ✅ **Word Categorization**: Automatic classification into 8 medical/psychological categories
- ✅ **Enhanced Explanations**: Comprehensive guides for all sentiment graphs
- ✅ **Pre-COVID vs Post-COVID Analysis**: Temporal comparison of discussions (2013-2019 vs 2020-2025)
- ✅ **Removed LLM Comparisons**: Focus on medical data, LLMs only used for keyword identification
- ✅ **TF-IDF Guide**: Complete integration guide for future keyword extraction
- ✅ **Detailed Insights**: Automatic generation of medical and community insights

📄 See `UPDATE_SUMMARY.md` for complete details of all previous changes.
📄 See `IMPLEMENTATION_SUMMARY.md` for technical details of new modules.

## Project Structure

```
CSS/
├── config.py                      # Configuration: subreddit weights, keywords, date ranges
├── firebase_manager.py            # Firebase data storage and retrieval
├── data_collector_praw.py         # Reddit data collection using PRAW
├── data_cleaning.py               # Text preprocessing and cleaning
├── sentiment_analysis.py          # Sentiment analysis with COVID comparison + validation + temporal shifts
├── wordcloud_generator.py         # Word clouds with categorization
├── eda_analysis.py                # EDA: temporal plots + TF-IDF analysis (NEW)
├── engagement_analysis.py         # Sentiment vs engagement analysis (NEW)
├── user_analysis.py               # User behavior and topic analysis (NEW)
├── run_all_analyses.py            # Master script to run all new analyses (NEW)
├── requirements.txt               # Python dependencies
├── NEW_ANALYSES_GUIDE.md          # Complete guide for new modules (NEW)
├── IMPLEMENTATION_SUMMARY.md      # Technical implementation details (NEW)
├── UPDATE_SUMMARY.md              # Detailed changelog (previous updates)
├── TFIDF_INTEGRATION_GUIDE.md     # TF-IDF implementation guide
├── eda_output/                    # EDA visualizations and statistics (NEW)
├── engagement_output/             # Engagement analysis results (NEW)
├── user_analysis_output/          # User behavior analysis (NEW)
├── sentiment_analysis_output/     # Sentiment visualizations and insights
├── wordcloud_output/              # Word clouds and category analysis
└── README.md                      # This file
```

## Features

### 1. Comprehensive Data Collection
- **Date Range**: June 29, 2013 (r/preeclampsia creation) to present
- **Subreddit Weights**: 0.3-1.0 based on medical relevance
  - High (1.0): r/preeclampsia, r/highriskpregnancy
  - Medium (0.5-0.8): r/BabyBumps, r/pregnant, r/birthstories
  - Broad (0.3-0.4): r/Parenting, r/TwoXChromosomes, r/Nursing
- **Smart Filtering**: Relevance scoring based on medical keywords

### 2. Advanced Sentiment Analysis
- **Multiple Algorithms**: VADER (social media optimized) + TextBlob
- **Medical Context**: Custom lexicon for pregnancy/pre-eclampsia terms
- **Sentiment Classification**: Positive, Neutral, Negative with thresholds
- **Temporal Analysis**: Pre-COVID (2013-2019) vs Post-COVID (2020-2025)
- **Medical Indicators**: Positive outcomes, complications, support-seeking, experience-sharing
- **VADER Validation**: Accuracy assessment with random examples (NEW)
- **Temporal Shifts**: Year-over-year sentiment trend analysis (NEW)

### 3. Exploratory Data Analysis (NEW)
- **Temporal Line Plots**: Posts and comments per year with COVID-19 markers
- **TF-IDF Analysis**: Extract distinctive unigrams and bigrams
- **Word Clouds**: TF-IDF-weighted visualization of key terms
- **Statistical Reports**: Detailed year-by-year statistics

### 4. Engagement Analysis (NEW)
- **Sentiment vs Upvotes**: Which sentiment gets more upvotes?
- **Sentiment vs Comments**: Which sentiment generates more discussion?
- **Statistical Tests**: One-Way ANOVA with p-values
- **Correlation Analysis**: Sentiment intensity vs engagement metrics
- **Box Plots & Scatter Plots**: Distribution and relationship visualization

### 5. User Behavior Analysis (NEW)
- **Posting Frequency**: Histogram of user activity patterns
- **User Categories**: One-time, occasional, regular, power users
- **User Overlap**: Heatmap showing cross-posting between subreddits
- **Active User Topics**: What do frequent posters discuss?
- **Topic Comparison**: Active users vs one-time posters word analysis
- **Top Contributors**: Identify the 20 most active users

### 3. Word Cloud Generation with Insights
- **8-Category Analysis**:
  - Symptoms (headaches, swelling, vision)
  - Diagnosis & Medical (preeclampsia, hypertension, tests)
  - Treatment & Interventions (magnesium, medications, hospital)
  - Pregnancy Outcomes (baby, birth, NICU, healthy)
  - Emotions & Experience (scared, grateful, support, story)
  - Complications & Risks (emergency, severe, critical)
  - Healthcare Providers (doctor, nurse, OB, specialist)
  - Support & Community (help, advice, question, anyone)
  
- **Multiple Visualizations**:
  - Overall word cloud
  - By subreddit (top 12)
  - By sentiment (positive/negative/neutral)
  - By time period (Pre-COVID vs Post-COVID)
  - Medical terms frequency analysis

### 4. Pre-COVID vs Post-COVID Analysis 🆕
- **COVID Boundary**: March 1, 2020
- **Comprehensive Comparisons**:
  - Sentiment distribution changes
  - Post volume growth metrics
  - Medical context indicator shifts
  - Terminology evolution
  - Community dynamics changes
  
- **Automated Insights**:
  - Statistical comparisons
  - Percentage changes
  - Growth rate calculations
  - Pattern identification

### 5. Comprehensive Documentation
- **Graph Explanations**: Detailed interpretation guides for all visualizations
- **Category Insights**: Automatic analysis of word usage patterns
- **Research Applications**: Identified opportunities for further study
- **TF-IDF Guide**: Future enhancement possibilities

## Setup Instructions

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

Required packages:
- pandas, numpy - Data manipulation
- matplotlib, seaborn - Visualizations
- wordcloud - Word cloud generation
- vaderSentiment, textblob - Sentiment analysis
- praw - Reddit API access
- firebase-admin - Database storage
- nltk - Natural language processing

### 2. Firebase Setup

#### Option A: Using Service Account (Recommended)
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project: `pre-eclampsia-analysis`
3. Go to Project Settings > Service Accounts
4. Click "Generate new private key"
5. Save as `firebase-credentials.json` in project directory
6. Update `firebase_manager.py` line 20:
   ```python
   cred = credentials.Certificate('firebase-credentials.json')
   ```

#### Option B: Run Setup Helper
```powershell
python service_account_setup.py
```

### 3. Reddit API Setup (for data collection)

Create a `.env` file with:
```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_secret_here
REDDIT_USER_AGENT=python:preeclampsia-analysis:v1.0 (by /u/yourusername)
```

## Usage

### 🚀 Complete Pipeline - From Data Collection to Analysis

Follow these steps in order for a complete analysis:

#### **Step 1: Data Collection** (Optional - if you need new data)
```powershell
python data_collector_praw.py
```
- Collects Reddit posts from June 2013 to present
- Applies weighted filtering based on subreddit relevance
- Stores raw data in Firebase
- **Output**: Data stored in Firebase database
- **Time**: 30-60 minutes (depending on date range)

#### **Step 2: Data Cleaning & Preprocessing** (Required)
```powershell
python data_cleaning.py
```
- Removes URLs, special characters, emojis
- Tokenizes and lemmatizes text
- Removes stopwords (preserves medical terms)
- Creates clean text columns
- **Output**: `cleaned_posts_[timestamp].csv`
- **Time**: 2-5 minutes (for 10,000 posts)

#### **Step 3: Sentiment Analysis** (Required for most other analyses)
```powershell
python sentiment_analysis.py
```
- Runs VADER and TextBlob sentiment analysis
- Classifies posts as positive/negative/neutral
- Creates Pre-COVID vs Post-COVID comparison
- Validates VADER accuracy with examples
- Analyzes year-over-year sentiment shifts
- **Output**: 
  - `sentiment_analysis_results_[timestamp].csv` (with sentiment scores)
  - Multiple visualizations in `sentiment_analysis_output/`
  - 6+ PNG files and 4+ TXT reports
- **Time**: 3-8 minutes (for 10,000 posts)

#### **Step 4: Word Cloud Generation** (Optional but recommended)
```powershell
python wordcloud_generator.py
```
- Generates word clouds by subreddit, sentiment, year, COVID period
- Categorizes words into 8 medical/psychological categories
- Creates medical terms frequency analysis
- **Output**: 8+ visualizations in `wordcloud_output/`
- **Time**: 5-10 minutes

#### **Step 5: Run All New Analyses** (Recommended - runs Steps 6-8 automatically)
```powershell
python run_all_analyses.py
```
This master script runs all new analyses in sequence:
- ✅ EDA with TF-IDF
- ✅ Engagement Analysis
- ✅ User Behavior Analysis
- **Output**: Outputs in `eda_output/`, `engagement_output/`, `user_analysis_output/`
- **Time**: 2-5 minutes total

**OR run them individually:**

#### **Step 6: Exploratory Data Analysis** (Individual)
```powershell
python eda_analysis.py
```
- Temporal line plots (posts/comments per year)
- TF-IDF analysis (unigrams and bigrams)
- Statistical summaries
- **Output**: 4 files in `eda_output/`
- **Time**: 30-60 seconds

#### **Step 7: Engagement Analysis** (Individual)
```powershell
python engagement_analysis.py
```
- Analyzes which sentiment gets more upvotes/comments
- Statistical significance testing (ANOVA)
- Correlation analysis
- **Output**: 2 files in `engagement_output/`
- **Time**: 20-40 seconds
- **Requires**: Step 3 (sentiment analysis) must be completed first

#### **Step 8: User Behavior Analysis** (Individual)
```powershell
python user_analysis.py
```
- Posting frequency histogram
- User overlap across subreddits
- Active user topic analysis
- **Output**: 4 files in `user_analysis_output/`
- **Time**: 40-80 seconds

---

### 📋 Quick Reference Guide

**First-time users (complete pipeline):**
```powershell
# 1. Collect data (if needed)
python data_collector_praw.py

# 2. Clean data
python data_cleaning.py

# 3. Run sentiment analysis
python sentiment_analysis.py

# 4. Run all new analyses
python run_all_analyses.py

# 5. (Optional) Generate word clouds
python wordcloud_generator.py
```

**Already have cleaned data? Start here:**
```powershell
# 1. Run sentiment analysis (if not done)
python sentiment_analysis.py

# 2. Run all new analyses
python run_all_analyses.py
```

**Already have sentiment results? Just run new analyses:**
```powershell
python run_all_analyses.py
```

**Total time estimate:**
- Data collection: 30-60 minutes (one-time)
- Data cleaning: 2-5 minutes
- Complete analysis pipeline: 10-20 minutes
- **Full first-time run: ~45-90 minutes**
- **Subsequent runs (with existing data): 10-20 minutes**

### Individual Components

#### Quick Run All New Analyses (Recommended)
```powershell
python run_all_analyses.py
```
This runs: EDA → Sentiment (with validation & shifts) → Engagement → User Analysis

#### Data Collection Only
```powershell
python data_collector_praw.py
```

This will:
- Collect posts from all configured subreddits (from June 2013)
- Apply weighted filtering based on medical relevance
- Store data in Firebase with metadata
- Track keyword matches

#### Sentiment Analysis with New Features
```python
from sentiment_analysis import SentimentAnalyzer

# Analyze latest cleaned data
analyzer = SentimentAnalyzer()
results = analyzer.run_sentiment_analysis()
analyzer.create_visualizations()
analyzer.create_precovid_postcovid_analysis()
analyzer.validate_vader_with_examples()  # NEW
analyzer.analyze_temporal_sentiment_shifts()  # NEW
analyzer.save_results()
```

#### EDA with TF-IDF (NEW)
```python
from eda_analysis import EDAAnalyzer

analyzer = EDAAnalyzer()
analyzer.run_complete_eda()  # Line plots + TF-IDF word clouds
```

#### Engagement Analysis (NEW)
```python
from engagement_analysis import EngagementAnalyzer

analyzer = EngagementAnalyzer()
analyzer.analyze_sentiment_engagement()  # Sentiment vs upvotes/comments
```

#### User Behavior Analysis (NEW)
```python
from user_analysis import UserAnalyzer

analyzer = UserAnalyzer()
analyzer.analyze_user_behavior()  # Frequency, overlap, topics
```

#### Word Cloud Generation Only
```python
from wordcloud_generator import WordCloudGenerator

generator = WordCloudGenerator()
generator.generate_all_wordclouds()  # Includes all features
```

## Output Files

### EDA Outputs (`eda_output/`) 🆕
- `temporal_line_plots.png` - Posts/comments per year (4-panel visualization)
- `temporal_statistics.txt` - Year-by-year detailed statistics
- `tfidf_wordclouds.png` - Unigram & bigram word clouds (4-panel)
- `tfidf_results.txt` - Top 50 unigrams and bigrams with TF-IDF scores

### Engagement Analysis Outputs (`engagement_output/`) 🆕
- `sentiment_engagement_analysis.png` - 9-panel comprehensive visualization
  - Average upvotes/comments by sentiment
  - Distribution box plots
  - Scatter plots with correlations
  - Statistical test results
- `engagement_analysis_report.txt` - Statistical tests, correlations, insights

### User Analysis Outputs (`user_analysis_output/`) 🆕
- `user_behavior_analysis.png` - 6-panel visualization
  - Posting frequency histogram
  - User overlap heatmap
  - User categories breakdown
  - Engagement by user type
- `active_users_topics.png` - 4-panel topic comparison
  - Word clouds: active vs one-time users
  - Top 20 words bar charts
- `user_behavior_report.txt` - Detailed statistics and concentration analysis
- `user_topics_comparison.txt` - Top 50 words for each user group

### Sentiment Analysis Outputs (`sentiment_analysis_output/`)
- `sentiment_overview.png` - Pie chart, histogram, polarity scatter, subreddit comparison
- `temporal_trends.png` - Sentiment over time, post volume trends
- `medical_context.png` - Medical indicators analysis
- `precovid_postcovid_analysis.png` - Comprehensive COVID comparison (3x3 grid)
- `precovid_postcovid_report.txt` - Detailed statistical report
- `sentiment_graphs_explanation.txt` - Complete interpretation guide
- `vader_validation_examples.txt` - 🆕 VADER accuracy assessment with examples
- `temporal_sentiment_shifts.png` - 🆕 Year-over-year sentiment trends (6-panel)
- `temporal_shift_report.txt` - 🆕 Statistical trend analysis with linear regression
- `sentiment_analysis_results_[timestamp].csv` - Raw results with all metrics

### Word Cloud Outputs (`wordcloud_output/`)
- `wordcloud_overall.png` - Overall discussion themes
- `wordcloud_overall_insights.txt` - Comprehensive interpretation guide
- `wordcloud_by_subreddit.png` - Top 12 subreddits (4x3 grid)
- `wordcloud_by_sentiment.png` - Positive, negative, neutral
- `wordcloud_precovid_postcovid.png` - Temporal comparison (2x2 grid)
- `precovid_postcovid_wordcloud_insights.txt` - COVID impact analysis
- `wordcloud_by_year.png` - Yearly evolution
- `word_categorization_analysis.png` - 8-category breakdown
- `word_categorization_insights.txt` - Category insights
- `wordcloud_medical_terms.png` - Medical vocabulary focus
- `medical_terms_frequency.png` - Top 20 medical terms bar chart

## Understanding the Analysis

### Reading Sentiment Graphs

📖 **See `sentiment_analysis_output/sentiment_graphs_explanation.txt`** for detailed explanations of:
- Polarity vs Subjectivity quadrants
- VADER compound score interpretation
- Temporal trend patterns
- Medical context indicators

### Understanding Word Categories

The analysis automatically categorizes words into 8 groups:

1. **Symptoms** (red) - Physical warning signs
2. **Diagnosis & Medical** (blue) - Clinical terminology
3. **Treatment** (green) - Medications and interventions
4. **Outcomes** (purple) - Birth results and baby health
5. **Emotions** (orange) - Psychological experiences
6. **Complications** (dark red) - Serious risks
7. **Providers** (teal) - Healthcare professionals
8. **Community** (gray) - Support and information-seeking

### Pre-COVID vs Post-COVID Insights

The analysis reveals changes in:
- **Sentiment patterns**: How emotional tone shifted during pandemic
- **Post volume**: Community growth and engagement changes
- **Terminology**: New medical vocabulary and concerns
- **Support-seeking**: Changes in how users request help
- **Medical context**: Shifts in outcome discussions

📊 **See `precovid_postcovid_report.txt`** for detailed statistical comparisons.

## Research Applications

This analysis can inform:

### Clinical Research
- **Patient Concerns**: Identify most common fears and questions
- **Symptom Patterns**: Track self-reported symptom frequencies
- **Treatment Experiences**: Understand patient perspectives on interventions
- **Outcome Tracking**: Monitor discussion of maternal and fetal outcomes

### Healthcare Communication
- **Patient Language**: Use community terminology in education materials
- **Information Gaps**: Identify misunderstood concepts needing clarification
- **Support Needs**: Design interventions based on expressed needs
- **Education Topics**: Prioritize commonly discussed concerns

### Public Health
- **Awareness Trends**: Track pre-eclampsia awareness over time
- **Pandemic Impact**: Understand COVID-19's effect on high-risk pregnancies
- **Community Dynamics**: Study online support patterns
- **Health Literacy**: Assess medical terminology understanding

### Future Enhancements

📄 **See `TFIDF_INTEGRATION_GUIDE.md`** for implementing:
- TF-IDF keyword extraction
- Topic modeling
- Document clustering
- Temporal trend analysis with statistical methods

## Data Privacy & Ethics

- ✅ All data is publicly available on Reddit
- ✅ No personal identifying information collected
- ✅ Usernames stored for deduplication only
- ✅ Analysis focuses on aggregate patterns, not individuals
- ✅ Results intended for medical education and research

## Technical Details

### Sentiment Analysis Algorithms

**VADER (Valence Aware Dictionary and sEntiment Reasoner)**
- Optimized for social media text
- Considers emoticons, slang, capitalization
- Custom medical lexicon additions for pre-eclampsia context
- Returns: positive, negative, neutral, compound scores

**TextBlob**
- General-purpose sentiment analysis
- Returns: polarity (-1 to +1), subjectivity (0 to 1)
- Useful for validating VADER results

### Word Processing Pipeline

1. **Text Cleaning**: Remove URLs, emojis, special characters
2. **Tokenization**: Split into individual words
3. **Stopword Removal**: Remove common words (preserved medical terms)
4. **Lemmatization**: Reduce words to root form
5. **Medical Term Detection**: Identify and categorize medical vocabulary
6. **Frequency Analysis**: Count and rank word occurrences
7. **Category Assignment**: Classify into 8 medical/psychological groups

### Data Collection Date Range

- **Start Date**: June 29, 2013 (r/preeclampsia creation)
- **COVID Boundary**: March 1, 2020
- **End Date**: Current date (continuously updated)
- **Collection Strategy**: Weighted sampling based on subreddit relevance

## Troubleshooting

### Common Issues

**Issue**: "No cleaned CSV file found!"
**Solution**: Run `data_cleaning.py` first to preprocess data

**Issue**: "Insufficient data for Pre-COVID comparison"
**Solution**: Ensure data collection includes posts from 2013-2019

**Issue**: NLTK download errors
**Solution**: Manually download required packages:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

**Issue**: Firebase connection errors
**Solution**: Verify `.env` file has correct credentials and service account JSON exists

### Performance Optimization

For large datasets (>10,000 posts):
- Process in batches
- Use multiprocessing for sentiment analysis
- Increase memory allocation
- Consider sampling for word clouds

## Contributing

Contributions welcome! Areas for improvement:
- Additional sentiment analysis algorithms
- Interactive visualizations (Plotly, Dash)
- Statistical significance testing
- Machine learning classification
- Network analysis of co-occurring terms

## Citation

If you use this code or analysis in research, please cite:

```
Pre-eclampsia Reddit Analysis Pipeline (2025)
Analysis of online health discussions about pre-eclampsia (2013-2025)
https://github.com/yourusername/CSS
```

## License

This project is for educational and research purposes. Reddit data subject to Reddit's Terms of Service and API usage policies.

## Contact & Support

- 📧 Questions: [Your contact]
- 🐛 Issues: [GitHub Issues]
- 📖 Documentation: See `UPDATE_SUMMARY.md` and `TFIDF_INTEGRATION_GUIDE.md`

## Acknowledgments

- Reddit community members sharing their pre-eclampsia experiences
- VADER Sentiment Analysis (Hutto & Gilbert, 2014)
- TextBlob Natural Language Processing
- WordCloud library by Andreas Mueller

---

**Last Updated**: November 22, 2025
**Version**: 2.0 (Major update with temporal analysis and categorization)
```python
firebase = FirebaseManager()
firebase.export_to_json('reddit_data_export.json')
```

## Configuration

### Adjust Date Range

Edit `config.py`:
```python
DATA_COLLECTION_CONFIG = {
    'start_date': '2022-01-01',  # Change start date
    'end_date': '2025-11-10',    # Change end date
    'max_posts_per_subreddit': 1000,
}
```

### Modify Subreddit Weights

Edit `config.py`:
```python
SUBREDDIT_WEIGHTS = {
    'preeclampsia': {
        'weight': 1.0,           # Change weight (0.0-1.0)
        'llm': ['claude'],       # Add/remove LLM attributions
        'focus': 'dedicated'     # 'dedicated', 'high', 'medium', 'broad'
    },
}
```

### Add New Keywords

Edit `config.py`:
```python
KEYWORDS_BY_LLM = {
    'claude': {
        'core_terms': [
            'preeclampsia',
            'your_new_keyword',  # Add here
        ],
    }
}
```

## Data Structure

### Post Data
```json
{
  "id": "abc123",
  "subreddit": "preeclampsia",
  "title": "Post title",
  "selftext": "Post content",
  "author": "username",
  "created_utc": 1699564800,
  "created_date": "2024-11-10T00:00:00",
  "score": 42,
  "num_comments": 15,
  "subreddit_weight": 1.0,
  "subreddit_focus": "dedicated",
  "llm_suggested_by": ["claude", "gemini"],
  "matched_keywords": {
    "claude": {
      "core_terms": ["preeclampsia"],
      "symptoms": ["high blood pressure"]
    }
  },
  "relevance_score": 0.875
}
```

### Comment Data
```json
{
  "id": "xyz789",
  "post_id": "abc123",
  "subreddit": "preeclampsia",
  "body": "Comment text",
  "author": "username",
  "created_utc": 1699565000,
  "score": 5,
  "matched_keywords": {...},
  "relevance_score": 0.750
}
```

## LLM Comparison Analysis

The pipeline tracks:
1. **Subreddit Suggestions**: Which LLM suggested which subreddits
2. **Keyword Matches**: Which keywords from which LLM matched most
3. **Data Quality**: Relevance scores by LLM attribution
4. **Coverage**: Unique vs. overlapping suggestions

This enables comparative analysis of:
- Which LLM suggested more relevant subreddits
- Which keyword sets captured more discussions
- Which approach yielded higher quality data

## Troubleshooting

### Rate Limiting
If you hit rate limits:
1. Increase sleep time in `data_collector.py` (line ~380)
2. Reduce `max_posts_per_subreddit` in `config.py`

### Firebase Connection Issues
1. Verify `.env` file is in the project directory
2. Check Firebase credentials are valid
3. Ensure database URL is correct

### No Data Collected
1. Check date range is valid
2. Verify subreddit names are correct
3. Try with dedicated subreddits first (r/preeclampsia)

## Next Steps

After data collection:
1. **Sentiment Analysis**: Analyze emotional tone of discussions
2. **Topic Modeling**: Identify key themes and concerns
3. **LLM Comparison**: Compare effectiveness of different LLM suggestions
4. **Temporal Analysis**: Track trends over time
5. **Risk Factor Identification**: Extract mentioned risk factors and symptoms

## License

This project is for academic research purposes.

## Contact

For questions or issues, please refer to the project documentation.
