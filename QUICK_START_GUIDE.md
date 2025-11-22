# 🎯 Quick Start Guide - Updated Analysis Pipeline

## What Changed?

Your pre-eclampsia analysis codebase has been significantly enhanced with:
1. ✅ **8-category word analysis** (symptoms, treatments, emotions, etc.)
2. ✅ **Detailed graph explanations** (especially polarity vs subjectivity)
3. ✅ **Pre-COVID vs Post-COVID analysis** (2013-2019 vs 2020-2025)
4. ✅ **Removed LLM comparisons** (cleaner focus on medical data)
5. ✅ **TF-IDF guide** (for future keyword extraction)
6. ✅ **Comprehensive insights** (automatic medical context analysis)

## 🚀 Quick Start (3 Steps)

### Step 1: Run Sentiment Analysis
```powershell
python sentiment_analysis.py
```
**What it does:**
- Analyzes all posts for sentiment (positive/negative/neutral)
- Creates Pre-COVID vs Post-COVID comparison
- Generates 12+ visualizations
- Creates detailed explanation documents

**New Outputs:**
- `sentiment_analysis_output/precovid_postcovid_analysis.png` ⭐ NEW
- `sentiment_analysis_output/precovid_postcovid_report.txt` ⭐ NEW
- `sentiment_analysis_output/sentiment_graphs_explanation.txt` ⭐ NEW

---

### Step 2: Generate Word Clouds
```powershell
python wordcloud_generator.py
```
**What it does:**
- Creates word clouds for overall, subreddits, sentiments, time periods
- Categorizes words into 8 medical/psychological groups
- Compares Pre-COVID vs Post-COVID terminology
- Generates automatic insights

**New Outputs:**
- `wordcloud_output/word_categorization_analysis.png` ⭐ NEW
- `wordcloud_output/word_categorization_insights.txt` ⭐ NEW
- `wordcloud_output/wordcloud_precovid_postcovid.png` ⭐ NEW
- `wordcloud_output/wordcloud_overall_insights.txt` ⭐ NEW

---

### Step 3: Read the Insights
Open these text files to understand your visualizations:

**Essential Reading:**
1. `UPDATE_SUMMARY.md` - Complete list of all changes
2. `sentiment_graphs_explanation.txt` - How to read sentiment graphs
3. `precovid_postcovid_report.txt` - Statistical COVID impact analysis
4. `word_categorization_insights.txt` - Medical term category breakdown

**Optional Reading:**
5. `TFIDF_INTEGRATION_GUIDE.md` - Future TF-IDF implementation guide

---

## 📊 What You'll Get

### Sentiment Analysis (12 Visualizations)

**File: `sentiment_overview.png`**
- Sentiment pie chart (positive/negative/neutral %)
- VADER score distribution histogram
- Polarity vs Subjectivity scatter plot ⭐ WITH EXPLANATIONS
- Average sentiment by subreddit

**File: `temporal_trends.png`**
- Sentiment trend over time (monthly)
- Post volume over time (colored by sentiment)

**File: `medical_context.png`**
- Positive outcomes bar chart
- Negative outcomes bar chart
- Support-seeking behavior
- Experience sharing posts

**File: `precovid_postcovid_analysis.png` ⭐ NEW**
- 3×3 grid of comparisons:
  - Sentiment distribution changes
  - Average score comparisons
  - Post volume by year
  - Medical context shifts
  - Score distribution overlays
  - Polarity vs subjectivity comparison
  - Top subreddits comparison

---

### Word Cloud Analysis (9+ Visualizations)

**File: `wordcloud_overall.png`**
- Beautiful overall word cloud (200 most common words)

**File: `wordcloud_by_subreddit.png`**
- 12 separate word clouds (4×3 grid)
- One for each top subreddit

**File: `wordcloud_by_sentiment.png`**
- 3 word clouds: positive, negative, neutral

**File: `wordcloud_precovid_postcovid.png` ⭐ NEW**
- 2×2 grid comparing time periods:
  - Pre-COVID word cloud (blue)
  - Post-COVID word cloud (orange)
  - Pre-COVID unique terms (bar chart)
  - Post-COVID unique terms (bar chart)

**File: `word_categorization_analysis.png` ⭐ NEW**
- 3×3 grid showing 8 categories:
  1. Symptoms (headache, swelling, pain)
  2. Diagnosis & Medical (preeclampsia, tests, diagnosis)
  3. Treatment & Interventions (magnesium, hospital, medication)
  4. Pregnancy Outcomes (baby, birth, NICU, healthy)
  5. Emotions & Experience (scared, grateful, worried)
  6. Complications & Risks (emergency, severe, critical)
  7. Healthcare Providers (doctor, nurse, OB)
  8. Support & Community (help, advice, question)

**File: `wordcloud_by_year.png`**
- Year-by-year evolution of terminology

**File: `wordcloud_medical_terms.png`**
- Focus on medical vocabulary only

---

## 🔍 Understanding Key Changes

### 1. Polarity vs Subjectivity - Now Explained!

**Before:** Confusing scatter plot with no context
**After:** Detailed quadrant explanations

```
Top-Left (Negative & Subjective): "I was terrified when diagnosed"
Top-Right (Positive & Subjective): "So grateful for my healthy baby"
Bottom-Left (Negative & Objective): "Emergency c-section at 34 weeks"
Bottom-Right (Positive & Objective): "Baby born healthy at 37 weeks"
```

📖 **Full explanation in:** `sentiment_graphs_explanation.txt`

---

### 2. Word Categorization - New Feature!

**What it does:**
- Automatically sorts all words into 8 medical/psychological categories
- Shows frequency of each category
- Identifies top terms in each group
- Generates insights about community focus

**Example Output:**
```
SYMPTOMS:
  Total mentions: 15,234
  Unique terms: 45
  Most frequent: headache (3,421), swelling (2,876), pain (2,543)
```

📖 **Full insights in:** `word_categorization_insights.txt`

---

### 3. Pre-COVID vs Post-COVID - Major Addition!

**COVID Boundary:** March 1, 2020

**What it compares:**
- Sentiment changes (are people more/less anxious?)
- Post volume growth (more people seeking support?)
- Terminology shifts (new medical vocabulary?)
- Medical context changes (more complications? better outcomes?)
- Community behavior (more support-seeking?)

**Example Insights:**
```
Post volume increased by 147% after COVID-19
Sentiment became 0.08 points more negative (VADER)
Support-seeking increased by 12.3%
New terms: "telemedicine", "virtual", "remote monitoring"
```

📊 **See:** `precovid_postcovid_analysis.png` and `precovid_postcovid_report.txt`

---

### 4. LLM References - Removed!

**Before:** Confusing "LLM suggested by" charts
**After:** Clean medical data analysis

**Why?** LLMs were only used to identify keywords/subreddits for collection, not for actual analysis. Removed to avoid confusion.

---

### 5. TF-IDF Guide - For Future Use

**What it is:** A complete 400-line guide for implementing TF-IDF (keyword importance analysis)

**Not implemented yet** - just a guide for when you want to add:
- Automatic keyword extraction
- Distinctive term identification per subreddit
- Temporal keyword trend analysis
- Document clustering

📖 **See:** `TFIDF_INTEGRATION_GUIDE.md`

---

## 🎨 Color Schemes

### Sentiment Colors
- 🟢 **Green**: Positive sentiment (hope, gratitude, success)
- 🔴 **Red**: Negative sentiment (fear, complications, emergency)
- ⚪ **Gray**: Neutral sentiment (informational, balanced)

### COVID Comparison Colors
- 🔵 **Blue**: Pre-COVID (2013-2019)
- 🟠 **Orange**: Post-COVID (2020-2025)

### Category Colors
- 🔴 **Red**: Symptoms, Complications
- 🔵 **Blue**: Diagnosis, Medical
- 🟢 **Green**: Treatment, Interventions
- 🟣 **Purple**: Pregnancy Outcomes
- 🟠 **Orange**: Emotions, Experience
- 🔷 **Teal**: Healthcare Providers
- ⬛ **Gray**: Community, Support

---

## 📈 Expected Results (Example)

Based on typical pre-eclampsia discussion data:

### Sentiment Distribution
- Negative: 35-45% (expected for serious medical condition)
- Neutral: 30-40% (informational posts)
- Positive: 20-30% (success stories, gratitude)

### Top Word Categories (by frequency)
1. Pregnancy Outcomes (~25%)
2. Symptoms (~20%)
3. Diagnosis & Medical (~18%)
4. Emotions & Experience (~15%)
5. Treatment (~12%)
6. Community Support (~10%)

### Pre-COVID vs Post-COVID
- Post volume likely increased (more Reddit usage during lockdowns)
- Sentiment may be more negative (pandemic anxiety)
- New terminology around telehealth and remote monitoring
- Possible increase in support-seeking behavior

---

## ⚠️ Important Notes

### Data Requirements
- ✅ Need CSV with `created_utc` column (Unix timestamp)
- ✅ Need CSV with `text_no_stopwords` column (cleaned text)
- ✅ Need posts from **2013-2019** for Pre-COVID analysis
- ✅ Need posts from **2020-2025** for Post-COVID analysis

### If You Don't Have Old Data
If your data only has 2020-2025 posts:
- Pre-COVID analysis will show "Insufficient data"
- All other features still work
- Consider collecting historical data or removing COVID comparison

### Performance
- Small dataset (<1,000 posts): ~30 seconds total
- Medium dataset (1,000-10,000 posts): ~2-5 minutes total
- Large dataset (>10,000 posts): ~10-20 minutes total

---

## 🐛 Troubleshooting

### "No cleaned CSV file found"
**Solution:** Run `data_cleaning.py` first

### "Insufficient data for Pre-COVID comparison"
**Solution:** Ensure you have posts from 2013-2019, or comment out COVID analysis

### "KeyError: 'created_datetime'"
**Solution:** The code will auto-create this from `created_utc`

### "Memory Error"
**Solution:** Process in smaller batches or increase system memory

### Graphs look wrong
**Solution:** Check that dates are in correct format (Unix timestamps)

---

## 📞 Quick Reference

### File Locations
- **Code**: `sentiment_analysis.py`, `wordcloud_generator.py`
- **Config**: `config.py` (dates and settings)
- **Outputs**: `sentiment_analysis_output/`, `wordcloud_output/`
- **Docs**: `UPDATE_SUMMARY.md`, `TFIDF_INTEGRATION_GUIDE.md`

### Key Functions
```python
# Sentiment analysis
analyzer = SentimentAnalyzer()
analyzer.run_sentiment_analysis()
analyzer.create_visualizations()
analyzer.create_precovid_postcovid_analysis()  # NEW

# Word clouds
generator = WordCloudGenerator()
generator.generate_all_wordclouds()  # Includes all new features
generator.categorize_and_analyze_words()  # NEW
```

---

## ✅ Checklist

Before running:
- [ ] Have cleaned CSV file (`cleaned_posts_*.csv`)
- [ ] CSV has `created_utc` and `text_no_stopwords` columns
- [ ] Data spans from 2013-2025 (or at least 2020-2025)
- [ ] Installed all requirements (`pip install -r requirements.txt`)

After running:
- [ ] Check `sentiment_analysis_output/` has 4-5 PNG files
- [ ] Check `wordcloud_output/` has 9+ PNG files
- [ ] Read `sentiment_graphs_explanation.txt`
- [ ] Read `precovid_postcovid_report.txt`
- [ ] Review category insights

---

## 🎓 Next Steps

1. **Run the analysis** with your existing data
2. **Review visualizations** - look for patterns
3. **Read insight files** - understand the medical context
4. **Share findings** - use for research or education
5. **Consider TF-IDF** - if you need keyword extraction later

---

**Need Help?** Check `UPDATE_SUMMARY.md` for complete technical details.

**Everything Working?** Great! You now have comprehensive pre-eclampsia discussion analysis with medical insights, sentiment tracking, and temporal comparisons. 🎉
