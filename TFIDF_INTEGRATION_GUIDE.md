# TF-IDF Integration Guide for Pre-eclampsia Analysis

## What is TF-IDF?

**TF-IDF** (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document in a collection of documents. It's commonly used in information retrieval and text mining.

### Formula:
- **TF (Term Frequency)**: How often a word appears in a document
- **IDF (Inverse Document Frequency)**: How rare/common a word is across all documents
- **TF-IDF Score** = TF × IDF

### What it reveals:
- Words that are frequent in a specific document but rare across all documents get high scores
- Common words (stopwords) get low scores
- Identifies distinctive/characteristic terms for each document/category

---

## Where TF-IDF Can Be Used in Your Pipeline

### 1. **KEYWORD EXTRACTION & TOPIC MODELING** ✅ RECOMMENDED
**Location**: `wordcloud_generator.py` or new `topic_analysis.py`

**Purpose**: 
- Identify most distinctive terms per subreddit
- Extract key themes automatically instead of manual categorization
- Find unique characteristics of different communities

**Implementation**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_keywords(self, n_keywords=20):
    """Extract top TF-IDF keywords for each subreddit"""
    
    # Group posts by subreddit
    for subreddit in self.df['subreddit'].unique():
        subreddit_texts = self.df[self.df['subreddit'] == subreddit]['text_no_stopwords'].tolist()
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(subreddit_texts)
        
        # Get top terms
        feature_names = vectorizer.get_feature_names_out()
        avg_tfidf = tfidf_matrix.mean(axis=0).A1
        top_indices = avg_tfidf.argsort()[-n_keywords:][::-1]
        
        print(f"r/{subreddit} distinctive terms:")
        for idx in top_indices:
            print(f"  {feature_names[idx]}: {avg_tfidf[idx]:.4f}")
```

**Benefits**:
- Discover what makes each subreddit unique
- Identify emerging terms over time
- Validate your manual keyword categorizations

---

### 2. **TEMPORAL TREND ANALYSIS** ✅ RECOMMENDED
**Location**: New `temporal_analysis.py` or extend `sentiment_analysis.py`

**Purpose**:
- Track which medical terms gain/lose importance over time
- Compare pre-COVID vs post-COVID terminology
- Identify shifts in community discourse

**Implementation**:
```python
def analyze_temporal_tfidf(self, time_periods=['pre_covid', 'post_covid']):
    """Compare TF-IDF scores across time periods"""
    
    covid_date = pd.Timestamp('2020-03-01')
    self.df['period'] = self.df['created_datetime'].apply(
        lambda x: 'pre_covid' if x < covid_date else 'post_covid'
    )
    
    # Calculate TF-IDF for each period
    for period in ['pre_covid', 'post_covid']:
        period_texts = self.df[self.df['period'] == period]['text_no_stopwords'].tolist()
        # ... TF-IDF calculation
```

**Benefits**:
- See if COVID changed how people discuss pre-eclampsia
- Track medical terminology evolution
- Understand pandemic's impact on pregnancy discussions

---

### 3. **DOCUMENT SIMILARITY & CLUSTERING** 🔄 OPTIONAL
**Location**: New `clustering_analysis.py`

**Purpose**:
- Group similar posts together
- Identify post archetypes (support-seeking, story-sharing, info-seeking)
- Find duplicate or near-duplicate content

**Implementation**:
```python
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def cluster_posts_by_content(self, n_clusters=5):
    """Cluster posts based on TF-IDF similarity"""
    
    vectorizer = TfidfVectorizer(max_features=200)
    tfidf_matrix = vectorizer.fit_transform(self.df['text_no_stopwords'])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    self.df['cluster'] = clusters
    
    # Analyze each cluster
    for cluster_id in range(n_clusters):
        cluster_posts = self.df[self.df['cluster'] == cluster_id]
        print(f"Cluster {cluster_id}: {len(cluster_posts)} posts")
        # Extract top terms for this cluster
```

**Benefits**:
- Understand post diversity
- Create automatic post categorization
- Identify outliers or unusual posts

---

### 4. **SUBREDDIT COMPARISON** ✅ RECOMMENDED
**Location**: `wordcloud_generator.py` or new analysis file

**Purpose**:
- Compare terminology differences between subreddits
- Identify which terms are unique to r/preeclampsia vs general pregnancy subs
- Validate subreddit weights in your config

**Implementation**:
```python
def compare_subreddit_vocabulary(self, subreddit1, subreddit2):
    """Compare distinctive terms between two subreddits using TF-IDF"""
    
    # Get texts for each subreddit
    texts1 = self.df[self.df['subreddit'] == subreddit1]['text_no_stopwords'].tolist()
    texts2 = self.df[self.df['subreddit'] == subreddit2]['text_no_stopwords'].tolist()
    
    # Calculate TF-IDF for each
    vectorizer = TfidfVectorizer(max_features=100)
    
    tfidf1 = vectorizer.fit_transform(texts1)
    tfidf2 = vectorizer.fit_transform(texts2)
    
    # Compare top terms
```

---

### 5. **SENTIMENT-AWARE TF-IDF** 🔄 OPTIONAL (Advanced)
**Location**: Extend `sentiment_analysis.py`

**Purpose**:
- Identify which terms are most distinctive in positive vs negative posts
- Understand language patterns in different emotional states
- Create sentiment-specific vocabularies

**Implementation**:
```python
def analyze_sentiment_vocabulary(self):
    """Find distinctive terms for positive vs negative posts"""
    
    for sentiment in ['positive', 'negative', 'neutral']:
        sentiment_texts = self.df[
            self.df['sentiment_class'] == sentiment
        ]['text_no_stopwords'].tolist()
        
        # Calculate TF-IDF
        # Identify characteristic terms for this sentiment
```

**Benefits**:
- Understand emotional language patterns
- Improve sentiment classification
- Create sentiment lexicons specific to pre-eclampsia

---

## Recommended Implementation Priority

### Phase 1: Essential (Implement Now) ✅
1. **Keyword Extraction per Subreddit** - Validates your manual categories
2. **Temporal TF-IDF Analysis** - Supports pre/post-COVID comparison

### Phase 2: Enhanced Analysis (Implement Later) 🔄
3. **Subreddit Vocabulary Comparison** - Deeper community understanding
4. **Document Clustering** - Automatic post categorization

### Phase 3: Advanced (Future Work) ⏭️
5. **Sentiment-Aware TF-IDF** - Emotional language patterns
6. **Topic Modeling (LDA + TF-IDF)** - Discover hidden themes

---

## Integration with Existing Code

### Add to `wordcloud_generator.py`:
```python
def generate_tfidf_analysis(self, output_dir='wordcloud_output'):
    """Generate TF-IDF-based keyword analysis"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Implementation here
    # Creates:
    # - tfidf_keywords_by_subreddit.csv
    # - tfidf_temporal_comparison.csv
    # - tfidf_analysis_visualization.png
```

### Add to `sentiment_analysis.py`:
```python
def analyze_tfidf_sentiment_vocabulary(self):
    """Analyze distinctive vocabulary by sentiment"""
    # Implementation here
```

---

## Expected Outputs

### 1. TF-IDF Keyword Tables
```
Subreddit: r/preeclampsia
Top TF-IDF Terms:
  magnesium sulfate: 0.2543
  hellp syndrome: 0.2198
  blood pressure: 0.1876
  ...
```

### 2. Temporal Comparison
```
Pre-COVID (2013-2019):
  Top terms: hospital, delivery, scared
  
Post-COVID (2020-2025):
  Top terms: virtual, telehealth, anxiety
  
New terms: telemedicine, remote monitoring
```

### 3. Visualizations
- Heatmap of TF-IDF scores across subreddits
- Word clouds weighted by TF-IDF (not raw frequency)
- Timeline showing term importance evolution

---

## When NOT to Use TF-IDF

❌ **Don't use TF-IDF for:**
1. **Sentiment Analysis** - TF-IDF is for importance, not emotion
2. **Word Frequency Counts** - Raw counts are better for simple frequency
3. **Small Documents** - Need sufficient text for meaningful IDF
4. **Real-time Analysis** - Requires entire corpus for IDF calculation

---

## Tools & Libraries Needed

Already in `requirements.txt` (likely):
```
scikit-learn  # TfidfVectorizer, clustering
pandas        # Data manipulation
numpy         # Numerical operations
matplotlib    # Visualization
seaborn       # Heatmaps
```

---

## Next Steps

1. ✅ **Read this guide** - Understand where TF-IDF fits
2. 🔧 **Implement Phase 1** - Start with keyword extraction
3. 📊 **Validate Results** - Compare with manual categories
4. 📈 **Expand Analysis** - Add temporal and subreddit comparisons
5. 📝 **Document Findings** - Update README with insights

---

## Questions to Answer with TF-IDF

1. What terms are most distinctive to r/preeclampsia vs general pregnancy subs?
2. How has medical terminology usage changed from 2013 to 2025?
3. Did COVID-19 introduce new vocabulary or concerns?
4. Which terms best distinguish positive from negative experiences?
5. Are there seasonal patterns in term usage?
6. What unique concerns appear in high-risk vs general pregnancy discussions?

---

## Summary

**TF-IDF is best used for:**
- ✅ Keyword extraction and ranking
- ✅ Finding distinctive terms per category
- ✅ Temporal trend analysis
- ✅ Document similarity and clustering
- ✅ Feature engineering for ML models

**Not ideal for:**
- ❌ Direct sentiment analysis
- ❌ Simple word counting
- ❌ Small text samples

**Recommended next step**: Implement TF-IDF keyword extraction for subreddit comparison and temporal analysis in the pre-COVID vs post-COVID section.
