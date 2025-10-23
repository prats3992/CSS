# Pre-eclampsia Reddit Analysis Project

## Overview
This project analyzes Reddit discussions about pre-eclampsia using Natural Language Processing (NLP) techniques including topic modeling with Latent Dirichlet Allocation (LDA) and sentiment analysis using VADER. The goal is to understand patient experiences, identify common topics of discussion, and analyze the emotional patterns in pre-eclampsia related conversations.

## Quick Start Checklist

✅ **Setup (10-15 minutes)**:
1. Install dependencies: `pip install -r requirements.txt`  
2. Install spaCy model: `python -m spacy download en_core_web_sm`
3. Copy `.env.example` to `.env` and add your API credentials
4. Get Reddit API credentials and update `.env` file
5. Set up Firebase project and update `.env` with Firebase config
6. Test connection: `python test_reddit_connection.py`

✅ **Run Analysis**:
7. Execute: `python main_analysis.py`
8. Check results in `visualizations/` folder

⚠️ **Security Reminder**:
- Never commit `.env` file or `firebase-service-account.json` to GitHub
- These files are protected by `.gitignore`
- Share credentials securely with team members (not via Git)

## Project Structure
```
pre-eclampsia-analysis/
├── config.py                     # Configuration and settings
├── requirements.txt              # Python dependencies  
├── data_collection.py            # Reddit data scraping with PRAW & Firebase
├── text_preprocessing.py         # Text cleaning and preprocessing
├── topic_modeling.py             # LDA topic modeling with Gensim
├── sentiment_analysis.py         # VADER sentiment analysis
├── main_analysis.py              # Main analysis pipeline
├── firebase_config.py            # Firebase database management
├── test_reddit_connection.py     # Reddit API connection testing
├── reddit_setup_guide.md         # Reddit API setup instructions
├── firebase_setup_guide.md       # Firebase setup instructions  
├── README.md                     # This file
├── Pre-eclamsia.pdf              # Research reference
├── data/                         # Local data storage directory
├── visualizations/               # Generated plots and charts
├── models/                       # Saved models
└── reports/                      # Analysis reports
```

## Features

### Data Collection
- Scrapes Reddit posts from multiple pregnancy and health-related subreddits
- Uses PRAW (Python Reddit API Wrapper) for ethical data collection
- Searches for pre-eclampsia related keywords across 9 subreddits
- Collects both posts and comments (up to 500 comments per post)
- **Firebase Integration**: Stores data in cloud database for team collaboration
- **Smart Caching**: Avoids duplicate API calls by checking existing data
- Anonymizes and stores data securely with local backup option

### Text Preprocessing  
- Removes personal identifiers and medical information
- Cleans Reddit-specific formatting (URLs, usernames, etc.)
- Tokenization and lemmatization using NLTK and spaCy
- Extracts medical terminology relevant to pre-eclampsia
- Filters out noise and irrelevant content
- **Advanced cleaning**: Handles contractions, medical abbreviations
- **Configurable parameters**: Minimum word length, post length filtering

### Topic Modeling
- Uses Gensim's LDA implementation with optimized parameters
- Identifies 8 main topics in discussions (configurable)
- Optimizes number of topics using coherence scores
- Creates interactive visualizations with pyLDAvis
- **Auto-labeling**: Topics labeled based on top words and medical context
- **Firebase Storage**: Saves trained models and results to cloud
- **Performance tuning**: 400 iterations, 10 passes for stable results

### Sentiment Analysis
- Uses VADER sentiment analyzer (optimized for social media text)
- Analyzes emotional patterns in posts with medical context awareness
- Creates comprehensive sentiment visualizations and word clouds  
- Analyzes sentiment trends over time and by subreddit
- **Multi-dimensional analysis**: Positive, negative, neutral, compound scores
- **Topic-sentiment correlation**: Links emotional patterns to discussion topics
- **Statistical insights**: Sentiment distribution and averages by subreddit

### Sentiment Analysis
- Uses VADER sentiment analyzer (optimized for social media)
- Analyzes emotional patterns in posts
- Creates sentiment visualizations and word clouds
- Analyzes sentiment trends over time
- Examines sentiment by topic and subreddit

### Visualization
- Interactive topic modeling visualizations (pyLDAvis HTML files)
- Comprehensive sentiment distribution charts and heatmaps
- Word clouds for different sentiments and topics
- Topic-sentiment correlation analysis
- Trend analysis over time and by subreddit
- **Firebase Dashboard**: Cloud-based result viewing and sharing
- **Publication-ready plots**: High-quality matplotlib/seaborn visualizations

## Installation

### ⚠️ IMPORTANT: Secure Setup

This project uses environment variables to keep API keys and secrets secure. **Never commit your `.env` file or Firebase service account JSON to version control!**

1. **Clone or download the project files**

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install spaCy language model**:
```bash
python -m spacy download en_core_web_sm
```

4. **Set up environment variables**:
   
   a. **Copy the example environment file**:
   ```bash
   cp .env.example .env
   ```
   
   b. **Edit `.env` file with your actual credentials** (see steps below for obtaining these)

5. **Set up Reddit API credentials**:
   - Go to https://www.reddit.com/prefs/apps
   - Click "Create App" or "Create Another App"
   - Fill in the form:
     - **name**: Pre-eclampsia Research (or any name)
     - **App type**: Select "script"
     - **description**: Research project (optional)
     - **about url**: Leave blank
     - **redirect uri**: http://localhost:8080 (required but not used)
   - Click "Create app"
   - Copy your credentials:
     - **client_id**: The string under "personal use script"
     - **client_secret**: The "secret" value
   - Update your `.env` file:
   ```
   REDDIT_CLIENT_ID=your_client_id_here
   REDDIT_CLIENT_SECRET=your_client_secret_here
   ```

6. **Set up Firebase (Required for cloud storage)**:
   
   a. **Create a Firebase project**:
   - Go to https://console.firebase.google.com/
   - Click "Add project"
   - Follow the setup wizard
   
   b. **Set up Realtime Database**:
   - In Firebase Console, go to "Build" > "Realtime Database"
   - Click "Create Database"
   - Choose a location (e.g., asia-southeast1)
   - Start in "test mode" (for development)
   
   c. **Get your database URL**:
   - Copy the database URL (looks like: `https://your-project-id-default-rtdb.region.firebasedatabase.app/`)
   - Update in `.env` file: `FIREBASE_DATABASE_URL=your_url_here`
   
   d. **Generate service account key**:
   - In Firebase Console, go to Project Settings (gear icon) > Service Accounts
   - Click "Generate new private key"
   - Save the JSON file as `firebase-service-account.json` in the project root
   - ⚠️ **Keep this file secure! It's already added to `.gitignore`**
   
   e. **Get Firebase web config** (for Pyrebase):
   - In Project Settings > General > Your apps
   - Click the web icon (</>)
   - Register your app
   - Copy the config values to your `.env` file:
   ```
   FIREBASE_API_KEY=your_api_key
   FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
   FIREBASE_PROJECT_ID=your-project-id
   FIREBASE_STORAGE_BUCKET=your-project.firebasestorage.app
   FIREBASE_MESSAGING_SENDER_ID=your_sender_id
   FIREBASE_APP_ID=your_app_id
   ```
   
   For detailed Firebase setup, see `firebase_setup_guide.md`

6. **Test your setup**:
```bash
python test_reddit_connection.py
```

## Usage

### Quick Start
Run the complete analysis pipeline:
```python
python main_analysis.py
```

This will automatically:
- Collect data from Reddit (or load from Firebase/local cache)
- Preprocess and clean the text
- Perform topic modeling with LDA
- Analyze sentiment with VADER
- Generate visualizations and save results

### Command Line Options
You can customize the analysis by editing the main execution section in `main_analysis.py`:
```python
# Run complete analysis
final_df, insights = analysis.run_complete_analysis(
    skip_collection=False,    # Set True to use existing data
    use_firebase_data=True    # Set False to use only local data
)
```

### Step-by-Step Execution

1. **Configure settings** in `config.py`:
   - Set Reddit API credentials
   - Adjust subreddits and keywords  
   - Modify analysis parameters (number of topics, sentiment thresholds, etc.)

2. **Test Reddit connection**:
```python
python test_reddit_connection.py
```

3. **Collect data** (or load existing):
```python
from data_collection import RedditDataCollector
collector = RedditDataCollector()
data = collector.collect_data()  # Collects new data
# OR
data = collector.load_data_from_firebase()  # Loads existing data
```

4. **Preprocess text**:
```python
from text_preprocessing import TextPreprocessor
preprocessor = TextPreprocessor()
processed_data = preprocessor.preprocess_dataframe(data)
```

5. **Perform topic modeling**:
```python
from topic_modeling import TopicModeler
modeler = TopicModeler()
modeler.train_lda_model(processed_data['tokens_lemmatized'].tolist())
topics = modeler.get_document_topics(processed_data['tokens_lemmatized'].tolist())
```

6. **Analyze sentiment**:
```python
from sentiment_analysis import SentimentAnalyzer
analyzer = SentimentAnalyzer()
results = analyzer.analyze_dataframe(processed_data)
analyzer.create_sentiment_visualizations(results)
```

## Configuration

### Reddit API Settings
- `REDDIT_CLIENT_ID`: Your Reddit app client ID  
- `REDDIT_CLIENT_SECRET`: Your Reddit app client secret
- `REDDIT_USER_AGENT`: Descriptive user agent string

### Firebase Settings (Optional)
- `FIREBASE_SERVICE_ACCOUNT_PATH`: Path to service account JSON file
- `FIREBASE_DATABASE_URL`: Your Firebase Realtime Database URL
- `FIREBASE_BATCH_SIZE`: Batch size for data uploads (default: 500)

### Data Collection Settings
- `SUBREDDITS`: 9 pregnancy and health-related subreddits
- `KEYWORDS`: 9 pre-eclampsia related search terms
- `MAX_POSTS_PER_SUBREDDIT`: Limit posts per subreddit (default: 1000)
- `MAX_COMMENTS_PER_POST`: Comments to collect per post (default: 500)
- `TIME_FILTER`: Time range for post collection ('all', 'year', 'month', etc.)

### Analysis Settings
- `NUM_TOPICS`: Number of topics for LDA (default: 8)
- `LDA_PASSES`: Number of training passes (default: 10)
- `LDA_ITERATIONS`: Training iterations (default: 400)
- `VADER_THRESHOLD_POSITIVE/NEGATIVE`: Sentiment classification thresholds
- `MIN_WORD_LENGTH`: Minimum word length for processing (default: 3)
- `MIN_POST_LENGTH`: Minimum post length in words (default: 10)

## Outputs

The analysis generates several outputs:

### Data Files
- **Local Storage** (backup):
  - `data/reddit_posts_raw.csv`: Original scraped data
  - `data/reddit_posts_processed.csv`: Cleaned and preprocessed data  
  - `data/reddit_posts_analyzed.csv`: Final data with topics and sentiment
- **Firebase Storage** (cloud):
  - Raw posts data with metadata
  - Processed text data with tokens
  - Final analyzed data with topics and sentiment scores

### Visualizations
- `visualizations/topic_coherence_scores.png`: Topic optimization chart
- `visualizations/lda_topics_interactive.html`: Interactive topic visualization  
- `visualizations/sentiment_analysis.png`: Comprehensive sentiment charts
- `visualizations/sentiment_wordclouds.png`: Word clouds by sentiment
- `visualizations/sentiment_by_topic.png`: Topic-sentiment heatmap
- `visualizations/topic_distribution.png`: Topic frequency analysis

### Models  
- `models/lda_model`: Trained LDA topic model (Gensim format)
- `models/dictionary.pkl`: Gensim dictionary object
- `models/corpus.pkl`: Document corpus for topic modeling
- **Firebase Models**: Cloud-stored model parameters and results

### Analysis Results
- **Firebase Analytics**: Real-time analysis results and statistics
- **Topic Labels**: Auto-generated topic descriptions
- **Sentiment Statistics**: Distribution by subreddit and topic
- **Coherence Scores**: Model performance metrics

## Ethical Considerations

This project follows ethical guidelines for social media research:
- Uses only public Reddit data
- Anonymizes all personal information
- Removes medical identifiers
- Aggregates data for analysis
- Complies with Reddit's API terms of service
- Respects user privacy and community guidelines

## Key Findings

The analysis typically reveals insights such as:
- Common symptoms and concerns discussed
- Emotional impact on patients and families
- Information-seeking behaviors
- Support-seeking patterns
- Treatment experiences
- Healthcare provider interactions

## Research Applications

This analysis can be valuable for:
- Healthcare providers understanding patient concerns
- Medical researchers studying patient experiences
- Public health officials monitoring health discussions
- Patient advocacy organizations
- Educational material development
- Telemedicine platform design

## Limitations

- Limited to English-language posts
- Reddit demographics may not represent all patients
- Self-reported information may have biases
- Cannot verify medical accuracy of posts
- Time-limited data collection

## Troubleshooting

### Common Issues

1. **Reddit API Connection Errors**:
   - Run `python test_reddit_connection.py` to verify credentials
   - Ensure CLIENT_ID and CLIENT_SECRET are correctly set
   - Check that your Reddit app is configured as "script" type

2. **Firebase Connection Issues**:
   - Verify service account JSON file path in config
   - Check Firebase database URL format
   - Ensure database security rules allow read/write access
   - Test with `firebase_config.py` directly

3. **NLTK/spaCy Model Errors**:
   - The application will automatically download required NLTK data
   - Ensure spaCy English model is installed: `python -m spacy download en_core_web_sm`

4. **Memory Issues with Large Datasets**:
   - Reduce `MAX_POSTS_PER_SUBREDDIT` in config
   - Use Firebase data loading instead of local files
   - Process subreddits individually

5. **Topic Modeling Convergence**:
   - Increase `LDA_ITERATIONS` for better convergence
   - Adjust `NUM_TOPICS` based on coherence scores
   - Ensure sufficient data volume (>100 posts recommended)

### Getting Help

- Check the setup guides: `reddit_setup_guide.md` and `firebase_setup_guide.md`  
- Review configuration in `config.py`
- Enable detailed logging by setting `logging.basicConfig(level=logging.DEBUG)`

## Contributing

To extend this project:

### Data Sources
- Add new subreddits to `SUBREDDITS` list in config
- Implement other social media APIs (Twitter, Facebook groups)  
- Add medical forum data sources

### Analysis Techniques  
- Implement additional topic modeling algorithms (CTM, BERTopic)
- Add emotion analysis beyond sentiment (fear, anxiety, hope)
- Include temporal trend analysis and seasonal patterns
- Add demographic analysis if available

### Visualization Enhancements
- Create interactive dashboards with Plotly/Dash
- Add real-time monitoring capabilities  
- Develop mobile-friendly result viewing
- Implement comparison tools across time periods

### Technical Improvements
- Add automated model retraining pipelines
- Implement data quality monitoring
- Add automated report generation
- Create REST API for analysis results
- Add support for multiple languages

## References

1. Goel, R., et al. (2023). Users' concerns about endometriosis on social media: Sentiment analysis and topic modeling study. Journal of Medical Internet Research, 25, e45381.

2. Emanuel, R. H. K., et al. (2024). Extracting features and sentiment from text posts and comments relating to polycystic ovary syndrome. IFAC-PapersOnLine, 58(24), 19–24.

3. Dhankar, A., & Katz, A. (2023). Tracking pregnant women's mental health through social media: An analysis of Reddit posts. JAMIA Open, 6(4), ooad094.

## License

This project is for research and educational purposes. Please ensure compliance with Reddit's terms of service and applicable data protection regulations when using this code.

## Contact

For questions about this project or pre-eclampsia research, please contact the research team at the respective institutions mentioned in the project documentation.
