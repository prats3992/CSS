"""
Main EDA Runner Script
Orchestrates all analyses and generates comprehensive report
"""

import os
import json
from datetime import datetime
import pandas as pd

from data_cleaning import DataCleaner
from sentiment_analysis import SentimentAnalyzer
from temporal_analysis import TemporalAnalyzer
from covid_comparison import CovidComparison
from overall_eda import OverallEDA


def create_html_report(posts_df, comments_df, results, output_dir='analysis_output'):
    """
    Create HTML report summarizing all analyses
    
    Args:
        posts_df (DataFrame): Posts with sentiment
        comments_df (DataFrame): Comments with sentiment
        results (dict): All analysis results
        output_dir (str): Output directory
    """
    # Build HTML content with proper escaping
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pre-eclampsia Reddit Data Analysis Report</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2E86AB;
                border-bottom: 3px solid #2E86AB;
                padding-bottom: 10px;
            }
            h2 {
                color: #118AB2;
                margin-top: 30px;
                border-bottom: 2px solid #118AB2;
                padding-bottom: 5px;
            }
            h3 {
                color: #06D6A0;
                margin-top: 20px;
            }
            .metric {
                display: inline-block;
                background-color: #f0f8ff;
                padding: 15px 25px;
                margin: 10px;
                border-radius: 8px;
                border-left: 4px solid #2E86AB;
            }
            .metric-value {
                font-size: 28px;
                font-weight: bold;
                color: #2E86AB;
            }
            .metric-label {
                font-size: 14px;
                color: #666;
            }
            .section {
                margin: 30px 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #2E86AB;
                color: white;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .timestamp {
                color: #999;
                font-size: 12px;
                text-align: right;
            }
            .image-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .image-container {
                border: 1px solid #ddd;
                padding: 10px;
                background-color: #fafafa;
            }
            .image-container img {
                width: 100%;
                height: auto;
            }
            .image-caption {
                text-align: center;
                margin-top: 10px;
                font-size: 14px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Pre-eclampsia Reddit Data Analysis Report</h1>
            <p class="timestamp">Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <div class="metric-value">""" + f"{len(posts_df):,}" + """</div>
                    <div class="metric-label">Total Posts</div>
                </div>
                <div class="metric">
                    <div class="metric-value">""" + f"{len(comments_df):,}" + """</div>
                    <div class="metric-label">Total Comments</div>
                </div>
                <div class="metric">
                    <div class="metric-value">""" + str(posts_df['subreddit'].nunique() if 'subreddit' in posts_df.columns else 'N/A') + """</div>
                    <div class="metric-label">Subreddits</div>
                </div>
                <div class="metric">
                    <div class="metric-value">""" + (f"{posts_df['sentiment_compound'].mean():.3f}" if 'sentiment_compound' in posts_df.columns else 'N/A') + """</div>
                    <div class="metric-label">Avg Sentiment</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Data Collection Period</h2>
                <p><strong>Start Date:</strong> """ + (posts_df['created_datetime'].min().strftime('%Y-%m-%d') if 'created_datetime' in posts_df.columns else 'N/A') + """</p>
                <p><strong>End Date:</strong> """ + (posts_df['created_datetime'].max().strftime('%Y-%m-%d') if 'created_datetime' in posts_df.columns else 'N/A') + """</p>
                <p><strong>Years Covered:</strong> """ + str(posts_df['year'].nunique() if 'year' in posts_df.columns else 'N/A') + """ years</p>
            </div>
            
            <div class="section">
                <h2>Sentiment Distribution</h2>
                <p>Overall sentiment breakdown across all posts and comments:</p>
    """
    
    if 'sentiment_category' in posts_df.columns:
        sentiment_counts = posts_df['sentiment_category'].value_counts()
        sentiment_pct = (sentiment_counts / len(posts_df) * 100).round(1)
        
        html_content += """
                <table>
                    <tr>
                        <th>Sentiment</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
        """
        
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in sentiment_counts.index:
                html_content += f"""
                    <tr>
                        <td style="text-transform: capitalize;">{sentiment}</td>
                        <td>{sentiment_counts[sentiment]:,}</td>
                        <td>{sentiment_pct[sentiment]}%</td>
                    </tr>
                """
        
        html_content += "</table>"
    
    html_content += """
            </div>
            
            <div class="section">
                <h2>Top Subreddits by Post Volume</h2>
    """
    
    if 'subreddit' in posts_df.columns:
        top_subs = posts_df['subreddit'].value_counts().head(10)
        
        html_content += """
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Subreddit</th>
                        <th>Posts</th>
                        <th>Percentage</th>
                    </tr>
        """
        
        for rank, (subreddit, count) in enumerate(top_subs.items(), 1):
            pct = round(count / len(posts_df) * 100, 1)
            html_content += f"""
                    <tr>
                        <td>{rank}</td>
                        <td>r/{subreddit}</td>
                        <td>{count:,}</td>
                        <td>{pct}%</td>
                    </tr>
            """
        
        html_content += "</table>"
    
    html_content += """
            </div>
            
            <div class="section">
                <h2>Analysis Visualizations</h2>
                
                <h3>Temporal Trends</h3>
                <div class="image-grid">
    """
    
    # Add temporal analysis images
    temporal_images = [
        ('sentiment_trend_over_years.png', 'Sentiment Trends Over Years'),
        ('post_volume_by_sentiment.png', 'Post Volume by Sentiment'),
        ('positive_percentage_by_year.png', 'Positive Posts Percentage by Year'),
        ('yoy_sentiment_change.png', 'Year-over-Year Sentiment Change'),
        ('wordcloud_by_year.png', 'Word Clouds by Year')
    ]
    
    for img_file, caption in temporal_images:
        img_path = f'temporal/{img_file}'
        if os.path.exists(os.path.join(output_dir, img_path)):
            html_content += f"""
                    <div class="image-container">
                        <img src="{img_path}" alt="{caption}">
                        <div class="image-caption">{caption}</div>
                    </div>
            """
    
    html_content += """
                </div>
                
                <h3>COVID-19 Comparison</h3>
                <div class="image-grid">
    """
    
    # Add COVID comparison images
    covid_images = [
        ('sentiment_distribution_comparison.png', 'Sentiment Distribution: Pre vs Post COVID'),
        ('vader_score_comparison.png', 'VADER Score Comparison'),
        ('post_volume_comparison.png', 'Post Volume Comparison'),
        ('sentiment_score_distributions.png', 'Detailed Sentiment Score Distributions'),
        ('subreddit_distribution_comparison.png', 'Subreddit Distribution Comparison'),
        ('wordcloud_comparison.png', 'Word Cloud Comparison')
    ]
    
    for img_file, caption in covid_images:
        img_path = f'covid_comparison/{img_file}'
        if os.path.exists(os.path.join(output_dir, img_path)):
            html_content += f"""
                    <div class="image-container">
                        <img src="{img_path}" alt="{caption}">
                        <div class="image-caption">{caption}</div>
                    </div>
            """
    
    html_content += """
                </div>
                
                <h3>Overall EDA</h3>
                <div class="image-grid">
    """
    
    # Add overall EDA images
    eda_images = [
        ('overall_sentiment_distribution.png', 'Overall Sentiment Distribution'),
        ('sentiment_by_subreddit.png', 'Sentiment by Subreddit'),
        ('tfidf_top_terms.png', 'Top TF-IDF Terms'),
        ('topic_modeling.png', 'Topic Modeling Results'),
        ('subreddit_tfidf_comparison.png', 'Subreddit TF-IDF Comparison'),
        ('medical_terms_analysis.png', 'Medical Terms Analysis')
    ]
    
    for img_file, caption in eda_images:
        img_path = f'overall/{img_file}'
        if os.path.exists(os.path.join(output_dir, img_path)):
            html_content += f"""
                    <div class="image-container">
                        <img src="{img_path}" alt="{caption}">
                        <div class="image-caption">{caption}</div>
                    </div>
            """
    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                <ul>
    """
    
    # Generate key findings
    if 'sentiment_compound' in posts_df.columns:
        avg_sentiment = posts_df['sentiment_compound'].mean()
        sentiment_trend = "positive" if avg_sentiment > 0 else "negative"
        html_content += f"""
                    <li>Overall sentiment is <strong>{sentiment_trend}</strong> with an average compound score of <strong>{avg_sentiment:.3f}</strong></li>
        """
    
    if 'covid_period' in posts_df.columns:
        pre_covid_count = len(posts_df[posts_df['covid_period'] == 'Pre-COVID'])
        post_covid_count = len(posts_df[posts_df['covid_period'] == 'Post-COVID'])
        change_pct = ((post_covid_count - pre_covid_count) / pre_covid_count * 100) if pre_covid_count > 0 else 0
        
        change_word = 'increased' if change_pct > 0 else 'decreased'
        html_content += f"""
                    <li>Post volume {change_word} by <strong>{abs(change_pct):.1f}%</strong> after COVID-19 pandemic</li>
        """
    
    if 'subreddit' in posts_df.columns:
        top_sub = posts_df['subreddit'].value_counts().index[0]
        html_content += f"""
                    <li>Most active subreddit: <strong>r/{top_sub}</strong></li>
        """
    
    html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Methodology</h2>
                <p>This analysis was performed using the following techniques:</p>
                <ul>
                    <li><strong>Data Collection:</strong> Reddit posts and comments from 23 pregnancy and health-related subreddits</li>
                    <li><strong>Text Cleaning:</strong> Removal of URLs, special characters, and normalization of text</li>
                    <li><strong>Sentiment Analysis:</strong> VADER (Valence Aware Dictionary and sEntiment Reasoner) for social media text</li>
                    <li><strong>TF-IDF:</strong> Term Frequency-Inverse Document Frequency for keyword extraction</li>
                    <li><strong>Topic Modeling:</strong> Latent Dirichlet Allocation (LDA) for discovering themes</li>
                    <li><strong>Statistical Testing:</strong> T-tests for comparing pre/post COVID sentiment differences</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Files Generated</h2>
                <ul>
                    <li><code>cleaned_data/cleaned_posts.csv</code> - Cleaned posts dataset</li>
                    <li><code>cleaned_data/cleaned_comments.csv</code> - Cleaned comments dataset</li>
                    <li><code>cleaned_data/posts_with_sentiment.csv</code> - Posts with sentiment scores</li>
                    <li><code>analysis_output/overall/sentiment_by_subreddit.csv</code> - Sentiment statistics by subreddit</li>
                    <li><code>analysis_output/overall/tfidf_scores.csv</code> - TF-IDF term scores</li>
                    <li><code>analysis_output/overall/medical_terms_frequency.csv</code> - Medical terms frequency</li>
                </ul>
            </div>
            
            <p class="timestamp">End of Report</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = os.path.join(output_dir, 'analysis_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nHTML report saved to: {report_path}")


def main():
    """Main execution function"""
    print("="*70)
    print(" "*10 + "PRE-ECLAMPSIA REDDIT DATA ANALYSIS")
    print("="*70)
    print("\nStarting comprehensive EDA pipeline...\n")
    
    # Step 1: Data Cleaning
    print("\n" + "█"*70)
    print("STEP 1: DATA CLEANING AND PREPROCESSING")
    print("█"*70)
    cleaner = DataCleaner()
    posts_df, comments_df = cleaner.run_full_cleaning(save_output=True)
    
    if posts_df.empty:
        print("\n❌ Error: No posts data available. Please run data collection first.")
        return
    
    # Step 2: Sentiment Analysis
    print("\n" + "█"*70)
    print("STEP 2: SENTIMENT ANALYSIS")
    print("█"*70)
    sentiment_analyzer = SentimentAnalyzer()
    posts_df, comments_df = sentiment_analyzer.run_full_analysis(
        posts_df, comments_df, save_output=True
    )
    
    # Step 3: Temporal Analysis
    print("\n" + "█"*70)
    print("STEP 3: TEMPORAL TREND ANALYSIS")
    print("█"*70)
    temporal = TemporalAnalyzer()
    temporal_results = temporal.run_full_analysis(posts_df)
    
    # Step 4: COVID Comparison
    print("\n" + "█"*70)
    print("STEP 4: PRE vs POST COVID COMPARISON")
    print("█"*70)
    covid_comp = CovidComparison()
    covid_results = covid_comp.run_full_analysis(posts_df)
    
    # Step 5: Overall EDA
    print("\n" + "█"*70)
    print("STEP 5: OVERALL EXPLORATORY DATA ANALYSIS")
    print("█"*70)
    eda = OverallEDA()
    eda_results = eda.run_full_analysis(posts_df, comments_df)
    
    # Step 6: Generate Report
    print("\n" + "█"*70)
    print("STEP 6: GENERATING COMPREHENSIVE REPORT")
    print("█"*70)
    
    all_results = {
        'temporal': temporal_results,
        'covid': covid_results,
        'eda': eda_results
    }
    
    create_html_report(posts_df, comments_df, all_results)
    
    # Final Summary
    print("\n" + "="*70)
    print(" "*20 + "ANALYSIS COMPLETE!")
    print("="*70)
    print("\n📊 SUMMARY:")
    print(f"   • Posts analyzed: {len(posts_df):,}")
    print(f"   • Comments analyzed: {len(comments_df):,}")
    print(f"   • Subreddits covered: {posts_df['subreddit'].nunique() if 'subreddit' in posts_df.columns else 'N/A'}")
    print(f"   • Time period: {posts_df['created_datetime'].min().strftime('%Y-%m-%d') if 'created_datetime' in posts_df.columns else 'N/A'} to {posts_df['created_datetime'].max().strftime('%Y-%m-%d') if 'created_datetime' in posts_df.columns else 'N/A'}")
    print(f"\n📁 OUTPUT LOCATIONS:")
    print(f"   • Cleaned data: cleaned_data/")
    print(f"   • Temporal analysis: analysis_output/temporal/")
    print(f"   • COVID comparison: analysis_output/covid_comparison/")
    print(f"   • Overall EDA: analysis_output/overall/")
    print(f"   • HTML Report: analysis_output/analysis_report.html")
    print("\n✅ All analyses completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
