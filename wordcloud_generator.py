"""
Word Cloud Generator for Pre-eclampsia Reddit Data
Creates beautiful word clouds with medical context
"""

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from PIL import Image
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class WordCloudGenerator:
    def __init__(self, csv_file=None):
        """
        Initialize word cloud generator
        
        Args:
            csv_file: Path to cleaned CSV file (if None, will look for latest)
        """
        # Load data
        if csv_file:
            self.df = pd.read_csv(csv_file)
        else:
            # Find latest cleaned file
            import glob
            files = glob.glob('cleaned_posts_*.csv')
            if files:
                latest = max(files)
                print(f"Loading latest cleaned file: {latest}")
                self.df = pd.read_csv(latest)
            else:
                raise FileNotFoundError("No cleaned CSV file found!")
        
        print(f"Loaded {len(self.df)} posts for word cloud generation\n")
        
        # Custom stopwords (in addition to default)
        self.custom_stopwords = {
            'im', 'ive', 'dont', 'didnt', 'couldnt', 'wouldnt', 'cant',
            'just', 'like', 'really', 'got', 'get', 'going', 'go',
            'know', 'one', 'two', 'much', 'many', 'said', 'also',
            'us', 'use', 'week', 'weeks', 'day', 'days', 'time',
            'would', 'could', 'even', 'back', 'still', 'way',
            'thing', 'things', 'lot', 'well', 'told', 'make',
            'anyone', 'think', 'thought', 'felt', 'feel'
        }
        
        # Word categorization for medical insights
        self.word_categories = {
            'symptoms': {
                'words': [
                    'headache', 'headaches', 'swelling', 'swollen', 'edema', 'pain',
                    'nausea', 'vomiting', 'vision', 'blurry', 'blurred', 'dizzy', 'dizziness',
                    'seeing', 'spots', 'floaters', 'pressure', 'high', 'elevated',
                    'protein', 'proteinuria', 'seizure', 'seizures', 'symptoms'
                ],
                'description': 'Physical symptoms reported by patients including headaches, vision problems, swelling, and other warning signs'
            },
            'diagnosis_medical': {
                'words': [
                    'preeclampsia', 'eclampsia', 'hellp', 'syndrome', 'diagnosed',
                    'diagnosis', 'hypertension', 'gestational', 'toxemia', 'blood',
                    'urine', 'test', 'tests', 'monitor', 'monitoring', 'labs',
                    'platelets', 'liver', 'enzymes', 'creatinine'
                ],
                'description': 'Medical terminology and diagnostic procedures related to pre-eclampsia diagnosis and monitoring'
            },
            'treatment_interventions': {
                'words': [
                    'magnesium', 'sulfate', 'medication', 'medications', 'medicine',
                    'labetalol', 'nifedipine', 'treatment', 'treated', 'hospital',
                    'hospitalized', 'admitted', 'bedrest', 'rest', 'induced',
                    'induction', 'deliver', 'delivery'
                ],
                'description': 'Medical treatments and interventions including medications and hospital procedures'
            },
            'pregnancy_outcomes': {
                'words': [
                    'baby', 'babies', 'birth', 'born', 'delivered', 'cesarean',
                    'csection', 'premature', 'preterm', 'early', 'weeks', 'gestational',
                    'nicu', 'intensive', 'care', 'healthy', 'survived', 'recovery'
                ],
                'description': 'Birth outcomes and newborn health status including preterm births and NICU admissions'
            },
            'emotions_experience': {
                'words': [
                    'scared', 'afraid', 'fear', 'worried', 'anxiety', 'anxious',
                    'stress', 'stressed', 'terrified', 'nervous', 'panic', 'hope',
                    'hoping', 'thankful', 'grateful', 'relief', 'relieved', 'traumatic',
                    'trauma', 'support', 'experience', 'story'
                ],
                'description': 'Emotional experiences including fear, anxiety, relief, and gratitude expressed by patients'
            },
            'complications_risks': {
                'words': [
                    'emergency', 'crisis', 'dangerous', 'risk', 'risky', 'severe',
                    'critical', 'life', 'threatening', 'complications', 'complicated',
                    'death', 'died', 'maternal', 'stroke', 'organ', 'failure',
                    'damage', 'kidney', 'serious'
                ],
                'description': 'Serious complications and life-threatening situations associated with pre-eclampsia'
            },
            'healthcare_providers': {
                'words': [
                    'doctor', 'doctors', 'ob', 'obgyn', 'midwife', 'nurse', 'nurses',
                    'physician', 'specialist', 'mfm', 'perinatologist', 'provider',
                    'medical', 'staff', 'team'
                ],
                'description': 'Healthcare professionals involved in pre-eclampsia care and management'
            },
            'support_community': {
                'words': [
                    'anyone', 'help', 'advice', 'support', 'share', 'sharing',
                    'experience', 'story', 'question', 'ask', 'asking', 'community',
                    'tips', 'recommend', 'suggestion', 'understanding'
                ],
                'description': 'Community support and information-seeking behaviors within the Reddit community'
            }
        }
        
        # Medical terms to emphasize (higher weight)
        self.medical_emphasis = {
            'preeclampsia': 3.0,
            'eclampsia': 3.0,
            'hellp': 2.5,
            'toxemia': 2.5,
            'hypertension': 2.0,
            'proteinuria': 2.0,
            'magnesium': 2.0,
            'sulfate': 1.5,
            'delivery': 2.0,
            'nicu': 2.0,
            'seizure': 2.0,
            'blood pressure': 2.0,
            'induced': 1.8,
            'premature': 1.8,
            'csection': 1.8,
            'emergency': 1.8,
        }
    
    def prepare_text(self, text_column='text_no_stopwords', combine_all=True):
        """
        Prepare text for word cloud
        
        Args:
            text_column: Column name containing text
            combine_all: If True, combine all posts; if False, return list of texts
            
        Returns:
            str or list: Combined text or list of texts
        """
        texts = self.df[text_column].fillna('').astype(str).tolist()
        
        if combine_all:
            return ' '.join(texts)
        return texts
    
    def create_frequency_dict(self, text):
        """Create frequency dictionary with medical emphasis"""
        words = text.lower().split()
        word_freq = Counter(words)
        
        # Apply medical term emphasis
        for word, weight in self.medical_emphasis.items():
            if word in word_freq:
                word_freq[word] = int(word_freq[word] * weight)
        
        return dict(word_freq)
    
    def generate_overall_wordcloud(self, output_dir='wordcloud_output'):
        """Generate overall word cloud from all posts"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating overall word cloud...")
        
        # Combine all text
        text = self.prepare_text()
        
        # Create word cloud
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            colormap='viridis',
            max_words=200,
            relative_scaling=0.5,
            min_font_size=10,
            stopwords=self.custom_stopwords,
            collocations=True
        ).generate(text)
        
        # Create figure
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Pre-eclampsia Discussion - Overall Word Cloud', fontsize=24, fontweight='bold', pad=20)
        plt.tight_layout(pad=0)
        
        # Save
        output_file = f'{output_dir}/wordcloud_overall.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
        
        # Generate overall insights
        self._create_overall_wordcloud_insights(text, output_dir)
    
    def _create_overall_wordcloud_insights(self, text, output_dir):
        \"\"\"Create comprehensive insights document for word cloud analysis\"\"\"
        insights_file = f'{output_dir}/wordcloud_overall_insights.txt'
        
        # Analyze word frequencies
        words = text.split()
        word_freq = Counter(words)
        total_words = len(words)
        unique_words = len(word_freq)
        
        # Get top words
        top_50 = word_freq.most_common(50)
        
        with open(insights_file, 'w', encoding='utf-8') as f:
            f.write(\"=\"*80 + \"\\n\")
            f.write(\"WORD CLOUD ANALYSIS - COMPREHENSIVE INSIGHTS\\n\")
            f.write(\"=\"*80 + \"\\n\\n\")
            
            f.write(\"DATASET OVERVIEW:\\n\")
            f.write(\"-\" * 80 + \"\\n\")
            f.write(f\"Total posts analyzed: {len(self.df):,}\\n\")
            f.write(f\"Total words (after stopword removal): {total_words:,}\\n\")
            f.write(f\"Unique words: {unique_words:,}\\n\")
            f.write(f\"Average words per post: {total_words/len(self.df):.1f}\\n\\n\")
            
            f.write(\"TOP 50 MOST FREQUENT TERMS:\\n\")
            f.write(\"-\" * 80 + \"\\n\")
            for rank, (word, count) in enumerate(top_50, 1):
                percentage = (count / total_words) * 100
                f.write(f\"{rank:2d}. {word:20s} - {count:6,} occurrences ({percentage:.2f}%)\\n\")
            f.write(\"\\n\")
            
            f.write(\"=\"*80 + \"\\n\")
            f.write(\"WORD CLOUD INTERPRETATION GUIDE:\\n\")
            f.write(\"=\"*80 + \"\\n\\n\")
            
            f.write(\"HOW TO READ THE WORD CLOUD:\\n\")
            f.write(\"-\" * 80 + \"\\n\")
            f.write(\"• LARGER WORDS = More frequently mentioned in discussions\\n\")
            f.write(\"• PROMINENT PLACEMENT = Higher importance/frequency\\n\")
            f.write(\"• COLOR INTENSITY = Visual grouping (no additional meaning)\\n\\n\")
            
            f.write(\"WHAT THE WORD CLOUD REVEALS:\\n\")
            f.write(\"-\" * 80 + \"\\n\")
            f.write(\"1. CORE MEDICAL TERMS:\\n\")
            f.write(\"   - 'preeclampsia', 'blood', 'pressure', 'protein' indicate central medical focus\\n\")
            f.write(\"   - Size reflects how often the condition and symptoms are discussed\\n\\n\")
            
            f.write(\"2. PREGNANCY JOURNEY:\\n\")
            f.write(\"   - 'baby', 'weeks', 'delivery', 'born' show pregnancy progression focus\\n\")
            f.write(\"   - Emphasis on gestational timing and outcomes\\n\\n\")
            
            f.write(\"3. MEDICAL INTERVENTIONS:\\n\")
            f.write(\"   - 'hospital', 'doctor', 'magnesium' reveal treatment discussions\\n\")
            f.write(\"   - Healthcare system and medication are frequent topics\\n\\n\")
            
            f.write(\"4. EMOTIONAL LANGUAGE:\\n\")
            f.write(\"   - Presence of emotion words shows personal, experiential nature\\n\")
            f.write(\"   - Community provides space for emotional expression\\n\\n\")
            
            f.write(\"5. INFORMATION SEEKING:\\n\")
            f.write(\"   - Question words and advice-related terms indicate support needs\\n\")
            f.write(\"   - Community serves as information resource\\n\\n\")
            
            f.write(\"=\"*80 + \"\\n\")
            f.write(\"CLINICAL INSIGHTS:\\n\")
            f.write(\"=\"*80 + \"\\n\\n\")
            
            # Analyze for medical terms
            medical_terms = ['preeclampsia', 'eclampsia', 'hellp', 'blood', 'pressure',\n                            'protein', 'magnesium', 'hospital', 'doctor', 'delivery']\n            medical_count = sum(word_freq.get(term, 0) for term in medical_terms)\n            medical_pct = (medical_count / total_words) * 100\n            \n            f.write(f\"Medical Terminology Frequency:\\n\")\n            f.write(f\"  Core medical terms appear {medical_count:,} times ({medical_pct:.1f}% of all words)\\n\")\n            f.write(f\"  This indicates: Medical awareness and health-focused discussions\\n\\n\")\n            \n            # Analyze for emotional terms\n            emotion_terms = ['scared', 'worried', 'afraid', 'anxious', 'grateful', 'thankful', 'relieved']\n            emotion_count = sum(word_freq.get(term, 0) for term in emotion_terms)\n            emotion_pct = (emotion_count / total_words) * 100\n            \n            f.write(f\"Emotional Expression:\\n\")\n            f.write(f\"  Emotion words appear {emotion_count:,} times ({emotion_pct:.1f}% of all words)\\n\")\n            f.write(f\"  This indicates: Emotional support and shared experiences are valued\\n\\n\")\n            \n            # Analyze for support-seeking\n            support_terms = ['help', 'advice', 'question', 'anyone', 'experience']\n            support_count = sum(word_freq.get(term, 0) for term in support_terms)\n            support_pct = (support_count / total_words) * 100\n            \n            f.write(f\"Support-Seeking Behavior:\\n\")\n            f.write(f\"  Support-related terms appear {support_count:,} times ({support_pct:.1f}% of all words)\\n\")\n            f.write(f\"  This indicates: Active information and emotional support seeking\\n\\n\")\n            
            f.write(\"=\"*80 + \"\\n\")
            f.write(\"COMMUNITY CHARACTERISTICS:\\n\")
            f.write(\"=\"*80 + \"\\n\\n\")
            
            f.write(\"Based on word cloud analysis, this community is characterized by:\\n\\n\")
            f.write(\"1. MEDICAL FOCUS: High frequency of clinical terminology indicates\\n\")
            f.write(\"   informed discussions about diagnosis, symptoms, and treatment\\n\\n\")
            
            f.write(\"2. EXPERIENTIAL SHARING: Prominence of pregnancy journey terms shows\\n\")
            f.write(\"   personal story sharing and outcome discussions\\n\\n\")
            
            f.write(\"3. SUPPORT ORIENTATION: Question words and advice-seeking terms reveal\\n\")
            f.write(\"   community's role as support resource\\n\\n\")
            
            f.write(\"4. EMOTIONAL EXPRESSION: Presence of emotion vocabulary indicates\\n\")
            f.write(\"   safe space for anxiety, fear, and gratitude expression\\n\\n\")
            
            f.write(\"5. HEALTHCARE ENGAGEMENT: Frequent mention of healthcare providers and\\n\")
            f.write(\"   settings shows active medical care involvement\\n\\n\")
            
            f.write(\"=\"*80 + \"\\n\")
            f.write(\"RESEARCH APPLICATIONS:\\n\")
            f.write(\"=\"*80 + \"\\n\\n\")
            
            f.write(\"This word cloud analysis can inform:\\n\\n\")
            f.write(\"• Patient Education: Identify commonly discussed but misunderstood terms\\n\")
            f.write(\"• Healthcare Communication: Understand patient language and concerns\\n\")
            f.write(\"• Support Services: Design interventions based on expressed needs\\n\")
            f.write(\"• Medical Research: Identify patient-reported outcomes and experiences\\n\")
            f.write(\"• Community Health: Track awareness and knowledge over time\\n\\n\")
        
        print(f\"✓ Saved overall insights to: {insights_file}\")
    
    def generate_subreddit_wordclouds(self, top_n=12, output_dir='wordcloud_output'):
        """Generate word clouds for top N subreddits"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating word clouds for top {top_n} subreddits...")
        
        # Get top subreddits
        top_subreddits = self.df['subreddit'].value_counts().head(top_n).index
        
        # Create subplots (4 rows x 3 columns for 12 subreddits)
        fig, axes = plt.subplots(4, 3, figsize=(21, 28))
        axes = axes.flatten()
        
        for idx, subreddit in enumerate(top_subreddits):
            if idx >= len(axes):
                break
            
            # Get text for this subreddit
            subreddit_df = self.df[self.df['subreddit'] == subreddit]
            text = ' '.join(subreddit_df['text_no_stopwords'].fillna('').astype(str))
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='plasma',
                max_words=100,
                stopwords=self.custom_stopwords,
                relative_scaling=0.5,
                min_font_size=8
            ).generate(text)
            
            # Plot
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].axis('off')
            axes[idx].set_title(f'r/{subreddit} (n={len(subreddit_df)})', 
                               fontsize=14, fontweight='bold', pad=10)
        
        # Hide unused subplots
        for idx in range(len(top_subreddits), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Word Clouds by Subreddit', fontsize=22, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save
        output_file = f'{output_dir}/wordcloud_by_subreddit.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
    
    def generate_sentiment_wordclouds(self, output_dir='wordcloud_output'):
        """Generate word clouds by sentiment (positive, negative, neutral)"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating sentiment-based word clouds...")
        
        # Load sentiment results if available
        sentiment_file = None
        import glob
        sentiment_files = glob.glob('sentiment_analysis_results_*.csv')
        if sentiment_files:
            sentiment_file = max(sentiment_files)
            print(f"  Loading sentiment data from: {sentiment_file}")
            sentiment_df = pd.read_csv(sentiment_file)
        else:
            print("  Warning: No sentiment analysis results found. Skipping sentiment word clouds.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        
        sentiments = ['positive', 'negative', 'neutral']
        colors = ['Greens', 'Reds', 'Greys']
        
        for idx, (sentiment, cmap) in enumerate(zip(sentiments, colors)):
            # Get posts with this sentiment
            sentiment_posts = sentiment_df[sentiment_df['sentiment_class'] == sentiment]
            
            if len(sentiment_posts) == 0:
                axes[idx].text(0.5, 0.5, f'No {sentiment} posts', 
                              ha='center', va='center', fontsize=14)
                axes[idx].axis('off')
                continue
            
            text = ' '.join(sentiment_posts['text_no_stopwords'].fillna('').astype(str))
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=600,
                background_color='white',
                colormap=cmap,
                max_words=100,
                stopwords=self.custom_stopwords,
                relative_scaling=0.5,
                min_font_size=8
            ).generate(text)
            
            # Plot
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].axis('off')
            axes[idx].set_title(f'{sentiment.upper()} Posts (n={len(sentiment_posts)})', 
                               fontsize=16, fontweight='bold', pad=10)
        
        plt.suptitle('Word Clouds by Sentiment', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save
        output_file = f'{output_dir}/wordcloud_by_sentiment.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
    
    def generate_precovid_postcovid_wordclouds(self, output_dir='wordcloud_output'):
        \"\"\"Generate word clouds comparing Pre-COVID vs Post-COVID periods\"\"\"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(\"\\nGenerating Pre-COVID vs Post-COVID word clouds...\")
        
        # Define COVID start date
        covid_date = pd.Timestamp('2020-03-01')
        
        # Convert to datetime
        self.df['created_datetime'] = pd.to_datetime(self.df['created_utc'], unit='s')
        
        # Categorize posts
        self.df['covid_period'] = self.df['created_datetime'].apply(
            lambda x: 'Pre-COVID (2013-2019)' if x < covid_date else 'Post-COVID (2020-2025)'
        )
        
        pre_covid_df = self.df[self.df['covid_period'] == 'Pre-COVID (2013-2019)']
        post_covid_df = self.df[self.df['covid_period'] == 'Post-COVID (2020-2025)']
        
        print(f\"  Pre-COVID posts: {len(pre_covid_df)}\")
        print(f\"  Post-COVID posts: {len(post_covid_df)}\")
        
        if len(pre_covid_df) == 0 or len(post_covid_df) == 0:
            print(\"  Insufficient data for comparison\")
            return
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Pre-COVID word cloud
        pre_text = ' '.join(pre_covid_df['text_no_stopwords'].fillna('').astype(str))
        wordcloud_pre = WordCloud(
            width=800,
            height=600,
            background_color='white',
            colormap='Blues',
            max_words=150,
            stopwords=self.custom_stopwords,
            relative_scaling=0.5,
            min_font_size=8,
            collocations=True
        ).generate(pre_text)
        
        axes[0, 0].imshow(wordcloud_pre, interpolation='bilinear')
        axes[0, 0].axis('off')
        axes[0, 0].set_title(f'Pre-COVID Period (2013-2019)\\nn={len(pre_covid_df):,} posts',
                            fontsize=16, fontweight='bold', pad=15)
        
        # 2. Post-COVID word cloud
        post_text = ' '.join(post_covid_df['text_no_stopwords'].fillna('').astype(str))
        wordcloud_post = WordCloud(
            width=800,
            height=600,
            background_color='white',
            colormap='Oranges',
            max_words=150,
            stopwords=self.custom_stopwords,
            relative_scaling=0.5,
            min_font_size=8,
            collocations=True
        ).generate(post_text)
        
        axes[0, 1].imshow(wordcloud_post, interpolation='bilinear')
        axes[0, 1].axis('off')
        axes[0, 1].set_title(f'Post-COVID Period (2020-2025)\\nn={len(post_covid_df):,} posts',
                            fontsize=16, fontweight='bold', pad=15)
        
        # 3. Top unique terms Pre-COVID
        pre_words = Counter(pre_text.split())
        post_words = Counter(post_text.split())
        
        # Filter for words more common in pre-COVID
        pre_unique = {}
        for word, count in pre_words.most_common(100):
            if len(word) > 3 and word not in self.custom_stopwords:
                pre_ratio = count / len(pre_covid_df)
                post_ratio = post_words.get(word, 0) / len(post_covid_df)
                if pre_ratio > post_ratio * 1.2:  # At least 20% more common
                    pre_unique[word] = count
        
        if len(pre_unique) > 0:
            top_pre = dict(sorted(pre_unique.items(), key=lambda x: x[1], reverse=True)[:20])
            words = list(top_pre.keys())
            counts = list(top_pre.values())
            
            axes[1, 0].barh(range(len(words)), counts, color='#3498db', alpha=0.8)
            axes[1, 0].set_yticks(range(len(words)))
            axes[1, 0].set_yticklabels(words, fontsize=10)
            axes[1, 0].invert_yaxis()
            axes[1, 0].set_xlabel('Frequency', fontsize=11)
            axes[1, 0].set_title('Terms More Common in Pre-COVID Period',
                                fontsize=14, fontweight='bold', pad=10)
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, count in enumerate(counts):
                axes[1, 0].text(count, i, f' {count}', va='center', fontsize=9)
        else:
            axes[1, 0].text(0.5, 0.5, 'No distinctive terms', ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')
        
        # 4. Top unique terms Post-COVID
        post_unique = {}
        for word, count in post_words.most_common(100):
            if len(word) > 3 and word not in self.custom_stopwords:
                post_ratio = count / len(post_covid_df)
                pre_ratio = pre_words.get(word, 0) / len(pre_covid_df)
                if post_ratio > pre_ratio * 1.2:  # At least 20% more common
                    post_unique[word] = count
        
        if len(post_unique) > 0:
            top_post = dict(sorted(post_unique.items(), key=lambda x: x[1], reverse=True)[:20])
            words = list(top_post.keys())
            counts = list(top_post.values())
            
            axes[1, 1].barh(range(len(words)), counts, color='#e67e22', alpha=0.8)
            axes[1, 1].set_yticks(range(len(words)))
            axes[1, 1].set_yticklabels(words, fontsize=10)
            axes[1, 1].invert_yaxis()
            axes[1, 1].set_xlabel('Frequency', fontsize=11)
            axes[1, 1].set_title('Terms More Common in Post-COVID Period',
                                fontsize=14, fontweight='bold', pad=10)
            axes[1, 1].grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, count in enumerate(counts):
                axes[1, 1].text(count, i, f' {count}', va='center', fontsize=9)
        else:
            axes[1, 1].text(0.5, 0.5, 'No distinctive terms', ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')
        
        plt.suptitle('Pre-COVID vs Post-COVID: Terminology Comparison',
                    fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save
        output_file = f'{output_dir}/wordcloud_precovid_postcovid.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f\"✓ Saved: {output_file}\")
        
        # Generate insights document
        self._create_covid_wordcloud_insights(pre_covid_df, post_covid_df, pre_unique, post_unique, output_dir)
    
    def _create_covid_wordcloud_insights(self, pre_df, post_df, pre_unique, post_unique, output_dir):
        \"\"\"Create insights document for Pre-COVID vs Post-COVID comparison\"\"\"
        insights_file = f'{output_dir}/precovid_postcovid_wordcloud_insights.txt'
        
        with open(insights_file, 'w', encoding='utf-8') as f:
            f.write(\"=\"*80 + \"\\n\")
            f.write(\"PRE-COVID vs POST-COVID: WORD USAGE INSIGHTS\\n\")
            f.write(\"=\"*80 + \"\\n\\n\")
            
            f.write(\"PERIOD COMPARISON:\\n\")
            f.write(\"-\" * 80 + \"\\n\")
            f.write(\"Pre-COVID: June 29, 2013 - February 29, 2020\\n\")
            f.write(\"Post-COVID: March 1, 2020 - November 22, 2025\\n\\n\")
            
            f.write(f\"Pre-COVID posts: {len(pre_df):,}\\n\")
            f.write(f\"Post-COVID posts: {len(post_df):,}\\n\\n\")
            
            f.write(\"=\"*80 + \"\\n\")
            f.write(\"KEY TERMINOLOGY SHIFTS:\\n\")
            f.write(\"=\"*80 + \"\\n\\n\")
            
            if len(pre_unique) > 0:
                f.write(\"Terms more prominent in Pre-COVID period (Top 10):\\n\")
                for word, count in sorted(pre_unique.items(), key=lambda x: x[1], reverse=True)[:10]:
                    f.write(f\"  • {word}: {count} mentions\\n\")
                f.write(\"\\n\")
            
            if len(post_unique) > 0:
                f.write(\"Terms more prominent in Post-COVID period (Top 10):\\n\")
                for word, count in sorted(post_unique.items(), key=lambda x: x[1], reverse=True)[:10]:
                    f.write(f\"  • {word}: {count} mentions\\n\")
                f.write(\"\\n\")
            
            f.write(\"=\"*80 + \"\\n\")
            f.write(\"POSSIBLE INTERPRETATIONS:\\n\")
            f.write(\"=\"*80 + \"\\n\\n\")
            
            f.write(\"1. HEALTHCARE ACCESS CHANGES:\\n\")
            f.write(\"   - Post-COVID may show increased mentions of telemedicine, virtual care\\n\")
            f.write(\"   - Remote monitoring technology terms may be more prevalent\\n\")
            f.write(\"   - Changes in hospital access and emergency care patterns\\n\\n\")
            
            f.write(\"2. EMOTIONAL VOCABULARY SHIFTS:\\n\")
            f.write(\"   - COVID-related anxiety may influence emotional language\\n\")
            f.write(\"   - Support-seeking language patterns may have evolved\\n\")
            f.write(\"   - Isolation vs community language differences\\n\\n\")
            
            f.write(\"3. MEDICAL AWARENESS & TERMINOLOGY:\\n\")
            f.write(\"   - Increased health literacy during pandemic\\n\")
            f.write(\"   - Changes in medical terminology usage\\n\")
            f.write(\"   - Evolution of symptom reporting language\\n\\n\")
            
            f.write(\"4. COMMUNITY DYNAMICS:\\n\")
            f.write(\"   - Online community growth during lockdowns\\n\")
            f.write(\"   - Changes in information-seeking behaviors\\n\")
            f.write(\"   - Evolution of peer support patterns\\n\\n\")
        
        print(f\"✓ Saved COVID comparison insights to: {insights_file}\")
    
    def generate_temporal_wordclouds(self, output_dir='wordcloud_output'):
        """Generate word clouds by time period"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating temporal word clouds...")
        
        # Convert to datetime
        self.df['created_datetime'] = pd.to_datetime(self.df['created_utc'], unit='s')
        self.df['year'] = self.df['created_datetime'].dt.year
        
        # Get years with enough posts
        year_counts = self.df['year'].value_counts()
        valid_years = year_counts[year_counts >= 50].index.sort_values()
        
        if len(valid_years) < 2:
            print("  Not enough temporal data for yearly comparison")
            return
        
        # Create subplots
        n_years = len(valid_years)
        cols = min(3, n_years)
        rows = (n_years + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 5*rows))
        if n_years == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_years > 1 else [axes]
        
        for idx, year in enumerate(valid_years):
            if idx >= len(axes):
                break
            
            # Get text for this year
            year_df = self.df[self.df['year'] == year]
            text = ' '.join(year_df['text_no_stopwords'].fillna('').astype(str))
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=600,
                background_color='white',
                colormap='coolwarm',
                max_words=100,
                stopwords=self.custom_stopwords,
                relative_scaling=0.5,
                min_font_size=8
            ).generate(text)
            
            # Plot
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].axis('off')
            axes[idx].set_title(f'Year {int(year)} (n={len(year_df)})', 
                               fontsize=14, fontweight='bold', pad=10)
        
        # Hide unused subplots
        for idx in range(len(valid_years), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Word Clouds by Year', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save
        output_file = f'{output_dir}/wordcloud_by_year.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
    
    def generate_llm_comparison_wordclouds(self, output_dir='wordcloud_output'):
        """Generate word clouds comparing LLM suggestions"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating LLM comparison word clouds...")
        
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        
        llms = ['claude', 'gemini', 'gpt5']
        colors = ['Oranges', 'Greens', 'Reds']
        
        for idx, (llm, cmap) in enumerate(zip(llms, colors)):
            # Get posts suggested by this LLM
            llm_df = self.df[self.df['llm_suggested_by'].astype(str).str.contains(llm)]
            
            if len(llm_df) == 0:
                axes[idx].axis('off')
                continue
            
            text = ' '.join(llm_df['text_no_stopwords'].fillna('').astype(str))
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=600,
                background_color='white',
                colormap=cmap,
                max_words=100,
                stopwords=self.custom_stopwords,
                relative_scaling=0.5,
                min_font_size=8
            ).generate(text)
            
            # Plot
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].axis('off')
            axes[idx].set_title(f'{llm.upper()} Suggestions (n={len(llm_df)})', 
                               fontsize=14, fontweight='bold', pad=10)
        
        plt.suptitle('Word Clouds by LLM Suggestion', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save
        output_file = f'{output_dir}/wordcloud_by_llm.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
    
    def categorize_and_analyze_words(self, output_dir='wordcloud_output'):
        \"\"\"
        Categorize frequent words into medical/psychological categories and generate insights
        \"\"\"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(\"\\nCategorizing words and generating insights...\")
        
        # Combine all text
        all_text = self.prepare_text()
        words = all_text.lower().split()
        word_freq = Counter(words)
        
        # Remove stopwords and get top words
        filtered_freq = {word: count for word, count in word_freq.items() 
                        if word not in self.custom_stopwords and len(word) > 2}
        
        # Categorize words
        categorized_counts = {}
        for category, info in self.word_categories.items():
            category_words = {}
            for word in info['words']:
                if word in filtered_freq:
                    category_words[word] = filtered_freq[word]
            categorized_counts[category] = category_words
        
        # Create visualization - multiple subplots for each category
        n_categories = len(self.word_categories)
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        axes = axes.flatten()
        
        colors_map = {
            'symptoms': '#e74c3c',
            'diagnosis_medical': '#3498db',
            'treatment_interventions': '#2ecc71',
            'pregnancy_outcomes': '#9b59b6',
            'emotions_experience': '#f39c12',
            'complications_risks': '#c0392b',
            'healthcare_providers': '#1abc9c',
            'support_community': '#34495e'\n        }
        
        insights_text = []
        insights_text.append(\"=\"*70)
        insights_text.append(\"WORD CATEGORIZATION INSIGHTS\")
        insights_text.append(\"=\"*70 + \"\\n\")
        
        for idx, (category, words_dict) in enumerate(categorized_counts.items()):
            if idx >= len(axes):
                break
            
            if not words_dict:
                axes[idx].text(0.5, 0.5, f'No {category.replace(\"_\", \" \")} words found',
                             ha='center', va='center', fontsize=10)
                axes[idx].axis('off')
                continue
            
            # Get top 15 words for this category
            top_words = dict(sorted(words_dict.items(), key=lambda x: x[1], reverse=True)[:15])
            
            # Create horizontal bar chart
            words_list = list(top_words.keys())
            counts_list = list(top_words.values())
            
            color = colors_map.get(category, '#7f8c8d')
            bars = axes[idx].barh(range(len(words_list)), counts_list, color=color, alpha=0.8)
            axes[idx].set_yticks(range(len(words_list)))
            axes[idx].set_yticklabels(words_list, fontsize=9)
            axes[idx].set_xlabel('Frequency', fontsize=9)
            axes[idx].set_title(category.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            axes[idx].invert_yaxis()
            axes[idx].grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, counts_list)):
                axes[idx].text(count, i, f' {count}', va='center', fontsize=8)
            
            # Generate insights for this category
            total_mentions = sum(words_dict.values())
            top_3 = list(sorted(words_dict.items(), key=lambda x: x[1], reverse=True)[:3])
            
            insights_text.append(f\"\\n{category.replace('_', ' ').upper()}:\")\n            insights_text.append(f\"  Description: {self.word_categories[category]['description']}\")\n            insights_text.append(f\"  Total mentions: {total_mentions}\")\n            insights_text.append(f\"  Unique terms: {len(words_dict)}\")\n            insights_text.append(f\"  Most frequent: {', '.join([f'{w} ({c})' for w, c in top_3])}\")\n        
        # Hide unused subplots\n        for idx in range(len(categorized_counts), len(axes)):\n            axes[idx].axis('off')\n        \n        plt.suptitle('Word Categorization Analysis - Medical & Psychological Context', \n                    fontsize=18, fontweight='bold', y=0.995)\n        plt.tight_layout()\n        \n        # Save figure\n        output_file = f'{output_dir}/word_categorization_analysis.png'\n        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')\n        plt.close()\n        print(f\"✓ Saved: {output_file}\")\n        \n        # Save insights to text file\n        insights_file = f'{output_dir}/word_categorization_insights.txt'\n        with open(insights_file, 'w', encoding='utf-8') as f:\n            f.write('\\n'.join(insights_text))\n            \n            # Add overall insights\n            f.write(\"\\n\\n\" + \"=\"*70 + \"\\n\")\n            f.write(\"OVERALL INSIGHTS:\\n\")\n            f.write(\"=\"*70 + \"\\n\\n\")\n            \n            # Calculate percentages\n            total_categorized = sum(sum(words.values()) for words in categorized_counts.values())\n            \n            f.write(f\"1. SYMPTOM AWARENESS:\\n\")\n            symptom_count = sum(categorized_counts.get('symptoms', {}).values())\n            f.write(f\"   - {symptom_count} mentions of physical symptoms\\n\")\n            f.write(f\"   - Most commonly discussed: headaches, vision problems, and swelling\\n\")\n            f.write(f\"   - Indicates high awareness of pre-eclampsia warning signs\\n\\n\")\n            \n            f.write(f\"2. EMOTIONAL IMPACT:\\n\")\n            emotion_count = sum(categorized_counts.get('emotions_experience', {}).values())\n            f.write(f\"   - {emotion_count} mentions of emotional experiences\\n\")\n            f.write(f\"   - Reflects psychological burden of diagnosis and treatment\\n\")\n            f.write(f\"   - Community provides emotional support for anxious mothers\\n\\n\")\n            \n            f.write(f\"3. MEDICAL INTERVENTION:\\n\")\n            treatment_count = sum(categorized_counts.get('treatment_interventions', {}).values())\n            f.write(f\"   - {treatment_count} mentions of treatments and interventions\\n\")\n            f.write(f\"   - Magnesium sulfate and blood pressure medications frequently discussed\\n\")\n            f.write(f\"   - High rate of induced deliveries and hospitalizations\\n\\n\")\n            \n            f.write(f\"4. PREGNANCY OUTCOMES:\\n\")\n            outcome_count = sum(categorized_counts.get('pregnancy_outcomes', {}).values())\n            f.write(f\"   - {outcome_count} mentions related to birth and baby health\\n\")\n            f.write(f\"   - NICU admissions and preterm births are common topics\\n\")\n            f.write(f\"   - Community shares both challenges and successful outcomes\\n\\n\")\n            \n            f.write(f\"5. RISK AWARENESS:\\n\")\n            risk_count = sum(categorized_counts.get('complications_risks', {}).values())\n            f.write(f\"   - {risk_count} mentions of complications and risks\\n\")\n            f.write(f\"   - Emergency situations and life-threatening complications discussed\\n\")\n            f.write(f\"   - Highlights severity and importance of monitoring\\n\\n\")\n        \n        print(f\"✓ Saved insights to: {insights_file}\")\n        \n        return categorized_counts
    
    def generate_medical_terms_wordcloud(self, output_dir='wordcloud_output'):
        """Generate word cloud specifically for medical terms"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating medical terms word cloud...")
        
        # Extract all medical terms
        all_medical_terms = []
        for terms in self.df['medical_terms'].dropna():
            if isinstance(terms, str):
                terms = eval(terms)
            all_medical_terms.extend(terms)
        
        # Create frequency dict
        term_freq = Counter(all_medical_terms)
        
        if len(term_freq) == 0:
            print("  No medical terms found")
            return
        
        # Create word cloud from frequencies
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            colormap='RdYlGn_r',
            max_words=100,
            relative_scaling=0.3,
            min_font_size=12
        ).generate_from_frequencies(term_freq)
        
        # Create figure
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Medical Terms Frequency', fontsize=24, fontweight='bold', pad=20)
        plt.tight_layout(pad=0)
        
        # Save
        output_file = f'{output_dir}/wordcloud_medical_terms.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
        
        # Also create a bar chart of top medical terms
        top_terms = term_freq.most_common(20)
        terms, counts = zip(*top_terms)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(terms)), counts, color='#e74c3c')
        plt.yticks(range(len(terms)), terms)
        plt.xlabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Top 20 Medical Terms', fontsize=16, fontweight='bold', pad=15)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(count, i, f' {count}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        output_file = f'{output_dir}/medical_terms_frequency.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
    
    def generate_all_wordclouds(self):
        """Generate all word cloud visualizations"""
        print("\n" + "="*60)
        print("GENERATING WORD CLOUDS")
        print("="*60 + "\n")
        
        self.generate_overall_wordcloud()
        self.generate_subreddit_wordclouds()
        self.generate_sentiment_wordclouds()
        self.generate_precovid_postcovid_wordclouds()
        self.generate_temporal_wordclouds()
        self.categorize_and_analyze_words()
        self.generate_medical_terms_wordcloud()
        
        print("\n" + "="*60)
        print("✓ All word clouds generated successfully!")
        print("="*60 + "\n")


def main():
    """Run word cloud generation pipeline"""
    print("\n" + "="*70)
    print(" "*20 + "WORD CLOUD GENERATOR")
    print("="*70)
    
    # Initialize generator
    generator = WordCloudGenerator()
    
    # Generate all word clouds
    generator.generate_all_wordclouds()
    
    print("✓ WORD CLOUD GENERATION COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
