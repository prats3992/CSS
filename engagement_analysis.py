"""
Engagement Analysis: Analyzing how sentiment relates to engagement (upvotes, comments)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)


class EngagementAnalyzer:
    def __init__(self, csv_file=None):
        """
        Initialize engagement analyzer
        
        Args:
            csv_file: Path to sentiment analysis results CSV
        """
        # Load data
        if csv_file:
            self.df = pd.read_csv(csv_file)
        else:
            # Find latest sentiment analysis file
            import glob
            files = glob.glob('sentiment_analysis_results_*.csv')
            if files:
                latest = max(files)
                print(f"Loading latest sentiment results: {latest}")
                self.df = pd.read_csv(latest)
            else:
                raise FileNotFoundError("No sentiment analysis results found!")
        
        print(f"Loaded {len(self.df)} posts for engagement analysis\n")
        
        # Ensure required columns exist
        required = ['score', 'num_comments', 'sentiment_class', 'vader_compound']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def analyze_sentiment_engagement(self, output_dir='engagement_output'):
        """Analyze relationship between sentiment and engagement metrics"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("SENTIMENT & ENGAGEMENT ANALYSIS")
        print("="*70 + "\n")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        
        # 1. Average engagement by sentiment class
        sentiment_order = ['negative', 'neutral', 'positive']
        engagement_by_sentiment = self.df.groupby('sentiment_class').agg({
            'score': 'mean',
            'num_comments': 'mean'
        }).reindex(sentiment_order)
        
        # Upvotes by sentiment
        bars = axes[0, 0].bar(range(3), engagement_by_sentiment['score'], 
                             color=['#e74c3c', '#95a5a6', '#2ecc71'], alpha=0.8, edgecolor='black')
        axes[0, 0].set_xticks(range(3))
        axes[0, 0].set_xticklabels(['Negative', 'Neutral', 'Positive'], fontsize=11)
        axes[0, 0].set_ylabel('Average Score (Upvotes)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Average Upvotes by Sentiment', fontsize=13, fontweight='bold', pad=10)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Comments by sentiment
        bars = axes[0, 1].bar(range(3), engagement_by_sentiment['num_comments'],
                             color=['#e74c3c', '#95a5a6', '#2ecc71'], alpha=0.8, edgecolor='black')
        axes[0, 1].set_xticks(range(3))
        axes[0, 1].set_xticklabels(['Negative', 'Neutral', 'Positive'], fontsize=11)
        axes[0, 1].set_ylabel('Average Comments', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Average Comments by Sentiment', fontsize=13, fontweight='bold', pad=10)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Box plots for engagement distribution
        sentiment_colors = {'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#2ecc71'}
        
        # Upvotes box plot (log scale for better visualization)
        bp_data = [self.df[self.df['sentiment_class'] == s]['score'].values 
                   for s in sentiment_order]
        bp = axes[0, 2].boxplot(bp_data, labels=['Negative', 'Neutral', 'Positive'],
                               patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], ['#e74c3c', '#95a5a6', '#2ecc71']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0, 2].set_ylabel('Score (Upvotes)', fontsize=11, fontweight='bold')
        axes[0, 2].set_title('Upvotes Distribution by Sentiment', fontsize=13, fontweight='bold', pad=10)
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        axes[0, 2].set_yscale('log')
        
        # 3. Comments box plot
        bp_data = [self.df[self.df['sentiment_class'] == s]['num_comments'].values
                   for s in sentiment_order]
        bp = axes[1, 0].boxplot(bp_data, labels=['Negative', 'Neutral', 'Positive'],
                               patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], ['#e74c3c', '#95a5a6', '#2ecc71']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 0].set_ylabel('Number of Comments', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Comments Distribution by Sentiment', fontsize=13, fontweight='bold', pad=10)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_yscale('log')
        
        # 4. Scatter: VADER compound vs upvotes
        scatter = axes[1, 1].scatter(self.df['vader_compound'], self.df['score'],
                                    c=self.df['vader_compound'], cmap='RdYlGn',
                                    alpha=0.5, s=30, edgecolors='none')
        axes[1, 1].set_xlabel('VADER Compound Score', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Score (Upvotes)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Sentiment vs Upvotes', fontsize=13, fontweight='bold', pad=10)
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='Sentiment')
        
        # Add correlation
        corr_score = self.df['vader_compound'].corr(self.df['score'])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_score:.3f}',
                       transform=axes[1, 1].transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. Scatter: VADER compound vs comments
        scatter = axes[1, 2].scatter(self.df['vader_compound'], self.df['num_comments'],
                                    c=self.df['vader_compound'], cmap='RdYlGn',
                                    alpha=0.5, s=30, edgecolors='none')
        axes[1, 2].set_xlabel('VADER Compound Score', fontsize=11, fontweight='bold')
        axes[1, 2].set_ylabel('Number of Comments', fontsize=11, fontweight='bold')
        axes[1, 2].set_title('Sentiment vs Comments', fontsize=13, fontweight='bold', pad=10)
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 2], label='Sentiment')
        
        # Add correlation
        corr_comments = self.df['vader_compound'].corr(self.df['num_comments'])
        axes[1, 2].text(0.05, 0.95, f'Correlation: {corr_comments:.3f}',
                       transform=axes[1, 2].transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. Engagement bins by sentiment score
        self.df['sentiment_bin'] = pd.cut(self.df['vader_compound'], 
                                         bins=[-1, -0.5, -0.05, 0.05, 0.5, 1],
                                         labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
        
        engagement_by_bin = self.df.groupby('sentiment_bin').agg({
            'score': 'mean',
            'num_comments': 'mean'
        })
        
        x = range(len(engagement_by_bin))
        bars1 = axes[2, 0].bar([i - 0.2 for i in x], engagement_by_bin['score'],
                              width=0.4, label='Upvotes', color='#3498db', alpha=0.8)
        bars2 = axes[2, 0].bar([i + 0.2 for i in x], engagement_by_bin['num_comments'],
                              width=0.4, label='Comments', color='#e67e22', alpha=0.8)
        axes[2, 0].set_xticks(x)
        axes[2, 0].set_xticklabels(engagement_by_bin.index, rotation=20, ha='right', fontsize=9)
        axes[2, 0].set_ylabel('Average Count', fontsize=11, fontweight='bold')
        axes[2, 0].set_title('Engagement by Sentiment Intensity', fontsize=13, fontweight='bold', pad=10)
        axes[2, 0].legend(fontsize=10)
        axes[2, 0].grid(True, alpha=0.3, axis='y')
        
        # 7. Statistical test results
        axes[2, 1].axis('off')
        
        # Perform ANOVA for upvotes
        groups_score = [self.df[self.df['sentiment_class'] == s]['score'].values
                       for s in sentiment_order]
        f_stat_score, p_val_score = stats.f_oneway(*groups_score)
        
        # Perform ANOVA for comments
        groups_comments = [self.df[self.df['sentiment_class'] == s]['num_comments'].values
                          for s in sentiment_order]
        f_stat_comments, p_val_comments = stats.f_oneway(*groups_comments)
        
        stats_text = f"""
        STATISTICAL TESTS
        {'='*40}
        
        One-Way ANOVA: Sentiment vs Upvotes
          F-statistic: {f_stat_score:.4f}
          P-value: {p_val_score:.4e}
          {'Significant' if p_val_score < 0.05 else 'Not significant'} (α=0.05)
        
        One-Way ANOVA: Sentiment vs Comments
          F-statistic: {f_stat_comments:.4f}
          P-value: {p_val_comments:.4e}
          {'Significant' if p_val_comments < 0.05 else 'Not significant'} (α=0.05)
        
        INTERPRETATION:
        {'─'*40}
        {'Sentiment significantly affects' if p_val_score < 0.05 else 'No significant difference in'} 
        upvotes across sentiment categories.
        
        {'Sentiment significantly affects' if p_val_comments < 0.05 else 'No significant difference in'}
        comments across sentiment categories.
        """
        
        axes[2, 1].text(0.1, 0.9, stats_text, transform=axes[2, 1].transAxes,
                       fontsize=10, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # 8. Top engaged posts by sentiment
        axes[2, 2].axis('off')
        
        top_engaged_text = "TOP ENGAGED POSTS BY SENTIMENT\n" + "="*40 + "\n\n"
        
        for sentiment in ['positive', 'neutral', 'negative']:
            sentiment_posts = self.df[self.df['sentiment_class'] == sentiment]
            if len(sentiment_posts) > 0:
                top_post = sentiment_posts.nlargest(1, 'num_comments').iloc[0]
                top_engaged_text += f"{sentiment.upper()}:\n"
                top_engaged_text += f"  Score: {int(top_post['score'])}\n"
                top_engaged_text += f"  Comments: {int(top_post['num_comments'])}\n"
                top_engaged_text += f"  VADER: {top_post['vader_compound']:.3f}\n\n"
        
        axes[2, 2].text(0.1, 0.9, top_engaged_text, transform=axes[2, 2].transAxes,
                       fontsize=10, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.suptitle('Sentiment vs Engagement Analysis', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = f'{output_dir}/sentiment_engagement_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
        
        # Create detailed report
        self._create_engagement_report(engagement_by_sentiment, f_stat_score, p_val_score,
                                      f_stat_comments, p_val_comments, corr_score, 
                                      corr_comments, output_dir)
    
    def _create_engagement_report(self, engagement_by_sentiment, f_stat_score, p_val_score,
                                 f_stat_comments, p_val_comments, corr_score, corr_comments, 
                                 output_dir):
        """Create detailed text report on engagement analysis"""
        report_file = f'{output_dir}/engagement_analysis_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SENTIMENT & ENGAGEMENT ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("RESEARCH QUESTION:\n")
            f.write("-" * 80 + "\n")
            f.write("Which sentiment (positive, negative, neutral) generates more engagement\n")
            f.write("in terms of upvotes and comments?\n\n")
            
            f.write("="*80 + "\n")
            f.write("1. AVERAGE ENGAGEMENT BY SENTIMENT\n")
            f.write("="*80 + "\n\n")
            
            for sentiment in ['negative', 'neutral', 'positive']:
                f.write(f"{sentiment.upper()}:\n")
                f.write(f"  Average upvotes: {engagement_by_sentiment.loc[sentiment, 'score']:.2f}\n")
                f.write(f"  Average comments: {engagement_by_sentiment.loc[sentiment, 'num_comments']:.2f}\n\n")
            
            # Find winner
            max_score_sentiment = engagement_by_sentiment['score'].idxmax()
            max_comments_sentiment = engagement_by_sentiment['num_comments'].idxmax()
            
            f.write("WINNERS:\n")
            f.write(f"  Most upvotes: {max_score_sentiment.upper()}\n")
            f.write(f"  Most comments: {max_comments_sentiment.upper()}\n\n")
            
            f.write("="*80 + "\n")
            f.write("2. STATISTICAL SIGNIFICANCE\n")
            f.write("="*80 + "\n\n")
            
            f.write("One-Way ANOVA Test Results:\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("Upvotes vs Sentiment:\n")
            f.write(f"  F-statistic: {f_stat_score:.4f}\n")
            f.write(f"  P-value: {p_val_score:.6f}\n")
            f.write(f"  Result: {'SIGNIFICANT' if p_val_score < 0.05 else 'NOT SIGNIFICANT'} (α=0.05)\n\n")
            
            f.write("Comments vs Sentiment:\n")
            f.write(f"  F-statistic: {f_stat_comments:.4f}\n")
            f.write(f"  P-value: {p_val_comments:.6f}\n")
            f.write(f"  Result: {'SIGNIFICANT' if p_val_comments < 0.05 else 'NOT SIGNIFICANT'} (α=0.05)\n\n")
            
            f.write("="*80 + "\n")
            f.write("3. CORRELATION ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"VADER Compound Score vs Upvotes: {corr_score:.4f}\n")
            if abs(corr_score) < 0.3:
                f.write("  → Weak correlation\n")
            elif abs(corr_score) < 0.7:
                f.write("  → Moderate correlation\n")
            else:
                f.write("  → Strong correlation\n")
            
            f.write(f"\nVADER Compound Score vs Comments: {corr_comments:.4f}\n")
            if abs(corr_comments) < 0.3:
                f.write("  → Weak correlation\n")
            elif abs(corr_comments) < 0.7:
                f.write("  → Moderate correlation\n")
            else:
                f.write("  → Strong correlation\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("4. KEY INSIGHTS\n")
            f.write("="*80 + "\n\n")
            
            # Generate insights
            if max_score_sentiment == max_comments_sentiment:
                f.write(f"• {max_score_sentiment.upper()} posts receive both the most upvotes and comments\n\n")
            else:
                f.write(f"• {max_score_sentiment.upper()} posts receive the most upvotes\n")
                f.write(f"• {max_comments_sentiment.upper()} posts receive the most comments\n\n")
            
            if p_val_score < 0.05:
                f.write("• Sentiment has a statistically significant effect on upvotes\n")
            else:
                f.write("• Sentiment does NOT significantly affect upvotes\n")
            
            if p_val_comments < 0.05:
                f.write("• Sentiment has a statistically significant effect on comments\n\n")
            else:
                f.write("• Sentiment does NOT significantly affect comments\n\n")
            
            if abs(corr_score) > 0.3 or abs(corr_comments) > 0.3:
                f.write("• There is a measurable relationship between sentiment intensity and engagement\n\n")
            
            f.write("="*80 + "\n")
            f.write("5. INTERPRETATION\n")
            f.write("="*80 + "\n\n")
            
            f.write("Possible explanations for engagement patterns:\n\n")
            
            if max_comments_sentiment in ['negative', 'neutral']:
                f.write("• Negative/neutral posts may generate more discussion as people:\n")
                f.write("  - Offer support and advice\n")
                f.write("  - Share similar experiences\n")
                f.write("  - Provide medical information\n\n")
            
            if max_score_sentiment == 'positive':
                f.write("• Positive posts may receive more upvotes as people:\n")
                f.write("  - Celebrate successful outcomes\n")
                f.write("  - Feel encouraged by recovery stories\n")
                f.write("  - Want to amplify good news\n\n")
            
            f.write("• Medical/health communities often show high engagement for:\n")
            f.write("  - Questions seeking advice (typically neutral/negative)\n")
            f.write("  - Success stories (typically positive)\n")
            f.write("  - Emotional support requests (typically negative)\n\n")
            
            f.write("="*80 + "\n")
            f.write("CONCLUSION\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"In pre-eclampsia discussions, {max_comments_sentiment} posts generate the most\n")
            f.write(f"comment engagement, while {max_score_sentiment} posts receive the most upvotes.\n")
            f.write("This reflects the supportive nature of health communities where users actively\n")
            f.write("engage with posts seeking help or sharing challenges.\n")
        
        print(f"✓ Saved engagement report to: {report_file}\n")


def main():
    """Run engagement analysis pipeline"""
    analyzer = EngagementAnalyzer()
    analyzer.analyze_sentiment_engagement()
    
    print("="*70)
    print("✓ ENGAGEMENT ANALYSIS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
