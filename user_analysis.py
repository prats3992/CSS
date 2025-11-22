"""
User Behavior Analysis: User overlap, posting frequency, and active user topic analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)


class UserAnalyzer:
    def __init__(self, csv_file=None):
        """
        Initialize user analyzer
        
        Args:
            csv_file: Path to cleaned CSV file
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
        
        print(f"Loaded {len(self.df)} posts for user analysis\n")
        
        # Ensure required columns
        if 'author' not in self.df.columns:
            raise ValueError("Missing 'author' column in data")
    
    def analyze_user_behavior(self, output_dir='user_analysis_output'):
        """Complete user behavior analysis"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("USER BEHAVIOR ANALYSIS")
        print("="*70 + "\n")
        
        # Remove deleted/removed users
        self.df_active = self.df[~self.df['author'].isin(['[deleted]', '[removed]', 'AutoModerator'])].copy()
        
        print(f"Active users (excluding deleted/bots): {self.df_active['author'].nunique()}")
        print(f"Total posts from active users: {len(self.df_active)}\n")
        
        # Analyze posting frequency
        user_post_counts = self.df_active['author'].value_counts()
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Posting frequency histogram
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Create bins
        bins = [1, 2, 3, 4, 5, 10, 20, 50, 100, user_post_counts.max()+1]
        bin_labels = ['1', '2', '3', '4', '5-9', '10-19', '20-49', '50-99', f'100+']
        
        hist_data = pd.cut(user_post_counts, bins=bins, labels=bin_labels[:len(bins)-1], right=False)
        hist_counts = hist_data.value_counts().sort_index()
        
        bars = ax1.bar(range(len(hist_counts)), hist_counts.values, 
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(hist_counts)))
        ax1.set_xticklabels(hist_counts.index, fontsize=11)
        ax1.set_xlabel('Number of Posts', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
        ax1.set_title('User Posting Frequency Distribution', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add statistics text
        stats_text = f"Total Users: {len(user_post_counts):,}\n"
        stats_text += f"One-time posters: {(user_post_counts == 1).sum():,} ({(user_post_counts == 1).sum()/len(user_post_counts)*100:.1f}%)\n"
        stats_text += f"Active users (5+ posts): {(user_post_counts >= 5).sum():,}"
        
        ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Top active users
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        top_users = user_post_counts.head(20)
        top_users_text = "TOP 20 ACTIVE USERS\n" + "="*35 + "\n\n"
        
        for rank, (user, count) in enumerate(top_users.items(), 1):
            pct = (count / len(self.df_active)) * 100
            top_users_text += f"{rank:2d}. {user[:15]:15s} {count:3d} ({pct:.1f}%)\n"
        
        ax2.text(0.05, 0.95, top_users_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 3. User overlap across subreddits
        ax3 = fig.add_subplot(gs[1, :])
        
        if 'subreddit' in self.df_active.columns:
            # Get top subreddits
            top_subreddits = self.df_active['subreddit'].value_counts().head(10).index.tolist()
            
            # Create overlap matrix
            overlap_matrix = np.zeros((len(top_subreddits), len(top_subreddits)))
            
            for i, sub1 in enumerate(top_subreddits):
                users_sub1 = set(self.df_active[self.df_active['subreddit'] == sub1]['author'].unique())
                for j, sub2 in enumerate(top_subreddits):
                    users_sub2 = set(self.df_active[self.df_active['subreddit'] == sub2]['author'].unique())
                    overlap = len(users_sub1 & users_sub2)
                    overlap_matrix[i, j] = overlap
            
            # Plot heatmap
            sns.heatmap(overlap_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                       xticklabels=[f'r/{s}' for s in top_subreddits],
                       yticklabels=[f'r/{s}' for s in top_subreddits],
                       ax=ax3, cbar_kws={'label': 'Number of Shared Users'})
            ax3.set_title('User Overlap Across Subreddits', fontsize=14, fontweight='bold', pad=15)
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Subreddit data not available', 
                    ha='center', va='center', fontsize=14)
            ax3.set_title('User Overlap Across Subreddits', fontsize=14, fontweight='bold')
        
        # 4. Activity metrics by user type
        ax4 = fig.add_subplot(gs[2, 0])
        
        # Categorize users
        self.df_active['user_type'] = self.df_active['author'].map(lambda x: user_post_counts[x])
        self.df_active['user_category'] = pd.cut(self.df_active['user_type'],
                                                 bins=[0, 1, 5, 20, float('inf')],
                                                 labels=['One-time', 'Occasional (2-5)', 
                                                        'Regular (6-20)', 'Power (20+)'])
        
        category_counts = self.df_active['user_category'].value_counts()
        colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        
        wedges, texts, autotexts = ax4.pie(category_counts.values, labels=category_counts.index,
                                           autopct='%1.1f%%', colors=colors, startangle=90)
        ax4.set_title('User Categories by Post Frequency', fontsize=13, fontweight='bold', pad=10)
        
        # 5. Engagement by user category
        ax5 = fig.add_subplot(gs[2, 1])
        
        if 'num_comments' in self.df_active.columns:
            engagement_by_category = self.df_active.groupby('user_category').agg({
                'num_comments': 'mean',
                'score': 'mean'
            })
            
            x = range(len(engagement_by_category))
            width = 0.35
            
            bars1 = ax5.bar([i - width/2 for i in x], engagement_by_category['num_comments'],
                           width, label='Avg Comments', color='#3498db', alpha=0.8)
            
            ax5_twin = ax5.twinx()
            bars2 = ax5_twin.bar([i + width/2 for i in x], engagement_by_category['score'],
                                width, label='Avg Upvotes', color='#e67e22', alpha=0.8)
            
            ax5.set_xticks(x)
            ax5.set_xticklabels(engagement_by_category.index, rotation=20, ha='right', fontsize=9)
            ax5.set_ylabel('Average Comments', fontsize=10, fontweight='bold', color='#3498db')
            ax5_twin.set_ylabel('Average Upvotes', fontsize=10, fontweight='bold', color='#e67e22')
            ax5.set_title('Engagement by User Category', fontsize=13, fontweight='bold', pad=10)
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Combine legends
            lines1, labels1 = ax5.get_legend_handles_labels()
            lines2, labels2 = ax5_twin.get_legend_handles_labels()
            ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        else:
            ax5.text(0.5, 0.5, 'Engagement data not available',
                    ha='center', va='center', fontsize=12, transform=ax5.transAxes)
            ax5.set_title('Engagement by User Category', fontsize=13, fontweight='bold')
        
        # 6. Statistics summary
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        stats_summary = "USER STATISTICS\n" + "="*35 + "\n\n"
        stats_summary += f"Total unique users: {len(user_post_counts):,}\n\n"
        stats_summary += f"Posts per user:\n"
        stats_summary += f"  Mean: {user_post_counts.mean():.2f}\n"
        stats_summary += f"  Median: {user_post_counts.median():.0f}\n"
        stats_summary += f"  Max: {user_post_counts.max():.0f}\n\n"
        
        stats_summary += f"User categories:\n"
        for cat, count in category_counts.items():
            pct = (count / len(self.df_active)) * 100
            stats_summary += f"  {cat}: {count:,} ({pct:.1f}%)\n"
        
        stats_summary += f"\nTop 10% users contribute:\n"
        top_10pct = int(len(user_post_counts) * 0.1)
        top_10pct_posts = user_post_counts.head(top_10pct).sum()
        contrib_pct = (top_10pct_posts / len(self.df_active)) * 100
        stats_summary += f"  {contrib_pct:.1f}% of all posts\n"
        
        ax6.text(0.05, 0.95, stats_summary, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.suptitle('User Behavior & Posting Patterns', fontsize=18, fontweight='bold', y=0.995)
        
        output_file = f'{output_dir}/user_behavior_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
        
        # Analyze active users' topics
        self.analyze_active_user_topics(user_post_counts, output_dir)
        
        # Create detailed report
        self._create_user_report(user_post_counts, output_dir)
    
    def analyze_active_user_topics(self, user_post_counts, output_dir):
        """Analyze what active users post about"""
        print("\nAnalyzing active users' topics...")
        
        # Define active users (5+ posts)
        active_users = user_post_counts[user_post_counts >= 5].index.tolist()
        active_posts = self.df_active[self.df_active['author'].isin(active_users)]
        
        # Get text data
        if 'text_no_stopwords' in active_posts.columns:
            text_col = 'text_no_stopwords'
        else:
            text_col = 'text_cleaned'
        
        # Combine all text from active users
        active_text = ' '.join(active_posts[text_col].fillna('').astype(str))
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Word cloud for active users
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='white',
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate(active_text)
        
        axes[0, 0].imshow(wordcloud, interpolation='bilinear')
        axes[0, 0].axis('off')
        axes[0, 0].set_title(f'Topics from Active Users (5+ posts, n={len(active_users)})',
                            fontsize=14, fontweight='bold', pad=10)
        
        # 2. Word cloud for one-time posters
        onetime_users = user_post_counts[user_post_counts == 1].index.tolist()
        onetime_posts = self.df_active[self.df_active['author'].isin(onetime_users)]
        onetime_text = ' '.join(onetime_posts[text_col].fillna('').astype(str))
        
        wordcloud_onetime = WordCloud(
            width=800,
            height=600,
            background_color='white',
            colormap='plasma',
            relative_scaling=0.5,
            min_font_size=10
        ).generate(onetime_text)
        
        axes[0, 1].imshow(wordcloud_onetime, interpolation='bilinear')
        axes[0, 1].axis('off')
        axes[0, 1].set_title(f'Topics from One-Time Posters (n={len(onetime_users)})',
                            fontsize=14, fontweight='bold', pad=10)
        
        # 3. Top words comparison
        active_words = Counter(active_text.split()).most_common(20)
        onetime_words = Counter(onetime_text.split()).most_common(20)
        
        words = [w[0] for w in active_words]
        counts = [w[1] for w in active_words]
        
        bars = axes[1, 0].barh(range(len(words)), counts, color='#3498db', alpha=0.8)
        axes[1, 0].set_yticks(range(len(words)))
        axes[1, 0].set_yticklabels(words, fontsize=10)
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Top 20 Words - Active Users', fontsize=13, fontweight='bold', pad=10)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        words = [w[0] for w in onetime_words]
        counts = [w[1] for w in onetime_words]
        
        bars = axes[1, 1].barh(range(len(words)), counts, color='#e67e22', alpha=0.8)
        axes[1, 1].set_yticks(range(len(words)))
        axes[1, 1].set_yticklabels(words, fontsize=10)
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_xlabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Top 20 Words - One-Time Posters', fontsize=13, fontweight='bold', pad=10)
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Topic Analysis: Active Users vs One-Time Posters',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = f'{output_dir}/active_users_topics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
        
        # Save topic comparison
        self._save_topic_comparison(active_words, onetime_words, output_dir)
    
    def _save_topic_comparison(self, active_words, onetime_words, output_dir):
        """Save topic comparison to file"""
        topic_file = f'{output_dir}/user_topics_comparison.txt'
        
        with open(topic_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TOPIC COMPARISON: ACTIVE USERS vs ONE-TIME POSTERS\n")
            f.write("="*80 + "\n\n")
            
            f.write("TOP 50 WORDS - ACTIVE USERS (5+ posts):\n")
            f.write("-" * 80 + "\n")
            for rank, (word, count) in enumerate(active_words[:50], 1):
                f.write(f"{rank:2d}. {word:25s} - {count:,}\n")
            
            f.write("\n\nTOP 50 WORDS - ONE-TIME POSTERS:\n")
            f.write("-" * 80 + "\n")
            for rank, (word, count) in enumerate(onetime_words[:50], 1):
                f.write(f"{rank:2d}. {word:25s} - {count:,}\n")
            
            f.write("\n\n" + "="*80 + "\n")
            f.write("KEY DIFFERENCES:\n")
            f.write("="*80 + "\n\n")
            
            active_words_set = set([w[0] for w in active_words[:50]])
            onetime_words_set = set([w[0] for w in onetime_words[:50]])
            
            active_unique = active_words_set - onetime_words_set
            onetime_unique = onetime_words_set - active_words_set
            
            f.write(f"Words unique to active users (in top 50): {len(active_unique)}\n")
            if active_unique:
                f.write(f"  Examples: {', '.join(list(active_unique)[:10])}\n\n")
            
            f.write(f"Words unique to one-time posters (in top 50): {len(onetime_unique)}\n")
            if onetime_unique:
                f.write(f"  Examples: {', '.join(list(onetime_unique)[:10])}\n\n")
            
            f.write("\nINTERPRETATION:\n")
            f.write("-" * 80 + "\n")
            f.write("• Active users tend to discuss: ongoing experiences, community support\n")
            f.write("• One-time posters tend to: seek immediate advice, share specific incidents\n")
        
        print(f"✓ Saved topic comparison to: {topic_file}")
    
    def _create_user_report(self, user_post_counts, output_dir):
        """Create detailed user behavior report"""
        report_file = f'{output_dir}/user_behavior_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("USER BEHAVIOR ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. USER OVERVIEW:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total unique users: {len(user_post_counts):,}\n")
            f.write(f"Total posts: {len(self.df_active):,}\n")
            f.write(f"Average posts per user: {user_post_counts.mean():.2f}\n")
            f.write(f"Median posts per user: {user_post_counts.median():.0f}\n\n")
            
            f.write("2. USER CATEGORIES:\n")
            f.write("-" * 80 + "\n")
            
            categories = {
                'One-time posters (1 post)': (user_post_counts == 1).sum(),
                'Occasional posters (2-5 posts)': ((user_post_counts >= 2) & (user_post_counts <= 5)).sum(),
                'Regular users (6-20 posts)': ((user_post_counts >= 6) & (user_post_counts <= 20)).sum(),
                'Power users (20+ posts)': (user_post_counts >= 20).sum()
            }
            
            for cat, count in categories.items():
                pct = (count / len(user_post_counts)) * 100
                posts = self.df_active[self.df_active['author'].isin(
                    user_post_counts[
                        (user_post_counts >= int(cat.split('(')[1].split()[0].replace('+', ''))) if '+' in cat
                        else (user_post_counts == 1) if 'One-time' in cat
                        else ((user_post_counts >= 2) & (user_post_counts <= 5)) if 'Occasional' in cat
                        else ((user_post_counts >= 6) & (user_post_counts <= 20))
                    ].index
                )].shape[0]
                posts_pct = (posts / len(self.df_active)) * 100
                f.write(f"{cat}:\n")
                f.write(f"  Users: {count:,} ({pct:.1f}%)\n")
                f.write(f"  Posts: {posts:,} ({posts_pct:.1f}%)\n\n")
            
            f.write("3. CONCENTRATION ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            
            top_10pct = int(len(user_post_counts) * 0.1)
            top_25pct = int(len(user_post_counts) * 0.25)
            
            top10_posts = user_post_counts.head(top_10pct).sum()
            top25_posts = user_post_counts.head(top_25pct).sum()
            
            f.write(f"Top 10% of users ({top_10pct:,} users):\n")
            f.write(f"  Contribute {top10_posts:,} posts ({top10_posts/len(self.df_active)*100:.1f}%)\n\n")
            f.write(f"Top 25% of users ({top_25pct:,} users):\n")
            f.write(f"  Contribute {top25_posts:,} posts ({top25_posts/len(self.df_active)*100:.1f}%)\n\n")
            
            f.write("4. TOP 20 MOST ACTIVE USERS:\n")
            f.write("-" * 80 + "\n")
            
            for rank, (user, count) in enumerate(user_post_counts.head(20).items(), 1):
                pct = (count / len(self.df_active)) * 100
                f.write(f"{rank:2d}. {user[:30]:30s} - {count:3d} posts ({pct:.1f}%)\n")
            
            f.write("\n\n5. USER OVERLAP (if subreddit data available):\n")
            f.write("-" * 80 + "\n")
            
            if 'subreddit' in self.df_active.columns:
                multi_sub_users = self.df_active.groupby('author')['subreddit'].nunique()
                users_multi_sub = (multi_sub_users > 1).sum()
                f.write(f"Users posting in multiple subreddits: {users_multi_sub:,} ")
                f.write(f"({users_multi_sub/len(user_post_counts)*100:.1f}%)\n\n")
                
                f.write("Average subreddits per user: ")
                f.write(f"{multi_sub_users.mean():.2f}\n")
            else:
                f.write("Subreddit data not available\n")
            
            f.write("\n\n6. KEY INSIGHTS:\n")
            f.write("="*80 + "\n\n")
            
            onetime_pct = (user_post_counts == 1).sum() / len(user_post_counts) * 100
            if onetime_pct > 50:
                f.write(f"• Majority ({onetime_pct:.1f}%) are one-time posters\n")
                f.write("  → Community serves as a one-time resource for many\n\n")
            
            if top10_posts / len(self.df_active) > 0.5:
                f.write(f"• Top 10% of users create {top10_posts/len(self.df_active)*100:.1f}% of content\n")
                f.write("  → Small core of highly engaged users drive the community\n\n")
            
            f.write("• This pattern is typical for health/support communities where:\n")
            f.write("  - Many users seek one-time advice during crisis\n")
            f.write("  - Few users stay engaged long-term to help others\n")
            f.write("  - Core members provide ongoing support and expertise\n")
        
        print(f"✓ Saved user behavior report to: {report_file}\n")


def main():
    """Run user analysis pipeline"""
    analyzer = UserAnalyzer()
    analyzer.analyze_user_behavior()
    
    print("="*70)
    print("✓ USER ANALYSIS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
