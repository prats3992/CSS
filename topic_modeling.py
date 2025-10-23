# Topic Modeling using LDA with Gensim
import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicModeler:
    def __init__(self):
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.coherence_model = None

    def prepare_corpus(self, texts):
        """Prepare corpus for LDA modeling"""
        logger.info("Preparing corpus for topic modeling...")

        # Create dictionary
        self.dictionary = corpora.Dictionary(texts)

        # Filter extremes
        self.dictionary.filter_extremes(
            no_below=2,  # Ignore tokens that appear in less than 2 documents
            no_above=0.5,  # Ignore tokens that appear in more than 50% of documents
            keep_n=2000  # Keep only the most frequent 2000 tokens
        )

        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]

        logger.info(f"Dictionary size: {len(self.dictionary)}")
        logger.info(f"Corpus size: {len(self.corpus)}")

        return self.dictionary, self.corpus

    def find_optimal_topics(self, texts, min_topics=2, max_topics=15):
        """Find optimal number of topics using coherence score"""
        logger.info("Finding optimal number of topics...")

        if not self.corpus:
            self.prepare_corpus(texts)

        coherence_scores = []
        topic_numbers = range(min_topics, max_topics + 1)

        for num_topics in topic_numbers:
            logger.info(f"Testing {num_topics} topics...")

            # Create LDA model
            lda_model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=Config.RANDOM_STATE,
                passes=5,  # Reduced for optimization search
                alpha=Config.LDA_ALPHA,
                eta=Config.LDA_BETA
            )

            # Calculate coherence score
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=texts,
                dictionary=self.dictionary,
                coherence='c_v'
            )

            coherence_score = coherence_model.get_coherence()
            coherence_scores.append(coherence_score)

            logger.info(f"Coherence score for {num_topics} topics: {coherence_score:.4f}")

        # Plot coherence scores
        plt.figure(figsize=(10, 6))
        plt.plot(topic_numbers, coherence_scores, marker='o')
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score')
        plt.title('Topic Coherence Scores')
        plt.grid(True)
        plt.savefig(f'{Config.VISUALIZATIONS_DIR}/topic_coherence_scores.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

        # Find optimal number
        optimal_topics = topic_numbers[np.argmax(coherence_scores)]
        logger.info(f"Optimal number of topics: {optimal_topics}")

        return optimal_topics, coherence_scores

    def train_lda_model(self, texts, num_topics=None):
        """Train LDA model"""
        if num_topics is None:
            num_topics = Config.NUM_TOPICS

        logger.info(f"Training LDA model with {num_topics} topics...")

        if not self.corpus:
            self.prepare_corpus(texts)

        # Train LDA model
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=Config.RANDOM_STATE,
            passes=Config.LDA_PASSES,
            iterations=Config.LDA_ITERATIONS,
            alpha=Config.LDA_ALPHA,
            eta=Config.LDA_BETA,
            per_word_topics=True
        )

        # Calculate coherence
        self.coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )

        coherence_score = self.coherence_model.get_coherence()
        logger.info(f"Model coherence score: {coherence_score:.4f}")

        # Save model
        Config.create_directories()
        self.lda_model.save(f'{Config.MODELS_DIR}/lda_model')
        self.dictionary.save(f'{Config.MODELS_DIR}/dictionary')

        with open(f'{Config.MODELS_DIR}/corpus.pkl', 'wb') as f:
            pickle.dump(self.corpus, f)

        return self.lda_model

    def print_topics(self, num_words=10):
        """Print topics with their top words"""
        if not self.lda_model:
            logger.error("LDA model not trained yet!")
            return

        logger.info("\nTopics and their top words:")
        topics = self.lda_model.print_topics(num_words=num_words)

        topic_labels = []
        for i, topic in enumerate(topics):
            print(f"\nTopic {i}:")
            print(topic[1])

            # Extract top words for labeling
            words = topic[1].split(' + ')
            top_words = [word.split('*')[1].strip('"') for word in words[:3]]
            label = f"Topic_{i}: {', '.join(top_words)}"
            topic_labels.append(label)

        return topic_labels

    def get_document_topics(self, texts):
        """Get topic distribution for each document"""
        if not self.lda_model:
            logger.error("LDA model not trained yet!")
            return None

        doc_topics = []
        for text in texts:
            if isinstance(text, list):
                bow = self.dictionary.doc2bow(text)
            else:
                bow = self.dictionary.doc2bow(text.split())

            topics = self.lda_model.get_document_topics(bow)

            # Convert to dictionary format
            topic_dist = {}
            for topic_id, prob in topics:
                topic_dist[f'topic_{topic_id}'] = prob

            # Find dominant topic
            if topics:
                dominant_topic = max(topics, key=lambda x: x[1])
                topic_dist['dominant_topic'] = dominant_topic[0]
                topic_dist['dominant_topic_prob'] = dominant_topic[1]
            else:
                topic_dist['dominant_topic'] = -1
                topic_dist['dominant_topic_prob'] = 0.0

            doc_topics.append(topic_dist)

        return doc_topics

    def visualize_topics(self):
        """Create interactive topic visualization"""
        if not self.lda_model:
            logger.error("LDA model not trained yet!")
            return

        logger.info("Creating topic visualization...")

        # Prepare visualization
        vis = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)

        # Save as HTML
        pyLDAvis.save_html(vis, f'{Config.VISUALIZATIONS_DIR}/lda_topics_interactive.html')

        # Display in notebook
        pyLDAvis.display(vis)

        return vis

    def create_topic_heatmap(self, doc_topics_df):
        """Create heatmap of topic distributions"""
        topic_columns = [col for col in doc_topics_df.columns if col.startswith('topic_')]

        if not topic_columns:
            logger.error("No topic columns found!")
            return

        # Calculate average topic distributions
        topic_means = doc_topics_df[topic_columns].mean()

        # Create heatmap data
        topic_matrix = doc_topics_df[topic_columns].values

        plt.figure(figsize=(12, 8))
        sns.heatmap(topic_matrix.T, 
                   cmap='YlOrRd', 
                   xticklabels=False,
                   yticklabels=[f'Topic {i}' for i in range(len(topic_columns))],
                   cbar_kws={'label': 'Topic Probability'})

        plt.title('Document-Topic Distribution Heatmap')
        plt.xlabel('Documents')
        plt.ylabel('Topics')
        plt.tight_layout()
        plt.savefig(f'{Config.VISUALIZATIONS_DIR}/topic_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv(Config.PROCESSED_DATA_FILE)

    # Convert token strings back to lists
    df['tokens_lemmatized'] = df['tokens_lemmatized'].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )

    # Initialize topic modeler
    modeler = TopicModeler()

    # Find optimal topics (optional)
    # optimal_topics, scores = modeler.find_optimal_topics(df['tokens_lemmatized'].tolist())

    # Train model
    lda_model = modeler.train_lda_model(df['tokens_lemmatized'].tolist())

    # Print topics
    topic_labels = modeler.print_topics()

    # Get document topics
    doc_topics = modeler.get_document_topics(df['tokens_lemmatized'].tolist())

    # Create visualization
    modeler.visualize_topics()
