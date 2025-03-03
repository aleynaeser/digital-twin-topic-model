import pandas as pd
import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import homogeneity_score, precision_score, recall_score, f1_score
import corextopic.corextopic as ct
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# UMass Coherence Score Calculation
def calculate_umass_coherence(topic_words, doc_word_matrix):
    coherence_scores = []
    for topic in topic_words:
        topic_score = 0
        for i, word1 in enumerate(topic):
            for j, word2 in enumerate(topic):
                if i <= j:
                    continue
                word1_idx = vectorizer.vocabulary_.get(word1)
                word2_idx = vectorizer.vocabulary_.get(word2)
                if word1_idx is not None and word2_idx is not None:
                    co_occur = np.log(((doc_word_matrix[:, word1_idx].multiply(doc_word_matrix[:, word2_idx])).sum() + 1) / doc_word_matrix[:, word2_idx].sum())
                    topic_score += co_occur
        coherence_scores.append(topic_score)
    return coherence_scores

# Load and preprocess data
df=pd.read_csv("digital_twin_dataset.csv",sep=";")
vectorizer = CountVectorizer(stop_words='english', max_features=20000, binary=True)
filtered_df = df[(df['Abstract'] == 'adv')  ]
doc_word_binary = vectorizer.fit_transform(filtered_df.gorus_eng_son)
vectorizer_count = CountVectorizer(stop_words='english', max_features=20000)
doc_word_count = vectorizer_count.fit_transform(filtered_df.gorus_eng_son)
words_binary = list(vectorizer.get_feature_names_out())
words_count = list(vectorizer_count.get_feature_names_out())

texts = [doc.split() for doc in filtered_df.gorus_eng_son]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Corex and LDA Model Evaluation for Optimal Topics
coherence_results = []
precision_results = []
recall_results = []
f1_results = []
homogeneity_results = []

topic_range = range(2, 21)
for n_topics in topic_range:
    # Train Corex
    corex_model = ct.Corex(n_hidden=n_topics, words=words_binary, max_iter=1000, verbose=False, seed=1)
    corex_model.fit(doc_word_binary)
    corex_topics = corex_model.get_topics()
    corex_topic_words = [[word for word, _, _ in topic] for topic in corex_topics]

    # Train LDA
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=1, max_iter=100)
    lda_model.fit(doc_word_count)
    lda_topics = lda_model.components_
    lda_topic_words = [
        [words_count[i] for i in topic.argsort()[-10:]] for topic in lda_topics
    ]

    # Calculate Coherence using Gensim
    corex_coherence = CoherenceModel(topics=corex_topic_words, texts=texts, dictionary=dictionary, coherence='c_v').get_coherence()
    lda_coherence = CoherenceModel(topics=lda_topic_words, texts=texts, dictionary=dictionary, coherence='c_v').get_coherence()

    # Simulated Metrics (Homogeneity, Precision, Recall, F1)
    true_labels = np.random.randint(0, n_topics, size=len(filtered_df))
    corex_doc_topics = corex_model.transform(doc_word_binary).argmax(axis=1)
    lda_doc_topics = lda_model.transform(doc_word_count).argmax(axis=1)

    homogeneity_results.append((n_topics, homogeneity_score(true_labels, corex_doc_topics), homogeneity_score(true_labels, lda_doc_topics)))
    precision_results.append((n_topics, precision_score(true_labels, corex_doc_topics, average='micro'), precision_score(true_labels, lda_doc_topics, average='micro')))
    recall_results.append((n_topics, recall_score(true_labels, corex_doc_topics, average='macro'), recall_score(true_labels, lda_doc_topics, average='macro')))
    f1_results.append((n_topics, f1_score(true_labels, corex_doc_topics, average='macro'), f1_score(true_labels, lda_doc_topics, average='macro')))

    coherence_results.append((n_topics, corex_coherence, lda_coherence))

# Plot Results
def plot_metric(results, metric_name):
    x = [r[0] for r in results]
    corex_scores = [r[1] for r in results]
    lda_scores = [r[2] for r in results]

    plt.plot(x, corex_scores, label="Corex", marker="o")
    plt.plot(x, lda_scores, label="LDA", marker="x")
    plt.title(f"{metric_name} Comparison")
    plt.xlabel("Number of Topics")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

plot_metric(coherence_results, "Coherence (C_V)")
plot_metric(homogeneity_results, "Homogeneity")
plot_metric(precision_results, "Precision (Micro)")
plot_metric(recall_results, "Recall (Macro)")
plot_metric(f1_results, "F1 Score (Macro)")