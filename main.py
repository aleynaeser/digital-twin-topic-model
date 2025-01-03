import nltk
import pandas as pd
import pyLDAvis
import pyLDAvis.lda_model
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# 1. Download necessary data for stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# 2. Data Loading
file_path = "digital_twin_dataset.csv"
data = pd.read_csv(file_path, sep=";", encoding="utf-8")

# 3. Combining and cleaning columns
data["Combined_Text"] = (
    data["Author Keywords"].fillna("") + " " + data["Abstract"].fillna("")
)


def preprocess_text(text):
    text = text.lower()
    text = " ".join(
        [word for word in text.split() if word not in stop_words]
    )  # Remove stopwords
    return text


data["Processed_Text"] = data["Combined_Text"].apply(preprocess_text)

# 4. LDA Model
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
doc_term_matrix = vectorizer.fit_transform(data["Processed_Text"])

lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(doc_term_matrix)


# 5. Subject extraction
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(
            " ".join(
                [feature_names[i] for i in topic.argsort()[: -no_top_words - 1 : -1]]
            )
        )


no_top_words = 10
display_topics(lda_model, vectorizer.get_feature_names_out(), no_top_words)

# 6. Visualization
visualization = pyLDAvis.lda_model.prepare(lda_model, doc_term_matrix, vectorizer)
pyLDAvis.save_html(visualization, "digital_twin_visualization.html")

print("Digital Twin Model File: digital_twin_visualization.html created successfully.")
