import numpy as np
import pandas as pd
import scipy.sparse as ss
import matplotlib.pyplot as plt

import corextopic.corextopic as ct
import corextopic.vis_topic as vt # jupyter notebooks will complain matplotlib is being loaded twice
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 

df=pd.read_csv("digital_twin_dataset.csv",sep=";")

def remove_duplicates(df, column_name='Abstract'):
    # Tekrar eden satırları kaldır, sadece ilk görünen satır kalır
    df_cleaned = df.drop_duplicates(subset=[column_name], keep='first')
    return df_cleaned
# Orijinal DataFrame'inizi kullanarak duplicate'leri temizleyin ve yeni bir DataFrame'e aktarın
df_cleaned = remove_duplicates(df, column_name='Abstract')

# Artık df_cleaned içinde tekrarlanan satırlar temizlenmiş veri bulunuyor


# nltk'nin gerekli veri setlerini indirme
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

# NLTK'dan İngilizce stopword listesini ve WordNet lemmatizer'ını yükle
english_stopwords = set(stopwords.words('english'))
english_words = set(words.words())



lemmatizer = WordNetLemmatizer()

def lemmatize_word(text):
    word_tokens = text.split()
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
    lemmas = ' '.join(lemmas)
    return lemmas

def preprocess_text(text):
    if pd.isna(text):  # Eğer NaN değeri varsa, boş string döndür
        return ''
   
    # Küçük harfe dönüştür
    text = text.lower()
    # Noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\s]', '', text)
    # Sayıları kaldır
    text = re.sub(r'\d+', '', text)
    # Stopword'leri kaldır, "no" ve "yes" istisna
    tokens = text.split()
   
    # Eğer token sayısı 1 ise ve bu token İngilizce bir kelime değilse, boş string döndür
    if len(tokens) == 1 and (tokens[0] not in english_words):
        return ''
   
    filtered_words = [word for word in tokens if (word.lower() not in english_stopwords) and (word.lower() in english_words)]
    filtered_words = ' '.join(filtered_words)

    return filtered_words

# Pandas DataFrame üzerinde 'metin_sütunu'na fonksiyonu uygulama
df_cleaned['islenmis'] = df_cleaned.Abstract.apply(preprocess_text)



# Transform 20 newsgroup data into a sparse matrix
vectorizer = CountVectorizer(stop_words='english', max_features=20000, binary=True)
doc_word = vectorizer.fit_transform(df_cleaned.islenmis)
doc_word = ss.csr_matrix(doc_word)

words = list(np.asarray(vectorizer.get_feature_names_out()))
not_digit_inds = [ind for ind, word in enumerate(words) if not word.isdigit()]
doc_word = doc_word[:, not_digit_inds]
words = [word for ind, word in enumerate(words) if not word.isdigit()]

# Train the Corex topic model
topic_model = ct.Corex(n_hidden=10, words=words, max_iter=1000, verbose=False, seed=1)
topic_model.fit(doc_word, words=words);

# Predict the primary topic for each document
document_topic_matrix = topic_model.transform(doc_word)
document_topics = document_topic_matrix.argmax(axis=1)

# Group documents by their primary topic
topic_documents = {n: [] for n in range(topic_model.n_hidden)}
for i, topic_num in enumerate(document_topics):
    topic_documents[topic_num].append(df_cleaned.islenmis)

# Output topics and associated documents
topics = topic_model.get_topics()
for n, topic in enumerate(topics):
    topic_words = [f"{word} ({weight:.4f})" for word, weight, _ in topic]
    print(f"Topic {n}: " + ', '.join(topic_words))
    print(f"Sample opinions for Topic {n}:")
    for doc in topic_documents[n][:5]:  # Adjust the number to show more or fewer examples
        print(f" - {doc[:500]}...")  # Adjust slicing to control the length of shown opinions
    print("\n")


