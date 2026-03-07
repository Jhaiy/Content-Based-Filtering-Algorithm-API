from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
import pymysql
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request


engine = create_engine('mysql+pymysql://root:@localhost/techsync_db')

query = """
    SELECT RoleDescription FROM jobroles
    """
role_description = pd.read_sql(query, engine)
role_description['RoleDescription'] = role_description['RoleDescription'].str.lower()
role_description['RoleDescription'] = role_description['RoleDescription'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))
role_description['RoleDescription'] = role_description['RoleDescription'].apply(lambda x: re.sub(r'\s+', ' ', x))
role_description['RoleDescription'] = role_description['RoleDescription'].apply(lambda x: nltk.word_tokenize(x))
role_description['RoleDescription'].tolist()
stop_words = nltk.corpus.stopwords.words('english')
description = []

for sentence in role_description['RoleDescription']:
    filtered_sentence = []
    for word in sentence:
        if word not in stop_words or len(word) < 3:
            filtered_sentence.append(word)
    description.append(filtered_sentence)

role_description['RoleDescription'] = [' '.join(words) for words in description]

tfidf = TfidfVectorizer()
features = tfidf.fit_transform(role_description['RoleDescription'])

cosine_similarity_matrix = cosine_similarity(features, features)

print(cosine_similarity_matrix)
print(role_description['RoleDescription'])