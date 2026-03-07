from flask import Flask, json, request, jsonify
from flask_mysqldb import MySQL, MySQLdb
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
nltk.download('stopwords')
import pymysql
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'techsync_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

engine = create_engine('mysql+pymysql://root:@localhost/techsync_db')

@app.route('/', methods=['GET'])
def process_role_description():
    applicant_id = request.args.get('applicant_id', 0, type=int)
    fetch_job_skills = """
        SELECT skills.SkillDescription, skills.SkillName
        FROM skills
    """
    applicant_skills_query = """
        SELECT applicantskills.ApplicantID, skills.SkillName, skills.SkillDescription
        FROM applicantskills
        INNER JOIN skills ON applicantskills.SkillID = skills.SkillID 
        WHERE applicantskills.ApplicantID = %s
    """
    applicant_category_query = """
        SELECT applicantcategories.ApplicantID, jobcategories.CategoryName, jobcategories.CategoryDescription
        FROM applicantcategories
        INNER JOIN jobcategories ON applicantcategories.JobCategoryID = jobcategories.JobCategoryID
        WHERE applicantcategories.ApplicantID = %s
    """ 
    applicant_role_query = """ 
        SELECT applicantroles.ApplicantID, jobroles.RoleName, jobroles.RoleDescription
        FROM applicantroles
        INNER JOIN rolecategories ON applicantroles.JobRoleID = jobroles.JobRoleID
        WHERE applicantroles.ApplicantID = %s
    """

    sql_cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    sql_cursor.execute(applicant_skills_query, (applicant_id,))
    applicant_skills = pd.DataFrame(sql_cursor.fetchall())

    skill_description = pd.read_sql(fetch_job_skills, con=engine)

    skill_description = skill_description.drop_duplicates(subset=['SkillDescription'])
    skill_description['SkillDescription'] = skill_description['SkillDescription'].str.lower()
    skill_description['SkillDescription'] = skill_description['SkillDescription'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))
    skill_description['SkillDescription'] = skill_description['SkillDescription'].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    stop_words = nltk.corpus.stopwords.words('english')
    skill_description['SkillDescription'] = skill_description['SkillDescription'].apply(
        lambda x: ' '.join([word for word in nltk.word_tokenize(x) if word not in stop_words ])
    )

    applicant_skills_filtered = applicant_skills[applicant_skills['ApplicantID'] == applicant_id]

    combined_skills = ' '.join(applicant_skills_filtered['SkillDescription'].str.lower().tolist())
    combined_skills = re.sub(r'[^a-zA-Z]', ' ', combined_skills)
    combined_skills = re.sub(r'\s+', ' ', combined_skills).strip()

    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(skill_description['SkillDescription'])
    applicant_skill_features = tfidf.transform([combined_skills])

    similarity_scores = cosine_similarity(applicant_skill_features, features).flatten()
    top_similar_skills = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)[:10]

    recommendations = {
        "JobIndex": applicant_id,
        "SkillDescription": combined_skills,
        "Recommendations": [
            {
                "JobSkillID": idx,
                "SimilarityScore": score,
                "SkillDescription": skill_description['SkillDescription'].iloc[idx],
                "SkillName": skill_description['SkillName'].iloc[idx]
            }
            for idx, score, in top_similar_skills
        ]
    }

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)