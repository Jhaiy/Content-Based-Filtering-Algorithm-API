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
    applicant_id = request.args.get('applicant_id', 10, type=int)
    fetch_job_skills = """
        SELECT company.CompanyID, companydetails.CompanyLogo, company.CompanyName, joblistings.JobListingID, joblistings.JobTitle, joblistings.JobDescription, jobcategories.CategoryName, jobcategories.CategoryDescription, jobroles.RoleName, jobroles.RoleDescription
        FROM joblistings
        INNER JOIN jobcategories ON joblistings.JobCategoryID = jobcategories.JobCategoryID
        INNER JOIN jobroles ON joblistings.JobRoleID = jobroles.JobRoleID
        INNER JOIN company ON joblistings.CompanyID = company.CompanyID
        INNER JOIN companydetails ON company.CompanyDetailsID = companydetails.CompanyDetailsID
        WHERE JobStatus = 'Open'
    """
    applicant_skills_query = """
        SELECT applicantskills.ApplicantID, skills.SkillName, skills.SkillDescription
        FROM applicantskills
        INNER JOIN skills ON applicantskills.SkillID = skills.SkillID 
        WHERE applicantskills.ApplicantID = %s
    """

    sql_cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    sql_cursor.execute(applicant_skills_query, (applicant_id,))
    applicant_skills = pd.DataFrame(sql_cursor.fetchall())

    category_description = pd.read_sql(fetch_job_skills, con=engine)
    category_description['CategoryDescription'] = category_description['CategoryDescription'].str.lower()
    category_description['CategoryDescription'] = category_description['CategoryDescription'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))
    category_description['CategoryDescription'] = category_description['CategoryDescription'].apply(lambda x: re.sub(r'\s+', ' ', x))

    category_description['RoleDescription'] = category_description['RoleDescription'].str.lower()
    category_description['RoleDescription'] = category_description['RoleDescription'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))
    category_description['RoleDescription'] = category_description['RoleDescription'].apply(lambda x: re.sub(r'\s+', ' ', x))

    stop_words = nltk.corpus.stopwords.words('english')
    category_description['CategoryDescription'] = category_description['CategoryDescription'].apply(
        lambda x: ' '.join([word for word in nltk.word_tokenize(x) if word not in stop_words ])
    )
    category_description['RoleDescription'] = category_description['RoleDescription'].apply(
        lambda x: ' '.join([word for word in nltk.word_tokenize(x) if word not in stop_words])
    )

    category_description['CombinedDescription'] = (
        category_description['CategoryDescription'] + ' ' + category_description['RoleDescription']
    )

    applicant_skills_filtered = applicant_skills[applicant_skills['ApplicantID'] == applicant_id]
    combined_skills = ' '.join(applicant_skills_filtered['SkillDescription'].str.lower().tolist())
    combined_skills = re.sub(r'[^a-zA-Z]', ' ', combined_skills)
    combined_skills = re.sub(r'\s+', ' ', combined_skills).strip()
    combined_skills_tokens = [word for word in nltk.word_tokenize(combined_skills) if word not in stop_words]
    combined_skills = ' '.join(combined_skills_tokens)

    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(category_description['CombinedDescription'])
    applicant_skill_features = tfidf.transform([combined_skills])

    category_similarity_scores = cosine_similarity(applicant_skill_features, features).flatten()

    top_similar_skills = [
        (idx, score) for idx, score in sorted(enumerate(category_similarity_scores), key=lambda x: x[1], reverse=True)
        if score > 0
    ][:10]

    if not top_similar_skills:
        return jsonify({"message": "No recommendations found."}), 404

    recommendations = {
        "JobIndex": applicant_id,
        "ApplicantSkillsDescription": combined_skills,
        "Recommendations": [
            {
                "CompanyID": int(category_description['CompanyID'].iloc[idx]),
                "JobListingID": int(category_description['JobListingID'].iloc[idx]),
                "CompanyName": category_description['CompanyName'].iloc[idx],
                "JobTitle": category_description['JobTitle'].iloc[idx],
                "JobDescription": category_description['JobDescription'].iloc[idx],
                "CategoryDescription": category_description['CategoryDescription'].iloc[idx],
                "CategoryName": category_description['CategoryName'].iloc[idx],
                "RoleDescription": category_description['RoleDescription'].iloc[idx],
                "RoleName": category_description['RoleName'].iloc[idx],
                "CompanyLogo": category_description['CompanyLogo'].iloc[idx],
                "SimilarityScore": float(score)
            }
            for idx, score, in top_similar_skills
        ]
    }

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)