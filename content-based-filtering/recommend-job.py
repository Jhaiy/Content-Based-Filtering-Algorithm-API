from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import nltk
import pymysql
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


engine = create_engine('mysql+pymysql://root:@localhost/techsync_db')

def fetch_applicant_skills(applicant_id):
    query = """
        SELECT skills.SkillName
        FROM applicantsskills
        INNER JOIN skills ON applicantskills.SkillID = skills.SkillID
        WHERE applicantskills.ApplicantID = %s
        """
    applicant_skills = pd.read_sql(query, engine, params=(applicant_id,))
    return applicant_skills['SkillName'].tolist()

def fetch_job_listings():
    query = """
        SELECT joblistings.JobListingID, joblistings.JobTitle, GROUP_CONCAT(skills.SkillName SEPARATOR ' ') AS RequiredSkills
        FROM joblistings
        INNER JOIN jobrequirements ON joblistings.JobListingID = jobrequirements.JobListingID
        INNER JOIN skills ON jobrequirements.SkillID = skills.SkillID
        GROUP BY joblistings.JobListingID
        """
    job_listings = pd.read_sql(query, engine)
    return job_listings

def recommend_jobs(applicant_id):
    applicant_skills = fetch_applicant_skills(applicant_id)
    job_listings = fetch_job_listings()
    job_listings['ApplicantSkills'] = applicant_skills

    vectorizer = CountVectorizer()
    skill_matrix = vectorizer.fit_transform(job_listings['ApplicantSkills'] + " " + job_listings['RequiredSkills'])
    similarity_scores = cosine_similarity(skill_matrix[0:1], skill_matrix[1:])

    job_listings['Similarity'] = similarity_scores.flatten()
    recommended_jobs = job_listings.sort_values(by='SimilarityScore', ascending=False)

    return recommended_jobs[['JobListingID', 'JobTitle', 'SimilarityScore']]

applicant_id = 1
recommended_jobs = recommend_jobs(applicant_id)
print(recommended_jobs)
    