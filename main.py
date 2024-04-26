import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import hamming_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pyresparser
import re
from pdfminer.high_level import extract_text
import spacy
from collections import Counter
import PyPDF2, pdfplumber
from fuzzywuzzy import fuzz
import joblib
import spacy
from collections import Counter


def calculate_fit_score(candidate_profile, job_role_requirements, weights):

    fit_score = 0

    st.markdown(f"<span style = 'font-size : 30px '><b>Key Points: </b></span> ", unsafe_allow_html= True)

    for criterion, weight in weights.items():

        criterion_score = calculate_criterion_score(candidate_profile, job_role_requirements, criterion)

        fit_score += weight * criterion_score

    return fit_score

def calculate_criterion_score(candidate_profile, job_role_requirements, criterion):

    if criterion == 'Position':
      for role in candidate_profile.get('Job Roles'):
        if role.lower() == job_role_requirements['Position'].lower():
            st.markdown(f"<span style = 'color : green; font-size : 20px'><b>The candidate has worked as {role} </b></span> ", unsafe_allow_html= True)           
            return 1
      return 0

    elif criterion == 'Experience':
        candidate_experience = int(re.search(r'\d+', candidate_profile['Number of Years of Experience']).group())
        required_experience = int(re.search(r'\d+', job_role_requirements['Experience']).group())
        if candidate_experience >= required_experience:
            st.markdown(f"<span style = 'color : green; font-size : 20px'><b>The candidate has {candidate_experience} years of experience</b></span> ", unsafe_allow_html= True)              
            return 1
        else:
            st.markdown(f"<span style = 'color : red; font-size :20px'><b>The candidate lacks experience </b></span> ", unsafe_allow_html= True) 
            return 0

    elif criterion == 'Education':
        candidate_degree = candidate_profile['Degree'].lower()
        required_qualifications = job_role_requirements.get('Qualifications')
        for qualification in required_qualifications:
            qualifications = qualification.lower()
            if fuzz.partial_ratio(candidate_degree, qualification.strip()) >= 80:
                st.markdown(f"<span style = 'color : green; font-size :20px'><b>The candidate's degree is {candidate_degree}</b></span> ", unsafe_allow_html= True)
                return 1
        return 0

    elif criterion == 'Skills':
        candidate_skills = [skill.lower() for skill in candidate_profile['Skills']]
        candidate_skills = set(candidate_skills)
        required_skills = [add_skill.lower() for add_skill in job_role_requirements['Skills']]
        required_skills = set(required_skills)

        if len(required_skills) == 0:
            return 0
        else:
            proportion_matched_skills = len(candidate_skills.intersection(required_skills)) / len(required_skills)
            st.markdown(f"<span style = 'color : green; font-size :20px'><b>The candidate has {len(candidate_skills.intersection(required_skills))} out of {len(required_skills)} skills! </b></span> ", unsafe_allow_html= True)

            if proportion_matched_skills >= 0.8:
                return 1
            elif proportion_matched_skills >= 0.6:
                return 0.8
            elif proportion_matched_skills >= 0.4:
                return 0.6
            elif proportion_matched_skills >= 0.2:
                return 0.4
            else:
                return 0

    elif criterion == 'Add_skills':
        candidate_skills = [skill.lower() for skill in candidate_profile['Skills']]
        add_skills = [add_skill.lower() for add_skill in job_role_requirements['Add_skills']]
        if any(add_skill.lower() in candidate_skills for add_skill in add_skills):
            return 1
        else:
            return 0

    else:
      return 0
    

def summarize_resume(data, job_role_requirements):
    # Load English tokenizer, tagger, parser, NER, and word vectors
    nlp = spacy.load("en_core_web_lg")

    json_object = json.dumps(data)

    # Extracting resume text from JSON object
    resume_text = json_object

    # Process the resume text with spaCy
    doc = nlp(resume_text)

    # Extract skills
    skills = [token.text for token in doc if token.pos_ == "NOUN" and token.text.lower() in data['Skills']]

    # Extract top skills
    top_skills = Counter(skills).most_common(5)

    # List of technology skills
    technology_skills = [
        'sql', 'mongodb', 'nosql', 'firebase', 'git', 'github', 'gitlab', 'docker', 'kubernetes',
        'linux', 'unix', 'aws', 'azure', 'google cloud platform', 'heroku', 'ci/cd', 'jenkins',
        'ansible', 'terraform', 'nginx', 'apache', 'redis', 'graphql', 'rabbitmq', 'apache kafka',
        'elasticsearch', 'prometheus', 'grafana', 'splunk', 'elk stack', 'tensorflow', 'pytorch',
        'scikit-learn', 'keras', 'nltk', 'spacy', 'numpy', 'pandas', 'matplotlib', 'seaborn',
        'bokeh', 'plotly', 'scrapy', 'beautiful soup', 'selenium', 'tensorflow.js', 'pygame',
        'unity3d', 'unreal engine', 'blender', 'maya', '3ds max', 'zbrush', 'adobe premiere pro',
        'final cut pro', 'davinci resolve', 'avid media composer', 'autodesk maya', 'autodesk 3ds max',
        'blender 3d'
    ]

    candidate_skills = data['Skills']
    company_skills = job_role_requirements['Skills']
    management_skills = job_role_requirements['Add_skills']


    # Filter out technology and management skills not in company skills
    filtered_technology_skills = [skill for skill in technology_skills if skill not in company_skills and skill in data['Skills']]
    filtered_management_skills = [skill for skill in management_skills if skill not in company_skills and skill in data['Skills']]

    candidate_skills = [skill.lower() for skill in candidate_skills]
    candidate_skills = set(candidate_skills)
    company_skills = [add_skill.lower() for add_skill in company_skills]
    company_skills = set(company_skills)
    matched_skills = list(candidate_skills.intersection(company_skills))

    # Extract domains worked in based on job roles
    domains_worked_in = [domain for domain in ['Data Science', 'Machine Learning', 'Web Development', 'Project Management', 'Frontend', 'Backend', 'Full Stack Development', 'UI/UX', 'Game Development'] if domain.lower() in resume_text.lower()]

    # Fit score
    weights = {
    'Position': 0.3,
    'Experience': 0.4,
    'Education': 0.1,
    'Skills': 0.5,
    'Add_skills':0.1
    }

    total_weights = sum(weights.values())

    normalized_weights = {criterion: weights[criterion] / total_weights for criterion in weights}

    fit_score = calculate_fit_score(data, job_role_requirements, normalized_weights)

    fit_score = fit_score * 100

    # classification

    # Load the trained model
    model = joblib.load('resume_classifier.pkl')

    # Load the vectorizer
    vectorizer = joblib.load('resume_vectorizer.pkl')
    class_labels = ['Backend', 'DevOps', 'Frontend', 'Full Stack Development','Game Development', 'Mobile Development', 'UI/UX Development','Web Development']


    input_combined = list(candidate_skills) + data['Degree'].split()

    input_combined = ' '.join(input_combined)
    input_vectorized = vectorizer.transform([input_combined])
    input_vectorized = pd.DataFrame(input_vectorized.toarray(), columns = vectorizer.get_feature_names_out())
    input_vectorized = np.array(input_vectorized)

    prediction= model.predict(input_vectorized)[0]
    indices = np.argsort(prediction)[::-1][:2]
    labels = [class_labels[i] for i in indices]

    # st.write("\nTop 2 Predicted Labels for candidate:")
    # st.write(labels)
    st.markdown(f"<span style = 'font-size : 20px ; color:#6495ED'><b>Top 2 Predicted Labels for candidate: </b></span>", unsafe_allow_html= True)
    st.markdown(f"<span style = 'color : #6495ED; font-size :18px'><b> {labels} </b></span> ", unsafe_allow_html= True)

    name = data['Name']
    email = data['E-mail']
    yoe = data['Number of Years of Experience']
    position = job_role_requirements['Position']
 
    return name, email, yoe, position, top_skills, domains_worked_in, matched_skills, filtered_technology_skills, filtered_management_skills, fit_score


def main():
    st.title("Resume Analyzer")

    st.sidebar.subheader("Upload Candidate Resume JSON file:")
    resume_file = st.sidebar.file_uploader("Upload JSON file", type="json", key="resume")

    st.sidebar.subheader("Upload Job Description JSON file:")
    job_desc_file = st.sidebar.file_uploader("Upload JSON file", type="json", key="job_desc")

    btn = st.sidebar.button('Generate Summary')

    try:

        if btn:
            if job_desc_file is None or resume_file is None:
                st.warning('Upload required Files!!')
            elif job_desc_file is not None and resume_file is not None:
                job_desc_data = json.load(job_desc_file)
                resume_data = json.load(resume_file)

                name, email, yoe, position, top_skills, domains_worked_in, matched_skills, filtered_technology_skills, filtered_management_skills, fit_score =  summarize_resume(resume_data, job_desc_data)

                st.markdown("<span style='font-size:30px'><b>Candidate Summary</b></span>", unsafe_allow_html=True)
                st.markdown(f"<span style = 'font-size : 20px '><b>Name: </b></span> {name} ", unsafe_allow_html= True)
                st.markdown(f"<span style = 'font-size : 20px '><b>E-mail: </b></span> {email} ", unsafe_allow_html= True)
                st.markdown(f"<span style = 'font-size : 20px '><b>Number of Years of Experience: </b></span> {yoe} ", unsafe_allow_html= True)         
                st.markdown(f"<span style = 'font-size : 20px '><b>Top Skills: </b></span>", unsafe_allow_html= True)
                for skill, count in top_skills:
                    st.write(f"- {skill}: {count} mentions\n")
                st.markdown(f"<span style = 'font-size : 20px '><b>Domains worked in: </b></span>", unsafe_allow_html= True)
                for domain in domains_worked_in:
                    st.write(f"- {domain}\n")
                st.markdown(f"<span style = 'font-size : 20px '><b>Job Relevant Technical skills: </b></span>", unsafe_allow_html= True)
                for skill in matched_skills:
                    st.write(f"- {skill} \n")
                st.markdown(f"<span style = 'font-size : 20px '><b>Other technical skills: </b></span>", unsafe_allow_html= True)
                for skill in filtered_technology_skills:
                    st.write(f"- {skill} \n")
                st.markdown(f"<span style = 'font-size : 20px '><b>Management skills: </b></span>", unsafe_allow_html= True)
                for skill in filtered_management_skills:
                    st.write(f"- {skill} \n")
                st.markdown(f"<span style = 'font-size : 20px '><b>Fit Score:</b></span>", unsafe_allow_html= True)

                if fit_score >= 60 :
                    st.markdown(f"<span style = 'color: green; font-size: 20px' ><b>The candidate is {fit_score:.2f}% fit for {position} </b></span>",  unsafe_allow_html= True)
                else:
                 st.markdown(f"<span style = 'color: red; font-size: 20px' ><b>The candidate is {fit_score:.2f}% fit for {position} </b></span>",    unsafe_allow_html= True )
            
    except Exception as e:
        st.error(f"Provide the JSON file in Proper format!!")

if __name__ == "__main__":
    main()
