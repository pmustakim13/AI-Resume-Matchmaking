import os
import pdfplumber
import nltk
import re
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Load BERT model & tokenizer (Using 'all-MiniLM-L6-v2' for efficiency)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


# Function to extract text from PDF resumes
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()


# Preprocess text (remove special characters, lowercase, remove stopwords)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)


# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


# Load resumes from a folder
def load_resumes(resume_folder):
    resumes = {}
    if not os.path.exists(resume_folder):
        raise FileNotFoundError(f"The folder '{resume_folder}' does not exist. Please provide a valid path.")
    for file in os.listdir(resume_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(resume_folder, file)
            text = extract_text_from_pdf(file_path)
            processed_text = preprocess_text(text)
            resumes[file] = processed_text
    return resumes


# Job Description (example)
job_description = """We are looking for a Data Scientist with experience in machine learning, NLP, Python, 
                     and deep learning frameworks such as TensorFlow and PyTorch."""

# Preprocess job description
job_description = preprocess_text(job_description)

# Specify the resume folder path
resume_folder = r"M:\TY\DMPM\resumes"

try:
    # Load resumes
    resumes = load_resumes(resume_folder)

    # Convert text to BERT embeddings
    job_embedding = get_bert_embedding(job_description)
    resume_embeddings = {file: get_bert_embedding(text) for file, text in resumes.items()}

    # Compute Cosine Similarity
    ranked_resumes = []
    for filename, embedding in resume_embeddings.items():
        similarity = cosine_similarity(job_embedding, embedding)[0][0]
        ranked_resumes.append((filename, similarity))

    # Sort resumes by similarity score
    ranked_resumes.sort(key=lambda x: x[1], reverse=True)

    # Display results
    print("\nTop Resume Matches:")
    for rank, (filename, score) in enumerate(ranked_resumes, start=1):
        print(f"{rank}. {filename} - Match Score: {score:.2f}")

except FileNotFoundError as fnf_error:
    print(fnf_error)
