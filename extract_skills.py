import sys
import json
import spacy
import PyPDF2
import re
from nltk.corpus import stopwords

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Common technical skills (extend this list as needed)
SKILL_LIST = set([
    "python","leadership","pandas" "c++", "c#", "javascript", "typescript", "ruby", "go", "rust", "kotlin", "swift", 
    "php", "r programming", "matlab", "scala", "perl", "html", "css", "sql", "bash",
    "react.js", "angular", "vue.js", "svelte", "node.js", "django", "flask", "spring boot", 
    "asp.net", "ruby on rails", "laravel", "bootstrap", "tailwind css", "jquery", "express.js",
    "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "keras", 
    "pandas", "numpy", "matplotlib", "seaborn", "rstudio", "data analysis", "predictive modeling", 
    "big data", "hadoop", "spark", "data visualization", "tableau", "power bi", "nlp", 
    "computer vision", "data wrangling", "jupyter notebook", "docker", "kubernetes", "jenkins", 
    "ansible", "terraform", "github actions", "azure devops", "aws", "google cloud platform", 
    "microsoft azure", "ci/cd pipelines", "serverless architecture", "cloudformation","API integration","java",
    "machine learning","data analytics"
])

# Stopwords to ignore
STOPWORDS = set(stopwords.words("english"))

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        return f"Error extracting text: {e}"

def extract_skills(text):
    """Extract skills from the given text."""
    extracted_skills = set()
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Use spaCy for Named Entity Recognition (NER)
    doc = nlp(text)
    
    for token in doc:
        if token.text in SKILL_LIST and token.text not in STOPWORDS:
            extracted_skills.add(token.text)

    return list(extracted_skills)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No PDF file provided"}))
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    text = extract_text_from_pdf(pdf_path)

    if "Error" in text:
        print(json.dumps({"error": text}))
        sys.exit(1)

    skills = extract_skills(text)
    print(json.dumps({"skills": skills}))
