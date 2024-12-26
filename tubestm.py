import spacy
from spacy.matcher import Matcher
from spacy.cli import download
from spacy.util import set_data_path
from PyPDF2 import PdfReader
import streamlit as st
import tempfile

class CVAnalyzer:
    def __init__(self, required_skills):
        temp_dir = tempfile.mkdtemp()
        set_data_path(temp_dir)

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.required_skills = set(required_skills)

    def extract_text_from_pdf(self, pdf_file):
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def analyze_cv(self, cv_text):
        doc = self.nlp(cv_text)

        matcher = Matcher(self.nlp.vocab)
        for skill in self.required_skills:
            pattern = [{"LOWER": token.lower()} for token in skill.split()]
            matcher.add(skill, [pattern])

        matches = matcher(doc)
        matched_skills = {self.nlp.vocab.strings[match_id] for match_id, start, end in matches}

        pos_identified_skills = set()
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and token.text in self.required_skills:
                pos_identified_skills.add(token.text)

        ner_identified_skills = set()
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "WORK_OF_ART", "SKILL"} and ent.text in self.required_skills:
                ner_identified_skills.add(ent.text)

        all_identified_skills = matched_skills.union(pos_identified_skills, ner_identified_skills)

        missing_skills = self.required_skills - all_identified_skills

        similarity_percentage = (len(all_identified_skills) / len(self.required_skills)) * 100

        return {
            "identified_skills": all_identified_skills,
            "missing_skills": missing_skills,
            "is_suitable": len(missing_skills) == 0,
            "similarity_percentage": similarity_percentage
        }

# Streamlit Application
st.title("CV Skill Analyzer")

# Upload file
uploaded_file = st.file_uploader("Upload your CV (PDF format only)", type="pdf")

# Input required skills
required_skills_input = st.text_area(
    "Enter required skills (comma-separated):", "Machine Learning, Tableau, Python, Power BI, Deep Learning"
)

if st.button("Analyze"):
    if uploaded_file and required_skills_input:
        required_skills = [skill.strip() for skill in required_skills_input.split(",")]
        analyzer = CVAnalyzer(required_skills)

        # Extract text from uploaded file
        cv_text = analyzer.extract_text_from_pdf(uploaded_file)

        # Analyze CV
        result = analyzer.analyze_cv(cv_text)

        # Display results
        st.write("### Results")
        st.write(f"**Identified Skills:** {', '.join(result['identified_skills'])}")
        st.write(f"**Missing Skills:** {', '.join(result['missing_skills'])}")
        st.write(f"**Similarity Percentage:** {result['similarity_percentage']:.2f}%")

        if result["is_suitable"]:
            st.success("The CV matches all required skills!")
        else:
            st.warning("The CV is missing some required skills.")
    else:
        st.error("Please upload a CV and enter required skills.")
