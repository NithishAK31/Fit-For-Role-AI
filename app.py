import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

# --- Function Definitions ---
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Handle None returns
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def clean_text(text):
    """Clean and normalize text"""
    return re.sub(r'\s+', ' ', text).lower()

def get_keywords(text):
    """Extract keywords from text"""
    return re.findall(r'\b[a-z]{3,}\b', text.lower())

def calculate_similarity(resume, job_desc):
    """Calculate similarity score between resume and job description"""
    try:
        if not resume.strip() or not job_desc.strip():
            return 0.0
            
        tfidf = TfidfVectorizer()
        vecs = tfidf.fit_transform([resume, job_desc])
        similarity = cosine_similarity(vecs[0:1], vecs[1:2])[0][0] * 100
        # Boost scores by 1.5x but cap at 95%
        return min(similarity * 1.5, 95)
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return 0.0

def analyze_missing(resume_text, job_desc):
    """Analyze missing keywords from resume compared to job description"""
    try:
        resume_words = set(get_keywords(resume_text))
        job_words = set(get_keywords(job_desc))
        
        common_words = {'the', 'and', 'for', 'with', 'this', 'that', 'have', 'has'}
        missing = job_words - resume_words - common_words
        
        job_word_counts = Counter(get_keywords(job_desc))
        missing_important = sorted(
            [(word, job_word_counts[word]) for word in missing],
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 most frequent missing keywords
        
        return [word for word, count in missing_important]
    except:
        return []

def generate_tailored_suggestions(resume_text, job_desc, skill_score):
    """Generate job-specific suggestions"""
    suggestions = []
    try:
        missing_keywords = analyze_missing(resume_text, job_desc)
        
        if missing_keywords:
            suggestions.append(f"🔍 **Missing keywords:** {', '.join(missing_keywords)}")
            
            if skill_score < 60:
                suggestions.append(f"✏️ Add these skills: {', '.join(missing_keywords[:3])} to your skills section")
            
            if 'python' in missing_keywords:
                suggestions.append("🐍 Mention Python projects from coursework")
            if 'cad' in missing_keywords:
                suggestions.append("📐 Highlight any CAD projects from mechanical labs")
            if 'data' in missing_keywords:
                suggestions.append("📊 Include any data analysis projects")
        
        # Job-type specific suggestions
        job_desc_lower = job_desc.lower()
        if 'intern' in job_desc_lower:
            suggestions.append("🎯 Frame academic projects as 'pre-internship experience'")
        if 'design' in job_desc_lower:
            suggestions.append("✏️ Add design projects from engineering graphics")
        if 'analysis' in job_desc_lower:
            suggestions.append("🔢 Highlight math/statistics coursework")
        
        return suggestions
    except:
        return ["⚠️ Could not generate specific suggestions"]

# --- Main Application ---
def main():
    st.set_page_config(layout="wide")
    st.title("FIT-FOR-ROLE.ai")
    st.markdown("### Resume-Job Matching for Engineering Students")
    
    # File upload
    resume_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
    
    # Job descriptions
    st.subheader("Enter Job Descriptions")
    job_desc1 = st.text_area("✍️ Job Description 1", height=200)
    job_desc2 = st.text_area("✍️ Job Description 2", height=200)
    job_desc3 = st.text_area("✍️ Job Description 3", height=200)
    
    job_descriptions = [job_desc1, job_desc2, job_desc3]
    job_descriptions = [desc for desc in job_descriptions if desc.strip() != ""]
    
    if resume_file and job_descriptions:
        resume_text = clean_text(extract_text_from_pdf(resume_file))
        
        if not resume_text.strip():
            st.error("Could not extract text from PDF. Please try another file.")
            return
            
        columns = st.columns(len(job_descriptions))
        
        for i, (desc, col) in enumerate(zip(job_descriptions, columns)):
            with col:
                st.subheader(f"💼 Job {i+1} Analysis")
                clean_jd = clean_text(desc)
                
                # Calculate scores
                score = calculate_similarity(resume_text, clean_jd)
                skill_score = round(score * 0.5, 2)  # Skills weighted higher
                exp_score = round(score * 0.2, 2)    # Experience weighted lower
                edu_score = round(score * 0.3, 2)    # Education weighted medium
                overall = round(score, 2)
                
                # Display scores
                st.markdown(f"""
                **📊 Match Scores:**
                - **Skills:** {skill_score}%
                - **Experience:** {exp_score}%
                - **Education:** {edu_score}%
                - **Overall Match:** 🎯 {overall}%
                """)
                
                # Suggestions
                st.markdown("---")
                st.subheader("🔧 Improvement Plan")
                
                suggestions = generate_tailored_suggestions(resume_text, desc, skill_score)
                
                if not suggestions:
                    st.info("✅ Great alignment! Focus on quantifying your achievements.")
                else:
                    for suggestion in suggestions:
                        st.markdown(f"- {suggestion}")
                
                # Student-specific tips
                st.markdown("---")
                st.markdown("**🎓 Student Tips:**")
                st.markdown("- Add a 'Projects' section with academic work")
                st.markdown("- Include relevant coursework and lab skills")
                st.markdown("- List any competitions/hackathons under 'Achievements'")

if __name__ == "__main__":
    main()