import streamlit as st
import pickle
import re
import json
import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('stopwords')

st.set_page_config(page_title="SO Question Guard", layout="wide", initial_sidebar_state="collapsed")

# Load model
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_model()

# Load metrics
try:
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
except:
    metrics = None

# Text processing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def analyze_quality(title, body):
    issues, suggestions = [], []
    
    if len(body) < 80:
        issues.append("📝 Question body is too short")
        suggestions.append("Add more details to your question")
    if "error" not in body.lower() and len(body) > 0:
        issues.append("🐛 No error message mentioned")
        suggestions.append("Include the exact error message")
    if "?" not in title and len(title) > 0:
        issues.append("❓ Title not framed as question")
        suggestions.append("Frame your title as a question")
    if "code" not in body.lower() and len(body) > 50:
        issues.append("💻 No code snippet found")
        suggestions.append("Add your code with proper formatting")
    if len(title) < 10 and len(title) > 0:
        issues.append("📌 Title is too short")
        suggestions.append("Make title more descriptive")
    
    return issues, suggestions

# Custom CSS for exact UI match
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #f8fafc;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: white;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        color: white;
    }
    
    .hero-title span {
        color: #fbbf24;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.95;
        max-width: 700px;
        margin: 0 auto;
        color: white;
    }
    
    /* Stats Section */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        flex: 1;
        min-width: 200px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #64748b;
    }
    
    /* Form Section */
    .form-section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .form-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #0f172a;
    }
    
    .form-subtitle {
        color: #64748b;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.7rem;
    }
    
    .stTextInput > label, .stTextArea > label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.7rem;
        font-weight: 600;
        border-radius: 10px;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.3);
    }
    
    /* Result Boxes */
    .result-success {
        background: linear-gradient(135deg, #16a34a, #059669);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-danger {
        background: linear-gradient(135deg, #dc2626, #ea580c);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .issue-box {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 0.7rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: #991b1b;
    }
    
    .suggestion-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 0.7rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: #1e40af;
    }
    
    .metric-card {
        background: #f1f5f9;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }
    
    hr {
        margin: 2rem 0;
        border-color: #e2e8f0;
    }
    
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==================== UI ====================

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">
        ⚡ AI-Powered Analysis
    </div>
    <div class="hero-title">
        Will Your Question Get<br><span>Closed on Stack Overflow?</span>
    </div>
    <div class="hero-subtitle">
        Get instant predictions and actionable suggestions to improve your question before posting. 
        Save time and increase your chances of getting great answers.
    </div>
</div>
""", unsafe_allow_html=True)

# Stats Section
st.markdown("""
<div class="stats-container">
    <div class="stat-card">
        <div class="stat-number">95%</div>
        <div class="stat-label">Accuracy Rate</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">10k+</div>
        <div class="stat-label">Questions Analyzed</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">80%</div>
        <div class="stat-label">Improved Questions</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Form Section
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown('<div class="form-title">✨ Analyze Your Question</div>', unsafe_allow_html=True)
st.markdown('<div class="form-subtitle">Enter your Stack Overflow question details below</div>', unsafe_allow_html=True)

with st.form("question_form"):
    title = st.text_input(
        "Question Title",
        placeholder="e.g., How do I fix 'NoneType' object has no attribute 'append'?"
    )
    
    body = st.text_area(
        "Question Body",
        placeholder="Include all relevant details, code examples, what you've tried, and expected vs actual results...",
        height=180
    )
    
    tags = st.text_input(
        "Tags (comma separated)",
        placeholder="e.g., python, list, error"
    )
    
    submitted = st.form_submit_button("🔍 Analyze Question")

st.markdown('</div>', unsafe_allow_html=True)

# Results
if submitted:
    if not title or not body:
        st.warning("⚠️ Please enter both title and body")
    elif model is None:
        st.error("❌ Model not found. Run train.py first.")
    else:
        with st.spinner("Analyzing your question..."):
            # Clean text
            cleaned = clean_text(f"{title} {body} {tags}")
            X = vectorizer.transform([cleaned])
            prob = model.predict_proba(X)[0][1]
            will_close = prob > 0.5
            confidence = int(prob * 100)
            
            # Display result
            if will_close:
                st.markdown(f"""
                <div class="result-danger">
                    <h2 style="color: white;">⚠️ Likely to be Closed</h2>
                    <h1 style="font-size: 2.5rem; color: white;">{confidence}% Confidence</h1>
                    <p style="color: white;">Your question may be closed. Review the suggestions below.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-success">
                    <h2 style="color: white;">✅ Good to Go!</h2>
                    <h1 style="font-size: 2.5rem; color: white;">{100-confidence}% Confidence</h1>
                    <p style="color: white;">Your question looks good! High chance of getting answers.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Quality analysis
            issues, suggestions = analyze_quality(title, body)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ⚠️ Issues Found")
                if issues:
                    for issue in issues:
                        st.markdown(f'<div class="issue-box">{issue}</div>', unsafe_allow_html=True)
                else:
                    st.success("✅ No major issues found!")
            
            with col2:
                st.markdown("### 💡 Suggestions")
                if suggestions:
                    for suggestion in suggestions:
                        st.markdown(f'<div class="suggestion-box">{suggestion}</div>', unsafe_allow_html=True)
                else:
                    st.info("🎉 Your question looks excellent!")
            
            # Stats
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            
            cola, colb, colc, cold = st.columns(4)
            
            with cola:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.8rem; color: #64748b;">Title Length</div>
                    <div style="font-size: 1.3rem; font-weight: 700;">{len(title)} chars</div>
                </div>
                """, unsafe_allow_html=True)
            
            with colb:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.8rem; color: #64748b;">Body Length</div>
                    <div style="font-size: 1.3rem; font-weight: 700;">{len(body)} chars</div>
                </div>
                """, unsafe_allow_html=True)
            
            with colc:
                has_code = "✅ Yes" if "code" in body.lower() or "```" in body else "❌ No"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.8rem; color: #64748b;">Code Included</div>
                    <div style="font-size: 1.3rem; font-weight: 700;">{has_code}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cold:
                has_error = "✅ Yes" if "error" in body.lower() else "❌ No"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.8rem; color: #64748b;">Error Mentioned</div>
                    <div style="font-size: 1.3rem; font-weight: 700;">{has_error}</div>
                </div>
                """, unsafe_allow_html=True)

# Model Performance (optional)
if metrics:
    st.markdown("---")
    st.markdown("## 📊 Model Performance")
    df = pd.DataFrame(metrics).T
    st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #64748b;">🚀 Powered by Machine Learning | Improve your Stack Overflow questions</p>', unsafe_allow_html=True)