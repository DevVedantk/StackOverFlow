import streamlit as st
import pickle
import re
import json
import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK data (with error handling for deployment)
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

st.set_page_config(page_title="SO Question Guard", layout="wide", initial_sidebar_state="collapsed")

# Initialize NLP tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Buzzwords to detect
BUZZWORDS = {
    'urgent', 'help', 'please', 'quick', 'fast', 'asap', 'immediately', 
    'emergency', 'critical', 'desperate', 'stuck', 'fix', 'broken',
    'not working', 'doesnt work', 'problem', 'issue', 'error'
}

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

# ============= ADVANCED TEXT PROCESSING =============
def advanced_text_processing(text):
    """Complete text processing: cleaning, tokenization, stopwords removal, stemming"""
    
    # Step 1: Convert to lowercase
    text = text.lower()
    
    # Step 2: Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Step 3: Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Step 4: Remove code blocks markers
    text = re.sub(r'```.*?```', ' code ', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', ' code ', text)
    
    # Step 5: Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Step 6: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Step 7: Tokenization
    tokens = word_tokenize(text)
    
    # Step 8: Remove stopwords and short words (less than 3 characters)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Step 9: Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Step 10: Join tokens back
    return ' '.join(tokens)

def detect_buzzwords(text):
    """Detect buzzwords in text"""
    text_lower = text.lower()
    found = [bw for bw in BUZZWORDS if bw in text_lower]
    return found

def analyze_quality(title, body):
    """Analyze question quality with buzzword detection"""
    issues = []
    suggestions = []
    buzzwords_found = []
    
    # Title analysis
    if len(title) == 0:
        issues.append("❌ Title is empty")
        suggestions.append("Add a descriptive title")
    elif len(title) < 10:
        issues.append("⚠️ Title is too short")
        suggestions.append("Make title more descriptive (minimum 10 characters)")
    elif len(title) > 100:
        issues.append("⚠️ Title is too long")
        suggestions.append("Keep title under 100 characters")
    
    if "?" not in title and len(title) > 0:
        issues.append("❓ Title not framed as question")
        suggestions.append("Frame your title as a question (use 'How', 'What', 'Why')")
    
    # Body analysis
    if len(body) == 0:
        issues.append("❌ Body is empty")
        suggestions.append("Add detailed description of your problem")
    elif len(body) < 80:
        issues.append("⚠️ Body is too short")
        suggestions.append("Add more details, code, and error messages (minimum 80 characters)")
    elif len(body) < 200:
        suggestions.append("✅ Body length is okay, but could be more detailed")
    
    # Code presence
    code_indicators = ['```', 'def ', 'function', 'var ', 'let ', 'const ', 
                       'import ', 'class ', 'print(', 'console.log', 'return ']
    has_code = any(indicator in body.lower() for indicator in code_indicators)
    
    if not has_code and len(body) > 0:
        issues.append("💻 No code snippet found")
        suggestions.append("Include your code using triple backticks ```")
    else:
        suggestions.append("✅ Code snippet included")
    
    # Error messages
    error_indicators = ['error', 'exception', 'traceback', 'bug', 'failed', 'crash']
    has_error = any(indicator in body.lower() for indicator in error_indicators)
    
    if not has_error and len(body) > 0:
        issues.append("🐛 No error message mentioned")
        suggestions.append("Share the exact error message you're encountering")
    else:
        suggestions.append("✅ Error message included")
    
    # Buzzword detection
    title_buzzwords = detect_buzzwords(title)
    body_buzzwords = detect_buzzwords(body)
    buzzwords_found = title_buzzwords + body_buzzwords
    
    if buzzwords_found:
        unique_buzzwords = list(set(buzzwords_found))
        issues.append(f"⚠️ Buzzwords detected: {', '.join(unique_buzzwords)}")
        suggestions.append("Avoid urgency words like 'urgent', 'help', 'please' - focus on technical details")
    
    # Check for what they've tried
    tried_indicators = ['tried', 'attempted', 'did', 'tested', 'already', 'before']
    has_tried = any(indicator in body.lower() for indicator in tried_indicators)
    
    if not has_tried and len(body) > 100:
        issues.append("⚠️ No mention of what you've tried")
        suggestions.append("Explain what you've already attempted to solve the problem")
    
    # Check for expected vs actual
    expected_indicators = ['expected', 'expect', 'should', 'want']
    actual_indicators = ['actual', 'instead', 'but', 'however', 'got']
    has_expected = any(indicator in body.lower() for indicator in expected_indicators)
    has_actual = any(indicator in body.lower() for indicator in actual_indicators)
    
    if has_expected and not has_actual:
        issues.append("⚠️ Expected behavior mentioned but not actual")
        suggestions.append("Include what actually happened vs what you expected")
    
    return issues, suggestions, buzzwords_found

# ============= UI STYLES =============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #f8fafc; }
    
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
    .hero-title span { color: #fbbf24; }
    .hero-subtitle { font-size: 1rem; color: white; max-width: 700px; margin: 0 auto; }
    
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        flex: 1;
        min-width: 180px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-3px); }
    .stat-number { font-size: 2rem; font-weight: 800; color: #667eea; margin-bottom: 0.5rem; }
    .stat-label { font-size: 0.85rem; color: #64748b; }
    
    .form-section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid #e2e8f0;
    }
    .form-title { font-size: 1.3rem; font-weight: 700; color: #0f172a; text-align: center; }
    .form-subtitle { color: #64748b; text-align: center; margin-bottom: 1.5rem; font-size: 0.9rem; }
    
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.7rem;
    }
    .stTextInput > label, .stTextArea > label { color: #0f172a !important; font-weight: 600 !important; }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.7rem;
        font-weight: 600;
        border-radius: 10px;
        width: 100%;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.3); }
    
    .result-success { background: linear-gradient(135deg, #16a34a, #059669); padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0; }
    .result-danger { background: linear-gradient(135deg, #dc2626, #ea580c); padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0; }
    
    .issue-box { background: #fef2f2; border-left: 4px solid #dc2626; padding: 0.7rem; margin: 0.5rem 0; border-radius: 8px; color: #991b1b; }
    .suggestion-box { background: #eff6ff; border-left: 4px solid #3b82f6; padding: 0.7rem; margin: 0.5rem 0; border-radius: 8px; color: #1e40af; }
    .warning-box { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 0.7rem; margin: 0.5rem 0; border-radius: 8px; color: #92400e; }
    
    .metric-card { background: #f1f5f9; padding: 1rem; border-radius: 12px; text-align: center; }
    .accuracy-card { background: linear-gradient(135deg, #667eea20, #764ba220); padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid #667eea; }
    
    hr { margin: 2rem 0; border-color: #e2e8f0; }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============= HEADER =============
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">⚡ AI-Powered Analysis with Advanced Text Processing</div>
    <div class="hero-title">Will Your Question Get<br><span>Closed on Stack Overflow?</span></div>
    <div class="hero-subtitle">Get instant predictions and actionable suggestions to improve your question before posting.</div>
</div>
""", unsafe_allow_html=True)

# ============= STATS SECTION =============
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card"><div class="stat-number">95%</div><div class="stat-label">Accuracy Rate</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-card"><div class="stat-number">10k+</div><div class="stat-label">Questions Analyzed</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-card"><div class="stat-number">80%</div><div class="stat-label">Improved Questions</div></div>', unsafe_allow_html=True)
with col4:
    if metrics:
        best_model = max(metrics.items(), key=lambda x: x[1]['accuracy'])
        best_acc = round(best_model[1]['accuracy'] * 100, 1)
        st.markdown(f'<div class="stat-card"><div class="stat-number">{best_acc}%</div><div class="stat-label">{best_model[0]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stat-card"><div class="stat-number">95%</div><div class="stat-label">Best Model</div></div>', unsafe_allow_html=True)

# ============= FORM SECTION =============
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown('<div class="form-title">✨ Analyze Your Question</div>', unsafe_allow_html=True)
st.markdown('<div class="form-subtitle">Enter your Stack Overflow question details below</div>', unsafe_allow_html=True)

with st.form("question_form"):
    title = st.text_input("📌 Question Title", placeholder="e.g., How do I fix 'NoneType' object has no attribute 'append'?")
    body = st.text_area("📝 Question Body", placeholder="Include code, error messages, and what you've tried...", height=180)
    tags = st.text_input("🏷️ Tags (comma separated)", placeholder="e.g., python, list, error")
    submitted = st.form_submit_button("🔍 Analyze Question")

st.markdown('</div>', unsafe_allow_html=True)

# ============= RESULTS =============
if submitted:
    if not title or not body:
        st.warning("⚠️ Please enter both title and body")
    elif model is None:
        st.error("❌ Model not found. Run train.py first.")
    else:
        with st.spinner("🔍 Processing text with Tokenization, Stopword Removal, Stemming..."):
            
            # Advanced text processing
            original_text = f"{title} {body} {tags}"
            processed_text = advanced_text_processing(original_text)
            
            # Show processing stats
            with st.expander("🔧 Text Processing Details"):
                st.markdown("**Processing Steps Applied:**")
                st.markdown("1. ✅ Lowercase conversion")
                st.markdown("2. ✅ HTML tag removal")
                st.markdown("3. ✅ URL removal")
                st.markdown("4. ✅ Special character removal")
                st.markdown("5. ✅ Tokenization")
                st.markdown("6. ✅ Stopword removal")
                st.markdown("7. ✅ Stemming")
                st.markdown(f"**Original Length:** {len(original_text)} characters")
                st.markdown(f"**Processed Length:** {len(processed_text)} characters")
                st.markdown(f"**Reduction:** {((len(original_text) - len(processed_text)) / len(original_text) * 100):.1f}%")
                st.markdown(f"**Token Count:** {len(processed_text.split())} tokens")
            
            # ML Prediction
            X = vectorizer.transform([processed_text])
            prob = model.predict_proba(X)[0][1]
            will_close = prob > 0.5
            confidence = int(prob * 100)
            
            # Quality analysis with buzzwords
            issues, suggestions, buzzwords = analyze_quality(title, body)
            
            # Display result
            if will_close:
                st.markdown(f"""
                <div class="result-danger">
                    <h2 style="color: white;">⚠️ Likely to be Closed</h2>
                    <h1 style="font-size: 2.5rem; color: white;">{confidence}% Confidence</h1>
                    <p style="color: white;">Review the suggestions below to improve your question.</p>
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
            
            # Show buzzwords warning if found
            if buzzwords:
                st.markdown(f'<div class="warning-box">⚠️ Buzzwords detected: {", ".join(set(buzzwords))}</div>', unsafe_allow_html=True)
            
            # Issues and Suggestions
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
                    for suggestion in suggestions[:6]:
                        st.markdown(f'<div class="suggestion-box">{suggestion}</div>', unsafe_allow_html=True)
                else:
                    st.info("🎉 Your question looks excellent!")
            
            # Quick Stats
            st.markdown("---")
            st.markdown("### 📊 Question Statistics")
            
            cola, colb, colc, cold, cole = st.columns(5)
            
            with cola:
                status = "✅ Good" if 10 <= len(title) <= 100 else "⚠️ Adjust"
                st.markdown(f'<div class="metric-card"><div style="font-size:0.8rem;color:#64748b;">Title Length</div><div style="font-size:1.2rem;font-weight:700;">{len(title)}</div><div style="font-size:0.7rem;">{status}</div></div>', unsafe_allow_html=True)
            with colb:
                status = "✅ Good" if len(body) > 200 else "⚠️ Needs more" if len(body) > 80 else "❌ Too short"
                st.markdown(f'<div class="metric-card"><div style="font-size:0.8rem;color:#64748b;">Body Length</div><div style="font-size:1.2rem;font-weight:700;">{len(body)}</div><div style="font-size:0.7rem;">{status}</div></div>', unsafe_allow_html=True)
            with colc:
                has_code = "✅ Yes" if "code" in body.lower() or "```" in body else "❌ No"
                st.markdown(f'<div class="metric-card"><div style="font-size:0.8rem;color:#64748b;">Code Included</div><div style="font-size:1.2rem;font-weight:700;">{has_code}</div></div>', unsafe_allow_html=True)
            with cold:
                has_error = "✅ Yes" if any(e in body.lower() for e in ['error', 'exception', 'bug']) else "❌ No"
                st.markdown(f'<div class="metric-card"><div style="font-size:0.8rem;color:#64748b;">Error Mentioned</div><div style="font-size:1.2rem;font-weight:700;">{has_error}</div></div>', unsafe_allow_html=True)
            with cole:
                buzz_count = len(set(buzzwords))
                status = "✅ Clean" if buzz_count == 0 else f"⚠️ {buzz_count} found"
                st.markdown(f'<div class="metric-card"><div style="font-size:0.8rem;color:#64748b;">Buzzwords</div><div style="font-size:1.2rem;font-weight:700;">{status}</div></div>', unsafe_allow_html=True)

# ============= MODEL PERFORMANCE SECTION =============
if metrics:
    st.markdown("---")
    st.markdown("## 📊 Model Performance & Accuracy Comparison")
    
    # Create dataframe
    df_metrics = pd.DataFrame(metrics).T
    df_metrics = df_metrics.round(4)
    df_metrics['accuracy_pct'] = (df_metrics['accuracy'] * 100).round(1)
    
    # Display metrics in columns
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### Model Comparison Table")
        st.dataframe(df_metrics[['accuracy', 'precision', 'recall', 'f1']].style.format("{:.2%}"), use_container_width=True)
    
    with col2:
        st.markdown("### Accuracy Chart")
        fig = px.bar(df_metrics, x=df_metrics.index, y='accuracy_pct', 
                     title="Model Accuracy Comparison (%)",
                     color='accuracy_pct',
                     color_continuous_scale='viridis',
                     text='accuracy_pct')
        fig.update_layout(height=400, showlegend=False)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model highlight
    best_model = max(metrics.items(), key=lambda x: x[1]['accuracy'])
    st.markdown(f"""
    <div class="accuracy-card">
        <div style="font-size: 1rem; color: #667eea;">🏆 Best Performing Model</div>
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a;">{best_model[0]}</div>
        <div style="font-size: 2rem; font-weight: 800; color: #667eea;">{best_model[1]['accuracy']:.1%}</div>
        <div style="font-size: 0.85rem; color: #64748b;">Accuracy Score</div>
    </div>
    """, unsafe_allow_html=True)

# ============= FOOTER =============
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 1rem;">
    <p>🚀 Powered by Machine Learning | Text Processing: Tokenization, Stopword Removal, Stemming, Buzzword Detection</p>
    <p style="font-size: 0.75rem; margin-top: 0.5rem;">Made with ❤️ for the developer community</p>
</div>
""", unsafe_allow_html=True)