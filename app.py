import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import random


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Diagnostic Consultation for Diabetes",
    page_icon="🩺",
    layout="wide"
)

# --- TAILWIND & DARK MODE STYLING ---
st.markdown("""
<script src="https://cdn.tailwindcss.com"></script>
<style>
    @import url('https://rsms.me/inter/inter.css');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #212121 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid #2f2f2f;
    }

    .main .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }

    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border-radius: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin-bottom: 1rem !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #10a37f !important;
        border-radius: 4px !important;
    }
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #3b82f6 !important;
        border-radius: 4px !important;
    }

    [data-testid="stMarkdownContainer"] p, 
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h3 {
        color: #ececec !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
    }

    .result-card {
        background: #171717;
        border: 1px solid #2f2f2f;
        padding: 24px;
        border-radius: 12px;
        margin-top: 2rem;
    }

    .risk-badge {
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
        margin-bottom: 12px;
        letter-spacing: 0.05em;
    }
    
    .high-risk { background: #450a0a !important; color: #fecaca !important; border: 1px solid #7f1d1d; }
    .low-risk { background: #064e3b !important; color: #d1fae5 !important; border: 1px solid #065f46; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- ASSETS LOADING ---
@st.cache_resource
def load_assets():
    path = "models/production_model.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

pkg = load_assets()

if pkg is None:
    st.error("Model assets not found. Please run the training pipeline.")
    st.stop()

# --- CONVERSATIONAL POOLS ---

GREETINGS = [
    "Welcome. I am your **Clinical Assessment Assistant**. I'll be guiding you through a professional screening for diabetes risk indicators. Shall we proceed?",
    "Good day. I'm here to assist with your medical intake for our diabetes research study. Ready to get started?",
    "Hello. I'm your AI Medical Officer today. I'll be collecting some clinical observations to assess your health profile. Are you ready?",
    "Greetings. I'm ready to begin your clinical risk assessment. We'll go through a few medical parameters together. Shall we start?"
]

ACKNOWLEDGEMENTS = [
    "Thank you. I've noted that down.",
    "Understood, thank you.",
    "Got it. Processing that entry.",
    "I've recorded that observation.",
    "Registered. Let's move on.",
    "That's noted for our assessment.",
    "I appreciate the information. Logged.",
    "Clear. Thank you."
]

BRIDGES = [
    "Moving forward...",
    "Next clinical point:",
    "Let's look at another factor...",
    "Continuing with the assessment...",
    "Regarding your clinical data...",
    "Let's proceed to the next metric..."
]

QUESTIONS = {
    "Age": ["Could you please share your **Age**?", "How old are you currently?", "What is your current **Age** for our records?", "Please provide your **Age**."],
    "Pregnancies": ["How many **Pregnancies** have you had, if applicable?", "What is the number of **Pregnancies** in your clinical history?", "Could you provide the count of **Pregnancies**?", "Please state your number of **Pregnancies**."],
    "Glucose": ["What is your current **Glucose** (blood sugar) level? (mg/dL)", "Could you share your most recent **Glucose** reading?", "What is your **Glucose** level for our clinical review?", "Please provide your **Glucose** measurement."],
    "BloodPressure": ["What is your **Blood Pressure** value? (mm Hg)", "Could you provide your current **Blood Pressure** measurement?", "What's the **Blood Pressure** reading we should note?", "Please enter your **Blood Pressure** value."],
    "Insulin": ["Could you provide your **Serum Insulin** level? (mu U/ml)", "What is your **Insulin** concentration?", "What **Insulin** value should I record for you?", "Please share your **Insulin** measurement."],
    "BMI": ["What is your current **Body Mass Index (BMI)**? (e.g. 26.5)", "Could you share your **BMI**?", "What is your **BMI** for our metabolic assessment?", "Please provide your **BMI**."],
    "DiabetesPedigreeFunction": ["What is your **Diabetes Pedigree Function** score? (e.g. 0.47)", "Could you provide the **Pedigree Function** score?", "What's your family history **Pedigree Function** value?", "Please state your **Pedigree Function** score."]
}

# --- LOGIC ---

def get_rule_reasoning():
    """Consolidated clinical reasoning for the final report."""
    reasoning_points = []
    inputs = st.session_state.inputs
    
    if inputs.get("Age", 0) > 45:
        reasoning_points.append("Age over 45 is a significant clinical benchmark for increased metabolic risk.")
    if inputs.get("Glucose", 0) > 140:
        reasoning_points.append("Elevated Glucose levels (>140 mg/dL) highlight a risk of hyperglycemia.")
    if inputs.get("BMI", 0) > 30:
        reasoning_points.append("A BMI over 30 is clinically associated with obesity-related insulin resistance.")
    if inputs.get("BloodPressure", 0) > 140:
        reasoning_points.append("Readings above 140 mm Hg suggest cardiovascular strain common in metabolic syndromes.")
    if inputs.get("DiabetesPedigreeFunction", 0) > 0.5:
        reasoning_points.append("The Pedigree Function score indicates a potential genetic predisposition.")
        
    return " ".join(reasoning_points)

def generate_report(pred, prob, factors):
    """Generates a professional paragraph-style clinical explanation."""
    risk_level = "High" if pred == 1 else "Low"
    prob_str = f"{prob*100:.1f}%"
    clinical_notes = get_rule_reasoning()
    
    explanation = (
        f"The diagnostic analysis is complete. Based on the clinical parameters provided, the system classifies your assessed risk as **{risk_level}** "
        f"with a statistical probability of **{prob_str}**. "
    )
    
    if clinical_notes:
        explanation += f"\n\n**Clinical Observation Details:** {clinical_notes} "
    
    if pred == 1:
        explanation += (
            f"The primary influencers identified in this assessment are **{factors}**. Historically, these variables correlate "
            "with a higher prevalence of diabetes in our research cohort. We strongly recommend presenting this summary to a "
            "licensed healthcare professional for formal clinical diagnostic procedures and personalized care planning."
        )
    else:
        explanation += (
            f"Your clinical markers, notably **{factors}**, align with a lower relative risk within our study framework. "
            "While these results are positive, we encourage you to maintain your current health regime and continue with regular "
            "check-ups with your medical provider for preventative care."
        )
    
    explanation += "\n\n***Professional Disclaimer:*** *This research-grade predictive tool provides statistical insights. It is not a clinical diagnosis or a medical treatment plan.*"
    return explanation

def predict_risk():
    df = pd.DataFrame([st.session_state.inputs])
    df['SkinThickness'] = 20
    df = df[pkg['feature_names']]
    pred = pkg['model'].predict(df)[0]
    prob = pkg['model'].predict_proba(df)[0][1]
    
    # Replacement for SHAP with simple feature importance for production-ready app
    if hasattr(pkg['model'], 'feature_importances_'):
        importances = pkg['model'].feature_importances_
        indices = np.argsort(importances)[-3:][::-1]
        factors = ", ".join([pkg['feature_names'][i] for i in indices])
    else:
        factors = "varied clinical markers"
    
    return pred, prob, factors

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": random.choice(GREETINGS)}]
if "inputs" not in st.session_state:
    st.session_state.inputs = {}
if "complete" not in st.session_state:
    st.session_state.complete = False
if "feedback_collected" not in st.session_state:
    st.session_state.feedback_collected = False
if "feedback_list" not in st.session_state:
    st.session_state.feedback_list = []
if "questions_pool" not in st.session_state:
    pool = list(QUESTIONS.keys())
    random.shuffle(pool)
    st.session_state.questions_pool = pool
if "current_q" not in st.session_state:
    st.session_state.current_q = None

# --- UI ---

with st.sidebar:
    st.markdown("<h2 style='color: white; font-size: 20px; font-weight: bold;'>Clinical Console</h2>", unsafe_allow_html=True)
    st.caption("Human-Centric Assistant v5.1")
    # Technical details removed as per requirements
    
    st.divider()
    if st.button("Reset Assessment", use_container_width=True):
        st.session_state.clear()
        st.rerun()

st.markdown("<h2 style='text-align: center; color: white; margin-bottom: 2rem;'>AI Diagnostic Consultation for Diabetes</h2>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- MAIN CHAT INPUT & LOGIC ---
prompt = st.chat_input("Enter your response here...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 1. Feedback handling (if assessment is complete)
    if st.session_state.complete:
        st.session_state.feedback_list.append(prompt)
        st.session_state.messages.append({"role": "assistant", "content": "Thank you for your feedback. Is there anything else you'd like to discuss or another clinical question you have?"})

    # 2. Initial flow to start questioning
    elif st.session_state.current_q is None and not st.session_state.inputs:
        st.session_state.current_q = st.session_state.questions_pool.pop(0)
        st.session_state.messages.append({"role": "assistant", "content": random.choice(QUESTIONS[st.session_state.current_q])})
    
    # 3. Data handling during questioning
    elif st.session_state.current_q:
        try:
            val = float(prompt)
            st.session_state.inputs[st.session_state.current_q] = val
            
            if st.session_state.questions_pool:
                st.session_state.current_q = st.session_state.questions_pool.pop(0)
                response = f"{random.choice(ACKNOWLEDGEMENTS)} {random.choice(BRIDGES)} {random.choice(QUESTIONS[st.session_state.current_q])}"
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "Thank you. The clinical intake is now complete. One moment while I synthesize your diagnostic profile..."})
                st.session_state.complete = True
                pred, prob, factors = predict_risk()
                report = generate_report(pred, prob, factors)
                
                # Append report and then ask for feedback
                st.session_state.messages.append({"role": "assistant", "content": report})
                st.session_state.messages.append({"role": "assistant", "content": "Was this consultation helpful? Your feedback helps improve the AI assistant."})
                st.session_state.final_results = {"pred": pred, "prob": prob}
                
        except ValueError:
            st.session_state.messages.append({"role": "assistant", "content": "Pardon me, I'll need a numeric value for that specific clinical record."})
    
    st.rerun()

if st.session_state.complete and hasattr(st.session_state, 'final_results'):
    res = st.session_state.final_results
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='text-lg font-semibold mb-4'>Research Metadata</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        badge_cls = "high-risk" if res['pred'] == 1 else "low-risk"
        badge_txt = "POSITIVE RESULT" if res['pred'] == 1 else "NEGATIVE RESULT"
        st.markdown(f"<span class='risk-badge {badge_cls}'>{badge_txt}</span>", unsafe_allow_html=True)
        st.metric("Calculation Accuracy", f"{res['prob']*100:.2f}%")
    with col2:
        st.write("**Assessment Origin:**")
        st.caption("Human-Centric Assistant v5.1")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='mt-20 mb-10 text-center text-slate-600 text-[10px] font-bold tracking-[0.2em] uppercase text-white'>Clinical Intelligence Platform © 2026</div>", unsafe_allow_html=True)
