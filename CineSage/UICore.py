from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Info Extractor", page_icon="🔍", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,400;0,500;1,400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0e0e12; color: #f0eee8; }

.page-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.page-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    color: #f0eee8;
    letter-spacing: -1px;
    margin: 0;
}
.page-header p {
    color: #888;
    font-size: 0.95rem;
    font-style: italic;
    margin-top: 0.4rem;
}

/* Textarea */
.stTextArea > label { color: #aaa !important; font-size: 0.9rem !important; }
.stTextArea textarea {
    background: #1c1c24 !important;
    border: 1px solid #2a2a38 !important;
    border-radius: 14px !important;
    color: #f0eee8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    line-height: 1.6 !important;
    padding: 1rem !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: #f7e26b !important;
    box-shadow: 0 0 0 2px rgba(247,226,107,0.15) !important;
}

/* Button */
.stButton > button {
    background: #f7e26b !important;
    color: #0e0e12 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-size: 1rem !important;
    width: 100%;
    transition: opacity 0.15s !important;
    margin-top: 0.5rem;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Result card */
.result-card {
    background: #1c1c24;
    border: 1px solid #2a2a38;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    white-space: pre-wrap;
    font-size: 0.96rem;
    line-height: 1.75;
    color: #f0eee8;
}

hr { border-color: #2a2a38 !important; }
</style>
""", unsafe_allow_html=True)

# ── Model & prompt (cached) ───────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

@st.cache_resource
def get_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert information extraction assistant.

Your task is to analyze the given paragraph and extract the most useful information in a clear, well-structured format.

Instructions:
- Read the paragraph carefully.
- Identify the main entity being discussed.
- Extract all important details explicitly mentioned.
- Do NOT invent missing information.
- If something is not mentioned, simply skip it.
- Generate a concise summary at the end.
- Keep the response clean, organized, and easy to read.

Extract useful information such as (but not limited to):
- Main entity / title / name
- Entity type (movie, book, product, company, person, event, etc.)
- Genre / category / classification
- Creator / founder / director / author
- Cast / important people / contributors
- Release year / important dates
- Ratings / scores / rankings
- Source of rating (if mentioned)
- Awards / recognition
- Key themes / topics
- Main storyline / purpose / description
- Notable features / highlights
- Critical reception / sentiment
- Important locations / setting
- Related concepts / keywords
- Any other meaningful insights

Output format:

Main Entity:
Entity Type:
Genre / Category:
Creator / Author / Director:
Important People:
Release Year / Date:
Ratings:
Source of Rating:
Awards / Recognition:
Key Themes:
Description / Plot:
Notable Features:
Reception / Sentiment:
Setting / Location:
Related Keywords:
Additional Insights:

Quick Summary:
(2–3 concise sentences)
            """
        ),
        ("human", "Extract Information from the paragraph\n{paragraph}")
    ])

model  = get_model()
prompt = get_prompt()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🔍 Info Extractor</h1>
    <p>Paste any paragraph and extract structured information instantly</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

paragraph = st.text_area(
    "Your Paragraph",
    placeholder="Paste your paragraph here...",
    height=200,
)

extract = st.button("Extract Information")

if extract:
    if not paragraph.strip():
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Extracting..."):
            final_prompt = prompt.invoke({"paragraph": paragraph})
            response     = model.invoke(final_prompt)

        st.markdown(f'<div class="result-card">{response.content}</div>', unsafe_allow_html=True)