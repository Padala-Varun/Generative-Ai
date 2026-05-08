from dotenv import load_dotenv
load_dotenv()

import streamlit as st


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Funny Chatbot",
    page_icon="😂",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,400;0,500;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: #0e0e12;
    color: #f0eee8;
}

/* Header */
.chat-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.chat-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    color: #f7e26b;
    letter-spacing: -1px;
    margin: 0;
}
.chat-header p {
    color: #888;
    font-size: 0.95rem;
    margin-top: 0.3rem;
    font-style: italic;
}

/* Chat bubbles */
.msg-row {
    display: flex;
    margin: 0.6rem 0;
    gap: 0.75rem;
    align-items: flex-end;
}
.msg-row.user { flex-direction: row-reverse; }

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.avatar.bot  { background: #f7e26b; }
.avatar.user { background: #3a3af5; }

.bubble {
    max-width: 72%;
    padding: 0.75rem 1.1rem;
    border-radius: 18px;
    font-size: 0.97rem;
    line-height: 1.55;
}
.bubble.bot {
    background: #1c1c24;
    border: 1px solid #2a2a38;
    border-bottom-left-radius: 4px;
    color: #f0eee8;
}
.bubble.user {
    background: #3a3af5;
    border-bottom-right-radius: 4px;
    color: #fff;
}

/* Input area */
.stTextInput > div > div > input {
    background: #1c1c24 !important;
    border: 1px solid #2a2a38 !important;
    border-radius: 12px !important;
    color: #f0eee8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #f7e26b !important;
    box-shadow: 0 0 0 2px rgba(247,226,107,0.15) !important;
}

/* Send button */
.stButton > button {
    background: #f7e26b !important;
    color: #0e0e12 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.4rem !important;
    font-size: 0.95rem !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Divider */
hr { border-color: #2a2a38 !important; }

/* Scrollable chat area */
.chat-box {
    height: 460px;
    overflow-y: auto;
    padding: 0.5rem 0.25rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Model (cached) ────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        max_tokens=20,
    )

model = get_model()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a funny chatbot.")
    ]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h1>😂 Funny Chatbot</h1>
    <p>Powered by Gemini · Always in a good mood</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Chat history display ──────────────────────────────────────────────────────
chat_html = '<div class="chat-box" id="chat-box">'
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        chat_html += f"""
        <div class="msg-row user">
            <div class="avatar user">🙂</div>
            <div class="bubble user">{msg.content}</div>
        </div>"""
    elif isinstance(msg, AIMessage):
        chat_html += f"""
        <div class="msg-row">
            <div class="avatar bot">🤖</div>
            <div class="bubble bot">{msg.content}</div>
        </div>"""
chat_html += "</div>"

st.markdown(chat_html, unsafe_allow_html=True)

# Auto-scroll to bottom
st.markdown("""
<script>
    const box = document.getElementById('chat-box');
    if (box) box.scrollTop = box.scrollHeight;
</script>
""", unsafe_allow_html=True)

# ── Input area ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        label="Message",
        placeholder="Type a message...",
        key="user_input",
        label_visibility="collapsed",
    )

with col2:
    send = st.button("Send")

# ── Handle send ───────────────────────────────────────────────────────────────
if send and user_input.strip():
    st.session_state.messages.append(HumanMessage(content=user_input.strip()))

    with st.spinner("Thinking..."):
        response = model.invoke(st.session_state.messages)

    st.session_state.messages.append(AIMessage(content=response.content))
    st.rerun()