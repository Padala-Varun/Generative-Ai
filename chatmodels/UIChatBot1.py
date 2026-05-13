from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,400;0,500;1,400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0e0e12; color: #f0eee8; }

.mode-screen {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 4rem 1rem; gap: 1.5rem;
}
.mode-title {
    font-family: 'Syne', sans-serif; font-size: 2.2rem;
    font-weight: 800; color: #f0eee8; text-align: center; letter-spacing: -1px;
}
.mode-sub { color: #888; font-style: italic; font-size: 0.95rem; text-align: center; }
.mode-card {
    background: #1c1c24; border: 2px solid #2a2a38;
    border-radius: 20px; padding: 2rem 2.5rem;
    text-align: center; min-width: 160px;
}
.mode-card .emoji { font-size: 3rem; }
.mode-card .label {
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 1.1rem; margin-top: 0.75rem; color: #f0eee8;
}
.mode-card .desc { color: #888; font-size: 0.85rem; margin-top: 0.3rem; }

.chat-header { text-align: center; padding: 1.5rem 0 0.5rem; }
.chat-header h1 {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 2rem; letter-spacing: -1px; margin: 0;
}
.chat-header p { color: #888; font-size: 0.9rem; font-style: italic; margin-top: 0.2rem; }

.msg-row { display: flex; margin: 0.55rem 0; gap: 0.7rem; align-items: flex-end; }
.msg-row.user { flex-direction: row-reverse; }
.avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.avatar.bot-funny { background: #f7e26b; }
.avatar.bot-angry { background: #e74c3c; }
.avatar.user      { background: #3a3af5; }
.bubble { max-width: 72%; padding: 0.7rem 1rem; border-radius: 18px; font-size: 0.96rem; line-height: 1.55; }
.bubble.bot  { background: #1c1c24; border: 1px solid #2a2a38; border-bottom-left-radius: 4px; color: #f0eee8; }
.bubble.user { background: #3a3af5; border-bottom-right-radius: 4px; color: #fff; }

.stTextInput > div > div > input {
    background: #1c1c24 !important; border: 1px solid #2a2a38 !important;
    border-radius: 12px !important; color: #f0eee8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important; padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #f7e26b !important;
    box-shadow: 0 0 0 2px rgba(247,226,107,0.15) !important;
}
.stButton > button {
    background: #f7e26b !important; color: #0e0e12 !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.6rem 1.4rem !important; font-size: 0.95rem !important;
    width: 100%; transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.chat-box { height: 440px; overflow-y: auto; padding: 0.5rem 0.25rem; margin-bottom: 1rem; }
hr { border-color: #2a2a38 !important; }
</style>
""", unsafe_allow_html=True)

# ── Model (cached) ────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

model = get_model()

# ── Session state ─────────────────────────────────────────────────────────────
if "mode"     not in st.session_state: st.session_state.mode     = None
if "messages" not in st.session_state: st.session_state.messages = []
if "exited"   not in st.session_state: st.session_state.exited   = False

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Mode selection
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode is None:
    st.markdown("""
    <div class="mode-screen">
        <div class="mode-title">Choose Your AI Mode</div>
        <div class="mode-sub">Pick a personality for your chatbot</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="mode-card">
            <div class="emoji">😂</div>
            <div class="label">Funny Mode</div>
            <div class="desc">Jokes, puns &amp; good vibes</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select Funny", key="btn_funny"):
            st.session_state.mode = "funny"
            st.session_state.messages = [SystemMessage(content="You are a funny chatbot.")]
            st.rerun()

    with col2:
        st.markdown("""
        <div class="mode-card">
            <div class="emoji">😤</div>
            <div class="label">Angry Mode</div>
            <div class="desc">Short fuse, zero patience</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select Angry", key="btn_angry"):
            st.session_state.mode = "angry"
            st.session_state.messages = [SystemMessage(content="You are an angry chatbot.")]
            st.rerun()

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Exit screen
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.exited:
    st.markdown("""
    <div style="text-align:center; padding:5rem 0;">
        <div style="font-size:4rem;">👋</div>
        <h2 style="font-family:'Syne',sans-serif; color:#f7e26b; margin-top:1rem;">Goodbye!</h2>
        <p style="color:#888;">You typed <code>0</code> to exit. Refresh the page to start over.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — Chat
# ══════════════════════════════════════════════════════════════════════════════
mode         = st.session_state.mode
header_emoji = "😂" if mode == "funny" else "😤"
header_label = "Funny Mode" if mode == "funny" else "Angry Mode"
header_sub   = "Jokes, puns & good vibes" if mode == "funny" else "Short fuse, zero patience"
avatar_class = "bot-funny" if mode == "funny" else "bot-angry"
bot_avatar   = "😂" if mode == "funny" else "😤"

st.markdown(f"""
<div class="chat-header">
    <h1>{header_emoji} {header_label}</h1>
    <p>{header_sub} · Type <strong>0</strong> to exit</p>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Chat bubbles
chat_html = '<div class="chat-box">'
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
            <div class="avatar {avatar_class}">{bot_avatar}</div>
            <div class="bubble bot">{msg.content}</div>
        </div>"""
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

# Input row
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("", placeholder="Type a message...", key="user_input", label_visibility="collapsed")
with col2:
    send = st.button("Send")

# Handle send
if send and user_input.strip():
    if user_input.strip() == "0":
        st.session_state.exited = True
        st.rerun()
    else:
        st.session_state.messages.append(HumanMessage(content=user_input.strip()))
        with st.spinner("Typing..."):
            response = model.invoke(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))
        st.rerun()