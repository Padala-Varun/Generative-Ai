from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from typing import Optional, List
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ----------------------------
# LLM
# ----------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2
)

# ----------------------------
# Schema
# ----------------------------
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    ratings: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)

# ----------------------------
# Prompt
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract movie information from the given paragraph.

{format_instructions}
"""
    ),
    ("human", "{paragraph}")
])

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="Movie Info Extractor",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Information Extractor")
st.write("Extract structured movie details from a paragraph using Gemini + LangChain.")

paragraph = st.text_area(
    "Paste Movie Paragraph",
    height=250,
    placeholder="Paste movie description here..."
)

if st.button("Extract Movie Info", use_container_width=True):

    if not paragraph.strip():
        st.warning("Please enter a movie paragraph.")
        st.stop()

    try:
        with st.spinner("Extracting movie details..."):

            final_prompt = prompt.invoke(
                {
                    "paragraph": paragraph,
                    "format_instructions": parser.get_format_instructions()
                }
            )

            response = model.invoke(final_prompt)

            movie = parser.parse(response.content)

        st.success("Extraction Completed!")

        # ----------------------------
        # Nice UI
        # ----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎥 Movie Details")
            st.write(f"**Title:** {movie.title}")
            st.write(f"**Release Year:** {movie.release_year}")
            st.write(f"**Director:** {movie.director}")
            st.write(f"**Rating:** {movie.ratings}")

        with col2:
            st.subheader("🎭 Genre")
            st.write(", ".join(movie.genre))

            st.subheader("👥 Cast")
            for actor in movie.cast:
                st.write(f"• {actor}")

        st.subheader("📝 Summary")
        st.write(movie.summary)

        st.subheader("📦 JSON Output")
        st.json(movie.model_dump())

    except Exception as e:
        st.error(f"Error: {str(e)}")