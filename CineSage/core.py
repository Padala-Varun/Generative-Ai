from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
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
    ("human","""
    Extract Information from the paaragraph
    {paragraph}
    """)
])

para=input("Give your paragraph : ")

final_prompt = prompt.invoke(
    {"paragraph": para}
)

response = model.invoke(final_prompt)

print(response.content)