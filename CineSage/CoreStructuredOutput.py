from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.output_parsers import PydanticOutputParser


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0.2)

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str] 
    cast: List[str]
    ratings: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)


prompt = ChatPromptTemplate.from_messages([
    ('system',"""
Extract movie information from the given paragraph 
     {format_instructions}
"""),

('human',"{paragraph}")
    
]
)

para=input("Give your paragraph : ")

final_prompt = prompt.invoke(
    {"paragraph": para,
     'format_instructions': parser.get_format_instructions()}

)

response = model.invoke(final_prompt)

print(response.content)