from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0.2,max_tokens=20)

response = model.invoke("Who is MS Dhoni?")

print(response.content)