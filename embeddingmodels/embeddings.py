from dotenv import load_dotenv

load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings



embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
texts=["hello, world!","goodbye, world!","hello, everyone!"]
vector = embeddings.embed_query("hello, world!")
print(len(vector))
print(vector[:3])

vectors = embeddings.embed_documents(texts)
print(len(vectors))
print(len(vectors[0]))
print(vectors[0][:3])