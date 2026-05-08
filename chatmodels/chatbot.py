from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0.2,max_tokens=20)

messages = [
    SystemMessage(content="You are a funny chatbot.")
]

print("-----------------------Welcome Type 0 to exit application-------------------------")


while True: 

    prompt = input("You : ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print("Bot = ",response.content)
print("Messages:", messages)