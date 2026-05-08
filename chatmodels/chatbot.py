from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0.2)

print("Choose Your AI Mode :")
print("Choose 1 to funny mode")
print("Choose 2 to angry mode")
mode = input("Enter your choice : ")

if mode == "1":
    mode = "You are a funny chatbot."
    
elif mode == "2":
    mode = "You are a angry chatbot."
   

messages = [
        SystemMessage(content=mode)
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