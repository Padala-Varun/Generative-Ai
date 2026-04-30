from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0.2,max_tokens=20)

messages = [

]

print("-----------------------Welcome Type 0 to exit application-------------------------")


while True: 

    prompt = input("You : ")
    messages.append(prompt)
    if prompt == "0":
        break
    response = model.invoke(messages)
    messages.append(response.content)

    print("Bot = ",response.content)