# chatbot/bot.py
from groq import Groq
import os
from dotenv import load_dotenv

class Chatbot:
    #Initializes the Groq LLM client.
    def __init__(self, api_key=None):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
        
    #Sends a user message to the LLM and returns the response.
    def get_response(self, user_message: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.7,
                max_completion_tokens=512,
                top_p=1,
                stream=False
            )

            return completion.choices[0].message.content

        except Exception as e:
            return f"Error: {e}"

#To fetch the API key from .env file
load_dotenv()
GROQ_KEY= os.getenv("GROQ_API_KEY")

#Chatbot testing, runs in this file directly.Runs a continuous chat loop in the terminal.
def chat_loop():
    bot = Chatbot(api_key=os.environ["GROQ_API_KEY"])
    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        # Stop the chat
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Get answer from the chatbot
        response = bot.get_response(user_input)
        print("Bot:", response, "\n")


if __name__ == "__main__":
    chat_loop()