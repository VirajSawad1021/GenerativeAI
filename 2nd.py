import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

generation_config = genai.types.GenerationConfig(
    temperature=0.9,
    top_p=0.95,
    top_k=40,
    max_output_tokens=2000,
)
history = [
       {'role':'user', 'parts': [{'text':'Briefly tell me about the Roman Empire.'}]},
       {'role':'model', 'parts': [{'text':'The Roman Empire was vast...'}]}
    ]
chat = model.start_chat(history=history)

print('Starting new chat session...! Enter "exit" to end the chat.')

while True:
    user_input = input('You: ')
    if user_input.lower() == 'exit':
        break
    response = chat.send_message(user_input, generation_config=generation_config)
    print(f'AI: {response.text}')




