import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")

    genai.configure(api_key=api_key)

    chat_generation_config = genai.types.GenerationConfig(
        temperature=0.7,
        max_output_tokens=2000 # Max tokens for the whole chat turn
    )

    model = genai.GenerativeModel(
        'gemini-1.5-flash-latest', # Or your preferred model
        generation_config=chat_generation_config
    )

    chat = model.start_chat(history=[])
    print("Starting a new chat session with STREAMING. Type 'quit' or 'exit' to end.")
    print("-" * 30)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting chat.")
            break

        if not user_input.strip():
            continue

        # --- Send User Message with Streaming ---
        print("Bot: ", end="", flush=True) # Print "Bot: " and stay on the same line

        # Use stream=True
        response_stream = chat.send_message(user_input, stream=True)

        # Iterate over the chunks in the stream
        for chunk in response_stream:
            # Each chunk has a .text attribute (among others)
            # Make sure the chunk has text before printing. Sometimes, initial chunks
            # might not have text or could be metadata.
            if hasattr(chunk, 'text') and chunk.text:
                print(chunk.text, end="", flush=True) # Print chunk text and stay on the same line
            elif hasattr(chunk, 'parts'): # Alternative way some SDKs structure it
                for part in chunk.parts:
                    if hasattr(part, 'text') and part.text:
                         print(part.text, end="", flush=True)


        print() # Move to the next line after the full response is streamed
        print("-" * 30)

        # Note: The chat.history is typically updated with the *full* response
        # once the stream is complete, not chunk by chunk.
        # You can verify this if you wish:
        # print("\n--- Current Chat History (after full stream) ---")
        # for message in chat.history:
        #     print(f"{message.role}: {message.parts[0].text if message.parts else ''}")
        # print("-" * 30)


except Exception as e:
    print(f"\nAn error occurred: {e}")