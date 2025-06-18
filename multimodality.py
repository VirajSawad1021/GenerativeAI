import google.generativeai as genai
from dotenv import load_dotenv
import os
from PIL import Image # For loading images

load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")

    genai.configure(api_key=api_key)

    # --- Prepare your Image and Text Prompt ---
    # 1. Image
    # IMPORTANT: Replace 'your_image.jpg' with the actual path to your image file.
    # For example: 'cat.jpg' if it's in the same directory,
    # or '/path/to/your/image.jpg' for an absolute path.
    image_path = 'me.jpeg' # <--- !!! CHANGE THIS !!!

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'. Please check the path.")
        exit()
    except Exception as e:
        print(f"Error opening image: {e}")
        exit()


    # 2. Text Prompt
    text_prompt = "Describe this image in detail. What do you see? and after that you have to ask user 'What kind of story should we create based on this image? (e.g., adventure, mystery, funny)' then User provides an opening line or a suggestion. you continue the story a paragraph.User adds to the story gives a new direction. and then will have to Repeat a few turns with user."
    
    model_name = 'gemini-1.5-flash-latest'
    model = genai.GenerativeModel(model_name)

    print(f"Using model: {model_name}")
    print(f"Processing image: {image_path}")
    print(f"With prompt: \"{text_prompt}\"")
    print("Generating response...")

    # --- Send the Image and Text to the Model ---
    # The 'contents' argument for generate_content can take a list
    # where elements can be text strings or image objects (from PIL).
    # The order can matter: often text first, then image, or interleaved.
    # The SDK handles converting the PIL Image object into the format the API needs.
    chat = model.start_chat()
    # You can also use streaming for multimodal if desired:
    response_stream = chat.send_message([text_prompt, img], stream=True)
    for chunk in response_stream:
        print(chunk.text, end="", flush=True)
    print()

    while True:

        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting chat.")
            break
        response_stream = chat.send_message(user_input, stream=True)
        print(f"AI: " , end="")
        for chunk in response_stream:
            print(f"{chunk.text}", end="", flush=True)
        print()


    print("\n--- Usage Metadata (Tokens) ---")
    if response_stream.usage_metadata:
      print(f"Prompt tokens: {response_stream.usage_metadata.prompt_token_count}")
      print(f"Candidates tokens: {response_stream.usage_metadata.candidates_token_count}")
      print(f"Total tokens: {response_stream.usage_metadata.total_token_count}")
    else:
      print("Usage metadata not available.")


except Exception as e:
    print(f"An error occurred: {e}")