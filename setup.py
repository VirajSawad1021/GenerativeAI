import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or your preferred model

    # --- Define your GenerationConfig ---
    generation_config = genai.types.GenerationConfig(
        temperature=0.9,      # Default is often around 0.9-1.0. Let's try explicit values.
        top_p=0.95,           # Example value
        top_k=40,             # Example value
        max_output_tokens=10, # Limit the output length
        candidate_count=3     # We only want one main answer for now
        # stop_sequences=["\n\n\n"] # Example: Stop if it generates three newlines
    )

    prompt = "Write a short, imaginative story (around 100 words) about a squirrel who dreams of flying to Mars."

    print(f"--- Generating with custom config: Temp={generation_config.temperature}, TopP={generation_config.top_p}, TopK={generation_config.top_k}, MaxTokens={generation_config.max_output_tokens} ---")

    response = model.generate_content(
        prompt,
        generation_config=generation_config # Pass the config here
    )

    print("\n--- Story with 3 candidates---")
    for i, candidate in enumerate(response.candidates):
       
       if candidate.content and candidate.content.parts:
          print(f"Candidate {i+1}: {candidate.content.parts[0].text}")
          print('finish reason: ', candidate.finish_reason)

    print('-'*10)

    
    print("\n--- Usage Metadata (Tokens) ---")
    if response.usage_metadata: # Check if usage_metadata exists
      print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
      print(f"Candidates tokens: {response.usage_metadata.candidates_token_count}")
      print(f"Total tokens: {response.usage_metadata.total_token_count}")
    else:
      print("Usage metadata not available.")


    # --- Experiment: Now try with a very different configuration ---
    low_temp_config = genai.types.GenerationConfig(
        temperature=0.1,
        top_p=0.7, # Lowering Top P as well for less randomness
        top_k=20,  # Lowering Top K
        max_output_tokens=150
    )

    print(f"\n--- Generating with LOW TEMP config: Temp={low_temp_config.temperature}, TopP={low_temp_config.top_p}, TopK={low_temp_config.top_k} ---")
    response_low_temp = model.generate_content(
        prompt, # Same prompt
        generation_config=low_temp_config
    )

    print("\n--- Story (Low Temp) ---")
    print(response_low_temp.text)

    print("\n--- Usage Metadata (Tokens) - Low Temp ---")
    if response_low_temp.usage_metadata:
        print(f"Prompt tokens: {response_low_temp.usage_metadata.prompt_token_count}")
        print(f"Candidates tokens: {response_low_temp.usage_metadata.candidates_token_count}")
        print(f"Total tokens: {response_low_temp.usage_metadata.total_token_count}")
    else:
        print("Usage metadata not available for low temp response.")


except Exception as e:
    print(f"An error occurred: {e}")
    # If the error is about genai.types.GenerationConfig not found,
    # it might be genai.GenerationConfig directly depending on SDK version.
    # Or, it could be that GenerationConfig is passed as a dictionary:
    # generation_config = { "temperature": 0.9, ... }
    # Always check latest SDK documentation for the exact structure.