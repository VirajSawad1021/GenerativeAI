import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

# 1. Define your Python function(s)
def get_current_weather(location: str, unit: str = "celsius"):
    """Get the current weather in a given location.

    Args:
        location (str): The city and state, e.g., "San Francisco, CA".
        unit (str): The temperature unit, either "celsius" or "fahrenheit". Defaults to "celsius".

    Returns:
        dict: A dictionary describing the weather.
    """
    print(f"--- Python function get_current_weather(location='{location}', unit='{unit}') AUTOMATICALLY called ---")
    weather_info = {"location": location, "unit": unit}
    if "tokyo" in location.lower():
        weather_info.update({"temperature": "10", "forecast": "snowy"})
    elif "san francisco" in location.lower(): # Corrected to match common spelling
        weather_info.update({"temperature": "72", "forecast": "sunny with patchy clouds"})
    elif "paris" in location.lower():
        weather_info.update({"temperature": "22", "forecast": "cloudy with a chance of rain"})
    else:
        weather_info.update({"temperature": "unknown", "forecast": "weather data not available"})
    return weather_info

# 2. Schema Declaration (still good to have for clarity, though SDK might infer)
# We might not strictly need to pass this explicitly if passing the function directly works
# and the SDK can infer the schema, but it doesn't hurt.
get_weather_func_declaration = genai.types.FunctionDeclaration(
    name="get_current_weather",
    description="Get the current weather in a given location.",
    parameters={
        'type': 'object',
        'properties': {
            'location': {'type': 'string', 'description': "The city and state, e.g., San Francisco, CA"},
            'unit': {'type': 'string', 'description': "The temperature unit, either 'celsius' or 'fahrenheit'", 'enum': ["celsius", "fahrenheit"]}
        },
        'required': ["location"]
    }
)
# We might not need weather_tool_declaration if passing get_current_weather function directly to model's tools
# weather_tool_declaration = genai.types.Tool(
#     function_declarations=[get_weather_func_declaration]
# )


load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        'gemini-1.5-flash-latest',
        # Pass the Python function(s) directly in a list.
        # The SDK will use these for execution and often try to infer their schema.
        tools=[get_current_weather]
    )

    chat = model.start_chat(enable_automatic_function_calling=True)

    print("AUTOMATIC Function Calling Chat. Ask about the weather. Type 'quit' to end.")
    print("-" * 30)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting chat.")
            break
        if not user_input.strip():
            continue

        print("Bot: Thinking...")
        response = chat.send_message(user_input)
        print(f"Bot: {response.text}")
        print("-" * 30)

except Exception as e:
    print(f"An error occurred: {e}")