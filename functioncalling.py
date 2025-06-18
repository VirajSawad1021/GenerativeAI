import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

# --- (Assume get_current_weather function is here) ---
def get_current_weather(location: str, unit: str = "celsius"):
    print(f"--- Python function get_current_weather(location='{location}', unit='{unit}') called ---")
    if "tokyo" in location.lower():
        return f'{{"location": "{location}", "temperature": "10", "unit": "{unit}", "forecast": "snowy"}}'
    elif "san francisco" in location.lower():
        return f'{{"location": "{location}", "temperature": "72", "unit": "{unit}", "forecast": "sunny with patchy clouds"}}'
    elif "paris" in location.lower():
        return f'{{"location": "{location}", "temperature": "22", "unit": "{unit}", "forecast": "cloudy with a chance of rain"}}'
    else:
        return f'{{"location": "{location}", "temperature": "unknown", "unit": "{unit}", "forecast": "weather data not available"}}'

# --- (Assume FunctionDeclaration and Tool setup is here using dicts for parameters) ---
get_weather_func = genai.types.FunctionDeclaration(
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
weather_tool = genai.types.Tool(
    function_declarations=[get_weather_func]
)

load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        'gemini-1.5-flash-latest',
        tools=[weather_tool]
    )

    chat = model.start_chat(enable_automatic_function_calling=False)
    print("Function Calling Chat. Ask about the weather. Type 'quit' to end.")
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
        
        function_call_to_process = None
        # Check for function call in the response
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call_to_process = part.function_call
                    break

        if function_call_to_process:
            fc = function_call_to_process
            print(f"Bot wants to call function: {fc.name}")
            print(f"With arguments: {dict(fc.args)}")

            if fc.name == "get_current_weather":
                location_arg = fc.args.get("location")
                unit_arg = fc.args.get("unit", "celsius")
                function_response_data_str = get_current_weather(location=location_arg, unit=unit_arg)

                try:
                    actual_function_payload = json.loads(function_response_data_str)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from function: {e}")
                    actual_function_payload = {"error": "failed to decode function response"}

                print(f"Bot: Sending function result back to model...")
                
                # --- NEW ATTEMPT TO SEND FUNCTION RESPONSE ---
                response_after_function_call = chat.send_message(
                    [ # Send a list containing one dictionary that represents the function response part
                        {
                            "function_response": {
                                "name": fc.name,
                                "response": actual_function_payload
                            }
                        }
                    ]
                )
                # --- END OF NEW ATTEMPT ---

                print(f"Bot: {response_after_function_call.text}")
            else:
                print(f"Bot: Unknown function call {fc.name}, not executing.")
        else:
            print(f"Bot: {response.text}")
        print("-" * 30)

except Exception as e:
    print(f"An error occurred: {e}")