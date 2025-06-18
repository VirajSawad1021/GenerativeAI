import google.generativeai as genai
from dotenv import load_dotenv
import os
# No json import needed if functions return dicts

# --- Define your Python functions ---
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
    elif "san francisco" in location.lower():
        weather_info.update({"temperature": "72", "forecast": "sunny with patchy clouds"})
    elif "paris" in location.lower():
        weather_info.update({"temperature": "22", "forecast": "cloudy with a chance of rain"})
    else:
        weather_info.update({"temperature": "unknown", "forecast": "weather data not available"})
    return weather_info

def get_meeting_details(meeting_id: str):
    """Get details for a specific meeting ID, like 'project_alpha_kickoff' or 'weekly_sync'.

    Args:
        meeting_id (str): The unique identifier for the meeting.

    Returns:
        dict: A dictionary containing meeting details or an error if not found.
    """
    print(f"--- Python function get_meeting_details(meeting_id='{meeting_id}') AUTOMATICALLY called ---")
    if meeting_id == "project_alpha_kickoff":
        return {"meeting_id": meeting_id, "topic": "Project Alpha Kickoff", "time": "Tomorrow at 10:00 AM PST", "attendees": ["Alice", "Bob", "Charlie"], "notes": "Agenda: Introductions, project goals, timelines."}
    elif meeting_id == "weekly_sync":
        return {"meeting_id": meeting_id, "topic": "Weekly Team Sync", "time": "Friday at 2:00 PM PST", "attendees": ["Alice", "Bob", "David", "Eve"], "notes": "Discuss weekly progress and blockers."}
    else:
        return {"meeting_id": meeting_id, "error": "Meeting not found."}
# --- End of Python functions ---

load_dotenv()
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)

    # --- Pass the Python function objects directly to the 'tools' parameter ---
    model = genai.GenerativeModel(
        'gemini-1.5-flash-latest',
        tools=[get_current_weather, get_meeting_details] # List of Python functions
    )

    chat = model.start_chat(enable_automatic_function_calling=True)

    print("AUTOMATIC Function Calling Chat (Weather & Meetings - Official Method). Type 'quit' to end.")
    print("-" * 30)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting chat.")
            break
        if not user_input.strip():
            continue

        print("Bot: Thinking...")
        # With automatic function calling, send_message handles the multi-step process
        response = chat.send_message(user_input)
        
        # The 'response' object should be the model's final natural language response.
        # The SDK handles checking for function_call, executing it, sending result, and getting final text.
        print(f"Bot: {response.text}")
        print("-" * 30)

except Exception as e:
    print(f"An error occurred: {e}")