from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Tool
from utils.weather_info import get_current_temperature
from validations.google_genai_verify import WeatherResponse
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

weather_function = {
    "name": "get_current_temperature",
    "description": "Gets the current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name, e.g. San Francisco",
            },
        },
        "required": ["location"],
    },
}

tools = Tool(function_declarations=[weather_function])
config = GenerateContentConfig(tools=[tools])

user_message = "What is the temperature in Delhi?"

# Turn 1: model decides to call the tool
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_message,
    config=config,
)

function_call = response.candidates[0].content.parts[0].function_call

if function_call:
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")

    raw_result = get_current_temperature(**function_call.args)

    # Pydantic validation — production pattern
    validated = WeatherResponse(**raw_result)
    print(f"Validated result: {validated}")

    # Turn 2: send tool result back to model for final answer
    tool_response_part = types.Part.from_function_response(
        name=function_call.name,
        response=validated.model_dump(),
    )

    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text=user_message)]),
            response.candidates[0].content,
            types.Content(role="user", parts=[tool_response_part]),
        ],
        config=config,
    )

    print(f"\nFinal answer: {final_response.text}")
else:
    print(response.text)
