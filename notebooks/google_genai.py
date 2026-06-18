from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="Give worlds top 3 cricket teams.",
    config={"response_mime_type": "application/json"},
)

for chunk in response:
    print(chunk.text, end="", flush=True)

# for model in client.models.list():
#     print(model.name, model.display_name)
