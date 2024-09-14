from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0125:aryzen::91pL6BvT",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)
print(completion.choices[0].message)