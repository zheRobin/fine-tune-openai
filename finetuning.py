from openai import OpenAI
from dotenv import load_dotenv
import os
import json
load_dotenv()
client = OpenAI()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0125:aryzen::97lUwbDH",
  messages=[
    {"role": "system", "content": "You are an Amazon Review Analysis chatbot, dedicated to evaluating the worth of a given review and discerning if it contains valuable insights, capable of influencing product improvements. My requirement involves merely identifying the value of the review, ranking it from low to high, excluding any explanatory outline. Thus, your response should be a simple, singular term."},
    {"role": "user", "content": "This is becoming a classic and a business must-read. Referred to me by a business partner, reading the book brought great ideas, tactics and strategic approaches. The book and author use the methodology in their own presentation and the psychology of sales. The book takes a few reads and references back to be able to execute - and while the concepts are relatively easy to grasp, they aren't as clearly conveyed as I had hoped. In conjunction with the website and other materials around Building A StoryBrand, you will be able to begin to execute on the strategy and tactics in here - but you'll need more than just the book in my opinion to do it right."}
  ]
)
print(completion.choices[0].message.content)
with open('response/chat_test_response.json', 'w') as f:
    json.dump(completion.choices[0].message.dict(), f, indent=4)