from dotenv import load_dotenv
import os
import json
from openai import OpenAI
load_dotenv()
client = OpenAI()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
file = 'dataset_amazon-reviews-scraper_2024-03-16_13-49-28-173.json'
file_path = os.path.join('resource', file)
result_file = os.path.join('result', os.path.splitext(file)[0] + '-result.json')

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(result_file, 'w', encoding='utf-8') as f:
    f.write("[")

    for i, review in enumerate(data):
        title, description = review['reviewTitle'], review['reviewDescription']
        user_content = f'Title -> {title}, Description -> {description}'

        completion = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:aryzen::93PMx1pD",
            messages=[
                {"role": "system", "content": "You are an Amazon Review Analysis chatbot. Your purpose is to assess the value of review content, determining whether it holds valuable insights that can be used for updating the product. Remember, the value of the review is not based on the reviewer's psychology, but the potential applicability of the feedback to product improvement. Analyze the content diligently and objectively. Your answers must be simple answers. For example, if the review is positive and valuable, you should answer 'high_value_positive'. If it is negative and valuable answer is 'high_value_negative'. If it is neither positive nor negative but valuable, answer 'high_value_middle'. You can't recognize any valuable entities, answer 'low_value'."},
                {"role": "user", "content": user_content}
            ]
        )

        ai_answer = completion.choices[0].message.content

        json.dump({"Title": title, "Description": description, "AI_Category": ai_answer}, f)

        if i < len(data) - 1:
            f.write(",\n")

    f.write("]")
