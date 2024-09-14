from openai import OpenAI
from dotenv import load_dotenv
import os
import json
load_dotenv()
client = OpenAI()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
with open("response/fine_tuning_response.json") as f:
    fine_tune_response = json.load(f)

# Get the job ID from the response
job_id = fine_tune_response.get('id')
# Load the fine tuning response from the file

fine_tune_events = client.fine_tuning.jobs.retrieve(job_id)
with open('response/event_check_response.json', 'w') as f:
    json.dump(fine_tune_events.dict(), f, indent=4)