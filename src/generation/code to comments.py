# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 11:02:11 2024

@author: ADMIN
"""

import json
import pandas as pd
import requests
import re

#%%
data = pd.read_csv("LLM Hard questions.csv")
data = data.dropna(subset = ["Prompt"])

with open("clean_code.json", "r") as fp:
    clean_code = json.load(fp)


#%%
"""
---------------
LLM2 - Hugging face models
---------------

Billing information is available at: https://ui.endpoints.huggingface.co/GireeshS/endpoints/dedicated
Please be mindful of the billing

"""


#codellama-python
API_URL = "https://c9ra0fh6n00g1w7i.us-east-1.aws.endpoints.huggingface.cloud"

#codellama-instruct
#API_URL = "https://f2bfcu86if7wsdfn.us-east-1.aws.endpoints.huggingface.cloud"


#%%

sys_prompt = """
Given a python program, explain the python program.
Give a one line title for the program and write a short brief description like docstring for the program
The output should be in a json format {"Title":"xxxxx", "Description":"yyyyy"}
Only return the json for Title and Description in the output. Do not return anything else.
"""

#%%
import pandas as pd
from openai import OpenAI


client = OpenAI(
    base_url="https://f2bfcu86if7wsdfn.us-east-1.aws.endpoints.huggingface.cloud/v1/",
    api_key="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"  # Replace with your actual API key
)

# Create an empty list to store responses for each row
all_responses = []

for prog in clean_code:
#    user_prompt = data["Actual answer"].iloc[i]
    user_prompt = prog
    print(prog)
  # Create the message list
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=messages,
        top_p=None,
        temperature=None,
        max_tokens=150,
        stream=False,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None
    )
    
    print(chat_completion)
    
    print(chat_completion.choices[0].message.content)

    # Add the list of responses for this row to the main list
    all_responses.append(chat_completion.choices[0].message.content)



#%%
headers = {
	"Accept" : "application/json",
	"Authorization": "Bearer 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",
	"Content-Type": "application/json",
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

sys_prompt = """
Given a python program, explain the python program.
Give a one line title for the program and write a short brief description like docstring for the program
The output should be in a json format {"Title":"xxxxx", "Description":"yyyyy"}
Only return the json for Title and Description in the output. Do not return anything else.
"""


def get_llm_response(user_prompt):
    output = query({
        "inputs": r"""
        [INST] """ + sys_prompt + """ [/INST]
        [USER] """ + user_prompt + """ [/USER]
        """,
        "parameters": {
		"top_k": 10,
		"top_p": 0.95,
		"temperature": 0.1,
		"max_new_tokens": 1500,
		"return_full_text": False
        }
    })
    
    print(output)
    
    if bool(output):
        response = output[0]["generated_text"]    
        return response
    
    else: return "No response"

#%%
llm_response = []

#%%

for i in range(0, len(data)):
    user_prompt = data["Actual answer"].iloc[i]
    print(i)
    response = get_llm_response(user_prompt)
    llm_response.append(response)
    
#%%
with open("code-to-comments-clean-code-codellama-7b-instruct-hf-xfe.json", "w") as fp:
    json.dump(all_responses, fp)
    
with open("code-to-comments-clean-code-codellama-7b-instruct-hf-xfe.json", "r") as fp:
    b = json.load(fp)
