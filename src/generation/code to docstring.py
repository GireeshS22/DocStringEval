# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 11:02:11 2024

@author: ADMIN
"""

import json
import pandas as pd
import requests
import re

import itertools 


#%%
import pickle

# Load the DataFrame from the file
with open('class_files_df.pkl', 'rb') as f:
    class_files_df = pickle.load(f)


with open("Output\code-to-docstring-codellama-34b-instruct-hf-kzi_NEW.json", "r") as fp:
    generated = json.load(fp)


#%%

sys_prompt = """
You are a python docstring generator. Given a python class, write a docstring for the given program.
Only return the docstring for the program. Do not generate any additional details. 
"""



#%%
import pandas as pd
from openai import OpenAI


client = OpenAI(
    base_url="https://ewvvf257bihy4qtj.us-east-1.aws.endpoints.huggingface.cloud/v1/",
    api_key="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"  # Replace with your actual API key
)

# Create an empty list to store responses for each row
all_responses = []

for (prog, generated_) in itertools.zip_longest(class_files_df["Code_without_comments"].tolist(), generated):
#    user_prompt = data["Actual answer"].iloc[i]
    user_prompt = prog
    #print(prog)
  # Create the message list
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
##        {"role": "user", "content": sys_prompt + "\n" + user_prompt},
##        {"role": "agent", "content": generated_},
##        {"role": "user", "content": """
##         Given the above docstring, make it better. 
##        Only return the docstring for the program. Do not generate any additional details. 
##        """}
    ]
    
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=messages,
        top_p=None,
        temperature=0.1,
        max_tokens=1000,
        stream=False,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None
    )
    
    #print(chat_completion)
    
    print(chat_completion.choices[0].message.content)

    # Add the list of responses for this row to the main list
    all_responses.append(chat_completion.choices[0].message.content)

    
#%%
with open("Output\code-to-docstring-qwen2-5-7b-instruct-qne.json", "w") as fp:
    json.dump(all_responses, fp)

"""    
with open("Output\code-to-docstring-clean_codegemma-7b-it-dfe.json", "r") as fp:
    b = json.load(fp)
"""

#%%
# If necessary, install the openai Python library by running 
# pip install openai

from openai import OpenAI

client = OpenAI(
	base_url="https://whk5k44myqhlqekc.us-east-1.aws.endpoints.huggingface.cloud/v1/", 
	api_key="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" 
)

chat_completion = client.chat.completions.create(
	model="tgi",
	messages=[
        {"role": "agent", "content": "yuou are a system explainer"},
	{
		"role": "user",
		"content": "What is deep learning?"
	}
],
	top_p=None,
	temperature=None,
	max_tokens=150,
	stream=True,
	seed=None,
	stop=None,
	frequency_penalty=None,
	presence_penalty=None
)

for message in chat_completion:
	print(message.choices[0].delta.content, end="")

#%%