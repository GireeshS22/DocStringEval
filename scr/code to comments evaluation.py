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

#%%
with open("code-to-comments-deepseek-coder-v2-lite-instr-zxn.json", "r") as fp:
    b = json.load(fp)
    
#%%
with open("code-to-comments-mistral-7b-instruct-v0-3-uly.json", "r") as fp:
    b = json.load(fp)


#%%
with open("code-to-comments-clean-code-codellama-7b-instruct-hf-xfe.json", "r") as fp:
    b = json.load(fp)

    
#%%
def extract_title_description(text):
    # Match text format
    title_match = re.search(r"\*\*Title:\*\*(.*?)\n\n", text, re.DOTALL)
    description_match = re.search(r"\*\*Description:\*\*(.*)", text, re.DOTALL)

    if title_match and description_match:
        title = title_match.group(1).strip()
        description = description_match.group(1).strip()
        return title, description
    
    # Match JSON-like format
    json_match = re.search(r'"Title":\s*"(.*?)",\s*"Description":\s*"(.*?)"', text, re.DOTALL)
    if json_match:
        title = json_match.group(1).strip('"')
        description = json_match.group(2).strip('"')
        return title, description

    title_match = re.search(r"\*\*Title\*\*(.*?)\n\n", text, re.DOTALL)
    description_match = re.search(r"\*\*Description\*\*(.*)", text, re.DOTALL)

    if title_match and description_match:
        title = title_match.group(1).strip()
        description = description_match.group(1).strip()
        return title, description

    title_match = re.search(r"\*\*Title\*\*(.*?)\n", text, re.DOTALL)
    description_match = re.search(r"\*\*Description\*\*(.*)", text, re.DOTALL)

    if title_match and description_match:
        title = title_match.group(1).strip()
        description = description_match.group(1).strip()
        return title, description
    
    pattern = r"Title:\s\"(.*?)\"\s*Description:\s\"(.*?)\""
    match = re.search(pattern, text, re.DOTALL)

    if match:
        title = match.group(1)
        description = match.group(2)
        return title, description

    return None, None

#%%
extract_title_description(b[1])

#%%
# Create a DataFrame
df = pd.DataFrame(columns=['Title_llm', 'Description_llm'])
for element in b:
    title, description = extract_title_description(element)
    df = pd.concat([df, pd.DataFrame({'Title_llm': [title], 'Description_llm': [description]})], ignore_index=True)


#%%
data_1 = data.merge(df,left_index=True, right_index=True)
data_1 = data_1[['Title', 'Prompt', 'Title_llm', 'Description_llm']]
data_1 = data_1.fillna('No Response')
data_1 = data_1.replace('', 'No Response')


#%%
import pandas as pd
from rouge import Rouge

def calculate_rouge(df, reference_column, hypothesis_column):
    rouge = Rouge()

    def calculate_score(row):
        scores = rouge.get_scores(row[hypothesis_column].lower(), row[reference_column].lower())
        return scores[0]['rouge-1']['f']

    df['ROUGE-1 ' + reference_column] = df.apply(calculate_score, axis=1)
    return df

# Calculate ROUGE-1 scores
data_1 = calculate_rouge(data_1, 'Title', 'Title_llm')
data_1 = calculate_rouge(data_1, 'Prompt', 'Description_llm')



#%%
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import nltk

def calculate_bleu(df, reference_column, hypothesis_column):
    nltk.download('punkt')

    def calculate_score(row):
        reference = [row[reference_column].lower().split()]
        hypothesis = row[hypothesis_column].lower().split()
        score = sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
        return score

    df['BLEU Score ' + reference_column] = df.apply(calculate_score, axis=1)
    return df


# Calculate BLEU scores
data_1 = calculate_bleu(data_1, 'Title', 'Title_llm')
data_1 = calculate_bleu(data_1, 'Prompt', 'Description_llm')



#%%
data_1['line_count'] = data['Actual answer'].str.count('\n') + 1