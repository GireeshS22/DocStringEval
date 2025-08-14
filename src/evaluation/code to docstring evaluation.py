# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 11:02:11 2024

@author: ADMIN
"""

import json
import pandas as pd
import requests
import re
import pickle

#%%
# Load the DataFrame from the file
with open('class_files_df.pkl', 'rb') as f:
    class_files_df = pickle.load(f)


#%%
with open("code-to-docstring-clean-codellama-13b-instruct-hf-abo.json", "r") as fp:
    b = json.load(fp)
    

#%%
# Create a DataFrame
df = pd.DataFrame({"Docstring_generated": b})

#%%
data_1 = class_files_df.merge(df,left_index=True, right_index=True)


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
data_1 = calculate_rouge(data_1, 'Comments', 'Docstring_generated')



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
data_1 = calculate_bleu(data_1, 'Comments', 'Docstring_generated')

#%%
data_1[['Comments', 'Code_without_comments', 'Docstring_generated',
       'ROUGE-1 Comments', 'BLEU Score Comments']].to_csv("sdfsdf.csv")
