# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:56:19 2025

@author: ADMIN
"""

from bert_score import score
import json

# import required module
import os
import re

import zlib
import sys

import pickle

import itertools 


#%%
# Load the DataFrame from the file
with open('class_files_df.pkl', 'rb') as f:
    class_files_df = pickle.load(f)

ground_truth = class_files_df["Comments"].to_list()


# Save the DataFrame to a file
with open('all_scoiring.pkl', 'rb') as f:
    scoring= pickle.load(f)


#%%

# Calculate BERT encoding score, using cosine similarity
def calculate_bert_score(ground_truth, generated):
    # Calculate BERT score
    _, _, bert_score_f1 = score([ground_truth], [generated], lang='en', model_type='bert-base-uncased')

    return bert_score_f1.item()    
    
#%%

# Calculate number of syllables in docstring
def count_syllables(word):
    # Remove punctuation
    word = re.sub(r'[^a-zA-Z]', '', word)
    
    # Vowel count
    vowels = 'aeiouy'
    syllables = 0
    last_was_vowel = False
    for char in word:
        if char.lower() in vowels:
            if not last_was_vowel:
                syllables += 1
            last_was_vowel = True
        else:
            last_was_vowel = False
    
    # Adjust syllable count for words ending in 'e'
    if word.endswith(('e', 'es', 'ed')):
        syllables -= 1
    
    # Adjust syllable count for words with no vowels
    if syllables == 0:
        syllables = 1
    
    return syllables

# Calculate Flesch reading score
def flesch_reading_ease(text):
    sentences = text.count('.') + text.count('!') + text.count('?') + 1
    words = len(re.findall(r'\b\w+\b', text))
    syllables = sum(count_syllables(word) for word in text.split())
    
    # Calculate Flesch Reading Ease score
    score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    
    return score


#%%
def compress(input):
	return zlib.compress(input.encode())


def conciness(ground_truth, generated):
    comp1 = compress(ground_truth)
    comp2 = compress(generated)
    return sys.getsizeof(comp2) / sys.getsizeof(comp1)


#%%
# assign directory
directory = 'Output'

#scoring = {}
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        if "_NEW" in f:
            print(f)
            with open(f, "r") as fp:
                generated = json.load(fp)
                
                list_append = []
                which_score = {}
                
                for (ground_truth_, generated_) in itertools.zip_longest(ground_truth, generated):
                    list_append.append(conciness(ground_truth_, generated_))
                which_score["Conciness_inv"] = list_append
                scoring[f] = which_score

                list_append = []
                
                for (ground_truth_, generated_) in itertools.zip_longest(ground_truth, generated):
                    list_append.append(flesch_reading_ease(generated_))
                scoring[f]["Ease"] = list_append
    
                list_append = []
                
                for (ground_truth_, generated_) in itertools.zip_longest(ground_truth, generated):
                    list_append.append(calculate_bert_score(ground_truth_, generated_))
                scoring[f]["Accuracy"] = list_append

#%%


# Save the DataFrame to a file
with open('all_scoiring.pkl', 'wb') as f:
    pickle.dump(scoring, f)

#%%