# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 10:35:45 2025

@author: ADMIN
"""

import pickle
import pandas as pd

import re

#%%

# Save the DataFrame to a file
with open('all_scoiring.pkl', 'rb') as f:
    scoring= pickle.load(f)
    
#%%

scoring_averages = {}

for key, value in scoring.items():
    scoring_averages[key] = {k:sum(v)/len(v) for k,v in value.items()}
    
#%%
x = pd.DataFrame(scoring_averages).T

#%%
# Load the DataFrame from the file
with open('class_files_df.pkl', 'rb') as f:
    class_files_df = pickle.load(f)

#%%

length_analysis = pd.DataFrame()

length = []

for prog in class_files_df["Code_without_comments"].tolist():
    length.append(len(prog.split('\n')))


fns = []

for prog in class_files_df["Code_without_comments"].tolist():
    words = prog.split()
    fns.append(words.count('def'))


length_analysis["Length"] = length
length_analysis["Functions"] = fns

#%%
class_files_df.iloc[0]["Clean_classes"]

#%%
for key, value in scoring.items():
    for k_, v_ in value.items():
        if k_ == "Conciness_inv":
            length_analysis[key] = v_

            