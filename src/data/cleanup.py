# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:28:41 2024

@author: ADMIN
"""


import json
import pandas as pd
import requests
import re

#%%
import pickle

# Load the DataFrame from the file
with open('class_files_df.pkl', 'rb') as f:
    class_files_df = pickle.load(f)

#%%

import re

def replace_def_statements(code_string):
    def_count = 0
    def replacement(match):
        nonlocal def_count
        def_count += 1
        func_name = match.group(1)
        args = match.group(2)
        return f"def dummy_def_{def_count}({args})"
    
    class_counter = 0
    def replacement_class(match):
        nonlocal class_counter
        class_counter += 1
        func_name = match.group(1)
        args = match.group(2)
        return f"class dummy_class_{class_counter}({args})"

    code_string = re.sub(r"def\s+(\w+)\s*\(([^)]*)\)", replacement, code_string)
    
    code_string = re.sub(r"class\s+(\w+)\s*\(([^)]*)\)", replacement_class, code_string)
    
    
    code_string = re.sub(r'""".*?"""', '', code_string, flags=re.DOTALL)
    code_string = re.sub(r"'''[^']*?'''", '', code_string, flags=re.DOTALL)
    
    code_string = re.sub(r'#.*\n', '', code_string)
    return code_string


#%%
clean_code = []

#%%
for i in range(0, len(class_files_df)):
    user_prompt = class_files_df["Code_without_comments"].iloc[i]
    print(i)
    response = replace_def_statements(code_string = user_prompt)
    clean_code.append(response)
    
#%%
class_files_df["Clean_classes"] = clean_code

#%%

with open('class_files_df.pkl', 'wb') as f:
    pickle.dump(class_files_df, f)
