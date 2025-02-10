# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:43:48 2024

@author: ADMIN
"""

import os

#%%

# specify the folder path
folder_path = r"C:\Users\ADMIN\Documents\SNU\paper-2\classes"  #'/path/to/your/folder'


# Initialize an empty list to store the file contents
file_contents = []

# List the Python files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.py'):
        # Open the file and read its contents
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            contents = file.read()
            file_contents.append(contents)
    
#%%
import re

def extract_comments_and_code(code):
    # Regular expression pattern to match comments within triple quotes
    comment_pattern = r'""".*?"""'
    
    # Find all comments in the code
    comments = re.findall(comment_pattern, code, re.DOTALL)
    
    # Remove comments from the code
    code_without_comments = re.sub(comment_pattern, '', code, flags=re.DOTALL)
    
    return comments, code_without_comments.strip()

#%%
comments = []
code_without_comments = []

for py in file_contents:
    comments_, code_without_comments_ = extract_comments_and_code(py)
    
    comments.append(comments_[0])
    code_without_comments.append(code_without_comments_)
#%%
import pandas as pd

df = pd.DataFrame({'Full_code': file_contents, 'Comments': comments, 'Code_without_comments': code_without_comments})
print(df)


#%%
import pickle

# Save the DataFrame to a file
with open('class_files_df.pkl', 'wb') as f:
    pickle.dump(df, f)

# Load the DataFrame from the file
with open('class_files_df.pkl', 'rb') as f:
    class_files_df = pickle.load(f)

print(class_files_df)
