# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:57:20 2024

@author: ADMIN
"""

import ast
import requests


def extract_classes(git_link):
    response = requests.get(git_link)
    code = response.text

    tree = ast.parse(code)

    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node)

    return classes

def write_class_to_file(class_node, file_path):
    with open(file_path, 'w') as file:
        file.write(ast.unparse(class_node))

#%%
import pandas as pd

#%%
data = pd.read_csv("class collection.csv")

#%%
for i in range(0, len(data)):
    link = data.iloc[i]["File path"]
    
    git_link = link
    classes = extract_classes(git_link)

    for i, class_node in enumerate(classes):
        class_name = class_node.name
        output_file_path = f'classes/{class_name}.py'
        write_class_to_file(class_node, output_file_path)
        print(f'Class {class_name} written to {output_file_path}')

