import requests

import requests

def download_file_from_git(url, filename):
    try:
        # Modify the URL to access the raw file contents
        raw_url = url.replace("(link unavailable)", "(link unavailable)").replace("/blob/", "/")
        
        response = requests.get(raw_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully: {filename}")
    except requests.exceptions.RequestException as err:
        print(f"Error downloading file: {err}")
        
# Example usage
git_url = "https://raw.githubusercontent.com/rasbt/mlxtend/refs/heads/master/mlxtend/regressor/linear_regression.py"
filename = "file.py"
download_file_from_git(git_url, filename)