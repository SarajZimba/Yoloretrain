import requests

# Replace 'YOUR_API_KEY' with your actual Roboflow API key
api_key = 'N7BgbRtie2bXDot8wul6'

# Set the Roboflow API URL for workspaces
url = "https://api.roboflow.com/workspaces"

# Send GET request with the API key
response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"})

# Check the response
if response.status_code == 200:
    # Print available workspaces
    workspaces = response.json()
    print(workspaces)
else:
    # Handle errors
    print(f"Error: {response.status_code}")
    print(response.text)
