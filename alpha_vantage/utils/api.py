import requests
import time

def fetch_data(url, retries=10):
    """
    A safe API call that will retry X times.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()

        if "Error Message" in data:
            print(f"API Error: {data['Error Message']}")
            return None
        elif "Note" in data:
            print(f"API Limit or Other Notice: {data['Note']}")
            return None
        else:
            return data  # Return the valid data

    except (requests.exceptions.Timeout, requests.exceptions.RequestException, ValueError) as e:
        if retries > 0:
            print(f"Error: {e}. Retrying in 1 second...")
            time.sleep(1)
            return self.fetch_data(url, retries - 1)
        else:
            print("Max retries reached. Failed to retrieve data.")
            return None