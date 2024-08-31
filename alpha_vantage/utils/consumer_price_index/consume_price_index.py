import requests
import time
import datetime
import os
from dotenv import load_dotenv
from utils.api import fetch_data

"""

WARNING!

THIS DATA IS MONTHLY!

"""

"""
Data from API looks like:

{
    "name": "Consumer Price Index for all Urban Consumers",
    "interval": "monthly",
    "unit": "index 1982-1984=100",
    "data": [
        {
            "date": "2024-07-01",
            "value": "314.540"
        },
        ]
}
"""

env_path = "../.env"
load_dotenv(env_path)
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
base_url = "https://www.alphavantage.co/query"

class ConsumerPriceIndex:
    """
    A collection of the consumer price index data.

    You can pull data from the API fresh with get().

    You can set information you've pulled elsewhere with set_data().
    """

    def __init__(self):
        self.data = []
        self.name = "consumer_price_index"

    def get(self):
        """
        Returns the full data pulled from the API. Sets to `self.data`

        Resets `self.data` if there was information set there.

        Data Shape:
        [
        ...
            {
                year:integer,
                month:integer,
                day:integer,
                value:float, 
            }
        ]
        """
        self.data=[]
        api_url=f"{base_url}?function=CPI&interval=daily&apikey={api_key}"
        response_data = fetch_data(api_url)
        if response_data is None:
            print(f"Failed to pull data for {self.name}")
            return None
        print(f"Parsing data for {self.name}...")
        daily_data = response_data['data']
        interval = 0
        total_intervals = len(daily_data)
        for item in daily_data:
            interval+=1
            print(f"Processing {self.name} - {interval}/{total_intervals}", end="\r")
            date_formatted = datetime.datetime.strptime(item['date'], "%Y-%m-%d")
            value = item['value']
            cleaned_data = {
                "year":date_formatted.year,
                "month":date_formatted.month,
                "day": date_formatted.day,
                "value":float(value)
            }
            self.data.append(cleaned_data)
        print("Completed consumer price index pull.")

    def set_data(self,data):
        """
        A function to set the consumer price index data.

        Expected data shape:
        [
        ...
            {
                year:integer,
                month:integer,
                day:integer,
                value:float, 
            }
        ]
        """
        self.data = data