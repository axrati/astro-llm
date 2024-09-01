import requests
import time
import datetime
import os
from dotenv import load_dotenv
from tbt.alpha_vantage.utils.api import fetch_data

"""
Data from API looks like:

{
    "name": "Effective Federal Funds Rate",
    "interval": "daily",
    "unit": "percent",
    "data": [
        {
            "date": "2024-08-29",
            "value": "5.33"
        }]
}
"""

env_path = ".env"
load_dotenv(env_path)
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
base_url = "https://www.alphavantage.co/query"

class FederalFundRate:
    """
    A collection of the federal funds rate data.

    You can pull data from the API fresh with get().

    You can set information you've pulled elswhere with set_data().

    The data shape is:
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
    def __init__(self):
        self.data = []
        self.name = "federal_fund_rate"
        self.min_date:datetime.datetime|None = None
        self.max_date:datetime.datetime|None = None
    
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
        api_url=f"{base_url}?function=FEDERAL_FUNDS_RATE&interval=daily&apikey={api_key}"
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
            # Parse important info
            date_formatted = datetime.datetime.strptime(item['date'], "%Y-%m-%d")
            value = item['value']
            cleaned_data = {
                "year":date_formatted.year,
                "month":date_formatted.month,
                "day": date_formatted.day,
                "value":float(value)
            }
            # Manage min/max
            if self.max_date is None or self.min_date is None:
                self.max_date = date_formatted
                self.min_date = date_formatted
            if date_formatted > self.max_date:
                self.max_date = date_formatted
            if date_formatted < self.min_date:
                self.min_date = date_formatted
            # Add to base
            self.data.append(cleaned_data)
        print("Completed federal fund rate pull.")
    
    def set_data(self,data):
        """
        A function to set the fedreal fund rate data.

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