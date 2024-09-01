"""
Response comes from API looking like this:

{
    "Meta Data": {
        "1. Information": "Daily Prices (open, high, low, close) and Volumes",
        "2. Symbol": "IBM",
        "3. Last Refreshed": "2024-08-30",
        "4. Output Size": "Full size",
        "5. Time Zone": "US/Eastern"
    },
    "Time Series (Daily)": {
            "2024-08-30": {
                "1. open": "199.1100",
                "2. high": "202.1700",
                "3. low": "198.7300",
                "4. close": "202.1300",
                "5. volume": "4750999"
            }
        }
"""

import requests
import time
import datetime
import os
from dotenv import load_dotenv

from utils.api import fetch_data

env_path = "../.env"
load_dotenv(env_path)
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
base_url = "https://www.alphavantage.co/query"



class Stock:
    """A stock is an instantiation of a stock data puller/setter.

    Has a min_date/max_date property for its price range.

    You can pull data from the API fresh with get().

    You can set information you've pulled elswhere with set_data().

    The data shape is:
    [
        ...
            {
                year:integer,
                month:integer,
                day:integer,
                open:float, 
                high:float, 
                low:float,
                close:float,
                volume:integer
            }
    ]
    
    """
    def __init__(self,name,ticker):
        self.name = name
        self.ticker = ticker
        self.data = []
        self.max_date:datetime.datetime|None = None
        self.min_date:datetime.datetime|None = None

            
    def get(self):
        """
        Returns the full data pulled from the API. Sets to `self.data`

        Resets `self.data` if there was information set there.

        Data Shape:
        [
        ...
            {
                open:float, 
                high:float, 
                low:float,
                close:float,
                volume:integer
            }
        ]
        """
        self.data = []
        print(f"Pulling for {self.ticker}...")
        # Build URL
        api_url = f"{base_url}?function=TIME_SERIES_DAILY&symbol={self.ticker}&outputsize=full&apikey={api_key}"
        gathered = False
        response_data = fetch_data(api_url)
        if response_data is None:
            print(f"Failed to pull data for {self.ticker}")
            return None
        print(f"Parsing data for {self.ticker}...")
        # Parse data
        daily_data = response_data['Time Series (Daily)']
        stringdates = list(daily_data.keys())
        interval = 0
        total_intervals = len(stringdates)
        for stringdate in stringdates:
            interval+=1
            print(f"Processing {self.name} ({self.ticker}) - {interval}/{total_intervals}", end="\r")
            try:
                # Parse important info
                date_formatted = datetime.datetime.strptime(stringdate, "%Y-%m-%d")
                date_info = daily_data[stringdate]
                cleaned_data = {
                    "year":date_formatted.year,
                    "month":date_formatted.month,
                    "day":date_formatted.day,
                    "open":float(date_info['1. open']),
                    "high":float(date_info['2. high']),
                    "low":float(date_info['3. low']),
                    "close":float(date_info['4. close']),
                    "volume":int(date_info['5. volume'])
                }
                # Manage stock min/max
                if self.max_date is None or self.min_date is None:
                    self.max_date = date_formatted
                    self.min_date = date_formatted
                if date_formatted > self.max_date:
                    self.max_date = date_formatted
                if date_formatted < self.min_date:
                    self.min_date = date_formatted
                # Add to class
                self.data.append(cleaned_data)
            except Exception as e:
                print(f"ERROR - failed parsing data {interval}/{total_intervals}")
                print(e)
        print(f"\nData pull for {self.ticker} complete.")
    
    def set_data(self, data:list):
        """
        A function to set data, in the event that its been loaded elsewhere.

        Expected data shape:
        [
        ...
            {
                open:float, 
                high:float, 
                low:float,
                close:float,
                volume:integer
            }
        ]
        """
        self.data=data
    