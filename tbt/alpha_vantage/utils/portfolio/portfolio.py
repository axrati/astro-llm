

from tbt.alpha_vantage.utils.stock.stock import Stock
from tbt.alpha_vantage.utils.federal_fund_rate.federal_fund_rate import FederalFundRate
import time
from typing import Dict, Literal, TypedDict, Any
from tbt.alpha_vantage.utils.data_linting import prepare_data

# Define the allowed value types
allowed_key_types = Literal["float", "int", "date", "string", "category", "boolean"]

class ModelData(TypedDict):
    source:list[Dict[str,Any]]
    target:list[Dict[str,Any]]

class Portfolio:
    """A Portfolio is a container for Stocks.

These are grouped together to build datasets that can be generated.

You can use the model prepared data at self.model_data.

Its recommended that you .initialize() before .prepare_data().

Portfolios can contain:
1) Federal Funds Rate (federal_fund_rate)- This is a dataset of the federal funds daily.


    """
    def __init__(self,name,federal:bool=True):
        self.name:str = name
        self.stocks:list[Stock] = []
        self.stocknames:list[str]=[]
        self.federal_fund_rate:FederalFundRate|None = None
        self.model_data:ModelData = {"source":[], "target":[]}
        self.model_keys:list[Dict[str, allowed_key_types]] = []
        self.key_map = {
            "open":{"type":"float"},
            "high":{"type":"float"},
            "low":{"type":"float"},
            "close":{"type":"float"},
            "volume":{"type":"float"}
        }
        if federal:
            self.federal_fund_rate = FederalFundRate()

    def register(self,stock:Stock):
        """
        Adds a stock to the portfolio.
        """
        if stock.name not in self.stocks and stock.name != "federal_fund_rate":
            self.stocknames.append(stock.name)
            self.stocks.append(stock)
        else:
            print("Couldnt add stock. It either exists or is using reserved name.")
            print("Current reserved names: federal_fund_rate")

    def remove(self,name):
        """
        Removes a stock from the portfolio.
        """
        stocks = []
        stocknames = []
        for i in self.stocks:
            if i.name != name:
                stocks.append(i)
                stocknames.append(i.name)
        self.stocks=stocks
        self.stocknames=stocknames

    def initialize(self):
        """
        Pulls the data for each stock from the API. Also does Federal Fund Rate if applicable.
        """
        print("Initalizing portfolio data pull...")
        interval = 0
        max_intervals = len(self.stocks)
        for stock in self.stocks:
            interval+=1
            print(f"Pulling stock {interval}/{max_intervals} ({stock.ticker})...", end="\r")
            time.sleep(1)
            stock.get()
        if self.federal_fund_rate is not None:
            print(f"Pulling federal fund rate...")
            time.sleep(1)
            self.federal_fund_rate.get()
        print("\nInitalization complete for portfolio.")

    def generate(self):
        """
        This sets the portfolio's `model_data` and `model_keys` information.

        `model_data` = {"source":[], "target":[]} formation
        `model_keys` = [...,"date"] which has each row's keyable info
        """
        print("Starting generation...")
        if self.federal_fund_rate:
            prepared_data = prepare_data(self.stocks, {"federal_fund_rate":{"data":self.federal_fund_rate.data, "keys":["value"], "item":self.federal_fund_rate}})
            self.model_data = prepared_data['data']
            self.model_keys = prepared_data['model_keys']
        else:
            prepared_data = prepare_data(self.stocks)
            self.model_data = prepared_data['data']
            self.model_keys = prepared_data['model_keys']
        print("Generation complete")
    