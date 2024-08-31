from utils.portfolio.portfolio import Portfolio
from utils.stock.stock import Stock
from utils.federal_fund_rate.federal_fund_rate import FederalFundRate
import time

class HealthcarePortfolio:
    def __init__(self,federal_fund_rate:bool=True):
        super().__init__("healthcare_portfolio",federal_fund_rate)
        stock_config = [
            {"name":"Pfizer", "ticker":"PFE"},
            {"name":"Johnson & Johnson", "ticker":"JNJ"},
            {"name":"Merck & Co", "ticker":"MRK"},
            {"name":"Eli Lilly & Co", "ticker":"LLY"},
            {"name":"AbbVie Inc", "ticker":"ABBV"},
        ]
        for i in stock_config:
            stock = Stock(i['name'],i['ticker'])
            self.register(stock)