from tbt.alpha_vantage.utils.portfolio.portfolio import Portfolio
from tbt.alpha_vantage.utils.stock.stock import Stock
from tbt.alpha_vantage.utils.federal_fund_rate.federal_fund_rate import FederalFundRate
import time

class HealthcarePortfolio:
    def __init__(self,federal_fund_rate:bool=True):
        super().__init__("healthcare_portfolio",federal_fund_rate)
        self.stock_config = [
            {"name":"Pfizer", "ticker":"PFE"},
            {"name":"Johnson & Johnson", "ticker":"JNJ"},
            {"name":"Merck & Co", "ticker":"MRK"},
            {"name":"Eli Lilly & Co", "ticker":"LLY"},
            {"name":"AbbVie Inc", "ticker":"ABBV"},
        ]
        for i in self.stock_config:
            stock = Stock(i['name'],i['ticker'])
            self.register(stock)