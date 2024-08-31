from utils.portfolio.portfolio import Portfolio
from utils.stock.stock import Stock
from utils.federal_fund_rate.federal_fund_rate import FederalFundRate
import time

class EnergyPortfolio(Portfolio):
    def __init__(self,federal_fund_rate:bool=True):
        super().__init__("energy_portfolio", federal_fund_rate)
        stock_config = [
            {"name":"Exxon", "ticker":"XOM"},
            {"name":"Chevron", "ticker":"CVX"},
            {"name":"Schlumberger", "ticker":"SLB"},
            {"name":"ConocoPhillips", "ticker":"COP"},
            {"name":"Baker Hughes", "ticker":"BKR"},
        ]
        for i in stock_config:
            stock = Stock(i['name'],i['ticker'])
            self.register(stock)