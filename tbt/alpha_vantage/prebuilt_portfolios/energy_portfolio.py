from tbt.alpha_vantage.utils.portfolio.portfolio import Portfolio
from tbt.alpha_vantage.utils.stock.stock import Stock
from tbt.alpha_vantage.utils.federal_fund_rate.federal_fund_rate import FederalFundRate
import time

class EnergyPortfolio(Portfolio):
    def __init__(self,federal_fund_rate:bool=True):
        super().__init__("energy_portfolio", federal_fund_rate)
        self.stock_config = [
            {"name":"Exxon", "ticker":"XOM"},
            {"name":"Chevron", "ticker":"CVX"},
            {"name":"Schlumberger", "ticker":"SLB"},
            {"name":"ConocoPhillips", "ticker":"COP"},
            {"name":"Baker Hughes", "ticker":"BKR"},
        ]
        for i in self.stock_config:
            stock = Stock(i['name'],i['ticker'])
            self.register(stock)