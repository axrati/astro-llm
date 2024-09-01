from tbt.alpha_vantage.utils.portfolio.portfolio import Portfolio
from tbt.alpha_vantage.utils.stock.stock import Stock
from tbt.alpha_vantage.utils.federal_fund_rate.federal_fund_rate import FederalFundRate
import time

class TechnologyPortfolio:
    def __init__(self,federal_fund_rate:bool=True):
        super().__init__("technology_portfolio",federal_fund_rate)
        self.stock_config = [
            {"name":"Apple", "ticker":"AAPL"},
            {"name":"Microsoft", "ticker":"MSFT"},
            {"name":"Google", "ticker":"GOOGL"},
            {"name":"Amazon", "ticker":"AMZN"},
            {"name":"NVIDIA", "ticker":"NVDA"},
        ]
        for i in self.stock_config:
            stock = Stock(i['name'],i['ticker'])
            self.register(stock)