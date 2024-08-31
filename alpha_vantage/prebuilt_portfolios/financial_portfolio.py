from utils.portfolio.portfolio import Portfolio
from utils.stock.stock import Stock
from utils.federal_fund_rate.federal_fund_rate import FederalFundRate
import time

class FinancialPortfolio(Portfolio):
    def __init__(self,federal_fund_rate:bool=True):
        super().__init__("financial_portfolio",federal_fund_rate)
        stock_config = [
            {"name":"JPMorgan", "ticker":"JPM"},
            {"name":"Bank of America", "ticker":"BAC"},
            {"name":"Goldman Sachs", "ticker":"GS"},
            {"name":"Wells Fargo", "ticker":"WFC"},
            {"name":"Morgan Stanley", "ticker":"MS"},
        ]
        for i in stock_config:
            stock = Stock(i['name'],i['ticker'])
            self.register(stock)