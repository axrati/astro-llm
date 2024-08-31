from utils.portfolio.portfolio import Portfolio
from utils.stock.stock import Stock
from utils.federal_fund_rate.federal_fund_rate import FederalFundRate
import time

class HealthInsurancePortfolio(Portfolio):
    def __init__(self,federal_fund_rate:bool=True):
        super().__init__("health_insurance_portfolio",federal_fund_rate)
        stock_config = [
            {"name":"UnitedHealthGroup", "ticker":"UNG"},
            {"name":"Elevance", "ticker":"ELV"},
            {"name":"Humana", "ticker":"HUM"},
            {"name":"Cigna", "ticker":"CI"},
            {"name":"CVS", "ticker":"CVS"},
        ]
        for i in stock_config:
            stock = Stock(i['name'],i['ticker'])
            self.register(stock)
