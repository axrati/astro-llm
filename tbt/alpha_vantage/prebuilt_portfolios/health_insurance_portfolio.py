from tbt.alpha_vantage.utils.portfolio.portfolio import Portfolio
from tbt.alpha_vantage.utils.stock.stock import Stock
from tbt.alpha_vantage.utils.federal_fund_rate.federal_fund_rate import FederalFundRate
import time

class HealthInsurancePortfolio(Portfolio):
    def __init__(self,federal_fund_rate:bool=True):
        super().__init__("health_insurance_portfolio",federal_fund_rate)
        self.stock_config = [
            {"name":"UnitedHealthGroup", "ticker":"UNG"},
            {"name":"Elevance", "ticker":"ELV"},
            {"name":"Humana", "ticker":"HUM"},
            {"name":"Cigna", "ticker":"CI"},
            {"name":"CVS", "ticker":"CVS"},
        ]
        for i in self.stock_config:
            stock = Stock(i['name'],i['ticker'])
            self.register(stock)
