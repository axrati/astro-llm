"""

This is largely a test file for alpha_vantage elements like:

Portfolio
Stock
FederalFundRate
data_linter
All AlphaVantage API calls

"""


import json
from utils.stock.stock import Stock
from utils.portfolio.portfolio import Portfolio
from utils.federal_fund_rate.federal_fund_rate import FederalFundRate
# from prebuilt_portfolios.health_insurance_portfolio import HealthInsurancePortfolio
from prebuilt_portfolios.energy_portfolio import EnergyPortfolio

# base_portfolio = Portfolio("health_insurance")
# cigna_stock = Stock("Cigna","CI")
# cigna_stock.get()
# base_portfolio.register(cigna_stock)

# hi_portfolio = HealthInsurancePortfolio()
# hi_portfolio.initialize()

e_port = EnergyPortfolio()
e_port.initialize()
e_port.generate()

# f = open("test.json","w")
# f.write(json.dumps(e_port.model_data))
# f.close()