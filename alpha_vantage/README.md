# Notes

### Important links

[Official API Documentation](https://www.alphavantage.co/documentation/)

### Implementation.

Sample code:

```python
from alpha_vantage.prebuilt_portfolios.healthcare_portfolio import HealthcarePortfolio

hc_portfolio = HealthcarePortfolio() # Optional - federal_fund_rate:bool
hc_portfolio.initialize()
hc_portfolio.prepare_data()

hc_portfolio.model_data # {"source":[...], "target":[...]}

```
