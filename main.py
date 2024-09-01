from tbt.model.model import DataTransformerModel
from tbt.config.config import ModelConfig
from tbt.trainer.trainer import Trainer
from tbt.cli.cli import CLI
import json
import string
from tbt.alpha_vantage.prebuilt_portfolios.energy_portfolio import EnergyPortfolio
from tbt.alpha_vantage.prebuilt_portfolios.health_insurance_portfolio import HealthInsurancePortfolio

p = HealthInsurancePortfolio()
p.initialize()
p.generate()

# Determine config for model
config = ModelConfig()
config.date("date","%Y-%m-%d")
for key in list(p.key_map.keys()):
    keymap = p.key_map[key]
    for name in p.stocknames:
        value = f"{name}_{key}"
        if keymap['type']=="float":
            config.float(value,50)
    if p.federal_fund_rate:
        config.float("federal_fund_rate_value",10)


model = DataTransformerModel(
    config=config,
    d_model=64,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=64,
    dropout=.1,
    max_len=5000,
    output_scale=1.0
)

source = p.model_data['source'][0:5]
target = p.model_data['target'][0:5]


trainer = Trainer(model, config)
trainer.add_data(source=source, target=target)
trainer.train(epochs=1)

# output = model(source,target)
# predictions = model.decode_output(output)
# print(predictions['original'])


cli = CLI(model,trainer)
cli.start()
