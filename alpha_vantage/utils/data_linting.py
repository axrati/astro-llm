"""

The purpose of this file is to lint stock data (and potentially other Portfolio data)

There needs to be a consistent start/end date, as well as # of records.

This will be interactive, requiring user input.

Stocks will be an array of the stocks. ie:  [...,Stock]

Metadata will be a dict of additional daily attibutes. ie:  {..., "federa_fund_rate":FederalFundRate}

"""

from utils.stock.stock import Stock
import datetime
from typing import Dict, Literal, TypedDict, Any, List
from utils.user_input import get_boolean_from_user
import traceback

class AssetData(TypedDict):
    data: List[Dict[str, Any]]  
    keys: List[str]             
    item: Any

class MetadataObject(TypedDict):
    Dict[str,AssetData]

def prepare_data(stocks:list[Stock],metadata:MetadataObject={}):
    """
    A function to organize the data into "source" and "target" form.

    Metadata needs to look like this:
    {"key_of_asset":{"data":data, "keys":[...,"key_in_data"], "item":Item}}

    This is into ensure that the data can be parsed for adding to the dataset.

    Returns: {"data":{"source":[...], "target":[...]}. "model_keys":[...,""]

    """
    # Wager what can be done
    total_key_map = {}
    true_min:datetime.datetime|None = None
    true_max:datetime.datetime|None = None
    for i in stocks:
        # Slim down "true_date"s to the smallest possible range all items share.
        # Commenting cause its awkward of < and > assignment
        if true_min is None or true_max is None:
            true_min = i.min_date
            true_max = i.max_date
        if i.max_date < true_max:
            true_max = i.max_date
        if i.min_date > true_min:
            true_min = i.min_date
        total_key_map[i.name]={"max":i.max_date, "min":i.min_date, "data":i.data, "item":i}
    # Repeat for the metadata hashmap
    for key in list(metadata.keys()):
        record = metadata[key]['item']
        if true_min is None or true_max is None:
            true_min = record.min_date
            true_max = record.max_date
        if record.max_date < true_max:
            true_max = record.max_date
        if record.min_date > true_min:
            true_min = record.min_date
        total_key_map[key]={"max":record.max_date, "min":record.min_date, "data":record.data, "item":record}
    
    # Mine for violators of this, ask if the user would like to parse them and proceed.
    violators = []
    for key in list(total_key_map.keys()):
        record = total_key_map[key]
        max_violation = False
        max_violation_amt = 0
        min_violation = False
        min_violation_amt = 0
        # Hoist violation for quick access
        absolute_violation = False
        if record['max']>true_max:
            max_violation = True
            max_violation_amt = (record['max']-true_max).days
            absolute_violation = True
        if record['min']<true_min:
            min_violation = True
            min_violation_amt = (record['min']-true_min).days
            absolute_violation = True
        violators.append({
            "keyname":key,
            "max_violation":max_violation,
            "max_violation_amt":max_violation_amt,
            "min_violation":min_violation,
            "min_violation_amt":min_violation_amt,
            "absolute_violation":absolute_violation
        })
    violation_readout = [i['absolute_violation'] for i in violators]
    if True in violation_readout:
        print(f"""WARNING --

The data pulled for this information only has the following range overlap when viewing min/max of your dataset.

---------              
CROSS_DATASET_ALLOWED_MIN: {true_min}
CROSS_DATASET_ALLOWED_MAX: {true_max}
TOTAL_DAYS: {(true_max-true_min).days}
---------
""")
        for key in list(total_key_map.keys()):
            record = total_key_map[key]
            print(f"""
{key}
MIN: {record['min']}
MAX: {record['max']}
RECORDS_DROPPED = {abs((true_max-true_min).days-len(record['data']))}
""")        
        print("""
              
Above are the stocks/assets provided. 
Would you like to trim these datasets to be equal? (y|n)""")
        user_response = get_boolean_from_user()
        if not user_response:
            print("Exiting data preparation...")
            return None
        # For each dataset, trim it to any row that has the required date.
        cleaned_key_map = {}
        for key in list(total_key_map.keys()):
            record = total_key_map[key]
            cleaned_record = {"max":true_max, "min":true_min, "item":record['item']}
            # Dates are used to hashmap for cross item stiching
            cleaned_row_data = {}
            for i in record['data']:
                parsed_date = datetime.datetime.strptime(f"{i['year']}-{i['month']}-{i['day']}", "%Y-%m-%d")
                if parsed_date >= true_min and parsed_date <= true_max:
                    cleaned_row_data[f"{i['year']}-{i['month']}-{i['day']}"]=i
            cleaned_record['data']=cleaned_row_data
            cleaned_key_map[key]=cleaned_record

        # Now that you have the clean dataset, loop through dates and spread the data into a common row item
        total_viable_days = abs(int((true_max-true_min).days))
        total_day_data = []
        total_model_keys = ["date"]
        for numdays in range(total_viable_days):
            try:
                # Days are returned from API in DESC order, so use max and go down  
                next_date = true_max + datetime.timedelta(days=numdays*-1)
                next_date_key = f"{next_date.year}-{next_date.month}-{next_date.day}"
                complete_row = {}
                for key in list(cleaned_key_map.keys()):
                        # Handle metadata differently
                        if key in list(metadata.keys()):
                            # Parse based on keys in data
                            parseable_keys = metadata[key]["keys"]
                            metadata_data = cleaned_key_map[key]['data']
                            for metakey in parseable_keys:
                                row_info = metadata_data[next_date_key][metakey]
                                colname = f"{key}_{metakey}"
                                complete_row[colname]=row_info
                                if colname not in total_model_keys:
                                    total_model_keys.append(colname)
                        else:
                            stock_data = cleaned_key_map[key]['data'][next_date_key]
                            stock_keys = ['open','high','low','close','volume']
                            for stock_key in stock_keys:
                                row_info = stock_data[stock_key]
                                colname = f"{key}_{stock_key}"
                                if colname not in total_model_keys:
                                    total_model_keys.append(colname)
                                complete_row[colname]=row_info
                complete_row['date']=next_date_key
                total_day_data = [complete_row]+total_day_data
            except Exception as e:
                # This is likely a weekend or holiday that the value wasnt reported
                continue

        # Now that you have the data in proper order, build source to target mapping.
        source = []
        target = []
        for i in range(len(total_day_data)):
            if i == len(total_day_data)-1:
                continue
            else:
                source.append(total_day_data[i])
                target.append(total_day_data[i+1])

        return {"data":{"source":source, "target":target}, "model_keys":total_model_keys}