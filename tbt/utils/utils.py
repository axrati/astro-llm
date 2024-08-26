import datetime

def stringdate(val, pattern):
    # Extract the year, month, and day from the dictionary
    year = val['year']
    month = val['month']
    day = val['day']
    
    # Handle the negative year manually
    if year < 0:
        year_str = f"-{-year:04d}"  # Ensure the year is in 4 digits and handle the negative sign
    else:
        year_str = f"{year:04d}"

    # Create a mapping from pattern to actual values
    replacements = {
        "%Y": year_str,                      # Year with century as a decimal number
        "%m": f"{month:02d}",                # Zero-padded month
        "%d": f"{day:02d}",                  # Zero-padded day of the month
        "%y": year_str[-2:],                 # Last two digits of the year
        "%b": datetime.date(1900, month, 1).strftime('%b'),  # Abbreviated month name
        "%B": datetime.date(1900, month, 1).strftime('%B'),  # Full month name
        "%a": datetime.date(1900, month, day).strftime('%a'), # Abbreviated weekday name
        "%A": datetime.date(1900, month, day).strftime('%A'), # Full weekday name
    }

    # Replace the pattern placeholders with actual values
    for placeholder, value in replacements.items():
        pattern = pattern.replace(placeholder, value)
    return pattern