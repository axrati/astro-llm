from datetime import datetime

def stringdate(val, pattern):
    # Extract the year, month, and day from the dictionary
    year = val['year']
    month = val['month']
    day = val['day']
    
    # Handle the negative year manually by adjusting the formatting
    if year < 0:
        year_str = f"-{-year:04d}"  # Ensure the year is in 4 digits and handle the negative sign
    else:
        year_str = f"{year:04d}"

    # Replace the year placeholder in the pattern with the manually formatted year
    formatted_date = pattern.replace("%Y", year_str)

    # Use datetime to format the rest of the date components
    date = datetime(abs(year), month, day)
    formatted_date = date.strftime(formatted_date)
    
    return formatted_date