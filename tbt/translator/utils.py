import re

def build_regex_from_pattern(pattern: str) -> str:
    """
    Converts a date pattern into a regex pattern with named groups.

    Args:
        pattern (str): The date pattern (e.g., "%Y-%d-%m").

    Returns:
        str: A regex pattern with named groups for year, month, and day.
    """
    # Mapping of strftime directives to regex patterns
    directive_mapping = {
        '%Y': r'(?P<year>-?\d{4})',  # Four-digit year, allowing negative
        '%y': r'(?P<year>-?\d{2})',  # Two-digit year, allowing negative
        '%m': r'(?P<month>\d{1,2})',  # One or two-digit month
        '%d': r'(?P<day>\d{1,2})',    # One or two-digit day
        '%b': r'(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
        '%B': r'(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)',
    }

    regex = pattern
    # Escape regex special characters except for % directives
    regex = re.escape(regex)
    # Replace escaped % directives with actual regex patterns
    for directive, regex_part in directive_mapping.items():
        regex = regex.replace(re.escape(directive), regex_part)

    return f"^{regex}$"

def get_year_date_month(val:str, date_pattern:str):
    regex_pattern = build_regex_from_pattern(date_pattern)
    match = re.match(regex_pattern, val)
    if not match:
        raise ValueError(f"Date string '{val}' does not match the date pattern '{date_pattern}'.")
    year = int(match.group('year'))
    month = int(match.group('month'))
    day = int(match.group('day'))
    return {"year":year, "month":month, "day":day}