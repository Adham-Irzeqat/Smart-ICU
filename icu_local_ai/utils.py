# utils.py
from datetime import datetime

def datetime_to_str(obj):
    """Convert datetime objects to ISO format string for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")
