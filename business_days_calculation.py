# simple_business_days.py
import holidays
from dateutil.parser import parse
from typing import Union, Optional, Set
from datetime import import datetime, timedelta
import logging

# Configurable constants
DEFAULT_COUNTRY = 'AU'
HOURS_PER_DAY = 24.0
SECONDS_PER_HOUR = 3600
MINIMUM_TIME = 0.0
BUSINESS_WEEKDAYS = {0, 1, 2, 3, 4}  # Monday=0 to Friday=4 (configurable for different cultures)

# Module-level holiday cache for performance
_holiday_cache: Optional[holidays.HolidayBase] = None


def _get_holidays(country: str = DEFAULT_COUNTRY) -> holidays.HolidayBase:
    """Get holidays instance with caching for performance."""
    global _holiday_cache
    if _holiday_cache is None:
        _holiday_cache = holidays.country_holidays(country)
    return _holiday_cache


def _normalize_date(date_input: Union[str, datetime]) -> datetime:
    """
    Normalize date input to datetime object.
    
    Args:
        date_input: Date as string or datetime object
    
    Returns:
        Normalized datetime object
    
    Raises:
        ValueError: If date cannot be parsed
    """
    if isinstance(date_input, str):
        return parse(date_input, ignoretz=True)
    elif isinstance(date_input, datetime):
        return date_input
    else:
        raise ValueError(f"Unsupported date type: {type(date_input)}")


def _is_business_day(date: datetime, holidays_set: holidays.HolidayBase,
                     business_weekdays: Set[int] = BUSINESS_WEEKDAYS) -> bool:
    """Check if a date is a business day (configurable weekdays and not holiday)."""
    return date.weekday() in business_weekdays and date.date() not in holidays_set


def calculate_business_days_fractional(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    country: str = DEFAULT_COUNTRY,
    business_weekdays: Set[int] = BUSINESS_WEEKDAYS,
    hours_per_day: float = HOURS_PER_DAY
) -> float:
    """
    Calculate fractional business days based on calendar time elapsed during business days only.
    
    This excludes weekends and holidays, then converts elapsed time to fractional days.
    
    Args:
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime)
        country: Country code for holidays (default: 'AU')
        business_weekdays: Set of weekday numbers that are business days (default: {0,1,2,3,4} = Mon-Fri)
        hours_per_day: Hours in a day for calculation (default: 24.0)
    
    Returns:
        Number of business days as float (e.g., 0.083 = 2 hours, 0.75 = 18 hours)
    
    Example:
        2 hours elapsed on business days = 2/24 = 0.083 business days
        18 hours elapsed on business days = 18/24 = 0.75 business days
    
    Raises:
        ValueError: If dates cannot be parsed or end_date < start_date
    """
    try:
        start_dt = _normalize_date(start_date)
        end_dt = _normalize_date(end_date)
        
        # Validate date order
        if end_dt < start_dt:
            raise ValueError(f"End date {end_dt.date()} cannot be before start date {start_dt.date()}")
        
        # Get holidays for filtering
        holiday_dates = _get_holidays(country)
        
        # Handle same day scenario
        if start_dt.date() == end_dt.date():
            if _is_business_day(start_dt, holiday_dates, business_weekdays):
                elapsed_hours = (end_dt - start_dt).total_seconds() / SECONDS_PER_HOUR
                return elapsed_hours / hours_per_day
            else:
                # Started and ended on non-business day
                return MINIMUM_TIME
        
        # Multi-day scenario: calculate elapsed time on each business day
        total_business_hours = MINIMUM_TIME
        current_date = start_dt
        
        while current_date.date() <= end_dt.date():
            if _is_business_day(current_date, holiday_dates, business_weekdays):
                
                if current_date.date() == start_dt.date():
                    # First day: from start time to end of day
                    day_end = datetime.combine(current_date.date(), datetime.max.time()).replace(microsecond=0))
                    elapsed_hours = (day_end - start_dt).total_seconds() / SECONDS_PER_HOUR
                elif current_date.date() == end_dt.date():
                    # Last day: from start of day to end time
                    day_start = datetime.combine(current_date.date(), datetime.min.time())
                    elapsed_hours = (end_dt - day_start).total_seconds() / SECONDS_PER_HOUR
                else:
                    # Full day between start and end
                    elapsed_hours = hours_per_day
                
                total_business_hours += elapsed_hours
            
            current_date += timedelta(days=1)
        
        # Convert total business hours to fractional business days
        return total_business_hours / hours_per_day
    
    except (ValueError, TypeError) as e:
        logging.error(f"Error calculating fractional business days between {start_date} and {end_date}: {e}")
        raise


def calculate_duration(start_date: str, end_date: str, **kwargs) -> float:
    """
    Calculate duration in fractional business days for executive reporting.
    
    This function maintains compatibility with existing data_processor.py
    while providing fractional business days calculation based on calendar time.
    
    Args:
        start_date: Start date as string
        end_date: End date as string
        **kwargs: Additional arguments passed to calculate_business_days_fractional
                  (country, business_weekdays, hours_per_day)
    
    Returns:
        Number of business days as float (e.g., 0.083, 0.75, 1.25)
    """
    return calculate_business_days_fractional(start_date, end_date, **kwargs)