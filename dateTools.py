from datetime import date, timedelta
import doctest
import pandas as pd
from datetime import date, datetime
from pandas.api.types import is_datetime64_any_dtype

def years_back_by_weekday(reference_date: date, n_years: int) -> date:
    """
    Return the date that is n_years back on the same weekday.

    >>> years_back_by_weekday(date(2024, 10, 15), 1)
    datetime.date(2023, 10, 17)
    >>> years_back_by_weekday(date(2025, 3, 1), 2)
    datetime.date(2023, 3, 4)
    """
    return reference_date - timedelta(weeks=52 * n_years)

print(years_back_by_weekday(date(2025, 3, 1), 2))

def previous_weeks(reference_date: date, weeks_back: int = 1, years_back: int = 0) -> tuple[date, date]:
    """
    The same weekday-aligned weeks_back weeks, years_back years earlier (0 = this year).

    >>> previous_weeks(date(2024, 10, 15))
    (datetime.date(2024, 10, 9), datetime.date(2024, 10, 15))
    >>> previous_weeks(date(2024, 10, 15), weeks_back=2)
    (datetime.date(2024, 10, 2), datetime.date(2024, 10, 15))
    >>> previous_weeks(date(2025, 3, 1), weeks_back=3, years_back=2)
    (datetime.date(2023, 2, 12), datetime.date(2023, 3, 4))
    """
    end = years_back_by_weekday(reference_date, years_back)
    start = end - timedelta(days=(6 + (7*max(0, weeks_back -1))))
    return start, end


def previous_weeks_years_back(reference_date: date, weeks_back: int = 1, years_back: int = 0) -> list[tuple[date, date]]:
    """
    The same weekday-aligned weeks_back weeks for years_back (0 = this year).

    >>> previous_weeks_years_back(date(2024, 10, 15))
    [(datetime.date(2024, 10, 9), datetime.date(2024, 10, 15))]
    >>> previous_weeks_years_back(date(2024, 10, 15), 2, 2)
    [(datetime.date(2024, 10, 2), datetime.date(2024, 10, 15)), (datetime.date(2023, 10, 4), datetime.date(2023, 10, 17)), (datetime.date(2022, 10, 5), datetime.date(2022, 10, 18))]
    """
    return [previous_weeks(reference_date, weeks_back, y) for y in range(0, years_back + 1)]


def next_weeks(reference_date: date, weeks_forward: int = 1, years_back: int = 1) -> tuple[date, date]:
    """
    The weeks_forward weeks starting the day after the reference_date for years_back (1 = last year).

    >>> next_weeks(date(2024, 10, 15))
    (datetime.date(2023, 10, 18), datetime.date(2023, 10, 24))
    >>> next_weeks(date(2024, 10, 15), weeks_forward=2)
    (datetime.date(2023, 10, 18), datetime.date(2023, 10, 31))
    """
    start = years_back_by_weekday(reference_date + timedelta(days=1), years_back)
    end = start + timedelta(days=6 + (max(0, weeks_forward - 1) * 7))
    return start, end


def next_weeks_years_back(reference_date: date, weeks_forward: int = 1, years_back: int = 1) -> list[tuple[date, date]]:
    """
    The weeks_forward weeks starting the day after the reference_date for years_back (1 = last year).

    >>> next_weeks_years_back(date(2024, 10, 15), weeks_forward=2, years_back=2)
    [(datetime.date(2023, 10, 18), datetime.date(2023, 10, 31)), (datetime.date(2022, 10, 19), datetime.date(2022, 11, 1))]
    """
    return [next_weeks(reference_date, weeks_forward=weeks_forward, years_back=y) for y in range(1, years_back + 1)]

def filter_in_ranges(df: pd.DataFrame, column: str, ranges: list[list]) -> pd.DataFrame:
    """
    Return rows where df[column] (compared at *date* precision) falls within any of the
    given [start, end] ranges. The DataFrame column may be converted to dates/strings,
    but the ranges are not converted (they are used and returned exactly as provided).

    Supported range bound types:
      - strings in 'YYYY-MM-DD' form
      - datetime.date objects

    The returned DataFrame includes 'matchStart' and 'matchEnd' columns containing the
    first matched range (exact original values).
    """
    s = df[column]

    # --- Detect how ranges are represented (without converting them) ---
    def bounds_are_str(rs):
        return all(isinstance(v, str) for lo, hi in rs for v in (lo, hi))

    def bounds_are_date(rs):
        return all(isinstance(v, date) and not isinstance(v, datetime)
                   for lo, hi in rs for v in (lo, hi))

    if bounds_are_str(ranges):
        # Compare as strings 'YYYY-MM-DD' on the column side
        # Convert the column to date first (if needed), then to the same string form
        if is_datetime64_any_dtype(s):
            s_cmp = s.dt.date.astype(str)
        else:
            # try to parse whatever is there to dates, then to 'YYYY-MM-DD'
            s_cmp = pd.to_datetime(s).dt.date.astype(str)
        # Build boolean matrix using string comparisons; ranges remain untouched
        masks = pd.concat(
            [pd.Series(s_cmp.between(lo, hi, inclusive="both"), index=df.index)
             for lo, hi in ranges],
            axis=1
        )

    elif bounds_are_date(ranges):
        # Compare as date objects on the column side
        if is_datetime64_any_dtype(s):
            s_cmp = s.dt.date
        else:
            # try to parse to dates (column conversion is allowed)
            s_cmp = pd.to_datetime(s).dt.date
        masks = pd.concat(
            [pd.Series(s_cmp.between(lo, hi, inclusive="both"), index=df.index)
             for lo, hi in ranges],
            axis=1
        )

    else:
        raise TypeError(
            "Unsupported range bound types. Provide ranges as either "
            "strings 'YYYY-MM-DD' or datetime.date objects. "
            "(Ranges are not converted by this function.)"
        )

    masks.columns = pd.RangeIndex(masks.shape[1])
    any_match = masks.any(axis=1)

    # Identify first matched range index per row (NA if no match)
    first_match_idx = masks.idxmax(axis=1).where(any_match, other=pd.NA)

    # Map the *original* (unconverted) bounds back into the result
    start_map = pd.Series([lo for lo, _ in ranges], index=masks.columns)
    end_map   = pd.Series([hi for _, hi in ranges], index=masks.columns)
    match_start = first_match_idx.map(start_map)
    match_end   = first_match_idx.map(end_map)

    out = df[any_match].copy()
    out["matchStart"] = match_start.loc[out.index].to_numpy()
    out["matchEnd"]   = match_end.loc[out.index].to_numpy()

    out["matchStart"] = pd.to_datetime(out["matchStart"])
    out["matchEnd"]   = pd.to_datetime(out["matchEnd"])
    return out