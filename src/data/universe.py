"""The nine-ETF opportunity set prescribed by the assignment (Part I).

Kept in one place so the download script, the cleaning script, the
environment calibration and the report all agree on tickers and ordering.
Column order everywhere in the project follows TICKERS.
"""

# ticker -> (segment, exposure)
UNIVERSE = {
    "SPY": ("Equity", "U.S. large-cap stocks"),
    "EFA": ("Equity", "Developed markets ex-U.S. and Canada"),
    "EEM": ("Equity", "Emerging-market stocks"),
    "IEF": ("Fixed income", "7-10 year U.S. Treasuries"),
    "LQD": ("Fixed income", "USD investment-grade corporate bonds"),
    "TIP": ("Fixed income", "U.S. Treasury inflation-protected securities"),
    "GLD": ("Real assets", "Gold bullion"),
    "VNQ": ("Real assets", "U.S. equity REITs"),
    "DBC": ("Real assets", "Broad commodity futures"),
}

TICKERS = list(UNIVERSE)

# DBC has the latest inception in the universe (Feb 2006), and the assignment
# requires a common sample beginning no earlier than February 2006. We request
# from 2006-01-01 so the raw pull straddles DBC's first trade and the true
# common start date is discovered from the data rather than assumed.
REQUEST_START = "2006-01-01"

# Kenneth French daily factor file; RF is the only column we use.
FF_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

# The French daily file publishes with roughly a two-month lag, so it, not the
# ETF data, sets how far the sample can run. Re-download it (download_raw.py
# --mode extend) when a new month posts, then re-run clean.py and the sample
# grows on its own.
#
# A daily 1-month Treasury yield from FRED (DGS1MO) was evaluated as a way to
# remove the lag. Over the 5,152-day overlap, DGS1MO/252 cumulates to within
# 0.008 of the French series on a base of 0.351, about 4bp per year. It was not
# adopted: the team chose to keep one consistent, assignment-named source rather
# than splice or switch mid-project.

# Optional hard cap on the sample end. None means "as far as the data supports".
# Pin a date here (or pass --end) to freeze the calibration window while new
# data is being appended, so report numbers stay reproducible.
DEFAULT_SAMPLE_END = None
