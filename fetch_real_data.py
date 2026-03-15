"""
Fetch real option chains for multiple asset classes and cache in SQLite.
"""

from vol_surface.fetcher import OptionChainFetcher


def main():
    tickers = ["AAPL", "SPY", "GC=F"]
    fetcher = OptionChainFetcher()
    
    for ticker in tickers:
        print(f"Fetching option chains for {ticker}...")
        try:
            option_chain = fetcher.fetch(ticker)
            print(f"  Fetched {len(option_chain.calls)} calls and {len(option_chain.puts)} puts for {ticker}.")
        except Exception as e:
            print(f"  Failed to fetch {ticker}: {e}")


if __name__ == "__main__":
    main()