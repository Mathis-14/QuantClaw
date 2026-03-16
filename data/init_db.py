"""
Initialize SQLite database for option chains.
"""

import sqlite3
import os


def init_db():
    """Initialize the SQLite database."""
    db_path = os.path.join(os.path.dirname(__file__), "option_chains.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS option_chains (
        ticker TEXT NOT NULL,
        expiration TEXT NOT NULL,
        strike REAL NOT NULL,
        option_type TEXT NOT NULL,  -- 'call' or 'put'
        last_price REAL,
        implied_volatility REAL,
        PRIMARY KEY (ticker, expiration, strike, option_type)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS forward_prices (
        ticker TEXT PRIMARY KEY,
        forward_price REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")


if __name__ == "__main__":
    init_db()