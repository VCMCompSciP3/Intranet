
import sqlite3
from pathlib import Path

DB_PATH = Path(r"\\192.168.1.95\CS-CLASSROOM\intraday_5m.sqlite")
DB_URI = "file:////192.168.1.95/CS-CLASSROOM/intraday_5m.sqlite?mode=ro&immutable=1"

# Customize these after examining which stocks and dates are available.
STOCK_TO_EXPLORE = "AAPL"
START_DATE = "2025-01-01"
END_DATE = "2025-01-10"
PREVIEW_ROWS = 10


def show_rows(title, columns, rows):
    print(f"\n=== {title} ===")
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(str(value) for value in row))


def query(conn, title, sql, params=()):
    cursor = conn.execute(sql, params)
    columns = [item[0] for item in cursor.description]
    rows = cursor.fetchall()
    show_rows(title, columns, rows)
    return rows


def main():
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot see {DB_PATH}. Connect to the NAS using the CS-Students "
            "account in PowerShell first."
        )

    # This database is a finalized historical snapshot. Never modify it.
    with sqlite3.connect(DB_URI, uri=True) as conn:
        conn.execute("PRAGMA query_only = ON")

        query(conn, "Database tables", """
            SELECT name FROM sqlite_master
            WHERE type = 'table' ORDER BY name
        """)

        query(conn, "bars_5m column names and types", "PRAGMA table_info(bars_5m)")

        # 1. What stocks are in the database?
        query(conn, "Available stock symbols", """
            SELECT DISTINCT symbol
            FROM bars_5m
            ORDER BY symbol
        """)

        # 2. How much historical data does each stock have?
        # This scans a large table and may take time over the network.
        query(conn, "History available by stock", """
            SELECT symbol,
                   COUNT(*) AS bar_count,
                   MIN(ts_utc) AS first_bar,
                   MAX(ts_utc) AS last_bar
            FROM bars_5m
            GROUP BY symbol
            ORDER BY bar_count DESC
        """)

        # 3. Look at five-minute prices for a stock YOU choose.
        query(conn, f"Recent five-minute bars for {STOCK_TO_EXPLORE}", """
            SELECT symbol, ts_utc, open, high, low, close, volume
            FROM bars_5m
            WHERE symbol = ?
            ORDER BY ts_utc DESC
            LIMIT 10
        """, (STOCK_TO_EXPLORE, PREVIEW_ROWS))

        # 4. Explore a date range YOU choose.
        # End date is exclusive; e.g. 2025-02-01 includes all of January.
        query(conn, f"{STOCK_TO_EXPLORE}: chosen date range", """
            SELECT symbol, ts_utc, open, high, low, close, volume
            FROM bars_5m
            WHERE symbol = ?
              AND ts_utc >= ?
              AND ts_utc < ?
            ORDER BY ts_utc ASC
            LIMIT ?
        """, (STOCK_TO_EXPLORE, START_DATE, END_DATE, PREVIEW_ROWS))

        # 5. Compare one stock's data volume and price range in that period.
        query(conn, f"{STOCK_TO_EXPLORE}: period summary", """
            SELECT symbol,
                   COUNT(*) AS bar_count,
                   MIN(ts_utc) AS first_bar,
                   MAX(ts_utc) AS last_bar,
                   MIN(low) AS lowest_price,
                   MAX(high) AS highest_price,
                   AVG(volume) AS average_bar_volume
            FROM bars_5m
            WHERE symbol = ?
              AND ts_utc >= ?
              AND ts_utc < ?
            GROUP BY symbol
        """, (STOCK_TO_EXPLORE, START_DATE, END_DATE))

    print("\nExploration complete. Decide:")
    print("1. Which stocks will you TRAIN on, and why?")
    print("2. Which stocks will you TRADE, and why?")
    print("3. Short-term (5-minute/intraday) or longer-term (days/weeks)?")
    print("4. What exactly will your linear regression model predict?")
    print("5. What input features and historical window will you use?")
    print("Do not assume good historical fit guarantees profitable trading.")


if __name__ == "__main__":
    main()