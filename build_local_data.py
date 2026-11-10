"""Step 1: copy chosen stocks from the big NAS database into a small local one."""
import sqlite3
import time
from datetime import date
from pathlib import Path

SOURCE_URI = "file:////192.168.1.95/CS-CLASSROOM/intraday_5m.sqlite?mode=ro&immutable=1"
LOCAL_DB = Path("my_stocks_5m.sqlite")

# "Innovation" watchlist, grouped by theme. Edit freely.
# Symbols not present in the source database are skipped and reported.
SYMBOLS = [
    # AI / compute
    "NVDA", "AMD", "TSM", "MSFT", "GOOGL", "PLTR",
    # Electric vehicles / energy
    "TSLA", "ENPH", "FSLR",
    # Biotech / health tech
    "ISRG", "CRSP", "ILMN", "TDOC",
    # Space / robotics / quantum
    "RKLB", "IONQ",
]

# Date filter compares strings, so this assumes ts_utc is ISO text.
# End date is exclusive. Keep this short: the NAS database is shared.
START_DATE = "2025-01-06"
END_DATE = "2025-01-27"   # 3 weeks

MAX_DAYS = 21             # hard safety cap on the window
PAUSE_SECONDS = 1.0       # breather between symbols to go easy on the NAS
CHUNK = 20_000

_days = (date.fromisoformat(END_DATE) - date.fromisoformat(START_DATE)).days
if _days > MAX_DAYS:
    raise SystemExit(f"Window is {_days} days; max allowed is {MAX_DAYS}. "
                     "The database is shared, so please keep it short.")


def main():
    src = sqlite3.connect(SOURCE_URI, uri=True)
    src.execute("PRAGMA query_only = ON")

    # Cheap existence check per symbol (no full-table scan of the shared DB)
    chosen, missing = [], []
    for s in SYMBOLS:
        hit = src.execute(
            "SELECT 1 FROM bars_5m WHERE symbol = ? LIMIT 1", (s,)).fetchone()
        (chosen if hit else missing).append(s)
    print(f"Copying {len(chosen)} symbols: {chosen}")
    if missing:
        print(f"Not in source database (skipped): {missing}")
    if not chosen:
        raise SystemExit("None of your symbols exist in the source database.")

    dst = sqlite3.connect(LOCAL_DB)
    dst.execute("""
        CREATE TABLE IF NOT EXISTS bars_5m (
            symbol TEXT NOT NULL,
            ts_utc TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)

    total = 0
    for sym in chosen:
        cur = src.execute("""
            SELECT symbol, ts_utc, open, high, low, close, volume
            FROM bars_5m
            WHERE symbol = ? AND ts_utc >= ? AND ts_utc < ?
            ORDER BY ts_utc
        """, (sym, START_DATE, END_DATE))
        n = 0
        while True:
            rows = cur.fetchmany(CHUNK)
            if not rows:
                break
            dst.executemany(
                "INSERT OR REPLACE INTO bars_5m VALUES (?,?,?,?,?,?,?)", rows)
            n += len(rows)
        dst.commit()
        total += n
        print(f"  {sym}: {n} bars")
        time.sleep(PAUSE_SECONDS)

    src.close()
    summary = dst.execute("""
        SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(ts_utc), MAX(ts_utc)
        FROM bars_5m
    """).fetchone()
    print(f"\nLocal DB {LOCAL_DB.resolve()}")
    print(f"rows={summary[0]} symbols={summary[1]} range={summary[2]} -> {summary[3]}")
    if total == 0:
        print("WARNING: 0 rows copied. Check ts_utc format (epoch vs ISO text) "
              "and your date range.")
    dst.close()


if __name__ == "__main__":
    main()