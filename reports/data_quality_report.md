# VN30 Data Quality Report

## Overall Status

**PASS**

A flag identifies a row requiring review. A single source row may receive multiple flags.

## Dataset Summary

- Source file: `sample_data/sample_ohlcv.csv`
- Report generated: 2026-06-22 14:54:21
- Number of rows: 30
- Number of tickers: 3
- Earliest date: 2024-01-02
- Latest date: 2024-01-15
- Total issue flags: 0

## Issue Counts by Type

_No rows to display_

## Flag Counts by Ticker

_No rows to display_

## Flag Counts by Date

_No rows to display_

## Detailed Problem Rows

_No rows to display_

## Treatment Policy

- Missing date or ticker: drop the row.
- Missing price: exclude until the source is verified.
- Exact duplicate: retain one copy.
- Conflicting duplicate: investigate manually.
- Zero volume: retain but mark as non-tradable.
- Negative volume: correct from source or drop.
- Non-positive price: correct from source or drop.
- Invalid OHLC relationship: correct or drop.
- Suspicious jump: investigate corporate actions.
- Stale price: investigate liquidity, suspension, or vendor errors.

## Current Limitations

- The current development data is synthetic.
- Corporate actions are not independently verified.
- Historical VN30 membership is not yet included.
- Survivorship bias remains unresolved.
- Official ceiling and floor prices are not yet included.
