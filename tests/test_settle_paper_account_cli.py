from __future__ import annotations

from scripts.settle_paper_account import parse_args


def test_settlement_cli_defaults_to_preview() -> None:
    args = parse_args([
        "--asof-date",
        "2026-07-24",
    ])

    assert args.asof_date == "2026-07-24"
    assert args.write is False


def test_settlement_cli_requires_explicit_write() -> None:
    args = parse_args([
        "--asof-date",
        "2026-07-24",
        "--write",
    ])

    assert args.write is True
