"""Adapter for the free CSV files published by Football-Data.co.uk."""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

from football_lab.data.schema import validate_matches

BASE_URL = "https://www.football-data.co.uk/mmz4281"
LEAGUE_CODES = {"premier_league": "E0", "la_liga": "SP1"}
ODDS_TRIPLETS = [
    ("AvgH", "AvgD", "AvgA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
]


def season_code(season: str | int) -> str:
    """Convert 2024-25, 2024/2025, or 2024 into the provider's 2425 format."""
    if isinstance(season, int):
        start = season
        end = season + 1
    else:
        cleaned = str(season).strip().replace("/", "-")
        parts = cleaned.split("-")
        if len(parts) == 1 and len(parts[0]) == 4:
            start = int(parts[0])
            end = start + 1
        elif len(parts) == 2:
            start = int(parts[0])
            end = int(parts[1])
            if end < 100:
                end += (start // 100) * 100
                if end <= start:
                    end += 100
        else:
            raise ValueError(f"Unsupported season format: {season}")
    if end != start + 1:
        raise ValueError("A league season must span consecutive years")
    return f"{start % 100:02d}{end % 100:02d}"


def season_label(season: str | int) -> str:
    code = season_code(season)
    start_two, end_two = int(code[:2]), int(code[2:])
    century = 2000 if start_two < 80 else 1900
    start = century + start_two
    end = (century + end_two) if end_two > start_two else (century + 100 + end_two)
    return f"{start}-{str(end)[-2:]}"


def build_url(league: str, season: str | int) -> str:
    if league not in LEAGUE_CODES:
        raise ValueError(f"Unsupported league: {league}")
    return f"{BASE_URL}/{season_code(season)}/{LEAGUE_CODES[league]}.csv"


def _market_probabilities(raw: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=raw.index, columns=["home", "draw", "away"], dtype=float)
    for home_col, draw_col, away_col in ODDS_TRIPLETS:
        if {home_col, draw_col, away_col}.issubset(raw.columns):
            odds = raw[[home_col, draw_col, away_col]].apply(pd.to_numeric, errors="coerce")
            valid = odds.notna().all(axis=1) & odds.gt(1.0).all(axis=1)
            implied = 1.0 / odds.loc[valid].to_numpy(dtype=float)
            implied /= implied.sum(axis=1, keepdims=True)
            result.loc[valid] = implied
            break
    return result


def parse_csv(
    source: str | Path | BinaryIO | TextIO | io.BytesIO,
    *,
    league: str,
    season: str | int,
) -> pd.DataFrame:
    """Parse one provider CSV into the canonical schema."""
    if league not in LEAGUE_CODES:
        raise ValueError(f"Unsupported league: {league}")
    raw = pd.read_csv(source)
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Football-Data.co.uk CSV is missing: {sorted(missing)}")

    dates = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", utc=True)
    frame = pd.DataFrame(
        {
            "date": dates,
            "league": league,
            "season": season_label(season),
            "home_team": raw["HomeTeam"],
            "away_team": raw["AwayTeam"],
            "home_goals": pd.to_numeric(raw["FTHG"], errors="coerce"),
            "away_goals": pd.to_numeric(raw["FTAG"], errors="coerce"),
        }
    )
    frame = frame.loc[frame["date"].notna() & frame["home_team"].notna()].copy()
    frame["status"] = np.where(
        frame[["home_goals", "away_goals"]].notna().all(axis=1),
        "FINISHED",
        "SCHEDULED",
    )
    market = _market_probabilities(raw).loc[frame.index]
    frame["market_home_prob"] = market["home"]
    frame["market_draw_prob"] = market["draw"]
    frame["market_away_prob"] = market["away"]
    return validate_matches(frame.reset_index(drop=True))


def fetch_season(
    league: str,
    season: str | int,
    *,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download and parse one historical season, optionally retaining the raw CSV."""
    request = urllib.request.Request(
        build_url(league, season),
        headers={"User-Agent": "football-forecasting-simulation-lab/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{league}_{season_label(season)}.csv"
        destination.write_bytes(payload)
    return parse_csv(io.BytesIO(payload), league=league, season=season)


def fetch_many(
    leagues: list[str],
    seasons: list[str | int],
    *,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    frames = [
        fetch_season(league, season, output_dir=output_dir)
        for league in leagues
        for season in seasons
    ]
    return validate_matches(pd.concat(frames, ignore_index=True))
