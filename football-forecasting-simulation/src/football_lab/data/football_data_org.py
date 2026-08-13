"""Client and parser for the football-data.org v4 API."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

from football_lab.data.schema import validate_matches

BASE_URL = "https://api.football-data.org/v4"
COMPETITION_CODES = {"premier_league": "PL", "la_liga": "PD"}


def parse_matches_payload(
    payload: dict[str, Any],
    *,
    league: str,
    season: str | int,
) -> pd.DataFrame:
    """Convert a v4 matches response into the canonical schema."""
    records: list[dict[str, Any]] = []
    for match in payload.get("matches", []):
        score = match.get("score", {}).get("fullTime", {}) or {}
        status = "FINISHED" if match.get("status") == "FINISHED" else "SCHEDULED"
        records.append(
            {
                "date": match.get("utcDate"),
                "league": league,
                "season": str(season),
                "home_team": match.get("homeTeam", {}).get("name"),
                "away_team": match.get("awayTeam", {}).get("name"),
                "home_goals": score.get("home"),
                "away_goals": score.get("away"),
                "status": status,
                "provider_match_id": match.get("id"),
            }
        )
    if not records:
        raise ValueError("football-data.org returned no matches")
    return validate_matches(pd.DataFrame.from_records(records))


def fetch_matches(
    league: str,
    *,
    season: int,
    token: str | None = None,
) -> pd.DataFrame:
    """Fetch current or historical matches for a free-tier competition."""
    if league not in COMPETITION_CODES:
        raise ValueError(f"Unsupported league: {league}")
    api_token = token or os.getenv("FOOTBALL_DATA_API_TOKEN")
    if not api_token:
        raise ValueError("Set FOOTBALL_DATA_API_TOKEN or pass token explicitly")
    query = urllib.parse.urlencode({"season": season})
    url = f"{BASE_URL}/competitions/{COMPETITION_CODES[league]}/matches?{query}"
    request = urllib.request.Request(url, headers={"X-Auth-Token": api_token})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return parse_matches_payload(payload, league=league, season=season)
