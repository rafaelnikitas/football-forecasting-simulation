"""Canonical match schema shared by every provider."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "league",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "status",
}

PROBABILITY_COLUMNS = ["market_home_prob", "market_draw_prob", "market_away_prob"]


def normalize_team_name(value: object) -> str:
    """Normalize whitespace while preserving the provider's display name."""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        raise ValueError("Team names cannot be empty")
    return text


def validate_matches(frame: pd.DataFrame, *, allow_scheduled: bool = True) -> pd.DataFrame:
    """Return a validated, time-sorted copy of the canonical match frame."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing canonical columns: {sorted(missing)}")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="coerce")
    if result["date"].isna().any():
        raise ValueError("All matches require a valid date")

    for column in ["league", "season"]:
        result[column] = result[column].astype(str).str.strip()
    for column in ["home_team", "away_team"]:
        result[column] = result[column].map(normalize_team_name)

    if (result["home_team"] == result["away_team"]).any():
        raise ValueError("A team cannot play itself")

    result["status"] = result["status"].astype(str).str.upper()
    allowed = {"FINISHED", "SCHEDULED"} if allow_scheduled else {"FINISHED"}
    invalid_status = set(result["status"]).difference(allowed)
    if invalid_status:
        raise ValueError(f"Unsupported statuses: {sorted(invalid_status)}")

    for column in ["home_goals", "away_goals"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    finished = result["status"].eq("FINISHED")
    if result.loc[finished, ["home_goals", "away_goals"]].isna().any().any():
        raise ValueError("Finished matches require both scores")
    if not allow_scheduled and (~finished).any():
        raise ValueError("Only finished matches are allowed")
    for column in ["home_goals", "away_goals"]:
        if (result.loc[finished, column] < 0).any():
            raise ValueError("Goal counts cannot be negative")

    available_probs = [column for column in PROBABILITY_COLUMNS if column in result]
    if available_probs and len(available_probs) != 3:
        raise ValueError("Market probabilities must contain the complete 1X2 triplet")
    if available_probs:
        values = result[PROBABILITY_COLUMNS].apply(pd.to_numeric, errors="coerce")
        complete = values.notna().all(axis=1)
        if ((values[complete] < 0) | (values[complete] > 1)).any().any():
            raise ValueError("Market probabilities must be between zero and one")
        sums = values.loc[complete].sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise ValueError("Complete market probabilities must sum to one")
        result[PROBABILITY_COLUMNS] = values

    result = result.sort_values(["date", "league", "home_team"]).reset_index(drop=True)
    return result


def completed_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only rows whose final score is known."""
    validated = validate_matches(frame)
    return validated.loc[validated["status"].eq("FINISHED")].reset_index(drop=True)
