"""Data contracts, source adapters, and synthetic fixtures."""

from football_lab.data.schema import validate_matches
from football_lab.data.synthetic import generate_synthetic_league

__all__ = ["generate_synthetic_league", "validate_matches"]
