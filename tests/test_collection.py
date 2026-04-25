"""Tests for the collection module."""

from collection import generate_seasons


def test_generate_seasons_inclusive_range() -> None:
    """Test that generate_seasons produces the correct inclusive range."""
    assert generate_seasons("2023-24", "2025-26") == [
        "2023-24",
        "2024-25",
        "2025-26",
    ]
