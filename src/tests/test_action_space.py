import numpy as np
import pytest

from src.action_space.grid import generate_weight_grid

# --- Regression anchors: exact values traced by hand / given in the assignment ---

ASSIGNMENT_GRID_COUNTS = {0.25: 495, 0.20: 1287, 0.10: 43758}


@pytest.mark.parametrize("increment,expected_count", ASSIGNMENT_GRID_COUNTS.items())
def test_grid_counts_match_assignment_table(increment, expected_count):
    """Cross-check against every count given in the assignment's grid-resolution table."""
    grid = generate_weight_grid(9, increment=increment)
    assert grid.shape == (expected_count, 9)


def test_matches_hand_traced_example():
    """3 assets, 50% increments -> should have these 6 portfolios """
    grid = generate_weight_grid(3, increment=0.5)
    expected = {
        (0.0, 0.0, 1.0),
        (0.0, 0.5, 0.5),
        (0.0, 1.0, 0.0),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (1.0, 0.0, 0.0),
    }
    actual = {tuple(row) for row in grid}
    assert actual == expected


# --- General properties: should hold for any (n_assets, increment) ---

GRID_CONFIGS = [
    (2, 0.5),
    (3, 0.25),
    (9, 0.2),
    (9, 0.1),
]


@pytest.fixture(params=GRID_CONFIGS, ids=lambda cfg: f"n{cfg[0]}_inc{cfg[1]}")
def grid(request):
    n_assets, increment = request.param
    return generate_weight_grid(n_assets, increment=increment)


def test_grid_weights_sum_to_one(grid):
    assert np.allclose(grid.sum(axis=1), 1.0)


def test_grid_weights_non_negative(grid):
    assert np.all(grid >= 0)


def test_grid_has_no_duplicate_portfolios(grid):
    unique_rows = np.unique(grid, axis=0)
    assert unique_rows.shape[0] == grid.shape[0]


# --- Edge cases ---

def test_single_asset_grid_is_fully_invested():
    """With one asset, there's only one feasible portfolio: 100% in it."""
    grid = generate_weight_grid(1, increment=0.2)
    assert grid.shape == (1, 1)
    assert grid[0, 0] == pytest.approx(1.0)


def test_two_asset_grid_matches_hand_trace():
    """2 assets, 50% increments -> the 3 obvious splits."""
    grid = generate_weight_grid(2, increment=0.5)
    expected = {(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)}
    actual = {tuple(row) for row in grid}
    assert actual == expected
