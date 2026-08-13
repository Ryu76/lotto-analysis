from dataclasses import dataclass
from random import Random

from recommender import calculate_combination_stats, generate, get_number_stats, run_backtest


@dataclass(frozen=True)
class Draw:
    round: int
    numbers: tuple[int, int, int, int, int, int]


def fixture_draws() -> list[Draw]:
    return [
        Draw(round_no, tuple(sorted(((round_no + shift * 7 - 1) % 45) + 1 for shift in range(6))))
        for round_no in range(1, 31)
    ]


def test_combination_stats_are_sorted_and_complete() -> None:
    result = calculate_combination_stats([42, 3, 27, 18, 31, 11])
    assert result.numbers == (3, 11, 18, 27, 31, 42)
    assert result.sum == 132
    assert result.odd == 4
    assert result.even == 2


def test_each_method_returns_unique_valid_combinations() -> None:
    stats = get_number_stats(fixture_draws())
    for method in ("balanced", "frequency", "random"):
        results = generate(method, 10, stats, Random(101))
        assert len(results) == 10
        assert len({result.numbers for result in results}) == 10
        assert all(len(set(result.numbers)) == 6 for result in results)
        assert all(1 <= number <= 45 for result in results for number in result.numbers)


def test_backtest_is_walk_forward_and_returns_all_thresholds() -> None:
    result = run_backtest(fixture_draws(), games=1, iterations=1, rng=Random(9))
    for method in ("balanced", "frequency", "random"):
        assert result[method]["tested_draws"] == 20
        assert 0 <= result[method]["average_match"] <= 6
        assert result[method]["match_1_plus"] >= result[method]["match_2_plus"]
        assert result[method]["match_5_plus"] >= result[method]["match_6_plus"]
