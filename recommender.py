"""Pure-Python Lotto 6/45 combination generators and leakage-free backtesting.

The functions receive draws in memory and never write recommendations or results to disk.
They generate combinations from descriptive historical patterns; they do not predict draws.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from random import Random
from statistics import median
from typing import Iterable, Literal, Protocol, Sequence

Method = Literal["balanced", "frequency", "random"]
VALID_METHODS = frozenset({"balanced", "frequency", "random"})
FREQUENCY_WEIGHTS = {
    "all": 0.30,
    "recent_30": 0.25,
    "recent_10": 0.15,
    "missing": 0.10,
    "random": 0.20,
}
BACKTEST_GAMES = 5
BACKTEST_ITERATIONS = 100


class DrawLike(Protocol):
    round: int
    numbers: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class Combination:
    numbers: tuple[int, int, int, int, int, int]
    sum: int
    odd: int
    even: int
    low: int
    high: int
    consecutive: bool

    def as_dict(self) -> dict[str, int | bool | list[int]]:
        return {
            "numbers": list(self.numbers),
            "sum": self.sum,
            "odd": self.odd,
            "even": self.even,
            "low": self.low,
            "high": self.high,
            "consecutive": self.consecutive,
        }


@dataclass(frozen=True)
class NumberStats:
    draws: tuple[DrawLike, ...]  # newest first
    all_counts: Counter[int]
    recent_30_counts: Counter[int]
    recent_10_counts: Counter[int]
    missing_rounds: dict[int, int]
    odd_patterns: Counter[int]
    low_patterns: Counter[int]
    sum_values: tuple[int, ...]
    sum_low: int
    sum_high: int
    sum_median: float
    consecutive_patterns: Counter[bool]
    overlap_patterns: Counter[int]
    latest_numbers: frozenset[int]


def _has_consecutive(numbers: Sequence[int]) -> bool:
    return any(right == left + 1 for left, right in zip(numbers, numbers[1:]))


def _quantile(sorted_values: Sequence[int], ratio: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[round((len(sorted_values) - 1) * ratio)]


def calculate_combination_stats(numbers: Iterable[int]) -> Combination:
    values = tuple(sorted(numbers))
    if len(values) != 6 or len(set(values)) != 6 or any(value not in range(1, 46) for value in values):
        raise ValueError("추천 조합은 1~45 사이에서 중복 없는 6개 번호여야 합니다.")
    odd = sum(value % 2 for value in values)
    low = sum(value <= 22 for value in values)
    return Combination(values, sum(values), odd, 6 - odd, low, 6 - low, _has_consecutive(values))


def get_number_stats(draws: Sequence[DrawLike]) -> NumberStats:
    """Calculate only from the draw sequence supplied by the caller."""
    ordered = tuple(sorted(draws, key=lambda draw: draw.round, reverse=True))
    if not ordered:
        raise ValueError("추천을 만들 과거 당첨번호가 없습니다.")
    all_counts = Counter(number for draw in ordered for number in draw.numbers)
    latest_round = ordered[0].round
    last_round: dict[int, int] = {}
    for draw in ordered:
        for number in draw.numbers:
            last_round.setdefault(number, draw.round)
    sums = tuple(sum(draw.numbers) for draw in ordered)
    overlaps = Counter(
        len(set(ordered[index].numbers) & set(ordered[index + 1].numbers))
        for index in range(len(ordered) - 1)
    )
    return NumberStats(
        draws=ordered,
        all_counts=all_counts,
        recent_30_counts=Counter(number for draw in ordered[:30] for number in draw.numbers),
        recent_10_counts=Counter(number for draw in ordered[:10] for number in draw.numbers),
        missing_rounds={number: latest_round - last_round[number] if number in last_round else len(ordered) for number in range(1, 46)},
        odd_patterns=Counter(sum(number % 2 for number in draw.numbers) for draw in ordered),
        low_patterns=Counter(sum(number <= 22 for number in draw.numbers) for draw in ordered),
        sum_values=sums,
        sum_low=_quantile(sorted(sums), 0.20),
        sum_high=_quantile(sorted(sums), 0.80),
        sum_median=median(sums),
        consecutive_patterns=Counter(_has_consecutive(draw.numbers) for draw in ordered),
        overlap_patterns=overlaps,
        latest_numbers=frozenset(ordered[0].numbers),
    )


def _normalised(counter: Counter[int], number: int) -> float:
    maximum = max(counter.values(), default=0)
    return counter[number] / maximum if maximum else 0.0


def _max_cluster(numbers: Sequence[int], window: int = 6) -> int:
    return max(sum(value - start < window for value in numbers) for start in numbers)


def _score_candidate(combination: Combination, stats: NumberStats) -> float:
    draw_count = len(stats.draws)
    score = 0.0
    score += 2.0 * stats.odd_patterns[combination.odd] / draw_count
    score += 2.0 * stats.low_patterns[combination.low] / draw_count
    if stats.sum_low <= combination.sum <= stats.sum_high:
        score += 2.0
    else:
        span = max(stats.sum_high - stats.sum_low, 1)
        score += max(0.0, 1.0 - abs(combination.sum - stats.sum_median) / span)
    score += stats.consecutive_patterns[combination.consecutive] / draw_count
    overlap = len(set(combination.numbers) & stats.latest_numbers)
    score += stats.overlap_patterns[overlap] / max(len(stats.draws) - 1, 1)
    cluster = _max_cluster(combination.numbers)
    if cluster >= 5:
        score -= 3.0
    elif cluster == 4:
        score -= 1.0
    return score


def score_combination(numbers: Iterable[int], stats: NumberStats) -> float:
    """Score a candidate against observed composition distributions, not draw odds."""
    return _score_candidate(calculate_combination_stats(numbers), stats)


def _weighted_numbers(stats: NumberStats, rng: Random) -> tuple[int, int, int, int, int, int]:
    available = list(range(1, 46))
    result: list[int] = []
    maximum_missing = max(stats.missing_rounds.values(), default=1) or 1
    while len(result) < 6:
        weights = [
            FREQUENCY_WEIGHTS["all"] * _normalised(stats.all_counts, number)
            + FREQUENCY_WEIGHTS["recent_30"] * _normalised(stats.recent_30_counts, number)
            + FREQUENCY_WEIGHTS["recent_10"] * _normalised(stats.recent_10_counts, number)
            + FREQUENCY_WEIGHTS["missing"] * stats.missing_rounds[number] / maximum_missing
            + FREQUENCY_WEIGHTS["random"] * rng.random()
            + 0.001
            for number in available
        ]
        selected = rng.choices(available, weights=weights, k=1)[0]
        result.append(selected)
        available.remove(selected)
    return tuple(sorted(result))


def generate_random(count: int, rng: Random | None = None) -> list[Combination]:
    rng = rng or Random()
    unique: dict[tuple[int, ...], Combination] = {}
    attempts = 0
    while len(unique) < count and attempts < count * 50:
        candidate = calculate_combination_stats(rng.sample(range(1, 46), 6))
        unique[candidate.numbers] = candidate
        attempts += 1
    if len(unique) != count:
        raise RuntimeError("서로 다른 랜덤 추천 조합을 충분히 만들지 못했습니다.")
    return list(unique.values())


def generate_frequency(count: int, stats: NumberStats, rng: Random | None = None) -> list[Combination]:
    rng = rng or Random()
    unique: dict[tuple[int, ...], Combination] = {}
    attempts = 0
    while len(unique) < count and attempts < count * 50:
        candidate = calculate_combination_stats(_weighted_numbers(stats, rng))
        unique[candidate.numbers] = candidate
        attempts += 1
    if len(unique) != count:
        raise RuntimeError("서로 다른 추천 조합을 충분히 만들지 못했습니다.")
    return list(unique.values())


def generate_balanced(count: int, stats: NumberStats, rng: Random | None = None, candidates_per_game: int = 300) -> list[Combination]:
    """Sample and rank candidates, then select randomly from the observed-pattern top group."""
    rng = rng or Random()
    candidate_pool: dict[tuple[int, ...], tuple[float, Combination]] = {}
    target = max(count * candidates_per_game, count * 4, 30)
    for _ in range(target):
        candidate = calculate_combination_stats(rng.sample(range(1, 46), 6))
        candidate_pool[candidate.numbers] = (_score_candidate(candidate, stats), candidate)
    ranked = sorted(candidate_pool.values(), key=lambda pair: pair[0], reverse=True)
    top_size = min(len(ranked), max(count * 12, 80))
    choices = ranked[:top_size]
    selected: list[Combination] = []
    while len(selected) < count and choices:
        _, candidate = choices.pop(rng.randrange(len(choices)))
        selected.append(candidate)
    if len(selected) != count:
        raise RuntimeError("서로 다른 균형 추천 조합을 충분히 만들지 못했습니다.")
    return selected


def generate(method: Method, count: int, stats: NumberStats | None = None, rng: Random | None = None, candidates_per_game: int = 300) -> list[Combination]:
    if method not in VALID_METHODS:
        raise ValueError("지원하지 않는 추천 방식입니다.")
    if count not in {1, 3, 5, 10}:
        raise ValueError("추천 게임 수는 1, 3, 5, 10만 가능합니다.")
    if method == "random":
        return generate_random(count, rng)
    if stats is None:
        raise ValueError("통계 기반 추천에는 과거 당첨번호가 필요합니다.")
    if method == "frequency":
        return generate_frequency(count, stats, rng)
    return generate_balanced(count, stats, rng, candidates_per_game)


def _empty_result() -> dict[str, int | float]:
    return {"average_match": 0.0, "tested_draws": 0, **{f"match_{matches}_plus": 0 for matches in range(1, 7)}}


def run_backtest(draws: Sequence[DrawLike], games: int = BACKTEST_GAMES, iterations: int = BACKTEST_ITERATIONS, rng: Random | None = None) -> dict[str, dict[str, int | float]]:
    """Walk forward: every target draw uses only strictly older rounds as its history."""
    if games not in {1, 3, 5, 10}:
        raise ValueError("백테스트 게임 수는 1, 3, 5, 10만 가능합니다.")
    if not 1 <= iterations <= 100:
        raise ValueError("백테스트 반복 횟수는 1~100만 가능합니다.")
    chronological = sorted(draws, key=lambda draw: draw.round)
    # Ten prior draws is the minimum meaningful rolling history in this small live window.
    targets = list(range(10, len(chronological)))
    results = {method: _empty_result() for method in VALID_METHODS}
    total_games = {method: 0 for method in VALID_METHODS}
    rng = rng or Random()
    for target_index in targets:
        history = chronological[:target_index]
        actual = set(chronological[target_index].numbers)
        stats = get_number_stats(history)
        for _ in range(iterations):
            for method in VALID_METHODS:
                recommendations = generate(method, games, stats, rng, candidates_per_game=6)
                for recommendation in recommendations:
                    matched = len(set(recommendation.numbers) & actual)
                    total_games[method] += 1
                    results[method]["average_match"] = float(results[method]["average_match"]) + matched
                    for threshold in range(1, 7):
                        if matched >= threshold:
                            key = f"match_{threshold}_plus"
                            results[method][key] = int(results[method][key]) + 1
    for method, result in results.items():
        result["tested_draws"] = len(targets)
        result["average_match"] = round(float(result["average_match"]) / total_games[method], 3) if total_games[method] else 0.0
    return results
