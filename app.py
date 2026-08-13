"""Live Lotto 6/45 descriptive analysis and combination-generation web app.

No lottery results or recommendations are stored locally. Each request retrieves the
current official public draw history, then calculates descriptive statistics in memory.
Recommendations are generated from past patterns and do not predict future results.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recommender import BACKTEST_GAMES, BACKTEST_ITERATIONS, VALID_METHODS, generate, get_number_stats, run_backtest

BASE_DIR = Path(__file__).resolve().parent
OFFICIAL_BASE = "https://www.dhlottery.co.kr"
RESULT_PAGE = f"{OFFICIAL_BASE}/lt645/result"
RESULT_API = f"{OFFICIAL_BASE}/lt645/selectPstLt645InfoNew.do"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LottoAnalysis/1.1)",
    "Accept": "application/json, text/plain, */*",
}

app = FastAPI(title="최근 로또 번호 분석", version="1.1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@dataclass(frozen=True)
class Draw:
    round: int
    numbers: tuple[int, int, int, int, int, int]
    bonus: int
    draw_date: str


class AnalysisResponse(BaseModel):
    source: str
    fetched_at: str
    latest_round: int
    oldest_round: int
    draw_count: int
    top_numbers: list[dict[str, int | None]]
    low_numbers: list[dict[str, int | None]]
    overdue_numbers: list[dict[str, int | None]]
    frequency: list[dict[str, int | None]]
    odd_even: dict[str, int]
    low_high: dict[str, int]
    sum_stats: dict[str, float | int]
    consecutive: dict[str, float | int]
    previous_overlap: dict[str, float | int]


class RecommendRequest(BaseModel):
    method: Literal["balanced", "frequency", "random"] = "balanced"
    count: Literal[1, 3, 5, 10] = 5


class RecommendationResponse(BaseModel):
    method: str
    count: int
    source_rounds: int
    recommendations: list[dict[str, int | bool | list[int]]]
    notice: str


def validate_draw(item: dict[str, Any]) -> Draw:
    """Convert an official API item while refusing malformed upstream data."""
    try:
        numbers = tuple(int(item[f"tm{i}WnNo"]) for i in range(1, 7))
        bonus = int(item["bnsWnNo"])
        round_no = int(item["ltEpsd"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("공식 응답에 필수 당첨번호가 없습니다.") from error
    if round_no < 1 or len(set(numbers)) != 6 or any(number not in range(1, 46) for number in numbers):
        raise ValueError(f"{round_no}회 공식 응답의 번호 형식이 올바르지 않습니다.")
    return Draw(round_no, tuple(sorted(numbers)), bonus, str(item.get("ltRflYmd", "")))


async def official_recent_draws(count: int = 50) -> list[Draw]:
    """Fetch current public results without persisting them locally."""
    timeout = httpx.Timeout(12.0, connect=8.0)
    async with httpx.AsyncClient(timeout=timeout, headers=HTTP_HEADERS, follow_redirects=True) as client:
        try:
            page = await client.get(RESULT_PAGE)
            page.raise_for_status()
            rounds = [int(value) for value in re.findall(r"data-value=[\"'](\d+)", page.text)]
            if not rounds:
                raise ValueError("공식 사이트에서 최신 회차를 찾지 못했습니다.")
            latest = max(rounds)
            response = await client.get(RESULT_API, params={"srchDir": "center", "srchLtEpsd": latest})
            response.raise_for_status()
            raw = response.json().get("data", {}).get("list", [])
            if not raw:
                raise ValueError("공식 사이트가 최신 회차 데이터를 반환하지 않았습니다.")
            by_round = {int(item["ltEpsd"]): item for item in raw}
            cursor = min(by_round)
            while len(by_round) < count:
                older = await client.get(RESULT_API, params={"srchDir": "older", "srchCursorLtEpsd": cursor})
                older.raise_for_status()
                items = older.json().get("data", {}).get("list", [])
                if not items:
                    break
                previous_count = len(by_round)
                by_round.update({int(item["ltEpsd"]): item for item in items})
                next_cursor = min(int(item["ltEpsd"]) for item in items)
                if len(by_round) == previous_count or next_cursor >= cursor:
                    break
                cursor = next_cursor
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            raise HTTPException(status_code=502, detail="동행복권 공식 결과를 가져오지 못했습니다. 잠시 후 다시 시도하세요.") from error
    try:
        draws = [validate_draw(item) for _, item in sorted(by_round.items(), reverse=True)[:count]]
    except ValueError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    if len(draws) < count:
        raise HTTPException(status_code=502, detail="공식 사이트에서 요청한 회차 수를 모두 가져오지 못했습니다.")
    return draws


def has_consecutive(numbers: tuple[int, ...]) -> bool:
    return any(right == left + 1 for left, right in zip(numbers, numbers[1:]))


def analyze(draws: list[Draw]) -> AnalysisResponse:
    """Produce descriptive statistics only; it does not predict future draws."""
    if not draws:
        raise ValueError("분석할 회차가 없습니다.")
    ordered = sorted(draws, key=lambda draw: draw.round, reverse=True)
    counts = Counter(number for draw in ordered for number in draw.numbers)
    last_seen: dict[int, int] = {}
    for draw in ordered:
        for number in draw.numbers:
            last_seen.setdefault(number, draw.round)
    latest = ordered[0].round
    frequency = [
        {
            "number": number,
            "count": counts[number],
            "last_round": last_seen.get(number),
            "missing_rounds": latest - last_seen[number] if number in last_seen else None,
        }
        for number in range(1, 46)
    ]
    top = sorted(frequency, key=lambda item: (-int(item["count"]), int(item["number"])))[:8]
    low = sorted(frequency, key=lambda item: (int(item["count"]), int(item["number"])))[:8]
    overdue = sorted(frequency, key=lambda item: (-(item["missing_rounds"] or 0), int(item["number"])))[:8]
    all_numbers = [number for draw in ordered for number in draw.numbers]
    sums = [sum(draw.numbers) for draw in ordered]
    odd = sum(number % 2 for number in all_numbers)
    low_band = sum(number <= 22 for number in all_numbers)
    consecutive_count = sum(has_consecutive(draw.numbers) for draw in ordered)
    overlaps = [len(set(ordered[index].numbers) & set(ordered[index + 1].numbers)) for index in range(len(ordered) - 1)]
    return AnalysisResponse(
        source="동행복권 로또6/45 공식 공개 결과 (최근 조회 시점)",
        fetched_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        latest_round=latest,
        oldest_round=ordered[-1].round,
        draw_count=len(ordered),
        top_numbers=top,
        low_numbers=low,
        overdue_numbers=overdue,
        frequency=frequency,
        odd_even={"odd": odd, "even": len(all_numbers) - odd},
        low_high={"low_1_22": low_band, "high_23_45": len(all_numbers) - low_band},
        sum_stats={"min": min(sums), "max": max(sums), "average": round(mean(sums), 1)},
        consecutive={"draws": consecutive_count, "rate_percent": round(consecutive_count / len(ordered) * 100, 1)},
        previous_overlap={"pairs": len(overlaps), "total": sum(overlaps), "average": round(mean(overlaps), 2) if overlaps else 0},
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/stats", response_model=AnalysisResponse)
async def stats(count: int = Query(50, ge=10, le=50)) -> AnalysisResponse:
    """Retrieve and analyze the requested recent public draws without a database."""
    return analyze(await official_recent_draws(count))


@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendRequest) -> RecommendationResponse:
    """Generate non-persistent combinations from the current official 50-draw window."""
    draws = await official_recent_draws(50)
    try:
        stats = get_number_stats(draws)
        recommendations = generate(request.method, request.count, stats)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return RecommendationResponse(
        method=request.method,
        count=request.count,
        source_rounds=len(draws),
        recommendations=[item.as_dict() for item in recommendations],
        notice="과거 당첨번호 통계를 기반으로 조합을 생성합니다. 모든 조합의 실제 당첨 확률은 동일하며 당첨을 보장하지 않습니다.",
    )


@app.get("/api/backtest")
async def backtest(
    games: int = Query(BACKTEST_GAMES),
    iterations: int = Query(BACKTEST_ITERATIONS, ge=1, le=100),
) -> dict[str, Any]:
    """Run a user-triggered, in-memory walk-forward comparison without future leakage."""
    if games not in {1, 3, 5, 10}:
        raise HTTPException(status_code=422, detail="백테스트 게임 수는 1, 3, 5, 10만 가능합니다.")
    draws = await official_recent_draws(50)
    try:
        result = await __import__("asyncio").to_thread(run_backtest, draws, games, iterations)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "source_rounds": len(draws),
        "games_per_draw": games,
        "iterations": iterations,
        "walk_forward": True,
        "notice": "각 대상 회차의 추천은 그보다 이전 회차만 사용했습니다. 결과는 우월성을 뜻하지 않으며, 무작위성에 따라 달라질 수 있습니다.",
        **result,
    }
