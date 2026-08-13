"""Recent 50-draw Lotto 6/45 descriptive analysis web application.

No lottery draw is saved locally. Each refresh requests current public draw data
from the official Donghaeng Lottery website and calculates descriptive statistics.
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
OFFICIAL_BASE = "https://www.dhlottery.co.kr"
RESULT_PAGE = f"{OFFICIAL_BASE}/lt645/result"
RESULT_API = f"{OFFICIAL_BASE}/lt645/selectPstLt645InfoNew.do"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LottoAnalysis/1.0)", "Accept": "application/json, text/plain, */*"}

app = FastAPI(title="최근 50회 로또 분석", version="1.0.0")
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
        return [validate_draw(item) for _, item in sorted(by_round.items(), reverse=True)[:count]]
    except ValueError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


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
        {"number": number, "count": counts[number], "last_round": last_seen.get(number), "missing_rounds": latest - last_seen[number] if number in last_seen else None}
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
