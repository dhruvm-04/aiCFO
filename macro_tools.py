from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import requests
import yfinance as yf
from langchain_core.tools import tool


# CONFIG

DATA_DIR = Path("data")

REPO_RATE_FILE = DATA_DIR / "repo_rate.xlsx"
WPI_FILE = DATA_DIR / "wpi.xlsx"
IIP_FILE = DATA_DIR / "iip.xlsx"

CPI_URL = (
    "https://indiandataproject.org/"
    "data/economy/2025-26/inflation.json"
)

FOREX_URL = (
    "https://api.frankfurter.dev/v1/latest"
    "?from=INR&to=USD,EUR,JPY"
)

warnings.filterwarnings(
    "ignore",
    message=(
        "Workbook contains no default style, "
        "apply openpyxl's default"
    ),
    category=UserWarning,
)


# HELPERS

def get_trend(value):
    if value is None:
        return "unknown"

    if value > 0:
        return "rising"

    if value < 0:
        return "falling"

    return "stable"


def safe_float(value):
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def request_json(url: str) -> dict | None:
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "aiCFO/1.0"
            },
        )

        response.raise_for_status()

        return response.json()

    except Exception:
        return None


# REPO RATE

def _read_repo_rate():
    if not REPO_RATE_FILE.exists():
        return None

    try:
        df = pd.read_excel(
            REPO_RATE_FILE,
            header=None,
        )

        # Existing project uses data from row 9 onward.
        data = df.iloc[8:].copy()

        data.columns = [
            "ignore",
            "date",
            "bank_rate",
            "repo",
            "reverse_repo",
            "sdf",
            "msf",
            "crr",
            "slr",
        ]

        data = data[
            data["repo"].notna()
        ]

        data = data[
            data["repo"].astype(str) != "-"
        ]

        data["repo"] = pd.to_numeric(
            data["repo"],
            errors="coerce",
        )

        data = data.dropna(
            subset=["repo"]
        )

        if len(data) < 1:
            return None

        current = float(
            data.iloc[0]["repo"]
        )

        previous = (
            float(data.iloc[1]["repo"])
            if len(data) > 1
            else None
        )

        change_bps = (
            round(
                (current - previous) * 100
            )
            if previous is not None
            else None
        )

        return {
            "current": current,
            "previous": previous,
            "change_bps": change_bps,
            "trend": get_trend(
                change_bps
            ),
            "source": str(
                REPO_RATE_FILE
            ),
        }

    except Exception as exc:
        return {
            "error": str(exc),
            "source": str(
                REPO_RATE_FILE
            ),
        }


@tool
def get_repo_rate() -> dict[str, Any]:
    """
    Get the latest RBI repo rate from the local RBI workbook.
    """

    result = _read_repo_rate()

    if result is None:
        return {
            "found": False,
            "metric": "repo_rate",
        }

    return {
        "found": True,
        "metric": "repo_rate",
        **result,
    }


# CPI / INFLATION

@tool
def get_inflation() -> dict[str, Any]:
    """
    Get latest CPI inflation data:
    headline, food and core.
    """

    data = request_json(CPI_URL)

    if not data:
        return {
            "found": False,
            "metric": "inflation",
        }

    try:
        series = data["series"]

        latest = series[-1]
        previous = series[-2]

        headline = safe_float(
            latest.get("cpiHeadline")
        )

        previous_headline = safe_float(
            previous.get("cpiHeadline")
        )

        if (
            headline is not None
            and previous_headline is not None
        ):
            trend = (
                "cooling"
                if headline < previous_headline
                else "heating"
            )
        else:
            trend = "unknown"

        status = (
            "within_target"
            if headline is not None
            and 2 <= headline <= 6
            else "outside_target"
        )

        return {
            "found": True,
            "metric": "inflation",
            "headline": headline,
            "food": safe_float(
                latest.get("cpiFood")
            ),
            "core": safe_float(
                latest.get("cpiCore")
            ),
            "status": status,
            "trend": trend,
            "source": CPI_URL,
        }

    except Exception as exc:
        return {
            "found": False,
            "metric": "inflation",
            "error": str(exc),
        }


# WPI

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def _load_wpi():
    if not WPI_FILE.exists():
        return None

    df = pd.read_excel(
        WPI_FILE
    )

    df["month_num"] = (
        df["month"]
        .astype(str)
        .str.strip()
        .map(MONTH_MAP)
    )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["index_value"] = pd.to_numeric(
        df["index_value"],
        errors="coerce",
    )

    return df.dropna(
        subset=[
            "year",
            "month_num",
            "index_value",
        ]
    )


def _wpi_yoy(
    df: pd.DataFrame,
    current_row,
):
    current_year = int(
        current_row["year"]
    )

    current_month = int(
        current_row["month_num"]
    )

    previous = df[
        (df["year"] == current_year - 1)
        & (
            df["month_num"]
            == current_month
        )
    ]

    if previous.empty:
        return None

    current_value = float(
        current_row["index_value"]
    )

    previous_value = float(
        previous.iloc[0]["index_value"]
    )

    if previous_value == 0:
        return None

    return round(
        (
            (
                current_value
                - previous_value
            )
            / previous_value
        )
        * 100,
        2,
    )


def _get_wpi_metric(
    df: pd.DataFrame,
    keywords: list[str],
):
    mask = pd.Series(
        False,
        index=df.index,
    )

    for keyword in keywords:

        mask |= (
            df["item"]
            .astype(str)
            .str.contains(
                keyword,
                case=False,
                na=False,
            )
        )

        if "subgroup" in df.columns:
            mask |= (
                df["subgroup"]
                .astype(str)
                .str.contains(
                    keyword,
                    case=False,
                    na=False,
                )
            )

    subset = df[
        mask
    ].copy()

    if subset.empty:
        return None

    subset = subset.sort_values(
        ["year", "month_num"],
        ascending=False,
    )

    current = subset.iloc[0]

    yoy = _wpi_yoy(
        subset,
        current,
    )

    return {
        "current": round(
            float(
                current["index_value"]
            ),
            2,
        ),
        "change_yoy": yoy,
        "trend": get_trend(yoy),
    }


@tool
def get_wpi_metrics() -> dict[str, Any]:
    """
    Get current WPI metrics relevant to automotive manufacturing:
    steel, rubber and aluminium.
    """

    df = _load_wpi()

    if df is None:
        return {
            "found": False,
            "metric": "wpi",
        }

    return {
        "found": True,
        "metric": "wpi",
        "steel": _get_wpi_metric(
            df,
            [
                "mild steel",
                "stainless steel",
                "alloy steel",
                "steel",
            ],
        ),
        "rubber": _get_wpi_metric(
            df,
            [
                "rubber",
                "tyre",
                "tire",
            ],
        ),
        "aluminium": _get_wpi_metric(
            df,
            [
                "aluminium",
            ],
        ),
        "source": str(
            WPI_FILE
        ),
    }


# IIP MOTOR VEHICLES

@tool
def get_iip_motor_vehicles() -> dict[str, Any]:
    """
    Get the latest IIP index and YoY growth for
    Manufacture of Motor Vehicles, Trailers and
    Semi-trailers.
    """

    if not IIP_FILE.exists():
        return {
            "found": False,
            "metric": "iip_motor_vehicles",
            "error": "IIP file not found",
        }

    try:
        df = pd.read_excel(IIP_FILE)

        # Actual value in the supplied iip.xlsx
        target = (
            "Manufacture of Motor Vehicles, "
            "Trailers and Semi-trailers"
        )

        mask = (
            df["sub_category"]
            .astype(str)
            .str.strip()
            .str.casefold()
            == target.casefold()
        )

        data = df[mask].copy()

        if data.empty:
            return {
                "found": False,
                "metric": "iip_motor_vehicles",
                "error": (
                    "Motor vehicle manufacturing "
                    "series not found"
                ),
            }

        data["index"] = pd.to_numeric(
            data["index"],
            errors="coerce",
        )

        data["growth_rate"] = pd.to_numeric(
            data["growth_rate"],
            errors="coerce",
        )

        data = data.dropna(
            subset=[
                "index",
                "growth_rate",
            ]
        )

        data = data.sort_values(
            "year",
            ascending=False,
        )

        current = data.iloc[0]

        return {
            "found": True,
            "metric": "iip_motor_vehicles",
            "index": float(
                current["index"]
            ),
            "change_yoy": float(
                current["growth_rate"]
            ),
            "year": str(
                current["year"]
            ),
            "trend": get_trend(
                float(
                    current["growth_rate"]
                )
            ),
            "source": str(IIP_FILE),
        }

    except Exception as exc:
        return {
            "found": False,
            "metric": "iip_motor_vehicles",
            "error": str(exc),
        }


# FOREX

@tool
def get_forex_rates() -> dict[str, Any]:
    """
    Get current USDINR, EURINR and JPYINR rates.
    """

    data = request_json(
        FOREX_URL
    )

    if not data or "rates" not in data:
        return {
            "found": False,
            "metric": "forex",
        }

    try:
        rates = data["rates"]

        return {
            "found": True,
            "metric": "forex",
            "USDINR": round(
                1 / rates["USD"],
                4,
            ),
            "EURINR": round(
                1 / rates["EUR"],
                4,
            ),
            "JPYINR": round(
                1 / rates["JPY"],
                4,
            ),
            "source": FOREX_URL,
        }

    except Exception as exc:
        return {
            "found": False,
            "metric": "forex",
            "error": str(exc),
        }


# BRENT

@tool
def get_brent() -> dict[str, Any]:
    """
    Get the latest Brent crude futures price.
    """

    for ticker in [
        "BZ=F",
        "CL=F",
    ]:

        try:
            history = (
                yf.Ticker(
                    ticker
                )
                .history(
                    period="5d"
                )
            )

            if history.empty:
                continue

            price = float(
                history["Close"].iloc[-1]
            )

            return {
                "found": True,
                "metric": "brent",
                "price": round(
                    price,
                    2,
                ),
                "unit": "USD/barrel",
                "ticker": ticker,
                "source": "Yahoo Finance",
            }

        except Exception:
            continue

    return {
        "found": False,
        "metric": "brent",
    }


# COMBINED SNAPSHOT

@tool
def get_macro_snapshot() -> dict[str, Any]:
    """
    Get a compact snapshot of the main automotive-relevant
    macroeconomic indicators.
    """

    return {
        "repo_rate": get_repo_rate.invoke({}),
        "inflation": get_inflation.invoke({}),
        "wpi": get_wpi_metrics.invoke({}),
        "iip_motor_vehicles": (
            get_iip_motor_vehicles.invoke({})
        ),
        "forex": get_forex_rates.invoke({}),
        "brent": get_brent.invoke({}),
    }


# DIRECT TEST

if __name__ == "__main__":

    import pprint

    print("\n" + "=" * 70)
    print("MACRO SNAPSHOT")
    print("=" * 70)

    result = get_macro_snapshot.invoke({})

    pprint.pprint(
        result,
        sort_dicts=False,
    )