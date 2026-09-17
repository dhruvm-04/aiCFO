from pathlib import Path
from typing import Any
import warnings
import pandas as pd
from langchain_core.tools import tool

# CONFIG

VAHAN_BASE_DIR = Path("data/vahan")

MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]

# VAHAN DATASET CONFIGURATION

DATASETS = {
    "maker": {
        "directory": "maker",
        "file_prefix": "maker",
        "label_column": "Maker",
    },
    "fuel": {
        "directory": "fuel",
        "file_prefix": "fuel",
        "label_column": "Fuel",
    },
    "norms": {
        "directory": "norms",
        "file_prefix": "norms",
        "label_column": "Norms",
    },
    "vehicle_category": {
        "directory": "vehicle_cat",
        "file_prefix": "vehicle_category",
        "label_column": "Vehicle Category",
    },
    "vehicle_class": {
        "directory": "vehicle_class",
        "file_prefix": "vehicle_class",
        "label_column": "Vehicle Class",
    },
}

# WARNING CONTROL

warnings.filterwarnings(
    "ignore",
    message=(
        "Workbook contains no default style, "
        "apply openpyxl's default"
    ),
    category=UserWarning,
)

# GENERAL HELPERS

def normalize_name(value: Any) -> str:
    """
    Normalize text for matching against VAHAN labels.
    """

    return (
        str(value)
        .replace("\xa0", " ")
        .strip()
        .upper()
    )


def clean_number(value: Any) -> int | float | None:
    """
    Convert VAHAN numeric values such as:

        '38,537'
        '1,21,89,294'
        38537

    into normal Python numbers.
    """

    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = (
            value
            .replace(",", "")
            .strip()
        )

        if not value:
            return None

    try:
        number = float(value)

        if number.is_integer():
            return int(number)

        return number

    except (ValueError, TypeError):
        return None

def dataset_config(dataset: str) -> dict:
    """
    Return configuration for a VAHAN dataset.
    """

    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Available datasets: {list(DATASETS)}"
        )

    return DATASETS[dataset]

def dataset_path(
    dataset: str,
    year: int,
) -> Path:
    """
    Build the path to a VAHAN workbook.
    """

    config = dataset_config(dataset)

    directory = (
        VAHAN_BASE_DIR
        / config["directory"]
    )

    filename = (f"{config['file_prefix']}_{year}.xlsx")
    return directory / filename

# LOADER

def load_vahan_year(
    dataset: str,
    year: int,
) -> pd.DataFrame:
    """
    Load and normalize one VAHAN workbook.

    Works with:
        maker
        fuel
        norms
        vehicle_category
        vehicle_class
    """

    config = dataset_config(dataset)

    path = dataset_path(
        dataset,
        year,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"VAHAN file not found: {path}"
        )

    # VAHAN has metadata rows before the actual table.
    raw = pd.read_excel(
        path,
        sheet_name="reportTable",
        header=None,
    )

    # Find the month header row.

    header_row = None

    for index, row in raw.iterrows():

        values = [
            normalize_name(value)
            for value in row.tolist()
            if pd.notna(value)
        ]

        if (
            "JAN" in values
            and "FEB" in values
        ):
            header_row = index
            break

    if header_row is None:
        raise ValueError(
            f"Could not find month header in {path}"
        )

    # Build column names.

    headers = []

    for value in raw.iloc[header_row].tolist():

        if pd.isna(value):
            headers.append("")
        else:
            headers.append(
                str(value).strip()
            )

    # VAHAN stores the dimension in column 2.
    if len(headers) < 2:
        raise ValueError(
            f"Unexpected VAHAN structure in {path}"
        )

    label_column = config["label_column"]

    headers[1] = label_column

    # Build the actual dataframe.

    data = raw.iloc[
        header_row + 1:
    ].copy()

    data.columns = headers

    data = data.dropna(
        how="all"
    )

    # Remove rows without labels.

    data = data[
        data[label_column].notna()
        & (
            data[label_column]
            .astype(str)
            .str.strip()
            != ""
        )
    ]

    # Clean numbers.

    for month in MONTHS:

        if month in data.columns:

            data[month] = (
                data[month]
                .apply(clean_number)
            )

    if "TOTAL" in data.columns:

        data["TOTAL"] = (
            data["TOTAL"]
            .apply(clean_number)
        )

    return data


# ROW MATCHING

def find_rows(
    dataset: str,
    year: int,
    query: str,
) -> list[dict[str, Any]]:
    """
    Find all rows matching a VAHAN dimension.
    """

    config = dataset_config(dataset)

    data = load_vahan_year(
        dataset,
        year,
    )

    label_column = config["label_column"]

    target = normalize_name(query)

    normalized = (
        data[label_column]
        .astype(str)
        .map(normalize_name)
    )

    # Exact match first.
    matches = data[
        normalized == target
    ]

    # Partial match if no exact result.
    if matches.empty:

        matches = data[
            normalized.str.contains(
                target,
                regex=False,
                na=False,
            )
        ]

    results = []

    for _, row in matches.iterrows():

        record = {
            "label": str(
                row[label_column]
            ).strip(),
        }

        # Monthly values.
        for month in MONTHS:

            if month not in data.columns:
                continue

            value = clean_number(
                row.get(month)
            )

            if value is not None:
                record[month] = value

        # Official TOTAL if present.
        if "TOTAL" in data.columns:

            total = clean_number(
                row.get("TOTAL")
            )

            if total is not None:
                record["TOTAL"] = total

        results.append(record)

    return results


# RESULT HELPERS

def available_months(
    row: dict[str, Any],
) -> list[str]:
    """
    Return months actually available in a result.
    """

    return [
        month
        for month in MONTHS
        if month in row
        and row[month] is not None
    ]


def calculate_total(
    row: dict[str, Any],
    months: list[str] | None = None,
) -> int:
    """
    Calculate a total from monthly values.
    """

    if months is None:
        months = available_months(row)

    return sum(
        int(row[month])
        for month in months
        if row.get(month) is not None
    )


def prepare_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add clean monthly/total structure to matched rows.
    """

    output = []

    for row in rows:

        months = available_months(
            row
        )

        if "TOTAL" in row:
            total = row["TOTAL"]
        else:
            total = calculate_total(
                row,
                months,
            )

        output.append(
            {
                "label": row["label"],
                "months_available": months,
                "monthly": {
                    month: row[month]
                    for month in months
                },
                "total": total,
            }
        )

    return output


# GENERIC DATA QUERY

def get_dimension_data(
    dataset: str,
    query: str,
    year: int,
) -> dict[str, Any]:
    """
    Generic VAHAN dimension lookup.
    """

    config = dataset_config(
        dataset
    )

    rows = find_rows(
        dataset=dataset,
        year=year,
        query=query,
    )

    if not rows:

        return {
            "found": False,
            "dataset": dataset,
            "query": query,
            "year": year,
        }

    return {
        "found": True,
        "dataset": dataset,
        "query": query,
        "year": year,
        "dimension": config["label_column"],
        "matches": prepare_rows(rows),
        "source": str(
            dataset_path(
                dataset,
                year,
            )
        ),
    }


# GENERIC TREND

def get_dimension_trend(
    dataset: str,
    query: str,
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    """
    Get a multi-year VAHAN trend.

    Partial years are compared using the same
    available months.
    """

    if start_year > end_year:
        raise ValueError(
            "start_year must be <= end_year"
        )

    yearly_matches = {}

    # Load available years.

    for year in range(
        start_year,
        end_year + 1,
    ):

        try:
            matches = find_rows(
                dataset=dataset,
                year=year,
                query=query,
            )

        except FileNotFoundError:
            continue

        if matches:
            yearly_matches[year] = matches

    if not yearly_matches:

        return {
            "found": False,
            "dataset": dataset,
            "query": query,
            "years": {},
        }

    # Determine how to resolve the requested dimension.

    latest_year = max(
        yearly_matches
    )

    latest_matches = (
        yearly_matches[latest_year]
    )

    # Don't silently merge multiple entities.
    if len(latest_matches) > 1:

        return {
            "found": True,
            "ambiguous": True,
            "dataset": dataset,
            "query": query,
            "latest_year": latest_year,
            "matches": [
                row["label"]
                for row in latest_matches
            ],
            "message": (
                "Multiple VAHAN rows matched. "
                "Specify the exact label."
            ),
        }

    target_label = (
        latest_matches[0]["label"]
    )

    # Select the same entity across years.

    selected = {}

    for year, rows in yearly_matches.items():

        exact_matches = [
            row
            for row in rows
            if normalize_name(
                row["label"]
            )
            == normalize_name(
                target_label
            )
        ]

        if exact_matches:
            selected[year] = (
                exact_matches[0]
            )

    if not selected:

        return {
            "found": False,
            "dataset": dataset,
            "query": query,
            "message": (
                "Could not resolve one "
                "consistent VAHAN entity."
            ),
        }

    # Determine comparable months.
    #
    # Example:
    # 2026 contains JAN-JUN
    #
    # Therefore all years use JAN-JUN.

    latest_row = selected[
        latest_year
    ]

    comparison_months = available_months(
        latest_row
    )

    # Calculate yearly totals.

    yearly_totals = {}

    for year, row in selected.items():

        months_used = [
            month
            for month in comparison_months
            if month in row
            and row[month] is not None
        ]

        total = calculate_total(
            row,
            months_used,
        )

        yearly_totals[year] = {
            "total": total,
            "months_used": months_used,
        }

    # YoY.

    yoy_growth = {}

    years = sorted(
        yearly_totals
    )

    for index in range(
        1,
        len(years),
    ):

        current_year = years[index]
        previous_year = years[index - 1]

        current_total = (
            yearly_totals[
                current_year
            ]["total"]
        )

        previous_total = (
            yearly_totals[
                previous_year
            ]["total"]
        )

        if previous_total == 0:
            growth = None
        else:
            growth = (
                (
                    current_total
                    - previous_total
                )
                / previous_total
            ) * 100

        yoy_growth[
            current_year
        ] = (
            round(growth, 2)
            if growth is not None
            else None
        )

    return {
        "found": True,
        "ambiguous": False,
        "dataset": dataset,
        "query": query,
        "label": target_label,
        "years": yearly_totals,
        "yoy_growth_pct": yoy_growth,
        "comparison_months": comparison_months,
        "latest_year": latest_year,
        "source_directory": str(
            VAHAN_BASE_DIR
            / dataset_config(
                dataset
            )["directory"]
        ),
    }


# LANGCHAIN TOOLS

@tool
def get_maker_data(
    maker: str,
    year: int,
) -> dict:
    """
    Get VAHAN registration data for a manufacturer.
    """

    return get_dimension_data(
        dataset="maker",
        query=maker,
        year=year,
    )


@tool
def get_maker_trend(
    maker: str,
    start_year: int = 2022,
    end_year: int = 2026,
) -> dict:
    """
    Get multi-year VAHAN registration trend for a manufacturer.
    """

    return get_dimension_trend(
        dataset="maker",
        query=maker,
        start_year=start_year,
        end_year=end_year,
    )


@tool
def get_fuel_data(
    fuel: str,
    year: int,
) -> dict:
    """
    Get VAHAN registration data by fuel type.
    """

    return get_dimension_data(
        dataset="fuel",
        query=fuel,
        year=year,
    )


@tool
def get_fuel_trend(
    fuel: str,
    start_year: int = 2022,
    end_year: int = 2026,
) -> dict:
    """
    Get multi-year VAHAN registration trend by fuel type.
    """

    return get_dimension_trend(
        dataset="fuel",
        query=fuel,
        start_year=start_year,
        end_year=end_year,
    )


@tool
def get_vehicle_category_data(
    category: str,
    year: int,
) -> dict:
    """
    Get VAHAN registration data by vehicle category.
    """

    return get_dimension_data(
        dataset="vehicle_category",
        query=category,
        year=year,
    )


@tool
def get_vehicle_category_trend(
    category: str,
    start_year: int = 2022,
    end_year: int = 2026,
) -> dict:
    """
    Get multi-year VAHAN registration trend by vehicle category.
    """

    return get_dimension_trend(
        dataset="vehicle_category",
        query=category,
        start_year=start_year,
        end_year=end_year,
    )


@tool
def get_vehicle_class_data(
    vehicle_class: str,
    year: int,
) -> dict:
    """
    Get VAHAN registration data by vehicle class.
    """

    return get_dimension_data(
        dataset="vehicle_class",
        query=vehicle_class,
        year=year,
    )


@tool
def get_vehicle_class_trend(
    vehicle_class: str,
    start_year: int = 2021,
    end_year: int = 2026,
) -> dict:
    """
    Get multi-year VAHAN registration trend by vehicle class.
    """

    return get_dimension_trend(
        dataset="vehicle_class",
        query=vehicle_class,
        start_year=start_year,
        end_year=end_year,
    )


@tool
def get_norms_data(
    norm: str,
    year: int,
) -> dict:
    """
    Get VAHAN registration data by emission norm.
    """

    return get_dimension_data(
        dataset="norms",
        query=norm,
        year=year,
    )


@tool
def get_norms_trend(
    norm: str,
    start_year: int = 2022,
    end_year: int = 2026,
) -> dict:
    """
    Get multi-year VAHAN registration trend by emission norm.
    """

    return get_dimension_trend(
        dataset="norms",
        query=norm,
        start_year=start_year,
        end_year=end_year,
    )


# DIRECT TESTS

if __name__ == "__main__":

    import pprint

    # Maker

    print("\n" + "=" * 70)
    print("MAKER")
    print("=" * 70)

    result = get_maker_data.invoke(
        {
            "maker": "TATA MOTORS LTD",
            "year": 2026,
        }
    )

    pprint.pprint(
        result,
        sort_dicts=False,
    )

    # Maker trend

    print("\n" + "=" * 70)
    print("MAKER TREND")
    print("=" * 70)

    result = get_maker_trend.invoke(
        {
            "maker": "TATA MOTORS LTD",
            "start_year": 2022,
            "end_year": 2026,
        }
    )

    pprint.pprint(
        result,
        sort_dicts=False,
    )

    # Fuel

    print("\n" + "=" * 70)
    print("FUEL")
    print("=" * 70)

    result = get_fuel_data.invoke(
        {
            "fuel": "ELECTRIC(BOV)",
            "year": 2026,
        }
    )

    pprint.pprint(
        result,
        sort_dicts=False,
    )

    # Vehicle Category

    print("\n" + "=" * 70)
    print("VEHICLE CATEGORY")
    print("=" * 70)

    result = get_vehicle_category_data.invoke(
        {
            "category": "LIGHT MOTOR VEHICLE",
            "year": 2026,
        }
    )

    pprint.pprint(
        result,
        sort_dicts=False,
    )

    # Vehicle Class

    print("\n" + "=" * 70)
    print("VEHICLE CLASS")
    print("=" * 70)

    result = get_vehicle_class_data.invoke(
        {
            "vehicle_class": "AGRICULTURAL TRACTOR",
            "year": 2026,
        }
    )

    pprint.pprint(
        result,
        sort_dicts=False,
    )

    # Norms

    print("\n" + "=" * 70)
    print("NORMS")
    print("=" * 70)

    result = get_norms_data.invoke(
        {
            "norm": "BHARAT STAGE VI",
            "year": 2026,
        }
    )

    pprint.pprint(
        result,
        sort_dicts=False,
    )