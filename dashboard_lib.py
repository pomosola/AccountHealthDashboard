"""Load practicum extracts and turn them into dashboard-ready tables."""

from __future__ import annotations

import io
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

NAVY = "#003057"
GOLD = "#B3A369"
COLORS = [
    "#003057",
    "#B3A369",
    "#3E7C78",
    "#C65D3B",
    "#5C6B7A",
    "#8D6A32",
    "#1F4E5F",
    "#A34155",
    "#6E8B74",
    "#4C5F7A",
]

PRACTICUM_FOLDER = Path(
    r"c:\Users\pomos\OneDrive\Documents\Ga_Tech_Courses\Applied_Analytics_Practicum\Dashboard datasets"
)
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xlsm", ".xls", ".json", ".parquet"}
AGGREGATIONS = ["Sum", "Average", "Median", "Minimum", "Maximum", "Count"]
FREQUENCIES = ["Day", "Week", "Month", "Quarter"]
_AGG_FUNCS = {
    "Sum": "sum",
    "Average": "mean",
    "Median": "median",
    "Minimum": "min",
    "Maximum": "max",
}
_FREQ_CODES = {"Day": "D", "Week": "W-MON", "Month": "MS", "Quarter": "QS"}

_DATE_RE = re.compile(r"\b(date|time|timestamp|month|day)\b")
_ID_RE = re.compile(r"\b(id|uuid|guid|sku|ssn|email|phone|postal|zipcode|zip)\b")
_MEASURE_RE = re.compile(
    r"\b(revenue|sales|amount|price|cost|profit|qty|quantity|units|score|rating|nps|value|total|target|budget|spend|income|aov|gmv|hours|orders)\b"
)
_PERCENT_RE = re.compile(r"\b(percent|pct|margin|discount)\b")
_MONEY_RE = re.compile(
    r"\b(revenue|sales|price|cost|profit|amount|ltv|gmv|aov|target|budget|spend|income|value)\b"
)
_PREFERRED_DIMENSION = re.compile(
    r"\b(region|category|segment|channel|status|type|product|state|department|group)\b"
)
_DATE_TOKEN = re.compile(
    r"(?:\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?)|(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
)


class DataLoadError(Exception):
    """A file in the data folder could not be read."""


@dataclass(frozen=True)
class DatasetRef:
    label: str
    path: str
    sheet: str | None = None


@dataclass(frozen=True)
class Roles:
    dates: list[str]
    measures: list[str]
    dimensions: list[str]
    identifiers: list[str]
    text: list[str]

    def role_of(self, column: str) -> str:
        if column in self.dates:
            return "Date"
        if column in self.measures:
            return "Measure"
        if column in self.dimensions:
            return "Category"
        if column in self.identifiers:
            return "Identifier"
        return "Text"


def bundled_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def practicum_folder() -> Path:
    return PRACTICUM_FOLDER


def default_folder() -> Path:
    """Prefer the practicum extracts, then an override, then the bundled health extract."""
    if PRACTICUM_FOLDER.exists():
        return PRACTICUM_FOLDER
    override = os.environ.get("DASHBOARD_DATA_DIR")
    if override and Path(override).expanduser().exists():
        return Path(override).expanduser()
    pack = bundled_data_dir() / "practicum"
    if (pack / "product_health_score.csv").exists() and (pack / "score_context.csv").exists():
        return pack
    health = bundled_data_dir() / "health"
    if health.exists() and any(path for path in health.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES):
        return health
    return bundled_data_dir()


def humanize(name: str) -> str:
    text = re.sub(r"[_\-]+", " ", str(name)).strip()
    if not text:
        return str(name)
    return text[:1].upper() + text[1:]


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def dataset_title(label: str) -> str:
    if "·" in label:
        file_part, sheet = [part.strip() for part in label.split("·", 1)]
        return f"{humanize(Path(file_part).stem)} — {sheet}"
    return humanize(Path(label).stem)


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.0f} KB"
    return f"{size / 1024**2:.1f} MB"


def is_text_like(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    )


def uses_percent(name: str, series: pd.Series | None = None) -> bool:
    if not _PERCENT_RE.search(normalize_name(name)):
        return False
    if series is None or not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    maximum = series.abs().max(skipna=True)
    if pd.isna(maximum):
        return True
    return float(maximum) <= 1.5


def is_money(name: str) -> bool:
    norm = normalize_name(name)
    return bool(_MONEY_RE.search(norm) and not _PERCENT_RE.search(norm))


def format_value(name: str, value, series: pd.Series | None = None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return "—"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%b %d, %Y")
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        number = int(value)
        if uses_percent(name, series):
            return f"{number:.1%}"
        if is_money(name):
            return f"${number:,.0f}"
        return f"{number:,}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "—"
        if uses_percent(name, series):
            return f"{number:.1%}"
        if is_money(name):
            return f"${number:,.0f}" if abs(number) >= 100 else f"${number:,.2f}"
        if abs(number) >= 100 or number.is_integer():
            return f"{number:,.0f}"
        return f"{number:,.2f}"
    return str(value)


def format_span(series: pd.Series) -> str:
    valid = series.dropna()
    if valid.empty:
        return "—"
    start = pd.Timestamp(valid.min())
    end = pd.Timestamp(valid.max())
    if start.year == end.year and start.month == end.month:
        return f"{start:%b %Y}"
    return f"{start:%b %Y} – {end:%b %Y}"


def suggest_measure(columns: list[str]) -> str | None:
    if not columns:
        return None

    def score(column: str) -> tuple:
        norm = normalize_name(column)
        primary = 1 if re.search(r"\b(revenue|sales|amount|profit|target)\b", norm) else 0
        money = 1 if is_money(column) else 0
        percent = 1 if _PERCENT_RE.search(norm) else 0
        return (primary, money, -percent, -columns.index(column))

    return max(columns, key=score)


def is_preferred_dimension(name: str) -> bool:
    return bool(_PREFERRED_DIMENSION.search(normalize_name(name)))


def suggest_dimension(frame: pd.DataFrame, columns: list[str]) -> str | None:
    usable = [column for column in columns if 2 <= int(frame[column].nunique(dropna=True)) <= 40]
    pool = usable or list(columns)
    if not pool:
        return None

    def score(column: str) -> tuple:
        distinct = int(frame[column].nunique(dropna=True))
        preferred = 1 if _PREFERRED_DIMENSION.search(normalize_name(column)) else 0
        compact = 1 if 2 <= distinct <= 12 else 0
        return (preferred, compact, -distinct)

    return max(pool, key=score)


def suggest_frequency(series: pd.Series) -> str:
    valid = series.dropna()
    if valid.empty:
        return "Month"
    span = pd.Timestamp(valid.max()) - pd.Timestamp(valid.min())
    days = int(getattr(span, "days", 0))
    if days > 120:
        return "Month"
    if days > 21:
        return "Week"
    return "Day"


def find_revenue_cost(columns) -> tuple[str | None, str | None]:
    lookup = {normalize_name(column): column for column in columns}
    revenue = (
        lookup.get("revenue")
        or lookup.get("total revenue")
        or lookup.get("net sales")
        or lookup.get("sales")
    )
    cost = lookup.get("cost") or lookup.get("total cost") or lookup.get("cogs")
    if revenue and cost and revenue != cost:
        return revenue, cost
    return None, None


def with_margin(frame: pd.DataFrame, revenue: str, cost: str) -> pd.DataFrame:
    out = frame.copy()
    if "gross_profit" not in out.columns:
        out["gross_profit"] = out[revenue] - out[cost]
    if "margin" not in out.columns:
        base = out[revenue].replace(0, np.nan)
        out["margin"] = out["gross_profit"] / base
    return out


def as_label(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        labels = series.map({True: "True", False: "False"}).astype("string")
        return labels.fillna("(blank)")
    if pd.api.types.is_datetime64_any_dtype(series):
        labels = series.dt.strftime("%Y-%m-%d").astype("string")
        return labels.fillna("(blank)")
    if pd.api.types.is_numeric_dtype(series):

        def numeric_label(value):
            if pd.isna(value):
                return "(blank)"
            number = float(value)
            if number.is_integer():
                return str(int(number))
            return f"{number:g}"

        return series.map(numeric_label).astype("string")
    labels = series.astype("string").fillna("(blank)")
    return labels.mask(labels.str.strip().fillna("") == "", "(blank)")


def category_options(series: pd.Series) -> list[str]:
    counts = as_label(series).value_counts(dropna=False)
    return [str(value) for value in counts.index.tolist()]


def discover(folder: Path) -> tuple[list[DatasetRef], list[str]]:
    if not folder.exists() or not folder.is_dir():
        return [], [f"Folder not found: {folder}"]
    try:
        files = sorted(
            (
                path
                for path in folder.iterdir()
                if path.is_file() and not path.name.startswith(("~$", "."))
            ),
            key=lambda path: path.name.lower(),
        )
    except OSError as exc:
        return [], [f"Could not open {folder}: {exc}"]

    refs: list[DatasetRef] = []
    errors: list[str] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            try:
                sheets = pd.ExcelFile(path).sheet_names
            except Exception as exc:  # noqa: BLE001 - surface the reader error as-is
                errors.append(f"{path.name}: {exc}")
                continue
            for sheet in sheets:
                label = path.name if len(sheets) == 1 else f"{path.name} · {sheet}"
                refs.append(DatasetRef(label=label, path=str(path), sheet=sheet))
        else:
            refs.append(DatasetRef(label=path.name, path=str(path), sheet=None))
    return refs, errors


def load_dataset(ref: DatasetRef) -> tuple[pd.DataFrame, list[str]]:
    try:
        raw, notes = _read_path(Path(ref.path), ref.sheet)
        frame, prep_notes = prepare(raw)
    except DataLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - attach the file name to any reader failure
        raise DataLoadError(f"Could not read {ref.label}: {exc}") from exc
    return frame, notes + prep_notes


def load_uploaded(name: str, payload: bytes) -> list[tuple[str, pd.DataFrame, list[str]]]:
    suffix = Path(name).suffix.lower()
    try:
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            workbook = pd.ExcelFile(io.BytesIO(payload))
            loaded = []
            for sheet in workbook.sheet_names:
                raw = pd.read_excel(workbook, sheet_name=sheet)
                frame, notes = prepare(raw)
                label = name if len(workbook.sheet_names) == 1 else f"{name} · {sheet}"
                loaded.append((label, frame, notes))
            return loaded
        raw, notes = _read_bytes(name, payload)
        frame, prep_notes = prepare(raw)
        return [(name, frame, notes + prep_notes)]
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Could not read {name}: {exc}") from exc


def prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    prepared = _promote_header(frame.copy(), notes)
    prepared = _tidy_columns(prepared, notes)
    prepared = _coerce(prepared, notes)
    prepared = prepared.dropna(how="all").reset_index(drop=True)
    if prepared.empty:
        notes.append("This file has no data rows.")
    return prepared, notes


def classify(frame: pd.DataFrame) -> Roles:
    dates: list[str] = []
    measures: list[str] = []
    dimensions: list[str] = []
    identifiers: list[str] = []
    text: list[str] = []
    row_count = len(frame)

    for column in frame.columns:
        series = frame[column]
        distinct = int(series.nunique(dropna=True))
        if pd.api.types.is_datetime64_any_dtype(series):
            dates.append(column)
            continue
        if _is_identifier(column):
            identifiers.append(column)
            continue
        if distinct <= 1:
            dimensions.append(column)
            continue
        if pd.api.types.is_bool_dtype(series):
            dimensions.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series):
            if _is_measure_name(column) or _is_percent_name(column):
                measures.append(column)
            elif distinct <= 12:
                dimensions.append(column)
            else:
                measures.append(column)
            continue
        ratio = distinct / row_count if row_count else 1
        if distinct <= 40 or (ratio <= 0.2 and distinct <= 200):
            dimensions.append(column)
        else:
            text.append(column)

    return Roles(dates, measures, dimensions, identifiers, text)


def effective_roles(
    frame: pd.DataFrame,
    dates: list[str],
    measures: list[str],
    dimensions: list[str],
    detected: Roles,
) -> Roles:
    columns = set(frame.columns)
    date_cols = [column for column in dates if column in columns]
    measure_cols = [column for column in measures if column in columns and column not in date_cols]
    dimension_cols = [
        column
        for column in dimensions
        if column in columns and column not in date_cols and column not in measure_cols
    ]
    used = set(date_cols + measure_cols + dimension_cols)
    identifiers = [column for column in detected.identifiers if column in columns and column not in used]
    text = [
        column
        for column in frame.columns
        if column not in used and column not in identifiers
    ]
    return Roles(date_cols, measure_cols, dimension_cols, identifiers, text)


def apply_date_overrides(frame: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in date_columns:
        if column not in out.columns:
            continue
        if not pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = _naive_dates(out[column])
    return out


def apply_filters(
    frame: pd.DataFrame,
    selections: dict[str, list[str]],
    date_ranges: dict[str, tuple],
) -> pd.DataFrame:
    filtered = frame
    for column, values in selections.items():
        if not values or column not in filtered.columns:
            continue
        filtered = filtered[as_label(filtered[column]).isin(values)]
    for column, bounds in date_ranges.items():
        if column not in filtered.columns:
            continue
        start, end = bounds
        if start is None or end is None:
            continue
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        filtered = filtered[filtered[column].between(start_ts, end_ts)]
    return filtered


def aggregate(
    frame: pd.DataFrame,
    dimension: str,
    measure: str | None,
    how: str,
    top_n: int = 12,
) -> pd.DataFrame:
    data = frame.copy()
    data["_group"] = as_label(data[dimension])
    if how == "Count" or not measure:
        table = data.groupby("_group", dropna=False).size().reset_index(name="value")
    else:
        table = (
            data.groupby("_group", dropna=False)[measure]
            .agg(_AGG_FUNCS[how])
            .reset_index(name="value")
        )
    table = table.dropna(subset=["value"]).sort_values("value", ascending=False)
    table["_group"] = table["_group"].astype(str)
    if len(table) > top_n:
        if how in {"Sum", "Count"}:
            other = float(table.iloc[top_n:]["value"].sum())
            table = table.iloc[:top_n]
            table = pd.concat(
                [table, pd.DataFrame({"_group": ["Other"], "value": [other]})],
                ignore_index=True,
            )
        else:
            table = table.iloc[:top_n]
    other_rows = table[table["_group"] == "Other"]
    kept = table[table["_group"] != "Other"]
    table = pd.concat([kept, other_rows], ignore_index=True)
    return table.rename(columns={"_group": dimension})


def trend(
    frame: pd.DataFrame,
    date_column: str,
    measure: str | None,
    how: str,
    frequency: str,
    split: str | None = None,
    top_series: int = 6,
) -> pd.DataFrame:
    data = frame.copy()
    data[date_column] = _naive_dates(data[date_column])
    data = data.dropna(subset=[date_column])
    if data.empty:
        return pd.DataFrame(columns=[date_column, "value"])
    keys: list = [pd.Grouper(key=date_column, freq=_FREQ_CODES[frequency])]
    if split:
        data["_split"] = as_label(data[split])
        if how == "Count" or not measure:
            ranking = data.groupby("_split").size().sort_values(ascending=False)
        else:
            ranking = data.groupby("_split")[measure].sum(min_count=1).sort_values(ascending=False)
        keep = list(ranking.head(top_series).index)
        data = data[data["_split"].isin(keep)]
        keys.append("_split")
    if how == "Count" or not measure:
        table = data.groupby(keys, dropna=False).size().reset_index(name="value")
    else:
        table = (
            data.groupby(keys, dropna=False)[measure]
            .agg(_AGG_FUNCS[how])
            .reset_index(name="value")
        )
    table = table.dropna(subset=["value"]).sort_values(date_column)
    if split:
        table = table.rename(columns={"_split": split})
    return table


def narrative(frame: pd.DataFrame, roles: Roles, measure: str | None, dimension: str | None) -> str:
    if frame.empty:
        return "No rows match the current filters."
    sentences = []
    if measure and measure in frame.columns and pd.api.types.is_numeric_dtype(frame[measure]):
        total = frame[measure].sum(min_count=1)
        if pd.notna(total):
            sentences.append(
                f"Total {humanize(measure).lower()} is {format_value(measure, total, frame[measure])} across {len(frame):,} rows."
            )
        else:
            sentences.append(f"{len(frame):,} rows.")
    else:
        sentences.append(f"{len(frame):,} rows and {frame.shape[1]} columns.")
    if roles.dates:
        dates = frame[roles.dates[0]].dropna()
        if not dates.empty:
            start = pd.Timestamp(dates.min())
            end = pd.Timestamp(dates.max())
            sentences.append(f"{humanize(roles.dates[0])} runs from {start:%b %d, %Y} to {end:%b %d, %Y}.")
    if (
        dimension
        and measure
        and dimension in frame.columns
        and measure in frame.columns
        and pd.api.types.is_numeric_dtype(frame[measure])
    ):
        grouped = (
            frame.groupby(as_label(frame[dimension]), dropna=False)[measure]
            .sum(min_count=1)
            .sort_values(ascending=False)
        )
        grouped = grouped.dropna()
        total = grouped.sum()
        if len(grouped) and pd.notna(total) and total != 0:
            share = float(grouped.iloc[0] / total)
            sentences.append(
                f"{grouped.index[0]} leads {humanize(dimension).lower()} with {share:.0%} of {humanize(measure).lower()}."
            )
    missing = frame.isna().mean()
    if not missing.empty and float(missing.max()) >= 0.1:
        column = str(missing.idxmax())
        sentences.append(f"{humanize(column)} is missing {float(missing.max()):.0%} of its values.")
    return " ".join(sentences)


def profile_table(frame: pd.DataFrame, roles: Roles) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        series = frame[column]
        row = {
            "Column": column,
            "Role": roles.role_of(column),
            "Type": _friendly_type(series),
            "Non-null": int(series.notna().sum()),
            "Missing %": round(float(series.isna().mean() * 100), 1),
            "Unique": int(series.nunique(dropna=True)),
            "Min": "",
            "Max": "",
            "Mean": "",
            "Top value": "",
        }
        if pd.api.types.is_bool_dtype(series):
            mode = as_label(series).mode(dropna=True)
            row["Top value"] = "" if mode.empty else str(mode.iloc[0])
        elif pd.api.types.is_numeric_dtype(series) and series.notna().any():
            row["Min"] = format_value(column, series.min(), series)
            row["Max"] = format_value(column, series.max(), series)
            row["Mean"] = format_value(column, series.mean(), series)
        elif pd.api.types.is_datetime64_any_dtype(series) and series.notna().any():
            row["Min"] = format_value(column, pd.Timestamp(series.min()))
            row["Max"] = format_value(column, pd.Timestamp(series.max()))
        else:
            mode = series.mode(dropna=True)
            row["Top value"] = "" if mode.empty else str(mode.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def value_axis_format(name: str, series: pd.Series | None, how: str) -> str | None:
    if how == "Count":
        return ","
    if uses_percent(name, series):
        return ".1%"
    if is_money(name):
        return "$,.0f"
    return ",.0f"


def style_figure(fig: go.Figure, title: str, height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=NAVY)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3, Segoe UI, sans-serif", color="#1C2834", size=13),
        colorway=COLORS,
        margin=dict(l=72, r=24, t=56, b=48),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        showlegend=legend,
    )
    fig.update_xaxes(automargin=True, showgrid=True, gridcolor="#E7E2D8", zeroline=False)
    fig.update_yaxes(automargin=True, showgrid=True, gridcolor="#E7E2D8", zeroline=False)
    return fig


def _apply_value_format(fig: go.Figure, fmt: str | None, horizontal: bool = False) -> None:
    if not fmt:
        return
    if horizontal:
        fig.update_xaxes(tickformat=fmt)
    else:
        fig.update_yaxes(tickformat=fmt)


def bar_figure(
    table: pd.DataFrame,
    dimension: str,
    measure_name: str,
    series: pd.Series | None,
    how: str,
    title: str,
    horizontal: bool = False,
) -> go.Figure:
    axis_title = _axis_title(measure_name, how)
    if horizontal:
        fig = px.bar(
            table,
            x="value",
            y=dimension,
            orientation="h",
            color_discrete_sequence=[NAVY],
            labels={"value": axis_title, dimension: humanize(dimension)},
        )
    else:
        fig = px.bar(
            table,
            x=dimension,
            y="value",
            color_discrete_sequence=[NAVY],
            labels={"value": axis_title, dimension: humanize(dimension)},
        )
    fig = style_figure(fig, title)
    if horizontal:
        fig.update_layout(margin=dict(l=128, r=24, t=56, b=40))
        fig.update_yaxes(automargin=True, categoryorder="total ascending", title_text=None)
    _apply_value_format(fig, value_axis_format(measure_name, series, how), horizontal=horizontal)
    return fig


def donut_figure(table: pd.DataFrame, dimension: str, title: str) -> go.Figure:
    fig = px.pie(
        table,
        names=dimension,
        values="value",
        color_discrete_sequence=COLORS,
        hole=0.58,
    )
    fig.update_traces(textposition="inside", textinfo="percent")
    return style_figure(fig, title, legend=True)


def line_figure(
    table: pd.DataFrame,
    date_column: str,
    split: str | None,
    measure_name: str,
    series: pd.Series | None,
    how: str,
    title: str,
) -> go.Figure:
    axis_title = _axis_title(measure_name, how)
    labels = {"value": axis_title, date_column: humanize(date_column)}
    if split:
        fig = px.line(
            table,
            x=date_column,
            y="value",
            color=split,
            markers=len(table) <= 80,
            color_discrete_sequence=COLORS,
            labels=labels,
        )
        fig.update_layout(hovermode="x unified")
    else:
        fig = px.area(
            table,
            x=date_column,
            y="value",
            color_discrete_sequence=[NAVY],
            labels=labels,
        )
        fig.update_traces(line_color=NAVY, fillcolor="rgba(0, 48, 87, 0.16)")
    fig = style_figure(fig, title, legend=bool(split))
    _apply_value_format(fig, value_axis_format(measure_name, series, how))
    return fig


def grouped_bar_figure(
    frame: pd.DataFrame,
    dimension: str,
    split: str,
    measure: str | None,
    how: str,
    title: str,
) -> go.Figure:
    data = frame.copy()
    data["_dim"] = as_label(data[dimension])
    data["_split"] = as_label(data[split])
    top_groups = list(data["_dim"].value_counts().head(8).index)
    if how == "Count" or not measure:
        top_series = list(data["_split"].value_counts().head(5).index)
    else:
        top_series = list(
            data.groupby("_split")[measure].sum(min_count=1).sort_values(ascending=False).head(5).index
        )
    data = data[data["_dim"].isin(top_groups) & data["_split"].isin(top_series)]
    if how == "Count" or not measure:
        table = data.groupby(["_dim", "_split"], dropna=False).size().reset_index(name="value")
    else:
        table = (
            data.groupby(["_dim", "_split"], dropna=False)[measure]
            .agg(_AGG_FUNCS[how])
            .reset_index(name="value")
        )
    fig = px.bar(
        table,
        x="_dim",
        y="value",
        color="_split",
        barmode="group",
        color_discrete_sequence=COLORS,
        labels={"value": _axis_title(measure or "Rows", how), "_dim": humanize(dimension), "_split": humanize(split)},
    )
    fig = style_figure(fig, title)
    _apply_value_format(fig, value_axis_format(measure or "Rows", frame[measure] if measure else None, how))
    return fig


def histogram_figure(frame: pd.DataFrame, measure: str, title: str) -> go.Figure:
    fig = px.histogram(
        frame,
        x=measure,
        nbins=40,
        color_discrete_sequence=[NAVY],
        labels={measure: humanize(measure)},
    )
    fig.update_layout(bargap=0.06)
    return style_figure(fig, title, legend=False)


def box_figure(frame: pd.DataFrame, dimension: str, measure: str, title: str) -> go.Figure:
    data = frame.copy()
    data["_dim"] = as_label(data[dimension])
    keep = list(data["_dim"].value_counts().head(8).index)
    data = data[data["_dim"].isin(keep)]
    fig = px.box(
        data,
        x="_dim",
        y=measure,
        color_discrete_sequence=[NAVY],
        labels={"_dim": humanize(dimension), measure: humanize(measure)},
    )
    return style_figure(fig, title, legend=False)


def scatter_figure(
    frame: pd.DataFrame,
    x: str,
    y: str,
    color: str | None,
    title: str,
) -> go.Figure:
    plot_frame = frame
    if len(plot_frame) > 4000:
        plot_frame = plot_frame.sample(4000, random_state=7)
    plot_frame = plot_frame.copy()
    if color:
        plot_frame[color] = as_label(plot_frame[color])
    fig = px.scatter(
        plot_frame,
        x=x,
        y=y,
        color=color,
        opacity=0.75,
        color_discrete_sequence=COLORS,
        labels={x: humanize(x), y: humanize(y)},
    )
    return style_figure(fig, title, legend=bool(color))


def correlation_figure(frame: pd.DataFrame, measures: list[str], title: str = "How the measures move together") -> go.Figure | None:
    usable = [
        column
        for column in measures
        if column in frame.columns and int(frame[column].nunique(dropna=True)) >= 2
    ]
    if len(usable) < 2:
        return None
    corr = frame[usable].replace([np.inf, -np.inf], np.nan).corr().round(2)
    text = ".2f" if len(usable) <= 8 else False
    fig = px.imshow(
        corr,
        color_continuous_scale=["#8C3A3A", "#F4F1EA", "#003057"],
        zmin=-1,
        zmax=1,
        text_auto=text,
        aspect="auto",
        labels={"color": ""},
    )
    height = max(420, 42 * len(usable) + 120)
    return style_figure(fig, title, height=height, legend=False)


def missing_figure(frame: pd.DataFrame) -> go.Figure | None:
    rates = (frame.isna().mean() * 100).sort_values(ascending=False)
    rates = rates[rates > 0].head(15)
    if rates.empty:
        return None
    chart = rates.rename("missing").reset_index()
    chart.columns = ["column", "missing"]
    fig = px.bar(
        chart,
        x="missing",
        y="column",
        orientation="h",
        color_discrete_sequence=["#C65D3B"],
        labels={"missing": "Missing %", "column": ""},
    )
    fig.update_yaxes(categoryorder="total ascending", title_text=None)
    fig.update_xaxes(ticksuffix="%")
    return style_figure(fig, "Columns with missing values", legend=False)


def _axis_title(measure_name: str, how: str) -> str:
    if how == "Count":
        return "Rows"
    pretty = humanize(measure_name)
    if how == "Sum":
        return f"Total {pretty.lower()}"
    return f"{how} {pretty.lower()}"


def _friendly_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date"
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if pd.api.types.is_integer_dtype(series):
        return "Integer"
    if pd.api.types.is_float_dtype(series):
        return "Number"
    return "Text"


def _is_identifier(name: str) -> bool:
    return bool(_ID_RE.search(normalize_name(name)))


def _is_measure_name(name: str) -> bool:
    return bool(_MEASURE_RE.search(normalize_name(name)))


def _is_percent_name(name: str) -> bool:
    return bool(_PERCENT_RE.search(normalize_name(name)))


def _promote_header(frame: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [str(column) for column in frame.columns]
    unnamed = sum(column.startswith("Unnamed") or column.strip() == "" for column in columns)
    if unnamed < max(2, math.ceil(len(columns) * 0.5)):
        return frame
    first = frame.iloc[0]
    if float(first.isna().mean()) > 0.5:
        return frame
    as_text = pd.Series(["" if pd.isna(value) else str(value).strip() for value in first.tolist()], dtype="string")
    numericish = float(pd.to_numeric(as_text.str.replace(",", "", regex=False), errors="coerce").notna().mean())
    if numericish > 0.5:
        return frame
    headers = []
    for index, value in enumerate(first.tolist()):
        if pd.isna(value) or str(value).strip() == "":
            headers.append(f"column_{index + 1}")
        else:
            headers.append(str(value).strip())
    promoted = frame.iloc[1:].copy()
    promoted.columns = headers
    notes.append("Used the first row as column names.")
    return promoted.reset_index(drop=True)


def _tidy_columns(frame: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    cleaned = []
    for index, column in enumerate(frame.columns):
        text = re.sub(r"\s+", " ", str(column)).strip(" :")
        if not text or text.lower().startswith("unnamed"):
            text = f"column_{index + 1}"
        cleaned.append(text)
    renamed = frame.copy()
    renamed.columns = _dedupe(cleaned)
    if renamed.shape[1] > 1:
        empty = [column for column in renamed.columns if renamed[column].isna().all()]
        if empty and len(empty) < renamed.shape[1]:
            renamed = renamed.drop(columns=empty)
            notes.append("Removed empty columns: " + ", ".join(str(column) for column in empty) + ".")
    return renamed


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        unique.append(name if count == 1 else f"{name}_{count}")
    return unique


def _coerce(frame: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    converted: list[str] = []
    out = frame.copy()
    for column in list(out.columns):
        series = out[column]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            if isinstance(series.dtype, pd.DatetimeTZDtype):
                out[column] = series.dt.tz_convert(None)
            continue
        if pd.api.types.is_numeric_dtype(series):
            parsed = _maybe_yyyymmdd(series, column)
            if parsed is not None:
                out[column] = parsed
                converted.append(column)
            continue
        if not is_text_like(series):
            continue
        if _is_identifier(column):
            continue
        parsed_bool = _maybe_bool(series)
        if parsed_bool is not None:
            out[column] = parsed_bool
            converted.append(column)
            continue
        parsed_dates = _maybe_dates(series, column)
        if parsed_dates is not None:
            out[column] = parsed_dates
            converted.append(column)
            continue
        parsed_numbers = _maybe_numeric(series)
        if parsed_numbers is not None:
            out[column] = parsed_numbers
            converted.append(column)
    if converted:
        notes.append("Interpreted columns as dates or numbers: " + ", ".join(converted) + ".")
    return out


def _maybe_dates(series: pd.Series, name: str) -> pd.Series | None:
    named = bool(_DATE_RE.search(normalize_name(name)))
    if not named and not _looks_like_dates(series):
        return None
    parsed = _naive_dates(series)
    present = int(series.notna().sum())
    if present == 0:
        return None
    reasonable = parsed.notna() & parsed.dt.year.between(1900, 2100)
    if reasonable.sum() / present >= 0.7:
        return parsed
    return None


def _looks_like_dates(series: pd.Series) -> bool:
    sample = series.dropna().astype("string").head(40)
    if len(sample) < 3:
        return False
    return bool(sample.str.contains(_DATE_TOKEN, na=False).mean() >= 0.8)


def _naive_dates(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert(None)
    return parsed


def _maybe_yyyymmdd(series: pd.Series, name: str) -> pd.Series | None:
    if not re.search(r"\b(date|day)\b", normalize_name(name)):
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    if not ((valid >= 19000101) & (valid <= 21001231)).all():
        return None
    parsed = pd.to_datetime(
        series.round().astype("Int64").astype("string"),
        errors="coerce",
        format="%Y%m%d",
    )
    if parsed.notna().sum() / int(series.notna().sum()) < 0.7:
        return None
    return parsed


def _maybe_bool(series: pd.Series) -> pd.Series | None:
    text = series.astype("string").str.strip().str.lower()
    present = text.dropna()
    if present.empty:
        return None
    mapping = {"true": True, "false": False, "yes": True, "no": False, "y": True, "n": False}
    if set(present.unique()).issubset(mapping):
        return text.map(mapping).astype("boolean")
    return None


def _maybe_numeric(series: pd.Series) -> pd.Series | None:
    text = series.astype("string").str.strip()
    markers = text.str.lower().isin(["", "nan", "none", "null", "n/a", "na", "-"]).fillna(False)
    missing = series.isna() | markers
    present = text[~missing]
    if present.empty:
        return None
    has_percent = bool(present.str.contains("%", regex=False).fillna(False).mean() >= 0.5)
    cleaned = (
        text.str.replace(r"[\$,]", "", regex=True)
        .str.replace("%", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    numeric = pd.to_numeric(cleaned.mask(missing), errors="coerce")
    comparable = int((~missing).sum())
    if comparable == 0 or numeric.notna().sum() / comparable < 0.9:
        return None
    if has_percent:
        numeric = numeric / 100.0
    return numeric


def _read_path(path: Path, sheet: str | None) -> tuple[pd.DataFrame, list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0), []
    return _read_bytes(path.name, path.read_bytes())


def _read_bytes(name: str, payload: bytes) -> tuple[pd.DataFrame, list[str]]:
    suffix = Path(name).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(payload)), []
    if suffix == ".json":
        return _read_json(payload), []
    if suffix == ".tsv":
        text, encoding_notes = _decode(payload)
        return pd.read_csv(io.StringIO(text), sep="\t", dtype="string"), encoding_notes
    text, notes = _decode(payload)
    text, stripped = _strip_preamble(text)
    if stripped:
        notes.append("Skipped a title line at the top of the file.")
    return _frame_from_text(text), notes


def _decode(payload: bytes) -> tuple[str, list[str]]:
    if not payload.strip():
        return "", ["The file is empty."]
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding), []
        except UnicodeDecodeError:
            continue
    return payload.decode("latin-1", errors="replace"), ["Some characters could not be decoded."]


def _strip_preamble(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    if len(lines) < 3:
        return text, False

    def score(line: str) -> int:
        return max(line.count(","), line.count(";"), line.count("\t"), line.count("|"))

    if score(lines[0]) + 1 < score(lines[1]) and score(lines[1]) >= 2:
        return "\n".join(lines[1:]), True
    return text, False


def _frame_from_text(text: str) -> pd.DataFrame:
    if not text.strip():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(io.StringIO(text), dtype="string")
    except pd.errors.ParserError:
        frame = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype="string")
    if frame.shape[1] == 1:
        header = str(frame.columns[0])
        if header.count(";") >= 1:
            return pd.read_csv(io.StringIO(text), sep=";", dtype="string")
        if "\t" in header:
            return pd.read_csv(io.StringIO(text), sep="\t", dtype="string")
        if header.count("|") >= 1:
            return pd.read_csv(io.StringIO(text), sep="|", dtype="string")
    return frame


def _read_json(payload: bytes) -> pd.DataFrame:
    try:
        return pd.read_json(io.BytesIO(payload))
    except ValueError:
        return pd.read_json(io.BytesIO(payload), lines=True)
