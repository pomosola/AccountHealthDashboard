"""Turn a folder of extracts into account health, hierarchy, and migration views.

The practicum folder is read as-is. Files are recognized by their names and
columns, then normalized. Sub-scores roll up with the weights in
``score_model`` when that file exists, and with equal weights when it does not.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

import dashboard_lib as lib

CHURN = "churn_health"
EXPANSION = "expansion"
OUTCOME_LABELS = {CHURN: "Churn health", EXPANSION: "Expansion potential"}

# Published model used when the extract has sub-scores but no weight file.
# Weights within an outcome sum to 1.
DEFAULT_MODEL = (
    (CHURN, "Product adoption", 0.30, "How consistently people complete the core workflow."),
    (CHURN, "Support experience", 0.20, "Case volume, reopen rate, and time to resolution."),
    (CHURN, "Engagement", 0.20, "Meetings, QBR attendance, and response to outreach."),
    (CHURN, "Commercial stability", 0.15, "Renewal timing, payment status, and contraction."),
    (CHURN, "Relationship coverage", 0.15, "Whether a champion and an economic buyer are active."),
    (EXPANSION, "Product whitespace", 0.30, "Purchased products versus the products the account could use."),
    (EXPANSION, "Engagement", 0.20, "Executive access and willingness to take an expansion meeting."),
    (EXPANSION, "Commercial headroom", 0.25, "Room in the contract and budget signals."),
    (EXPANSION, "Relationship coverage", 0.15, "A sponsor who can say yes to more spend."),
    (EXPANSION, "Adoption breadth", 0.10, "How many sites or teams are active, not just one champion."),
)

STAGE_ORDER = [
    "Onboarding",
    "Adoption",
    "Value realization",
    "Renewal",
    "Expansion",
    "Recovery",
]
BANDS = ("Healthy", "Watch", "Critical")
FRESH_DAYS = 45
SMALL_SAMPLE = 3

_HEALTH_HINTS = (
    "health",
    "churn",
    "expansion",
    "subscore",
    "sub score",
    "nps",
    "risk score",
    "driver",
    "touchpoint",
    "journey",
    "migration",
)


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def find_column(columns, aliases: list[str], exclude: set[str] | None = None) -> str | None:
    """Return the first column whose name matches an alias, most specific first."""
    blocked = exclude or set()
    pairs = [(norm(column), column) for column in columns if column not in blocked]
    for alias in aliases:
        for normalized, original in pairs:
            if normalized == alias:
                return original
    for alias in aliases:
        pattern = rf"\b{re.escape(alias)}\b"
        for normalized, original in pairs:
            if re.search(pattern, normalized):
                return original
    return None


def as_bool(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="boolean")
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.astype("string").str.strip().str.lower()
    return text.isin(["yes", "true", "y", "1"]).fillna(False)


def as_number(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.replace(r"[,$%]", "", regex=True).str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def as_dates(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def band(score: float | None) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "Not scored"
    if score >= 75:
        return "Healthy"
    if score >= 55:
        return "Watch"
    return "Critical"


def expansion_band(score: float | None) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "Not scored"
    if score >= 70:
        return "Ready"
    if score >= 45:
        return "Developing"
    return "Limited"


def domain_score(
    values: dict[str, float | None],
    weights: dict[str, float],
) -> tuple[float | None, list[dict]]:
    """Weighted mean of the domains that actually have a score.

    Missing domains are shown with a blank contribution. Their model weight is
    spread across the domains that were scored, and that effective weight is
    returned so the rollup can be checked by hand.
    """
    rows: list[dict] = []
    available: list[str] = []
    for domain, weight in weights.items():
        raw = values.get(domain)
        missing = raw is None or (isinstance(raw, float) and math.isnan(raw)) or pd.isna(raw)
        rows.append(
            {
                "domain": domain,
                "model_weight": float(weight),
                "sub_score": None if missing else float(raw),
                "missing": missing,
            }
        )
        if not missing:
            available.append(domain)
    total = sum(weights[domain] for domain in available)
    score_acc = 0.0
    for row in rows:
        if row["missing"] or total == 0:
            row["effective_weight"] = None
            row["contribution"] = None
            continue
        row["effective_weight"] = weights[row["domain"]] / total
        row["contribution"] = row["sub_score"] * row["effective_weight"]
        score_acc += row["contribution"]
    if not available:
        return None, rows
    score = round(score_acc, 1)
    rounded: list[float | None] = []
    for row in rows:
        if row["contribution"] is None:
            rounded.append(None)
        else:
            rounded.append(round(row["contribution"], 1))
    drift = round(score - sum(value for value in rounded if value is not None), 1)
    if drift != 0:
        index = max(
            (i for i, row in enumerate(rows) if row["contribution"] is not None),
            key=lambda i: rows[i]["contribution"],
        )
        rounded[index] = round((rounded[index] or 0) + drift, 1)
    for row, value in zip(rows, rounded):
        row["contribution"] = value
    return score, rows


def default_model_frame() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_MODEL, columns=["outcome", "domain", "weight", "definition"])


def weights_for(model: pd.DataFrame, outcome: str) -> dict[str, float]:
    subset = model[model["outcome"] == outcome]
    if subset.empty:
        return {}
    return {str(row.domain): float(row.weight) for row in subset.itertuples(index=False)}


def definition_for(model: pd.DataFrame, outcome: str, domain: str) -> str:
    subset = model[(model["outcome"] == outcome) & (model["domain"] == domain)]
    if subset.empty or "definition" not in subset.columns:
        return ""
    value = subset.iloc[0]["definition"]
    return "" if pd.isna(value) else str(value)


@dataclass
class SourceFile:
    label: str
    rows: int
    roles: list[str]
    notes: list[str] = field(default_factory=list)


@dataclass
class Portfolio:
    folder: str
    using_sample: bool
    files: list[SourceFile]
    errors: list[str]
    model: pd.DataFrame
    model_is_default: bool
    accounts: pd.DataFrame
    score_history: pd.DataFrame
    subscores: pd.DataFrame
    drivers: pd.DataFrame
    contacts: pd.DataFrame
    role_scores: pd.DataFrame
    touchpoints: pd.DataFrame
    migration: pd.DataFrame
    latest: pd.DataFrame
    actions: pd.DataFrame
    cutover: pd.Timestamp | None
    extract_as_of: pd.Timestamp | None
    go_forward_system: str
    legacy_system: str
    handoff_note: str
    ready: bool
    gaps: list[str]
    tables: dict = field(default_factory=dict)

    @property
    def file_labels(self) -> list[str]:
        return [item.label for item in self.files]


def _empty_portfolio(folder: str, using_sample: bool, errors: list[str], gaps: list[str]) -> Portfolio:
    empty = pd.DataFrame()
    return Portfolio(
        folder=folder,
        using_sample=using_sample,
        files=[],
        errors=errors,
        model=default_model_frame(),
        model_is_default=True,
        accounts=empty,
        score_history=empty,
        subscores=empty,
        drivers=empty,
        contacts=empty,
        role_scores=empty,
        touchpoints=empty,
        migration=empty,
        latest=empty,
        actions=empty,
        cutover=None,
        extract_as_of=None,
        go_forward_system="Salesforce",
        legacy_system="Legacy CRM",
        handoff_note="The go-forward system is Salesforce. Legacy-only items stay visible and are flagged.",
        ready=False,
        gaps=gaps,
    )


def _hints(label: str, columns) -> str:
    return norm(label) + " " + " ".join(norm(column) for column in columns)


def _looks_related(label: str, columns) -> bool:
    blob = _hints(label, columns)
    return any(token in blob for token in _HEALTH_HINTS)


def _take(frame: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    found: dict[str, pd.Series] = {}
    used: set[str] = set()
    for canonical, aliases in mapping.items():
        column = find_column(frame.columns, aliases, exclude=used)
        if column is None:
            continue
        found[canonical] = frame[column]
        used.add(column)
    if not found:
        return pd.DataFrame()
    return pd.DataFrame(found)


def _parse_model(frame: pd.DataFrame) -> pd.DataFrame | None:
    taken = _take(
        frame,
        {
            "outcome": ["outcome", "score", "score name", "health domain"],
            "domain": ["domain", "sub score", "subscore", "component", "driver group"],
            "weight": ["weight", "model weight"],
            "definition": ["definition", "description", "basis", "what it measures"],
        },
    )
    if not {"outcome", "domain", "weight"}.issubset(taken.columns):
        return None
    taken["weight"] = as_number(taken["weight"])
    if find_column(frame.columns, ["account id", "acct id", "accountid"]) is not None:
        return None
    taken = taken.dropna(subset=["domain", "weight"])
    if taken.empty or len(taken) > 40:
        return None
    taken["outcome"] = taken["outcome"].map(_canonical_outcome)
    taken = taken.dropna(subset=["outcome"])
    if "definition" not in taken.columns:
        taken["definition"] = ""
    return taken.reset_index(drop=True)


def _canonical_outcome(value: object) -> str | None:
    text = norm(value)
    if not text or text == "nan":
        return None
    if "expan" in text or "upsell" in text or "whitespace" in text or "cross sell" in text:
        return EXPANSION
    if "churn" in text or "health" in text or "risk" in text or "retain" in text:
        return CHURN
    return None


def _parse_accounts(frame: pd.DataFrame) -> pd.DataFrame:
    taken = _take(
        frame,
        {
            "account_id": ["account id", "acct id", "accountid", "salesforce account id", "sf account id"],
            "account_name": ["account name", "account", "customer name", "customer"],
            "parent_account_id": ["parent account id", "parent id", "parentaccountid"],
            "parent_account_name": ["parent account name", "parent account", "parent name"],
            "segment": ["segment", "tier", "account tier"],
            "region": ["region", "territory", "geo"],
            "industry": ["industry", "vertical"],
            "owner": ["owner", "account owner", "csm", "account manager", "owner name"],
            "arr": ["arr", "annual recurring revenue", "acv", "annual revenue", "revenue"],
            "renewal_date": ["renewal date", "renewal", "contract end", "renewal due"],
            "salesforce_account_id": ["salesforce account id", "sfid", "sfdc id", "new salesforce id"],
            "legacy_account_id": ["legacy account id", "legacy id", "old crm id", "legacy crm id"],
            "system_of_record": ["system of record", "source system", "system", "crm"],
            "migration_date": ["migration date", "cutover date", "migrated on"],
        },
    )
    if "account_name" not in taken.columns and "parent_account_id" not in taken.columns and "arr" not in taken.columns:
        return pd.DataFrame()
    if "account_id" not in taken.columns and "account_name" not in taken.columns:
        return pd.DataFrame()
    if "account_id" not in taken.columns:
        taken["account_id"] = taken["account_name"].map(lambda value: norm(value).replace(" ", "-"))
    if "account_name" not in taken.columns:
        taken["account_name"] = taken["account_id"]
    taken["account_id"] = taken["account_id"].map(_text_id)
    if "parent_account_id" in taken.columns:
        taken["parent_account_id"] = taken["parent_account_id"].map(_text_id)
    else:
        taken["parent_account_id"] = ""
    if "arr" in taken.columns:
        taken["arr"] = as_number(taken["arr"])
    else:
        taken["arr"] = pd.NA
    if "renewal_date" in taken.columns:
        taken["renewal_date"] = as_dates(taken["renewal_date"])
    if "migration_date" in taken.columns:
        taken["migration_date"] = as_dates(taken["migration_date"])
    for column in ("segment", "region", "industry", "owner", "parent_account_name", "salesforce_account_id", "legacy_account_id", "system_of_record"):
        if column not in taken.columns:
            taken[column] = ""
        else:
            taken[column] = taken[column].map(_text)
    taken = taken.dropna(subset=["account_id"])
    taken = taken[taken["account_id"] != ""]
    return taken.drop_duplicates("account_id").reset_index(drop=True)


def _parse_scores(frame: pd.DataFrame) -> pd.DataFrame:
    taken = _take(
        frame,
        {
            "account_id": ["account id", "acct id", "accountid", "account name", "account"],
            "as_of_date": ["as of date", "as of", "score date", "snapshot date", "period month", "month"],
            "churn_health": ["churn health", "health score", "account health", "retention score"],
            "churn_risk": ["churn risk", "risk score", "churn score"],
            "expansion_score": ["expansion score", "expansion potential", "expansion health", "upsell score"],
            "confidence": ["confidence", "score confidence", "confidence level"],
            "evidence_count": ["evidence count", "evidence", "sample size", "activity count"],
            "caveats": ["caveats", "caveat", "notes", "warnings"],
            "primary_source": ["primary source", "source system", "scored from"],
            "score_window": ["score window", "window", "basis window"],
        },
    )
    if "account_id" not in taken.columns:
        return pd.DataFrame()
    score_columns = [column for column in ("churn_health", "churn_risk", "expansion_score") if column in taken.columns]
    if not score_columns or not any(as_number(taken[column]).notna().any() for column in score_columns):
        return pd.DataFrame()
    taken["account_id"] = taken["account_id"].map(_text_id)
    if "as_of_date" in taken.columns:
        taken["as_of_date"] = as_dates(taken["as_of_date"])
    for column in ("churn_health", "churn_risk", "expansion_score", "evidence_count"):
        if column in taken.columns:
            taken[column] = as_number(taken[column])
    if "churn_health" not in taken.columns and "churn_risk" in taken.columns:
        taken["churn_health"] = 100 - taken["churn_risk"]
    if "churn_risk" not in taken.columns and "churn_health" in taken.columns:
        taken["churn_risk"] = (100 - taken["churn_health"]).round(1)
    return taken.dropna(subset=["account_id"]).reset_index(drop=True)


def _parse_subscores(frame: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    taken = _take(
        frame,
        {
            "account_id": ["account id", "acct id", "accountid", "account name", "account"],
            "as_of_date": ["as of date", "as of", "score date", "snapshot date", "period month", "month"],
            "outcome": ["outcome", "score", "score name"],
            "domain": ["domain", "component", "sub score name", "subscore name"],
            "sub_score": ["sub score", "subscore", "component score", "score value", "value"],
            "weight": ["weight", "model weight"],
            "evidence_count": ["evidence count", "evidence", "sample size"],
            "gap_flag": ["gap flag", "data gap", "missing flag", "gap"],
            "legacy_points": ["legacy points", "legacy only points"],
        },
    )
    long_ready = {"account_id", "domain", "sub_score"}.issubset(taken.columns)
    if long_ready:
        taken["account_id"] = taken["account_id"].map(_text_id)
        if "as_of_date" in taken.columns:
            taken["as_of_date"] = as_dates(taken["as_of_date"])
        taken["sub_score"] = as_number(taken["sub_score"])
        if "outcome" in taken.columns:
            taken["outcome"] = taken["outcome"].map(_canonical_outcome)
        else:
            taken["outcome"] = CHURN
        taken = taken.dropna(subset=["outcome", "domain"])
        if "gap_flag" in taken.columns:
            taken["gap_flag"] = as_bool(taken["gap_flag"])
            taken.loc[taken["gap_flag"], "sub_score"] = pd.NA
        if "evidence_count" in taken.columns:
            taken["evidence_count"] = as_number(taken["evidence_count"])
        if "weight" in taken.columns:
            taken["weight"] = as_number(taken["weight"])
        return taken.reset_index(drop=True)
    return _melt_wide_subscores(frame, model)


def _melt_wide_subscores(frame: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    if not _looks_related("", frame.columns) and frame.shape[1] > 40:
        return pd.DataFrame()
    account = find_column(frame.columns, ["account id", "acct id", "accountid", "account name", "account"])
    if account is None:
        return pd.DataFrame()
    as_of = find_column(frame.columns, ["as of date", "as of", "score date", "snapshot date", "period month", "month"], {account})
    rows: list[dict] = []
    domains = [(str(row.outcome), str(row.domain), norm(row.domain)) for row in model.itertuples(index=False)]
    for column in frame.columns:
        if column in {account, as_of}:
            continue
        normalized = norm(column)
        outcome = None
        if normalized.startswith("churn ") or normalized.startswith("health "):
            outcome = CHURN
        elif normalized.startswith("expansion ") or normalized.startswith("upsell "):
            outcome = EXPANSION
        stripped = re.sub(r"^(churn health|churn|health|expansion potential|expansion|upsell)\s+", "", normalized)
        stripped = re.sub(r"\s+(score|sub score|subscore)$", "", stripped)
        match = None
        for model_outcome, domain, domain_norm in domains:
            if stripped == domain_norm and (outcome is None or outcome == model_outcome):
                match = (model_outcome if outcome is None else outcome, domain)
                break
        if match is None or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        for _, record in frame.iterrows():
            rows.append(
                {
                    "account_id": _text_id(record[account]),
                    "as_of_date": pd.to_datetime(record[as_of], errors="coerce") if as_of else pd.NaT,
                    "outcome": match[0],
                    "domain": match[1],
                    "sub_score": pd.to_numeric(record[column], errors="coerce"),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _parse_drivers(frame: pd.DataFrame) -> pd.DataFrame:
    taken = _take(
        frame,
        {
            "account_id": ["account id", "acct id", "accountid", "account name", "account"],
            "as_of_date": ["as of date", "as of", "score date", "snapshot date", "period month", "month"],
            "outcome": ["outcome", "score", "score name"],
            "domain": ["domain", "component", "sub score name"],
            "driver": ["driver", "factor", "reason", "signal"],
            "impact": ["impact", "points", "contribution", "weight of evidence"],
            "evidence": ["evidence", "basis", "detail", "comment"],
            "legacy_only": ["legacy only", "legacy flag", "not in salesforce"],
            "source_system": ["source system", "system", "source"],
        },
    )
    if not {"account_id", "driver", "impact"}.issubset(taken.columns):
        return pd.DataFrame()
    taken["account_id"] = taken["account_id"].map(_text_id)
    if "as_of_date" in taken.columns:
        taken["as_of_date"] = as_dates(taken["as_of_date"])
    if "impact" in taken.columns:
        taken["impact"] = as_number(taken["impact"])
    if "outcome" in taken.columns:
        taken["outcome"] = taken["outcome"].map(_canonical_outcome).fillna(CHURN)
    else:
        taken["outcome"] = CHURN
    if "domain" not in taken.columns:
        taken["domain"] = "Overall"
    if "legacy_only" in taken.columns:
        taken["legacy_only"] = as_bool(taken["legacy_only"])
    else:
        taken["legacy_only"] = False
    if "source_system" not in taken.columns:
        taken["source_system"] = ""
    else:
        taken["source_system"] = taken["source_system"].map(_text)
    if "evidence" not in taken.columns:
        taken["evidence"] = ""
    else:
        taken["evidence"] = taken["evidence"].map(_text)
    return taken.dropna(subset=["account_id", "driver"]).reset_index(drop=True)


def _parse_contacts(frame: pd.DataFrame) -> pd.DataFrame:
    taken = _take(
        frame,
        {
            "contact_id": ["contact id", "person id", "contactid"],
            "account_id": ["account id", "acct id", "accountid", "account name", "account"],
            "contact_name": ["contact name", "full name", "person name", "person"],
            "title": ["title", "job title"],
            "role": ["role", "buying role", "contact role", "persona"],
            "email": ["email", "email address"],
            "salesforce_contact_id": ["salesforce contact id", "sf contact id"],
            "legacy_contact_id": ["legacy contact id", "legacy id"],
            "active": ["active", "is active"],
            "legacy_only": ["legacy only", "not in salesforce"],
        },
    )
    if "account_id" not in taken.columns or "contact_name" not in taken.columns:
        return pd.DataFrame()
    if "contact_id" not in taken.columns:
        taken["contact_id"] = [
            f"{_text_id(account)}-{index + 1}"
            for index, account in enumerate(taken["account_id"].tolist())
        ]
    taken["contact_id"] = taken["contact_id"].map(_text_id)
    taken["account_id"] = taken["account_id"].map(_text_id)
    for column in ("contact_name", "title", "role", "email", "salesforce_contact_id", "legacy_contact_id"):
        if column not in taken.columns:
            taken[column] = ""
        else:
            taken[column] = taken[column].map(_text)
    if "active" in taken.columns:
        taken["active"] = as_bool(taken["active"])
    else:
        taken["active"] = True
    if "legacy_only" in taken.columns:
        taken["legacy_only"] = as_bool(taken["legacy_only"])
    else:
        taken["legacy_only"] = False
    return taken.dropna(subset=["contact_id"]).reset_index(drop=True)


def _parse_role_scores(frame: pd.DataFrame) -> pd.DataFrame:
    taken = _take(
        frame,
        {
            "contact_id": ["contact id", "person id"],
            "account_id": ["account id", "acct id", "accountid", "account name"],
            "contact_name": ["contact name", "full name", "person"],
            "role": ["role", "buying role", "contact role"],
            "relationship_health": ["relationship health", "role health", "role score", "contact health"],
            "driver": ["driver", "factor", "reason"],
            "churn_influence": ["churn influence", "churn weight"],
            "expansion_influence": ["expansion influence", "expansion weight"],
            "last_touch_date": ["last touch date", "last touch", "last activity"],
            "sample_size": ["sample size", "touch count", "evidence count", "activities"],
            "confidence": ["confidence"],
            "caveat": ["caveat", "caveats", "note"],
        },
    )
    if "relationship_health" not in taken.columns:
        return pd.DataFrame()
    if "contact_id" not in taken.columns and "contact_name" not in taken.columns:
        return pd.DataFrame()
    if "contact_id" not in taken.columns:
        taken["contact_id"] = taken["contact_name"].map(_text_id)
    taken["contact_id"] = taken["contact_id"].map(_text_id)
    if "account_id" in taken.columns:
        taken["account_id"] = taken["account_id"].map(_text_id)
    taken["relationship_health"] = as_number(taken["relationship_health"])
    if "sample_size" in taken.columns:
        taken["sample_size"] = as_number(taken["sample_size"])
    if "last_touch_date" in taken.columns:
        taken["last_touch_date"] = as_dates(taken["last_touch_date"])
    for column in ("contact_name", "role", "driver", "churn_influence", "expansion_influence", "confidence", "caveat"):
        if column not in taken.columns:
            taken[column] = ""
        else:
            taken[column] = taken[column].map(_text)
    return taken.dropna(subset=["contact_id"]).reset_index(drop=True)


def _parse_touchpoints(frame: pd.DataFrame) -> pd.DataFrame:
    taken = _take(
        frame,
        {
            "touchpoint_id": ["touchpoint id", "activity id", "task id", "event id"],
            "account_id": ["account id", "acct id", "accountid", "account name", "account"],
            "contact_id": ["contact id", "person id"],
            "touch_date": ["touch date", "activity date", "event date", "date", "completed date"],
            "channel": ["channel", "type", "activity type", "touch type"],
            "journey_stage": ["journey stage", "stage", "customer stage"],
            "phase": ["phase", "migration phase", "period"],
            "system": ["system", "source system", "crm"],
            "legacy_only": ["legacy only", "not in salesforce"],
            "sentiment": ["sentiment", "tone"],
            "summary": ["summary", "subject", "description", "notes"],
            "owner": ["owner", "assigned to", "created by"],
        },
    )
    if "account_id" not in taken.columns or "touch_date" not in taken.columns:
        return pd.DataFrame()
    if not any(column in taken.columns for column in ("channel", "summary", "journey_stage")):
        return pd.DataFrame()
    taken["account_id"] = taken["account_id"].map(_text_id)
    taken["touch_date"] = as_dates(taken["touch_date"])
    if "legacy_only" in taken.columns:
        taken["legacy_only"] = as_bool(taken["legacy_only"])
    else:
        taken["legacy_only"] = False
    for column in ("touchpoint_id", "contact_id", "channel", "journey_stage", "phase", "system", "sentiment", "summary", "owner"):
        if column not in taken.columns:
            taken[column] = ""
        else:
            taken[column] = taken[column].map(_text)
    if (taken["touchpoint_id"] == "").all():
        taken["touchpoint_id"] = [f"TP-{index + 1:05d}" for index in range(len(taken))]
    return taken.dropna(subset=["touch_date"]).reset_index(drop=True)


def _parse_migration(frame: pd.DataFrame) -> pd.DataFrame:
    taken = _take(
        frame,
        {
            "account_id": ["account id", "acct id", "accountid", "account name", "account"],
            "period_month": ["period month", "month", "period", "as of date", "snapshot date"],
            "legacy_health": ["legacy health", "legacy score", "legacy crm health", "pre migration health"],
            "salesforce_health": ["salesforce health", "sf health", "post migration health", "new crm health"],
            "reconciled_health": ["reconciled health", "reconciled score", "bridged health", "health"],
            "source_used": ["source used", "winning source", "source"],
            "variance": ["variance", "difference", "gap"],
            "note": ["note", "notes", "comment"],
        },
    )
    if "period_month" not in taken.columns:
        return pd.DataFrame()
    if not any(column in taken.columns for column in ("legacy_health", "salesforce_health", "reconciled_health")):
        return pd.DataFrame()
    if "account_id" in taken.columns:
        taken["account_id"] = taken["account_id"].map(_text_id)
    taken["period_month"] = as_dates(taken["period_month"])
    for column in ("legacy_health", "salesforce_health", "reconciled_health", "variance"):
        if column in taken.columns:
            taken[column] = as_number(taken[column])
    for column in ("source_used", "note"):
        if column not in taken.columns:
            taken[column] = ""
        else:
            taken[column] = taken[column].map(_text)
    return taken.dropna(subset=["period_month"]).reset_index(drop=True)


def _parse_profile(frame: pd.DataFrame) -> dict:
    taken = _take(
        frame,
        {
            "cutover_date": ["cutover date", "migration date", "go live date"],
            "from_system": ["from system", "legacy system", "source system"],
            "to_system": ["to system", "go forward system", "target system"],
            "note": ["note", "notes", "handoff note"],
        },
    )
    if "cutover_date" not in taken.columns:
        return {}
    cutover = pd.to_datetime(taken["cutover_date"].iloc[0], errors="coerce")
    return {
        "cutover": None if pd.isna(cutover) else pd.Timestamp(cutover),
        "legacy_system": _text(taken["from_system"].iloc[0]) if "from_system" in taken.columns else "Legacy CRM",
        "go_forward_system": _text(taken["to_system"].iloc[0]) if "to_system" in taken.columns else "Salesforce",
        "note": _text(taken["note"].iloc[0]) if "note" in taken.columns else "",
    }


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return ""
    return str(value).strip()


def _text_id(value: object) -> str:
    text = _text(value)
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    return pd.concat(usable, ignore_index=True)


def _file_roles(label: str, frame: pd.DataFrame, model: pd.DataFrame) -> set[str]:
    blob = _hints(label, frame.columns)
    roles: set[str] = set()
    if "score model" in blob or {"outcome", "domain", "weight"}.issubset({norm(column) for column in frame.columns}):
        parsed = _parse_model(frame)
        if parsed is not None:
            roles.add("model")
    if "migration profile" in blob or "cutover" in blob and "cutover date" in blob:
        roles.add("profile")
    if _parse_drivers(frame).shape[0] and ("driver" in blob or "factor" in blob or "impact" in {norm(column) for column in frame.columns}):
        roles.add("drivers")
    subs = _parse_subscores(frame, model)
    if not subs.empty and ("domain" in subs.columns):
        roles.add("subscores")
    if not _parse_migration(frame).empty and any(token in blob for token in ("migration", "reconcil", "bridge", "legacy")):
        roles.add("migration")
    touches = _parse_touchpoints(frame)
    if not touches.empty and any(token in blob for token in ("touch", "activit", "journey", "event", "timeline")):
        roles.add("touchpoints")
    if not _parse_role_scores(frame).empty and any(token in blob for token in ("role", "relationship", "contact health")):
        roles.add("role_scores")
    contacts = _parse_contacts(frame)
    if not contacts.empty and any(token in blob for token in ("contact", "people", "person", "buyer")):
        roles.add("contacts")
    scores = _parse_scores(frame)
    if not scores.empty and any(token in blob for token in ("score", "health", "churn", "expansion", "risk")):
        roles.add("scores")
    accounts = _parse_accounts(frame)
    if not accounts.empty and any(token in blob for token in ("account", "hierarch", "parent", "customer")):
        if "contacts" not in roles or "parent" in blob or "arr" in blob:
            roles.add("accounts")
    if not roles and _looks_related(label, frame.columns):
        if not scores.empty:
            roles.add("scores")
        if not accounts.empty:
            roles.add("accounts")
    return roles


def _phase(when: pd.Timestamp | None, cutover: pd.Timestamp | None) -> str:
    if when is None or pd.isna(when) or cutover is None or pd.isna(cutover):
        return ""
    if when < pd.Timestamp(cutover).replace(day=1):
        return "Pre-migration"
    if when.to_period("M") == pd.Timestamp(cutover).to_period("M"):
        return "Cutover"
    return "Post-migration"


def _confidence(evidence: float | None, age_days: float | None, missing: bool, legacy_share: float, near_cutover: bool) -> str:
    evidence = 0 if evidence is None or pd.isna(evidence) else float(evidence)
    if missing or evidence < SMALL_SAMPLE or legacy_share > 0.25:
        return "Low"
    if (age_days is not None and age_days > FRESH_DAYS) or evidence < 8 or near_cutover or legacy_share > 0:
        return "Medium"
    return "High"


def _near_cutover(when: pd.Timestamp | None, cutover: pd.Timestamp | None) -> bool:
    if when is None or cutover is None or pd.isna(when) or pd.isna(cutover):
        return False
    return abs((pd.Timestamp(when) - pd.Timestamp(cutover)).days) <= 45


def _align_model(model: pd.DataFrame, subscores: pd.DataFrame, model_is_default: bool) -> tuple[pd.DataFrame, bool]:
    """Keep the published weights. If there is no weight file, use the standard model when the domains match it, otherwise equal weights on the domains that were actually supplied."""
    if subscores.empty or not model_is_default or "domain" not in subscores.columns:
        return model, model_is_default
    rows: list[tuple] = []
    outcomes = subscores["outcome"].dropna().unique().tolist() if "outcome" in subscores.columns else [CHURN]
    for outcome in outcomes:
        observed = [
            str(domain)
            for domain in subscores.loc[subscores["outcome"].eq(outcome), "domain"].dropna().unique()
        ]
        default = weights_for(model, outcome)
        if not observed:
            continue
        if set(observed).issubset(set(default)):
            for domain, weight in default.items():
                rows.append((outcome, domain, weight, definition_for(model, outcome, domain)))
            continue
        weight = 1 / len(observed)
        for domain in sorted(observed):
            rows.append(
                (
                    outcome,
                    domain,
                    weight,
                    "Equal weight. The folder did not include a score model for this domain.",
                )
            )
    if not rows:
        return model, model_is_default
    return pd.DataFrame(rows, columns=["outcome", "domain", "weight", "definition"]), True


def assemble(frames: list[tuple[str, pd.DataFrame, list[str]]], folder: str, using_sample: bool) -> Portfolio:
    """Build the working portfolio from already-loaded tables."""
    errors: list[str] = []
    model_parts: list[pd.DataFrame] = []
    profile: dict = {}
    classified: list[tuple[str, pd.DataFrame, list[str], set[str]]] = []
    preliminary_model = default_model_frame()
    for label, frame, notes in frames:
        roles = _file_roles(label, frame, preliminary_model)
        classified.append((label, frame, notes, roles))
        if "model" in roles:
            parsed = _parse_model(frame)
            if parsed is not None:
                model_parts.append(parsed)
        if "profile" in roles:
            profile = _parse_profile(frame) or profile
    model = _concat(model_parts) if model_parts else default_model_frame()
    model_is_default = not model_parts
    if model_parts:
        # Re-classify with the published domain names so wide sub-score columns match.
        classified = [
            (label, frame, notes, _file_roles(label, frame, model) or roles)
            for label, frame, notes, roles in classified
        ]

    accounts = _concat([_parse_accounts(frame) for label, frame, _, roles in classified if "accounts" in roles])
    score_history = _concat([_parse_scores(frame) for label, frame, _, roles in classified if "scores" in roles])
    subscores = _concat([_parse_subscores(frame, model) for label, frame, _, roles in classified if "subscores" in roles])
    drivers = _concat([_parse_drivers(frame) for label, frame, _, roles in classified if "drivers" in roles])
    contacts = _concat([_parse_contacts(frame) for label, frame, _, roles in classified if "contacts" in roles])
    role_scores = _concat([_parse_role_scores(frame) for label, frame, _, roles in classified if "role_scores" in roles])
    touchpoints = _concat([_parse_touchpoints(frame) for label, frame, _, roles in classified if "touchpoints" in roles])
    migration = _concat([_parse_migration(frame) for label, frame, _, roles in classified if "migration" in roles])

    if not accounts.empty:
        rank = accounts["account_name"].ne(accounts["account_id"]).astype(int)
        if "arr" in accounts.columns:
            rank = rank + pd.to_numeric(accounts["arr"], errors="coerce").notna().astype(int)
        if "parent_account_id" in accounts.columns:
            rank = rank + accounts["parent_account_id"].map(_text).ne("").astype(int)
        accounts = accounts.assign(_rank=rank).sort_values("_rank", ascending=False).drop_duplicates("account_id").drop(columns="_rank")
        accounts = _link_parents(accounts)
    if not model.empty:
        model = model.drop_duplicates(["outcome", "domain"], keep="first").reset_index(drop=True)
    model, model_is_default = _align_model(model, subscores, model_is_default)
    if not contacts.empty:
        richness = contacts["title"].map(_text).str.len().gt(0).astype(int) if "title" in contacts.columns else 0
        if "email" in contacts.columns:
            richness = richness + contacts["email"].map(_text).str.len().gt(0).astype(int)
        contacts = contacts.assign(_rank=richness).sort_values("_rank", ascending=False).drop_duplicates("contact_id").drop(columns="_rank")
    if not role_scores.empty:
        role_scores = role_scores.drop_duplicates(["contact_id"], keep="first")
    cutover = profile.get("cutover")
    if cutover is None and not accounts.empty and "migration_date" in accounts.columns:
        dates = pd.to_datetime(accounts["migration_date"], errors="coerce").dropna()
        if not dates.empty:
            cutover = pd.Timestamp(dates.mode().iloc[0])
    if not touchpoints.empty:
        touchpoints = touchpoints.copy()
        missing_phase = touchpoints["phase"].eq("") | touchpoints["phase"].isna()
        touchpoints.loc[missing_phase, "phase"] = [
            _phase(value, cutover) for value in touchpoints.loc[missing_phase, "touch_date"]
        ]

    files = [
        SourceFile(label=label, rows=int(len(frame)), roles=sorted(roles) or ["unmapped"], notes=list(notes))
        for label, frame, notes, roles in classified
    ]
    gaps = _coverage_gaps(accounts, score_history, subscores, drivers, contacts, role_scores, touchpoints, migration)
    portfolio = Portfolio(
        folder=folder,
        using_sample=using_sample,
        files=files,
        errors=errors,
        model=model.reset_index(drop=True),
        model_is_default=model_is_default,
        accounts=accounts,
        score_history=score_history,
        subscores=subscores,
        drivers=drivers,
        contacts=contacts,
        role_scores=role_scores,
        touchpoints=touchpoints,
        migration=migration,
        latest=pd.DataFrame(),
        actions=pd.DataFrame(),
        cutover=cutover,
        extract_as_of=None,
        go_forward_system=profile.get("go_forward_system") or "Salesforce",
        legacy_system=profile.get("legacy_system") or "Legacy CRM",
        handoff_note=profile.get("note")
        or "The go-forward system is Salesforce. Anything that exists only in the legacy CRM is flagged and is not a field the new owner should maintain.",
        ready=not accounts.empty or not score_history.empty or not subscores.empty,
        gaps=gaps,
    )
    portfolio.extract_as_of = _extract_as_of(portfolio)
    portfolio.tables = {label: frame for label, frame, _, _ in classified}
    portfolio.latest = _build_latest(portfolio)
    portfolio.role_scores = _annotate_roles(portfolio)
    portfolio.actions = build_actions(portfolio)
    return portfolio


def _link_parents(accounts: pd.DataFrame) -> pd.DataFrame:
    linked = accounts.copy()
    names = dict(zip(linked["account_id"], linked["account_name"]))
    linked["parent_account_id"] = linked["parent_account_id"].fillna("").map(_text_id)
    linked.loc[linked["parent_account_id"].eq(linked["account_id"]), "parent_account_id"] = ""
    known = set(names)
    linked["unresolved_parent"] = linked["parent_account_id"].ne("") & ~linked["parent_account_id"].isin(known)
    if "parent_account_name" not in linked.columns:
        linked["parent_account_name"] = ""
    resolved_name = linked["parent_account_id"].map(lambda value: names.get(value, ""))
    linked["parent_account_name"] = resolved_name.where(resolved_name.ne(""), linked["parent_account_name"].map(_text))
    child_parents = set(linked.loc[linked["parent_account_id"].isin(known), "parent_account_id"])

    def level(row: pd.Series) -> str:
        if row["account_id"] in child_parents:
            return "Parent"
        if row["parent_account_id"] in known:
            return "Child"
        return "Standalone"

    linked["hierarchy_level"] = linked.apply(level, axis=1)
    linked["family_id"] = linked.apply(
        lambda row: row["parent_account_id"] if row["parent_account_id"] in known else row["account_id"],
        axis=1,
    )
    linked["family_name"] = linked["family_id"].map(lambda value: names.get(value, value))
    return linked.reset_index(drop=True)


def _coverage_gaps(*frames: pd.DataFrame) -> list[str]:
    labels = [
        "accounts",
        "health score history",
        "sub-scores",
        "drivers",
        "contacts",
        "role scores",
        "touchpoints",
        "migration bridge",
    ]
    gaps = []
    for label, frame in zip(labels, frames):
        if frame is None or frame.empty:
            gaps.append(label)
    return gaps


def _extract_as_of(portfolio: Portfolio) -> pd.Timestamp | None:
    candidates: list[pd.Series] = []
    for frame, column in (
        (portfolio.score_history, "as_of_date"),
        (portfolio.subscores, "as_of_date"),
        (portfolio.touchpoints, "touch_date"),
        (portfolio.migration, "period_month"),
    ):
        if not frame.empty and column in frame.columns:
            candidates.append(pd.to_datetime(frame[column], errors="coerce"))
    if not candidates:
        return None
    combined = pd.concat(candidates, ignore_index=True).dropna()
    if combined.empty:
        return None
    return pd.Timestamp(combined.max())


def _build_latest(portfolio: Portfolio) -> pd.DataFrame:
    accounts = portfolio.accounts.copy()
    if accounts.empty and portfolio.score_history.empty and portfolio.subscores.empty:
        return pd.DataFrame()
    if accounts.empty:
        ids = []
        if not portfolio.score_history.empty:
            ids.extend(portfolio.score_history["account_id"].dropna().tolist())
        if not portfolio.subscores.empty:
            ids.extend(portfolio.subscores["account_id"].dropna().tolist())
        accounts = pd.DataFrame({"account_id": sorted(set(ids))})
        accounts["account_name"] = accounts["account_id"]
        accounts["parent_account_id"] = ""
        accounts["parent_account_name"] = ""
        accounts["family_id"] = accounts["account_id"]
        accounts["family_name"] = accounts["account_id"]
        accounts["hierarchy_level"] = "Standalone"
        accounts["arr"] = pd.NA
        for column in ("segment", "region", "industry", "owner", "system_of_record", "salesforce_account_id", "legacy_account_id"):
            accounts[column] = ""
    history = portfolio.score_history.copy()
    if not history.empty and "as_of_date" in history.columns:
        history = history.sort_values("as_of_date")
        published = history.groupby("account_id", as_index=False).tail(1)
    else:
        published = history
    computed = _latest_from_subscores(portfolio)
    latest = accounts.merge(computed, on="account_id", how="left", suffixes=("", "_computed"))
    if not published.empty:
        keep = [column for column in ("account_id", "as_of_date", "churn_health", "churn_risk", "expansion_score", "confidence", "evidence_count", "caveats", "primary_source", "score_window") if column in published.columns]
        latest = latest.merge(published[keep], on="account_id", how="left", suffixes=("", "_published"))
        for column in ("churn_health", "expansion_score", "churn_risk"):
            published_column = f"{column}_published"
            if published_column not in latest.columns:
                continue
            if column not in latest.columns:
                latest[column] = latest[published_column]
            else:
                mismatch = latest[column].notna() & latest[published_column].notna() & ((latest[column] - latest[published_column]).abs() > 0.6)
                latest["published_mismatch"] = mismatch if "published_mismatch" not in latest.columns else latest["published_mismatch"] | mismatch
                latest[column] = latest[column].where(latest[column].notna(), latest[published_column])
        if "as_of_date" not in latest.columns and "as_of_date_published" in latest.columns:
            latest["as_of_date"] = latest["as_of_date_published"]
        elif "as_of_date_published" in latest.columns:
            latest["as_of_date"] = latest["as_of_date"].where(latest["as_of_date"].notna(), latest["as_of_date_published"])
        for column in ("confidence", "evidence_count", "caveats", "primary_source", "score_window"):
            published_column = f"{column}_published" if f"{column}_published" in latest.columns else column
            if column not in latest.columns and published_column in latest.columns:
                latest[column] = latest[published_column]
    latest = _annotate_latest(latest, portfolio)
    latest = _add_family_rollups(latest)
    latest["_level_order"] = latest["hierarchy_level"].map({"Parent": 0, "Standalone": 0, "Child": 1}).fillna(2)
    return latest.sort_values(["family_name", "_level_order", "account_name"]).drop(columns="_level_order").reset_index(drop=True)


def _latest_from_subscores(portfolio: Portfolio) -> pd.DataFrame:
    subs = portfolio.subscores
    if subs.empty:
        return pd.DataFrame(columns=["account_id"])
    working = subs.copy()
    if "as_of_date" not in working.columns:
        working["as_of_date"] = pd.NaT
    working["as_of_date"] = pd.to_datetime(working["as_of_date"], errors="coerce")
    dated = working.dropna(subset=["as_of_date"])
    if dated.empty:
        snapshot = working
        as_of = {account: pd.NaT for account in working["account_id"].unique()}
    else:
        latest_dates = dated.groupby("account_id")["as_of_date"].max()
        as_of = latest_dates.to_dict()
        snapshot = dated.merge(latest_dates.rename("latest_date"), on="account_id")
        snapshot = snapshot[snapshot["as_of_date"].eq(snapshot["latest_date"])]
    rows = []
    drivers = portfolio.drivers
    for account_id, group in snapshot.groupby("account_id"):
        record: dict = {"account_id": account_id, "as_of_date": as_of.get(account_id, pd.NaT)}
        evidence_values = []
        for outcome in (CHURN, EXPANSION):
            weights = weights_for(portfolio.model, outcome)
            if not weights:
                continue
            values = {}
            outcome_rows = group[group["outcome"] == outcome] if "outcome" in group.columns else group.iloc[0:0]
            for _, item in outcome_rows.iterrows():
                values[str(item["domain"])] = None if pd.isna(item["sub_score"]) else float(item["sub_score"])
                if "evidence_count" in outcome_rows.columns and pd.notna(item.get("evidence_count")):
                    evidence_values.append(float(item["evidence_count"]))
            score, _ = domain_score(values, weights)
            if outcome == CHURN:
                record["churn_health"] = score
                record["churn_risk"] = None if score is None else round(100 - score, 1)
                record["churn_missing_domains"] = [domain for domain, value in values.items() if value is None and domain in weights]
                record["churn_missing_domains"] += [domain for domain in weights if domain not in values]
                record["churn_missing_domains"] = sorted(set(record["churn_missing_domains"]))
            else:
                record["expansion_score"] = score
                record["expansion_missing_domains"] = sorted(
                    {domain for domain in weights if values.get(domain) is None}
                )
            legacy_points = _legacy_points(drivers, account_id, record["as_of_date"], outcome)
            record[f"{outcome}_legacy_points"] = legacy_points
            if legacy_points and score is not None:
                adjusted = dict(values)
                # Spread the legacy points back off the domains that carried them.
                for domain, points in _legacy_points_by_domain(drivers, account_id, record["as_of_date"], outcome).items():
                    if adjusted.get(domain) is not None:
                        adjusted[domain] = max(0.0, adjusted[domain] - points)
                adjusted_score, _ = domain_score(adjusted, weights)
                record[f"{outcome}_salesforce_only"] = adjusted_score
        if evidence_values:
            record["evidence_count"] = int(round(sum(evidence_values) / len(evidence_values)))
        rows.append(record)
    return pd.DataFrame(rows)


def _legacy_points(drivers: pd.DataFrame, account_id: str, as_of, outcome: str) -> float:
    by_domain = _legacy_points_by_domain(drivers, account_id, as_of, outcome)
    return float(sum(by_domain.values()))


def _legacy_points_by_domain(drivers: pd.DataFrame, account_id: str, as_of, outcome: str) -> dict[str, float]:
    if drivers.empty or "legacy_only" not in drivers.columns:
        return {}
    subset = drivers[(drivers["account_id"] == account_id) & (drivers["legacy_only"])]
    if "outcome" in subset.columns:
        subset = subset[subset["outcome"].eq(outcome) | subset["outcome"].isna()]
    if subset.empty:
        return {}
    if "as_of_date" in subset.columns and as_of is not None and not pd.isna(as_of):
        dated = subset[pd.to_datetime(subset["as_of_date"], errors="coerce").eq(pd.Timestamp(as_of))]
        if not dated.empty:
            subset = dated
    totals: dict[str, float] = {}
    for domain, group in subset.groupby(subset["domain"].astype(str) if "domain" in subset.columns else ""):
        impacts = pd.to_numeric(group["impact"], errors="coerce") if "impact" in group.columns else pd.Series(dtype=float)
        total = float(impacts.fillna(0).sum())
        if total:
            totals[str(domain)] = total
    return totals


def _annotate_latest(latest: pd.DataFrame, portfolio: Portfolio) -> pd.DataFrame:
    frame = latest.copy()
    extract = portfolio.extract_as_of
    if "as_of_date" not in frame.columns:
        frame["as_of_date"] = pd.NaT
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    if "churn_health" not in frame.columns:
        frame["churn_health"] = pd.NA
    if "expansion_score" not in frame.columns:
        frame["expansion_score"] = pd.NA
    if "churn_risk" not in frame.columns:
        frame["churn_risk"] = pd.NA
    frame["churn_band"] = [band(_maybe_float(value)) for value in frame["churn_health"]]
    frame["expansion_band"] = [expansion_band(_maybe_float(value)) for value in frame["expansion_score"]]
    ages = []
    confidences = []
    caveats = []
    bases = []
    for _, row in frame.iterrows():
        as_of = row["as_of_date"]
        age = None if extract is None or pd.isna(as_of) else float((pd.Timestamp(extract) - pd.Timestamp(as_of)).days)
        ages.append(age)
        evidence = _maybe_float(row["evidence_count"]) if "evidence_count" in frame.columns else None
        missing = bool(row["churn_missing_domains"]) if "churn_missing_domains" in frame.columns and isinstance(row["churn_missing_domains"], list) and row["churn_missing_domains"] else False
        legacy_points = _maybe_float(row["churn_health_legacy_points"]) if "churn_health_legacy_points" in frame.columns else 0.0
        health = _maybe_float(row["churn_health"])
        legacy_share = 0.0 if not health else (legacy_points or 0) / health
        level = _confidence(evidence, age, missing or health is None, legacy_share, _near_cutover(as_of, portfolio.cutover))
        published = _text(row["confidence"]) if "confidence" in frame.columns else ""
        if published in {"High", "Medium", "Low"} and published != level and published == "High" and level == "Low":
            level_note = f"The file marks confidence High. The evidence on the row supports {level}."
        else:
            level_note = ""
        if published in {"High", "Medium", "Low"}:
            shown = published
        else:
            shown = level
        confidences.append(shown)
        notes = []
        if "caveats" in frame.columns and _text(row["caveats"]):
            notes.append(_text(row["caveats"]))
        if missing:
            missing_names = ", ".join(row["churn_missing_domains"])
            notes.append(
                f"Data gap: {missing_names} was not scored. Churn health uses the remaining domains and their weights were renormalized."
            )
        if age is not None and age > FRESH_DAYS:
            notes.append(
                f"Stale: this score is as of {pd.Timestamp(as_of).strftime('%d %b %Y')}, {int(age)} days before the extract date {pd.Timestamp(extract).strftime('%d %b %Y')}."
            )
        if evidence is not None and evidence < SMALL_SAMPLE:
            notes.append(f"Small sample: the score rests on {int(evidence)} pieces of evidence.")
        if _near_cutover(as_of, portfolio.cutover):
            notes.append(
                f"Migration window: scored within 45 days of the {pd.Timestamp(portfolio.cutover).strftime('%d %b %Y')} cutover. Treat this month as bridged."
            )
        if legacy_points:
            sf_only = _maybe_float(row["churn_health_salesforce_only"]) if "churn_health_salesforce_only" in frame.columns else None
            extra = f" Without those points the Salesforce-only churn health is {sf_only:.1f}." if sf_only is not None else ""
            share_note = " Confidence is held down because those legacy points are more than a quarter of the score." if legacy_share > 0.25 else ""
            notes.append(
                f"Legacy-only: {legacy_points:.0f} sub-score points come from drivers that were not rebuilt on {portfolio.go_forward_system}.{extra}{share_note}"
            )
        system = _text(row["system_of_record"]) if "system_of_record" in frame.columns else ""
        if system.lower().startswith("legacy"):
            notes.append(
                f"This account is still {system}. It is not a go-forward {portfolio.go_forward_system} record yet."
            )
        if level_note:
            notes.append(level_note)
        if "published_mismatch" in frame.columns and bool(row["published_mismatch"]):
            notes.append("The published score and the sub-score rollup differ by more than 0.6 points. The rollup is the number used here.")
        no_sf = _no_salesforce_touch(portfolio, str(row["account_id"]))
        if no_sf:
            notes.append(f"No {portfolio.go_forward_system} touchpoint after cutover. The journey still depends on the {portfolio.legacy_system} history.")
        caveats.append(notes)
        window = _text(row["score_window"]) if "score_window" in frame.columns else ""
        if not window:
            window = "Sub-scores on the as-of date, weighted by the published model."
            if portfolio.model_is_default:
                window = "Sub-scores on the as-of date. No weight file was in the folder, so domains are combined with the standard weights shipped with this app."
        source = _text(row["primary_source"]) if "primary_source" in frame.columns else ""
        if not source:
            source = portfolio.go_forward_system if not legacy_points else f"{portfolio.go_forward_system} plus flagged {portfolio.legacy_system} drivers"
        bases.append(
            {
                "basis": window,
                "source": source,
                "evidence": "not stated" if evidence is None else f"{int(evidence)} evidence points",
            }
        )
    frame["freshness_days"] = ages
    frame["confidence"] = confidences
    frame["caveat_list"] = caveats
    frame["basis"] = [item["basis"] for item in bases]
    frame["source"] = [item["source"] for item in bases]
    frame["evidence_label"] = [item["evidence"] for item in bases]
    frame["open_gaps"] = [
        len(value) if isinstance(value, list) else 0
        for value in (frame["churn_missing_domains"] if "churn_missing_domains" in frame.columns else [[]] * len(frame))
    ]
    return frame


def _no_salesforce_touch(portfolio: Portfolio, account_id: str) -> bool:
    if portfolio.cutover is None or portfolio.touchpoints.empty:
        return False
    touches = portfolio.touchpoints[portfolio.touchpoints["account_id"].eq(account_id)]
    if touches.empty:
        return False
    after = touches[pd.to_datetime(touches["touch_date"]) >= pd.Timestamp(portfolio.cutover)]
    if after.empty:
        return True
    systems = after["system"].astype(str).str.lower()
    return bool(systems.str.contains("salesforce|sfdc").sum() == 0 and systems.str.contains("legacy").any())


def _maybe_float(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_family_rollups(latest: pd.DataFrame) -> pd.DataFrame:
    frame = latest.copy()
    if "family_id" not in frame.columns:
        frame["family_id"] = frame["account_id"]
        frame["family_name"] = frame.get("account_name", frame["account_id"])
    family_rows = []
    for family_id, group in frame.groupby("family_id"):
        weights = pd.to_numeric(group["arr"], errors="coerce").fillna(0)
        total = float(weights.sum())
        def weighted(column: str) -> float | None:
            if column not in group.columns or total <= 0:
                values = pd.to_numeric(group[column], errors="coerce") if column in group.columns else pd.Series(dtype=float)
                values = values.dropna()
                return None if values.empty else round(float(values.mean()), 1)
            values = pd.to_numeric(group[column], errors="coerce")
            mask = values.notna() & weights.gt(0)
            if not mask.any():
                return None
            return round(float((values[mask] * weights[mask]).sum() / weights[mask].sum()), 1)
        family_rows.append(
            {
                "family_id": family_id,
                "family_churn_health": weighted("churn_health"),
                "family_expansion": weighted("expansion_score"),
                "family_arr": total if total else pd.NA,
                "family_accounts": int(len(group)),
            }
        )
    families = pd.DataFrame(family_rows)
    return frame.merge(families, on="family_id", how="left")


def _annotate_roles(portfolio: Portfolio) -> pd.DataFrame:
    roles = portfolio.role_scores.copy()
    if roles.empty:
        return roles
    touches = portfolio.touchpoints
    if "sample_size" not in roles.columns or roles["sample_size"].isna().all():
        if not touches.empty and "contact_id" in touches.columns:
            counts = touches.groupby("contact_id").size()
            roles["sample_size"] = roles["contact_id"].map(counts).fillna(0)
        else:
            roles["sample_size"] = pd.NA
    caveats = []
    confidences = []
    for _, row in roles.iterrows():
        notes = []
        existing = _text(row["caveat"]) if "caveat" in roles.columns else ""
        if existing:
            notes.append(existing)
        sample = _maybe_float(row["sample_size"]) if "sample_size" in roles.columns else None
        if sample is not None and sample < SMALL_SAMPLE and "small-sample" not in existing.lower():
            notes.append(
                f"Small-sample role: {int(sample)} touch{'es' if int(sample) != 1 else ''} in the extract. Do not treat this role score as stable."
            )
            level = "Low"
        elif sample is not None and sample < 8:
            level = "Medium"
        else:
            level = "High"
        published = _text(row["confidence"]) if "confidence" in roles.columns else ""
        confidences.append(published if published in {"High", "Medium", "Low"} else level)
        if "last_touch_date" in roles.columns and portfolio.extract_as_of is not None and pd.notna(row["last_touch_date"]):
            age = (pd.Timestamp(portfolio.extract_as_of) - pd.Timestamp(row["last_touch_date"])).days
            if age > FRESH_DAYS:
                notes.append(f"Last touch was {int(age)} days before the extract date.")
        caveats.append(notes)
    roles["confidence"] = confidences
    roles["caveat_list"] = caveats
    if not portfolio.contacts.empty:
        extra = portfolio.contacts[["contact_id", "contact_name", "title", "role", "account_id", "legacy_only"]].rename(
            columns={"role": "contact_role", "contact_name": "contact_name_master"}
        )
        roles = roles.merge(extra, on="contact_id", how="left", suffixes=("", "_contact"))
        if "account_id_contact" in roles.columns:
            roles["account_id"] = roles["account_id"].where(roles["account_id"].astype(str).str.len() > 0, roles["account_id_contact"])
        master = roles["contact_name_master"] if "contact_name_master" in roles.columns else ""
        current = roles["contact_name"].map(_text)
        roles["contact_name"] = current.where(current.ne(""), master if isinstance(master, str) else master.map(_text))
        if "role" in roles.columns and "contact_role" in roles.columns:
            roles["role"] = roles["role"].where(roles["role"].astype(str).str.len() > 0, roles["contact_role"])
    return roles


def build_actions(portfolio: Portfolio) -> pd.DataFrame:
    """Playbook rows for the operational and process-executor views."""
    if portfolio.latest.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    names = dict(zip(portfolio.latest["account_id"], portfolio.latest["account_name"]))
    for _, account in portfolio.latest.iterrows():
        health = _maybe_float(account["churn_health"])
        expansion = _maybe_float(account["expansion_score"])
        confidence = _text(account["confidence"])
        as_of = account["as_of_date"]
        as_of_label = "an unknown date" if pd.isna(as_of) else pd.Timestamp(as_of).strftime("%d %b %Y")
        common = {
            "account_id": account["account_id"],
            "account_name": account["account_name"],
            "owner": _text(account.get("owner", "")),
            "contact_name": "",
            "role": "",
            "as_of": as_of_label,
            "confidence": confidence or "Not stated",
            "score": "" if health is None else f"{health:.1f}",
        }
        if health is not None and health < 55:
            rows.append(
                {
                    **common,
                    "priority": 1,
                    "audience": "Operational",
                    "action": "Put this account on the recovery list and assign an owner for this week.",
                    "why": f"Churn health is {health:.1f} ({band(health)}). Confidence is {confidence or 'not stated'} as of {as_of_label}.",
                    "system": portfolio.go_forward_system,
                    "legacy_only": False,
                }
            )
            rows.append(
                {
                    **common,
                    "priority": 1,
                    "audience": "Process executor",
                    "action": f"Book a recovery call and log it on the {portfolio.go_forward_system} account.",
                    "why": f"Churn health is {health:.1f}. The call is the next touch, not a new score.",
                    "system": portfolio.go_forward_system,
                    "legacy_only": False,
                }
            )
        if expansion is not None and expansion >= 70 and (health is None or health >= 55):
            rows.append(
                {
                    **common,
                    "priority": 2,
                    "audience": "Operational",
                    "action": "Qualify the expansion and name the executive sponsor.",
                    "why": f"Expansion potential is {expansion:.1f} and churn health is not Critical.",
                    "system": portfolio.go_forward_system,
                    "legacy_only": False,
                }
            )
            rows.append(
                {
                    **common,
                    "priority": 2,
                    "audience": "Process executor",
                    "action": f"Open or update the expansion opportunity in {portfolio.go_forward_system} and attach the sponsor.",
                    "why": f"Expansion potential is {expansion:.1f}, as of {as_of_label}.",
                    "system": portfolio.go_forward_system,
                    "legacy_only": False,
                }
            )
        if int(account.get("open_gaps") or 0) > 0:
            gap_notes = [note for note in account["caveat_list"] if str(note).startswith("Data gap")] if isinstance(account["caveat_list"], list) else []
            rows.append(
                {
                    **common,
                    "priority": 1,
                    "audience": "Process executor",
                    "action": "Fill the missing sub-score before anyone acts on this number.",
                    "why": gap_notes[0] if gap_notes else "A domain has a data gap.",
                    "system": portfolio.go_forward_system,
                    "legacy_only": False,
                }
            )
        freshness = _maybe_float(account.get("freshness_days"))
        if freshness is not None and freshness > FRESH_DAYS:
            rows.append(
                {
                    **common,
                    "priority": 2,
                    "audience": "Operational",
                    "action": "Refresh the score before the next review.",
                    "why": f"The latest score is {int(freshness)} days older than the extract.",
                    "system": portfolio.go_forward_system,
                    "legacy_only": False,
                }
            )
        legacy_points = _maybe_float(account.get("churn_health_legacy_points")) or 0
        if legacy_points > 0:
            rows.append(
                {
                    **common,
                    "priority": 2,
                    "audience": "Process executor",
                    "action": f"Replace the legacy-only driver. Do not carry it into {portfolio.go_forward_system}.",
                    "why": f"{legacy_points:.0f} churn-health points still come from a legacy-only driver.",
                    "system": portfolio.legacy_system,
                    "legacy_only": True,
                }
            )
        system = _text(account.get("system_of_record", ""))
        if system.lower().startswith("legacy"):
            rows.append(
                {
                    **common,
                    "priority": 1,
                    "audience": "Process executor",
                    "action": f"Create or match this account in {portfolio.go_forward_system} before using the score there.",
                    "why": f"System of record is {system}.",
                    "system": portfolio.legacy_system,
                    "legacy_only": True,
                }
            )
        renewal = account["renewal_date"] if "renewal_date" in portfolio.latest.columns else pd.NaT
        if portfolio.extract_as_of is not None and pd.notna(renewal):
            days = (pd.Timestamp(renewal) - pd.Timestamp(portfolio.extract_as_of)).days
            if 0 <= days <= 120:
                rows.append(
                    {
                        **common,
                        "priority": 1 if days <= 60 else 2,
                        "audience": "Operational",
                        "action": "Start the renewal. The contract date is inside 120 days of the extract.",
                        "why": f"Renewal date {pd.Timestamp(renewal).strftime('%d %b %Y')} is {int(days)} days after the extract date.",
                        "system": portfolio.go_forward_system,
                        "legacy_only": False,
                    }
                )
    if not portfolio.role_scores.empty:
        for _, role in portfolio.role_scores.iterrows():
            sample = _maybe_float(role.get("sample_size"))
            relationship = _maybe_float(role.get("relationship_health"))
            role_name = _text(role.get("role"))
            contact_name = _text(role.get("contact_name"))
            account_id = _text(role.get("account_id"))
            common = {
                "account_id": account_id,
                "account_name": names.get(account_id, account_id),
                "owner": "",
                "contact_name": contact_name,
                "role": role_name,
                "as_of": "",
                "confidence": _text(role.get("confidence")) or "Low",
                "score": "" if relationship is None else f"{relationship:.0f}",
            }
            if sample is not None and sample < SMALL_SAMPLE:
                rows.append(
                    {
                        **common,
                        "priority": 3,
                        "audience": "Process executor",
                        "action": "Log more touches before treating this role score as stable.",
                        "why": f"{contact_name or 'This contact'} ({role_name or 'role'}) has {int(sample)} touch{'es' if int(sample) != 1 else ''}.",
                        "system": portfolio.go_forward_system,
                        "legacy_only": False,
                    }
                )
            if role_name == "Champion" and relationship is not None and relationship < 55:
                rows.append(
                    {
                        **common,
                        "priority": 1,
                        "audience": "Process executor",
                        "action": f"Re-establish the champion and record the meeting in {portfolio.go_forward_system}.",
                        "why": f"Champion relationship health is {relationship:.0f}. Driver: {_text(role.get('driver')) or 'not stated'}.",
                        "system": portfolio.go_forward_system,
                        "legacy_only": False,
                    }
                )
    if not rows:
        return pd.DataFrame()
    actions = pd.DataFrame(rows)
    if not portfolio.latest.empty:
        owners = portfolio.latest.set_index("account_id")["owner"]
        missing = actions["owner"].eq("") | actions["owner"].isna()
        actions.loc[missing, "owner"] = actions.loc[missing, "account_id"].map(owners)
    return actions.sort_values(["priority", "account_name", "audience"]).reset_index(drop=True)


def scope_accounts(
    latest: pd.DataFrame,
    segments: list[str] | None = None,
    owners: list[str] | None = None,
    bands: list[str] | None = None,
    families: list[str] | None = None,
) -> pd.DataFrame:
    if latest.empty:
        return latest
    frame = latest
    if segments:
        frame = frame[frame["segment"].isin(segments)]
    if owners:
        frame = frame[frame["owner"].isin(owners)]
    if bands:
        frame = frame[frame["churn_band"].isin(bands)]
    if families:
        frame = frame[frame["family_id"].isin(families)]
    return frame.copy()


def rollup_table(portfolio: Portfolio, account_id: str, outcome: str) -> pd.DataFrame:
    """One row per domain for the account's latest as-of, with the math visible."""
    weights = weights_for(portfolio.model, outcome)
    latest_row = portfolio.latest[portfolio.latest["account_id"].eq(account_id)]
    if latest_row.empty or not weights:
        return pd.DataFrame()
    as_of = latest_row.iloc[0]["as_of_date"]
    subs = portfolio.subscores
    values: dict[str, float | None] = {domain: None for domain in weights}
    evidence: dict[str, float | None] = {}
    if not subs.empty:
        group = subs[subs["account_id"].eq(account_id) & subs["outcome"].eq(outcome)]
        if "as_of_date" in group.columns and pd.notna(as_of):
            exact = group[pd.to_datetime(group["as_of_date"], errors="coerce").eq(pd.Timestamp(as_of))]
            if not exact.empty:
                group = exact
        for _, item in group.iterrows():
            domain = str(item["domain"])
            if domain in values:
                values[domain] = None if pd.isna(item["sub_score"]) else float(item["sub_score"])
                if "evidence_count" in group.columns:
                    evidence[domain] = None if pd.isna(item["evidence_count"]) else float(item["evidence_count"])
    _, detail = domain_score(values, weights)
    legacy = _legacy_points_by_domain(portfolio.drivers, account_id, as_of, outcome)
    rows = []
    for item in detail:
        domain = item["domain"]
        rows.append(
            {
                "Domain": domain,
                "Sub-score": None if item["sub_score"] is None else round(item["sub_score"], 1),
                "Model weight": item["model_weight"],
                "Effective weight": None if item["effective_weight"] is None else round(item["effective_weight"], 3),
                "Contribution": item["contribution"],
                "Evidence": evidence.get(domain),
                "Legacy points": legacy.get(domain, 0) or None,
                "Definition": definition_for(portfolio.model, outcome, domain),
                "Gap": "Data gap" if item["missing"] else "",
            }
        )
    return pd.DataFrame(rows)


def driver_table(portfolio: Portfolio, account_id: str, outcome: str) -> pd.DataFrame:
    drivers = portfolio.drivers
    if drivers.empty:
        return pd.DataFrame()
    latest_row = portfolio.latest[portfolio.latest["account_id"].eq(account_id)]
    if latest_row.empty:
        return pd.DataFrame()
    as_of = latest_row.iloc[0]["as_of_date"]
    group = drivers[drivers["account_id"].eq(account_id) & drivers["outcome"].eq(outcome)]
    if "as_of_date" in group.columns and pd.notna(as_of):
        exact = group[pd.to_datetime(group["as_of_date"], errors="coerce").eq(pd.Timestamp(as_of))]
        if not exact.empty:
            group = exact
    if group.empty:
        return pd.DataFrame()
    show = pd.DataFrame(
        {
            "Domain": group["domain"],
            "Driver": group["driver"],
            "Impact": group["impact"] if "impact" in group.columns else pd.NA,
            "Evidence": group["evidence"] if "evidence" in group.columns else "",
            "Source": group["source_system"] if "source_system" in group.columns else "",
            "Legacy only": ["Yes" if flag else "" for flag in group["legacy_only"]],
        }
    )
    return show.sort_values(["Domain", "Driver"]).reset_index(drop=True)


def account_history(portfolio: Portfolio, account_id: str) -> pd.DataFrame:
    history = portfolio.score_history
    if history.empty or "as_of_date" not in history.columns:
        return pd.DataFrame()
    group = history[history["account_id"].eq(account_id)].sort_values("as_of_date")
    return group


def portfolio_trend(portfolio: Portfolio, account_ids: list[str] | None = None) -> pd.DataFrame:
    history = portfolio.score_history
    if history.empty or "as_of_date" not in history.columns or "churn_health" not in history.columns:
        return pd.DataFrame()
    frame = history if not account_ids else history[history["account_id"].isin(account_ids)]
    if frame.empty:
        return pd.DataFrame()
    arr = portfolio.latest.set_index("account_id")["arr"] if not portfolio.latest.empty and "arr" in portfolio.latest.columns else pd.Series(dtype=float)
    frame = frame.copy()
    frame["arr"] = frame["account_id"].map(arr)
    rows = []
    for when, group in frame.groupby(pd.to_datetime(frame["as_of_date"])):
        weights = pd.to_numeric(group["arr"], errors="coerce").fillna(0)
        def weighted(column: str) -> float | None:
            if column not in group.columns:
                return None
            values = pd.to_numeric(group[column], errors="coerce")
            mask = values.notna()
            if not mask.any():
                return None
            if float(weights[mask].sum()) <= 0:
                return round(float(values[mask].mean()), 1)
            return round(float((values[mask] * weights[mask]).sum() / weights[mask].sum()), 1)
        rows.append(
            {
                "as_of_date": pd.Timestamp(when),
                "churn_health": weighted("churn_health"),
                "expansion_score": weighted("expansion_score") if "expansion_score" in group.columns else None,
                "accounts": int(group["account_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("as_of_date")


def suggested_account(latest: pd.DataFrame) -> str | None:
    if latest.empty:
        return None
    gaps = latest[latest["open_gaps"].fillna(0).astype(int) > 0] if "open_gaps" in latest.columns else latest.iloc[0:0]
    if not gaps.empty:
        return str(gaps.sort_values("arr", ascending=False).iloc[0]["account_id"])
    scored = latest[pd.to_numeric(latest["churn_health"], errors="coerce").notna()]
    pool = scored if not scored.empty else latest
    ordered = pool.assign(_health=pd.to_numeric(pool["churn_health"], errors="coerce").fillna(100), _arr=pd.to_numeric(pool["arr"], errors="coerce").fillna(0))
    ordered = ordered.sort_values(["_health", "_arr"], ascending=[True, False])
    return str(ordered.iloc[0]["account_id"])


def load_portfolio(folder: Path, using_sample: bool = False) -> Portfolio:
    if not folder.exists():
        return _empty_portfolio(str(folder), using_sample, [f"Folder not found: {folder}"], ["the folder itself"])
    refs, errors = lib.discover(folder)
    loaded: list[tuple[str, pd.DataFrame, list[str]]] = []
    for ref in refs:
        try:
            frame, notes = lib.load_dataset(ref)
        except lib.DataLoadError as exc:
            errors.append(str(exc))
            continue
        loaded.append((ref.label, frame, notes))
    portfolio = assemble(loaded, str(folder), using_sample)
    portfolio.errors = errors + portfolio.errors
    return portfolio


def is_sample_folder(folder: Path) -> bool:
    try:
        return folder.resolve() == (lib.bundled_data_dir() / "health").resolve()
    except OSError:
        return False


def folder_signature(folder: Path) -> str:
    if not folder.exists():
        return "missing"
    parts = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in lib.SUPPORTED_SUFFIXES and not path.name.startswith("~$"):
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def money(value) -> str:
    number = _maybe_float(value)
    if number is None:
        return "—"
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number >= 1_000_000:
        return f"{sign}${number / 1_000_000:.1f}M"
    if number >= 10_000:
        return f"{sign}${number / 1_000:.0f}K"
    return f"{sign}${number:,.0f}"


def pretty_date(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%d %b %Y")


def stamp() -> str:
    return datetime.now().strftime("%d %b %Y %H:%M")
