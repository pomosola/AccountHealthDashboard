"""Load the practicum health extracts without inventing scores the files do not contain.

Product health is 100 times one minus the calibrated 120-day churn probability.
Expansion opportunity score is 100 times the expansion probability. Neither is a
weighted rollup of sub-scores. Driver rows are signed log-odds contributions:
intercept plus the contributions, passed through a sigmoid, recovers the
probability. Product adoption is a separate snapshot, activated units over
contracted units, and its source extract date is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MIGRATION_REFERENCE = pd.Timestamp("2025-09-30")
SCORE_AS_OF = pd.Timestamp("2025-01-01")
GO_FORWARD = "New MCF Salesforce"
ID_COLUMNS = {
    "account_number",
    "parent_account_number",
    "candidate_parent_account_number",
    "validated_parent_account_number",
    "salesforce_opportunity_id",
    "source_salesforce_opportunity_id",
    "target_salesforce_opportunity_id",
    "contact",
    "contact_id",
    "case_number",
}


def is_practicum_pack(folder: Path) -> bool:
    return (folder / "product_health_score.csv").exists() and (folder / "score_context.csv").exists()


def sigmoid(log_odds: float) -> float:
    return float(1 / (1 + np.exp(-log_odds)))


def reconstruct_probability(intercept: float, contributions: pd.Series) -> float:
    """Intercept plus every driver contribution, then the logistic function."""
    return sigmoid(float(intercept) + float(pd.to_numeric(contributions, errors="coerce").fillna(0).sum()))


def _read(folder: Path, name: str, usecols: list[str] | None = None) -> pd.DataFrame:
    path = folder / name
    header = pd.read_csv(path, nrows=0).columns.tolist()
    dtype = {column: str for column in header if column in ID_COLUMNS or column.endswith("_id") or column.endswith("_number")}
    frame = pd.read_csv(path, dtype=dtype, usecols=usecols, low_memory=False)
    for column in frame.columns:
        if column in dtype:
            frame[column] = frame[column].fillna("").map(lambda value: "" if value in {"nan", "None"} else str(value).strip())
    return frame


@dataclass
class Pack:
    folder: str
    accounts: pd.DataFrame
    definitions: pd.DataFrame
    drivers: pd.DataFrame
    review: pd.DataFrame
    performance: pd.DataFrame
    calibration: pd.DataFrame
    capacity: pd.DataFrame
    contacts: pd.DataFrame
    contact_activity: pd.DataFrame
    products: pd.DataFrame
    events: pd.DataFrame
    monthly: pd.DataFrame
    migration: pd.DataFrame
    portfolio_migration: dict
    handoff: dict


def load_pack(folder: Path) -> Pack:
    health = _read(folder, "product_health_score.csv")
    adoption = _read(
        folder,
        "product_adoption_score.csv",
        [
            "account_number",
            "product_adoption_score",
            "score_status",
            "reason_codes",
            "verified_contracted_units",
            "verified_activated_units",
            "eligible_active_opportunities",
            "assessment_basis",
            "assessment_run_date",
            "release_generated_date",
            "source_extract_date",
        ],
    )
    state = _read(
        folder,
        "account_data_state.csv",
        [
            "account_number",
            "account_status",
            "platform",
            "parent_link_status",
            "parent_validation_status",
            "candidate_parent_account_number",
            "source_coverage",
            "freshness_status",
            "target_mapping_status",
            "live_lookup_status",
            "customer_role_status",
            "has_historical_model_scores",
            "legacy_touchpoint_events",
            "mcf_touchpoint_events",
            "first_recorded_touchpoint",
            "last_recorded_touchpoint",
            "migration_reference_date",
            "actual_account_migration_date",
            "state_interpretation",
            "withheld_active_opportunities",
            "administrative_end_date_records",
        ],
    )
    hierarchy = _read(
        folder,
        "account_hierarchy.csv",
        [
            "account_number",
            "account_name_token",
            "parent_account_token",
            "parent_account_number",
            "parent_link_status",
            "hierarchy_path",
            "recorded_depth",
            "parent_validation_status",
            "parent_investigation",
            "validated_parent_account_number",
            "candidate_parent_account_number",
        ],
    )
    accounts = hierarchy.merge(state.drop(columns=[c for c in ("parent_link_status", "parent_validation_status", "candidate_parent_account_number") if c in state.columns]), on="account_number", how="outer")
    accounts = accounts.merge(health, on="account_number", how="left").merge(adoption, on="account_number", how="left", suffixes=("", "_adoption"))
    accounts["product_health_score"] = pd.to_numeric(accounts["product_health_score"], errors="coerce")
    accounts["full_churn_probability_120d"] = pd.to_numeric(accounts["full_churn_probability_120d"], errors="coerce")
    accounts["expansion_probability_120d"] = pd.to_numeric(accounts["expansion_probability_120d"], errors="coerce")
    accounts["expansion_opportunity_score"] = pd.to_numeric(accounts["expansion_opportunity_score"], errors="coerce")
    accounts["upgrade_probability_120d"] = pd.to_numeric(accounts["upgrade_probability_120d"], errors="coerce")
    accounts["product_adoption_score"] = pd.to_numeric(accounts["product_adoption_score"], errors="coerce")
    accounts["churn_rank"] = accounts["full_churn_probability_120d"].rank(ascending=False, method="min")
    accounts["expansion_rank"] = accounts["expansion_probability_120d"].rank(ascending=False, method="min")
    accounts["upgrade_rank"] = accounts["upgrade_probability_120d"].rank(ascending=False, method="min")
    accounts["recorded_depth"] = pd.to_numeric(accounts["recorded_depth"], errors="coerce")

    context = _read(
        folder,
        "score_context.csv",
        [
            "score_name",
            "outcome",
            "score_scale",
            "definition",
            "assessment_as_of",
            "freshness_status",
            "confidence_basis",
            "caveats",
            "migration_caveat",
            "role_sample_caveat",
            "source_comparability_caveat",
            "score_status",
            "selected_model",
            "test_accounts",
            "test_positive_accounts",
            "history_start_inclusive",
            "outcome_end_exclusive",
            "target_mapping_status",
        ],
    )
    definitions = context.drop_duplicates("score_name").reset_index(drop=True)

    drivers = _read(folder, "account_prediction_drivers.csv")
    drivers["contribution_calibrated_log_odds"] = pd.to_numeric(drivers["contribution_calibrated_log_odds"], errors="coerce")
    drivers["calibrated_intercept"] = pd.to_numeric(drivers["calibrated_intercept"], errors="coerce")
    drivers["driver_rank"] = pd.to_numeric(drivers["driver_rank"], errors="coerce")
    drivers["is_top_three_driver"] = drivers["is_top_three_driver"].map(_as_bool)

    review = _read(folder, "opportunity_review_lists.csv")
    review["probability_120d"] = pd.to_numeric(review["probability_120d"], errors="coerce")
    review["rank"] = pd.to_numeric(review["rank"], errors="coerce")
    review["is_primary_top_50"] = review["is_primary_top_50"].map(_as_bool)
    review = review.merge(hierarchy[["account_number", "account_name_token"]], on="account_number", how="left")

    events = _read(
        folder,
        "customer_journey_touchpoints.csv",
        [
            "account_number",
            "event_at",
            "event_type",
            "journey_stage",
            "recorded_system",
            "legacy_only",
            "migration_reference_period",
            "date_meaning",
            "source_instance_context",
            "migration_date_basis",
        ],
    )
    events["event_at"] = pd.to_datetime(events["event_at"], errors="coerce")
    events["legacy_only"] = events["legacy_only"].map(_as_bool)
    events["month"] = events["event_at"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        events.groupby(["month", "recorded_system", "migration_reference_period", "legacy_only"], dropna=False)
        .size()
        .rename("events")
        .reset_index()
    )

    migration_rows = _read(
        folder,
        "migration_reconciliation.csv",
        [
            "account_number",
            "reconciliation_status",
            "verified",
            "administrative_end_date_flag",
            "reported_migrated_units",
            "reported_failed_units",
            "reported_native_units",
            "installed_migrated_units",
            "installed_failed_units",
            "installed_native_units",
            "opportunity_type",
            "issues",
            "contract_end_interpretation",
        ],
    )
    for column in (
        "reported_migrated_units",
        "reported_failed_units",
        "reported_native_units",
        "installed_migrated_units",
        "installed_failed_units",
        "installed_native_units",
    ):
        migration_rows[column] = pd.to_numeric(migration_rows[column], errors="coerce").fillna(0)
    migration_rows["verified"] = migration_rows["verified"].map(_as_bool)
    migration_rows["administrative_end_date_flag"] = migration_rows["administrative_end_date_flag"].map(_as_bool)
    linked = migration_rows[migration_rows["account_number"].ne("")]
    migration = linked.groupby("account_number", as_index=False).agg(
        opportunities=("opportunity_type", "size"),
        verified_opportunities=("verified", "sum"),
        withheld_opportunities=("verified", lambda s: int((~s).sum())),
        admin_end_dates=("administrative_end_date_flag", "sum"),
        reported_migrated_units=("reported_migrated_units", "sum"),
        reported_failed_units=("reported_failed_units", "sum"),
        reported_native_units=("reported_native_units", "sum"),
        installed_migrated_units=("installed_migrated_units", "sum"),
        installed_failed_units=("installed_failed_units", "sum"),
        installed_native_units=("installed_native_units", "sum"),
    )
    portfolio_migration = {
        "opportunities": int(len(migration_rows)),
        "unlinked_opportunities": int(migration_rows["account_number"].eq("").sum()),
        "verified": int(migration_rows["verified"].sum()),
        "withheld": int((~migration_rows["verified"]).sum()),
        "admin_end_dates": int(migration_rows["administrative_end_date_flag"].sum()),
        "reported_migrated_units": int(migration_rows["reported_migrated_units"].sum()),
        "reported_failed_units": int(migration_rows["reported_failed_units"].sum()),
        "reported_native_units": int(migration_rows["reported_native_units"].sum()),
    }

    crosswalk = _read(
        folder,
        "salesforce_handoff_crosswalk.csv",
        [
            "account_number",
            "legacy_only_contract_identifiers",
            "required_owner_action",
            "target_mapping_status",
            "live_lookup_status",
            "target_salesforce_account_id",
            "contract_identifier_basis",
            "account_link_status",
        ],
    )
    crosswalk["legacy_only_contract_identifiers"] = crosswalk["legacy_only_contract_identifiers"].map(_as_bool)
    handoff = {
        "rows": int(len(crosswalk)),
        "legacy_only_contracts": int(crosswalk["legacy_only_contract_identifiers"].sum()),
        "missing_account_number": int(crosswalk["account_link_status"].eq("missing_account_number").sum()),
        "blank_account_id": int(crosswalk["target_salesforce_account_id"].eq("").sum()) if "target_salesforce_account_id" in crosswalk.columns else int(len(crosswalk)),
        "owner_action": str(crosswalk["required_owner_action"].dropna().iloc[0]) if crosswalk["required_owner_action"].ne("").any() else "",
        "live_lookup_status": str(crosswalk["live_lookup_status"].dropna().iloc[0]) if len(crosswalk) else "",
        "target_mapping_status": str(crosswalk["target_mapping_status"].mode().iloc[0]) if len(crosswalk) else "",
        "legacy_accounts": int(crosswalk.loc[crosswalk["legacy_only_contract_identifiers"] & crosswalk["account_number"].ne(""), "account_number"].nunique()),
    }
    legacy_accounts = (
        crosswalk.loc[crosswalk["legacy_only_contract_identifiers"] & crosswalk["account_number"].ne("")]
        .groupby("account_number")
        .size()
        .rename("legacy_only_contracts")
        .reset_index()
    )
    accounts = accounts.merge(legacy_accounts, on="account_number", how="left")
    accounts["legacy_only_contracts"] = accounts["legacy_only_contracts"].fillna(0).astype(int)
    accounts = accounts.merge(migration, on="account_number", how="left")

    contacts = _read(
        folder,
        "contact_account_bridge.csv",
        [
            "contact",
            "account_number",
            "link_status",
            "source_rows",
            "distinct_interactions",
            "distinct_cases",
            "first_activity",
            "last_activity",
            "customer_role",
            "person_identity_verified",
            "token_type",
            "relationship_scope",
        ],
    )
    contacts["distinct_interactions"] = pd.to_numeric(contacts["distinct_interactions"], errors="coerce")
    contacts["distinct_cases"] = pd.to_numeric(contacts["distinct_cases"], errors="coerce")

    return Pack(
        folder=str(folder),
        accounts=accounts,
        definitions=definitions,
        drivers=drivers,
        review=review,
        performance=_read(folder, "model_performance.csv"),
        calibration=_calibration(folder),
        capacity=_read(folder, "review_capacity_metrics.csv"),
        contacts=contacts,
        contact_activity=_contact_activity(folder),
        products=_products(folder),
        events=events,
        monthly=monthly,
        migration=migration,
        portfolio_migration=portfolio_migration,
        handoff=handoff,
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _calibration(folder: Path) -> pd.DataFrame:
    frame = _read(folder, "calibration_summary.csv")
    for column in (
        "accounts",
        "positives",
        "min_score",
        "max_score",
        "observed_rate",
        "rate_ci_low",
        "rate_ci_high",
        "mean_predicted_probability",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _products(folder: Path) -> pd.DataFrame:
    return _read(
        folder,
        "account_product_relationships.csv",
        [
            "account_number",
            "opportunity_type",
            "mcf_contract_number",
            "legacy_contract_number",
            "legacy_product_name",
            "product_names",
            "product_types",
            "adoption_contract_verified",
            "adoption_contract_issues",
            "product_linkage_status",
            "product_type_status",
            "salesforce_identifier_scope",
            "source_extract_date",
        ],
    )


def _contact_activity(folder: Path) -> pd.DataFrame:
    frame = _read(
        folder,
        "contact_touchpoint_links.csv",
        [
            "contact",
            "account_number",
            "case_number",
            "link_status",
            "activity_date",
            "customer_role",
            "identity_basis",
            "data_role",
        ],
    )
    frame["activity_date"] = pd.to_datetime(frame["activity_date"], errors="coerce")
    return frame


def calibration_band(pack: Pack, outcome: str, probability: float | None) -> pd.Series | None:
    """Held-out band that contains this probability. The band rate is not an account interval."""
    if probability is None or pd.isna(probability):
        return None
    rows = pack.calibration[pack.calibration["outcome"].eq(outcome)]
    hit = rows[(rows["min_score"] <= probability) & (rows["max_score"] >= probability)]
    if hit.empty:
        return None
    return hit.iloc[0]


def definition(pack: Pack, score_name: str) -> pd.Series | None:
    rows = pack.definitions[pack.definitions["score_name"].eq(score_name)]
    if rows.empty:
        return None
    return rows.iloc[0]


def account_drivers(pack: Pack, account_number: str, outcome: str) -> pd.DataFrame:
    rows = pack.drivers[pack.drivers["account_number"].eq(account_number) & pack.drivers["outcome"].eq(outcome)].copy()
    return rows.sort_values("driver_rank")


def children_of(pack: Pack, account_number: str) -> pd.DataFrame:
    parent_cols = [column for column in ("parent_account_number", "candidate_parent_account_number") if column in pack.accounts.columns]
    if not parent_cols:
        return pack.accounts.iloc[0:0]
    mask = False
    for column in parent_cols:
        mask = mask | pack.accounts[column].eq(account_number)
    return pack.accounts.loc[mask]


def scope_accounts(accounts: pd.DataFrame, platform: str, status: str, coverage: str, scored_only: bool) -> pd.DataFrame:
    frame = accounts
    if platform and platform != "All":
        frame = frame[frame["platform"].eq(platform)]
    if status and status != "All":
        frame = frame[frame["account_status"].eq(status)]
    if coverage and coverage != "All":
        frame = frame[frame["source_coverage"].eq(coverage)]
    if scored_only:
        frame = frame[frame["full_churn_probability_120d"].notna()]
    return frame
