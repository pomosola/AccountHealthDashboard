"""Pages for the practicum extracts.

Home is the book summary. The sidebar moves to one account, the journey, or
the Salesforce handoff. Leadership, process executors, support, and product
read the same pages and are told which number is theirs. Scores keep the
published definition, the as-of date, and the caveats in score_context.csv.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import practicum_pack as packlib

CHART = {"displaylogo": False, "displayModeBar": False, "responsive": True}
PAGES = ["Home", "Health scores", "Hierarchy and people", "Journey and migration", "Handoff", "Source tables"]


@st.cache_data(show_spinner="Reading the practicum extracts")
def cached_pack(folder: str, signature: str) -> packlib.Pack:
    del signature
    return packlib.load_pack(Path(folder))


def render(folder: Path) -> None:
    if st.session_state.get("page") not in PAGES:
        st.session_state["page"] = "Home"
    st.sidebar.radio("Go to", PAGES, key="page")
    st.sidebar.caption("Home is the summary. Health scores opens one account. Journey is for support. Handoff is for a Salesforce write.")
    data = cached_pack(str(folder.resolve()), _signature(folder))
    with st.sidebar.expander("Narrow the book"):
        scoped = filters(data)
    with st.sidebar.expander("Data folder"):
        st.text_input("Folder", key="data_folder")
        st.button("Reload files", on_click=cached_pack.clear)
    page = st.session_state["page"]
    if page == "Home":
        home(data, scoped)
    elif page == "Health scores":
        health(data, scoped)
    elif page == "Hierarchy and people":
        hierarchy(data, scoped)
    elif page == "Journey and migration":
        journey(data, scoped)
    elif page == "Handoff":
        handoff(data, scoped)
    else:
        sources(folder)


def filters(data: packlib.Pack) -> pd.DataFrame:
    accounts = data.accounts
    platform = st.selectbox("Platform", ["All"] + sorted(v for v in accounts["platform"].dropna().unique() if str(v).strip()))
    status = st.selectbox("Account status", ["All"] + sorted(v for v in accounts["account_status"].dropna().unique() if str(v).strip()))
    coverage = st.selectbox("Touchpoint coverage", ["All"] + sorted(v for v in accounts["source_coverage"].dropna().unique() if str(v).strip()))
    scored = st.checkbox("Only accounts with a historical score", value=False)
    scoped = packlib.scope_accounts(accounts, platform, status, coverage, scored)
    st.caption(f"{len(scoped):,} of {len(accounts):,} accounts in scope.")
    return scoped


def home(data: packlib.Pack, scoped: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class='hero'>
          <div class='eyebrow'>Applied analytics practicum · supplied extracts</div>
          <h1>Account health handoff</h1>
          <p>This page is the book summary for 1 Jan 2025. Use Go to in the sidebar when you want one account, the journey, or the Salesforce handoff.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _audiences()
    _kpis(data, scoped)
    st.markdown("### Book summary")
    st.caption("Leadership: read the model table before comparing accounts. Expansion is the ranking that held up. Product health is the churn probability flipped onto a 0–100 scale, and that ranking did not beat chance.")
    _performance(data)
    _calibration_table(data)
    _portfolio_timeline(data)
    scored = scoped[scoped["expansion_probability_120d"].notna()]
    if not scored.empty:
        figure = px.histogram(scored, x="expansion_probability_120d", nbins=40, title="Expansion probability on 1 Jan 2025", color_discrete_sequence=["#003057"])
        figure.update_layout(height=320, margin=dict(l=8, r=8, t=48, b=8), xaxis_title="Probability", yaxis_title="Accounts")
        figure.update_xaxes(tickformat=".0%", automargin=True)
        st.plotly_chart(figure, width="stretch", theme="streamlit", config=CHART, key="exp-hist")
        st.caption("Churn probabilities sit in a narrow band. A 0–100 health chart would draw almost a single bar, so expansion is the portfolio chart.")
    st.markdown("### Where to go next")
    st.markdown(
        "Open **Health scores** to read one account after this summary. "
        "**Journey and migration** is the support timeline. "
        "**Hierarchy and people** is contact tokens and product rows. "
        "**Handoff** is the checklist before a Salesforce write."
    )
    with st.expander("What each file is for"):
        st.markdown(
            """
            - `product_health_score.csv` — churn probability, product health, expansion, and upgrade as of 1 Jan 2025.
            - `account_prediction_drivers.csv` — log-odds contributions that add back to those probabilities.
            - `score_context.csv` — definition, freshness, confidence basis, and caveats for every score.
            - `product_adoption_score.csv` — deployment completion. Unscored and reconciliation-required rows stay visible.
            - `account_hierarchy.csv` — candidate parent links. Validated parent IDs are blank.
            - `contact_account_bridge.csv` — point-of-contact tokens. Customer role is unknown.
            - `customer_journey_touchpoints.csv` — dated events, with legacy-only flagged.
            - `migration_reconciliation.csv` — migrated, failed, and native units. An administrative end date is not churn.
            - `salesforce_handoff_crosswalk.csv` — new-MCF opportunity IDs. Account.Id was not supplied. Live lookup was not performed.
            - `opportunity_review_lists.csv` — the historical top-50 expansion and upgrade queues.
            - `model_performance.csv` and `calibration_summary.csv` — held-out quality, not an account-level confidence interval.
            """
        )


def health(data: packlib.Pack, scoped: pd.DataFrame) -> None:
    st.markdown(
        "<div class='hero'><div class='eyebrow'>One account, after the book</div><h1>Health scores</h1>"
        "<p>The summary is first. The account picker is under it. Leadership and process executors read the same score differently.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Leadership.** Use the table to decide whether a ranking is worth acting on. Expansion separated accounts that expanded from those that did not. Full churn did not. Product health is 100 × (1 − the churn probability), so a high health number is not a second finding.")
    st.markdown("**Process executors.** Pick one account below. The drivers show what pushed that probability. They are not a call list. Before a Salesforce write, open Handoff. There is no contact-level score.")
    st.markdown("**Support** should use Journey and migration for events, not this score. **Product** should read the adoption card on the account, which is installed units over contracted units and is separate from product health.")
    _performance(data)
    _calibration_table(data)
    st.markdown("### Accounts to review")
    st.caption("Leadership and process executors can start from this list. It is the historical top 50 for expansion and for upgrade.")
    _operational_home(data, scoped)
    st.markdown("### Explore an account")
    row = _pick(data, scoped, "health-account")
    if row is None:
        return
    _account_line(row)
    churn_p = _num(row.get("full_churn_probability_120d"))
    health_score = _num(row.get("product_health_score"))
    expansion_p = _num(row.get("expansion_probability_120d"))
    expansion_s = _num(row.get("expansion_opportunity_score"))
    upgrade_p = _num(row.get("upgrade_probability_120d"))
    adoption = _num(row.get("product_adoption_score"))
    columns = st.columns(4)
    columns[0].metric("Churn probability, 120 days", _pct(churn_p), delta=_rank(row.get("churn_rank"), "highest risk"), delta_color="off")
    columns[1].metric("Product health", _points(health_score), delta="100 × (1 − churn probability)", delta_color="off")
    columns[2].metric("Expansion probability", _pct(expansion_p), delta=_rank(row.get("expansion_rank"), "highest likelihood"), delta_color="off")
    columns[3].metric("Adoption", "—" if adoption is None else f"{adoption:.0f}", delta=_blank(row.get("score_status"), "no status"), delta_color="off")
    st.caption(f"Upgrade probability {_pct(upgrade_p)} · rank {_rank(row.get('upgrade_rank'), 'highest likelihood')}. Expansion score { _points(expansion_s) } is 100 × the expansion probability. It is not combined with product health.")
    outcome = st.radio(
        "Drivers for",
        ["full_churn", "expansion", "upgrade"],
        format_func=lambda value: {"full_churn": "Churn", "expansion": "Expansion", "upgrade": "Upgrade"}[value],
        horizontal=True,
        key="driver_outcome",
    )
    _basis(data, {"full_churn": "full_churn_probability_120d", "expansion": "expansion_probability_120d", "upgrade": "upgrade_probability_120d"}[outcome])
    _account_band(data, outcome, churn_p if outcome == "full_churn" else expansion_p if outcome == "expansion" else upgrade_p)
    if outcome == "full_churn":
        st.caption("Product health on the card above is that probability rescaled. It is not a second model, and it is not a weighted average of domain scores.")
    _drivers(data, str(row["account_number"]), outcome, churn_p if outcome == "full_churn" else expansion_p if outcome == "expansion" else upgrade_p)
    _adoption_block(data, row)
    st.caption("Product: adoption is deployment completion. A blank extract date means the snapshot date was not supplied. Process executors: reason codes are holds, not a score of zero.")


def hierarchy(data: packlib.Pack, scoped: pd.DataFrame) -> None:
    st.markdown(
        "<div class='hero'><div class='eyebrow'>Support and product</div><h1>Hierarchy and people</h1>"
        "<p>Support reads the contact tokens. Product reads the contract rows. Parent links stay candidates until an owner confirms them.</p></div>",
        unsafe_allow_html=True,
    )
    row = _pick(data, scoped, "tree-account")
    if row is None:
        return
    _account_line(row)
    parent_id = str(row.get("parent_account_number") or row.get("candidate_parent_account_number") or "")
    parent_name = ""
    if parent_id:
        match = data.accounts[data.accounts["account_number"].eq(parent_id)]
        if not match.empty:
            parent_name = str(match.iloc[0].get("account_name_token") or "")
    st.markdown(
        f"**Path.** {html.escape(str(row.get('hierarchy_path') or row.get('account_name_token') or row['account_number']))}"
    )
    st.caption(
        f"Parent link: {row.get('parent_link_status') or '—'}. "
        f"Validation: {row.get('parent_validation_status') or '—'}. "
        f"Candidate parent: {parent_id or 'none'} {parent_name}. "
        f"Validated parent: {row.get('validated_parent_account_number') or 'blank'}. "
        f"Recorded depth: {row.get('recorded_depth') if pd.notna(row.get('recorded_depth')) else '—'}."
    )
    if str(row.get("parent_investigation") or "").strip():
        st.caption(str(row.get("parent_investigation")))
    kids = packlib.children_of(data, str(row["account_number"]))
    st.markdown(f"**Child accounts recorded under this account: {len(kids)}**")
    if kids.empty:
        st.caption("No child account points at this account number. Most accounts have no parent recorded.")
    else:
        show = kids[["account_number", "account_name_token", "account_status", "parent_validation_status", "product_health_score", "expansion_probability_120d", "product_adoption_score", "score_status"]].copy()
        show.columns = ["Account", "Token", "Status", "Parent validation", "Product health", "Expansion probability", "Adoption", "Adoption status"]
        st.dataframe(show, width="stretch", hide_index=True)
    people = data.contacts[data.contacts["account_number"].eq(str(row["account_number"]))]
    st.markdown(f"**Support: point-of-contact tokens on this account: {len(people)}**")
    st.caption("Support can see who was recorded on a case. These are source tokens. Customer role was not supplied, and the person was not verified.")
    role_note = packlib.definition(data, "product_health_score")
    if role_note is not None:
        st.caption(str(role_note.get("role_sample_caveat") or ""))
    if people.empty:
        st.caption("No contact token is linked to this account in the case bridge.")
    else:
        table = people[["contact", "token_type", "link_status", "customer_role", "person_identity_verified", "distinct_interactions", "distinct_cases", "first_activity", "last_activity"]].copy()
        table.columns = ["Token", "Token type", "Link", "Customer role", "Person verified", "Interactions", "Cases", "First activity", "Last activity"]
        st.dataframe(table, width="stretch", hide_index=True, height=360)
        st.caption(str(people.iloc[0].get("relationship_scope") or "Links are recorded interactions, not a current contact directory."))
    _activity(data, str(row["account_number"]))
    _products(data, str(row["account_number"]))


def journey(data: packlib.Pack, scoped: pd.DataFrame) -> None:
    st.markdown(
        "<div class='hero'><div class='eyebrow'>Support</div><h1>Journey and migration</h1>"
        "<p>Support uses this page for recorded events. Leadership can read the book timeline as coverage, not as proof that migration is finished. The dashed line is 30 Sep 2025, an administrative reference, not an account migration date.</p></div>",
        unsafe_allow_html=True,
    )
    _portfolio_timeline(data)
    migration = data.portfolio_migration
    st.markdown(
        f"**Release reconciliation.** {migration['opportunities']:,} opportunity rows · "
        f"{migration['verified']:,} verified in the release · {migration['withheld']:,} withheld · "
        f"{migration['admin_end_dates']:,} carry the 30 Sep 2025 administrative end date. "
        f"Reported units: migrated {migration['reported_migrated_units']:,}, failed {migration['reported_failed_units']:,}, native {migration['reported_native_units']:,}."
    )
    st.caption("An administrative end date is the bulk-update reference. Read it as that date. Failed units are a source category.")
    st.markdown("### One account")
    row = _pick(data, scoped, "journey-account")
    if row is None:
        return
    _account_line(row)
    events = data.events[data.events["account_number"].eq(str(row["account_number"]))].sort_values("event_at")
    legacy = int(events["legacy_only"].sum()) if not events.empty else 0
    st.caption(
        f"{len(events):,} recorded events · {legacy:,} legacy-only · "
        f"coverage {row.get('source_coverage') or '—'} · "
        f"first { _date(row.get('first_recorded_touchpoint')) } · last { _date(row.get('last_recorded_touchpoint')) }. "
        "No recorded event is not proof that nothing happened."
    )
    if events.empty:
        st.info("This account has no touchpoints in the extract.")
    else:
        monthly = events.groupby([events["event_at"].dt.to_period("M").dt.to_timestamp(), "recorded_system"], dropna=False).size().rename("events").reset_index()
        figure = px.bar(monthly, x="event_at", y="events", color="recorded_system", title="Recorded events by month", color_discrete_sequence=["#5C6B7A", "#003057", "#C65D3B", "#B3A369"])
        figure.add_vline(x=packlib.MIGRATION_REFERENCE, line_dash="dash", line_color="#C65D3B")
        figure.add_annotation(x=packlib.MIGRATION_REFERENCE, y=1.04, yref="paper", text="Reference 30 Sep 2025", showarrow=False, font=dict(color="#C65D3B", size=12))
        figure.update_layout(height=380, margin=dict(l=8, r=8, t=48, b=8), legend=dict(orientation="h", y=-0.22), barmode="stack")
        figure.update_xaxes(automargin=True)
        figure.update_yaxes(automargin=True, title="Events")
        st.plotly_chart(figure, width="stretch", theme="streamlit", config=CHART, key=f"journey-{row['account_number']}")
        period = events["migration_reference_period"].value_counts()
        st.caption("Relative to the reference date: " + ", ".join(f"{name} {int(count):,}" for name, count in period.items()))
        show = events[["event_at", "journey_stage", "event_type", "recorded_system", "legacy_only", "migration_reference_period", "date_meaning"]].tail(40).copy()
        show.columns = ["When", "Stage", "Event", "System", "Legacy only", "Reference period", "What the date means"]
        show["When"] = show["When"].map(lambda value: "—" if pd.isna(value) else pd.Timestamp(value).strftime("%d %b %Y"))
        show["Legacy only"] = show["Legacy only"].map(lambda value: "Yes" if value else "")
        st.dataframe(show, width="stretch", hide_index=True, height=360)
    mig = data.migration[data.migration["account_number"].eq(str(row["account_number"]))]
    st.markdown("**Opportunity reconciliation for this account**")
    if mig.empty:
        st.caption("No migration-reconciliation rows are tied to this account number.")
    else:
        item = mig.iloc[0]
        st.markdown(
            f"{int(item.opportunities)} opportunities · {int(item.verified_opportunities)} verified in the release · "
            f"{int(item.withheld_opportunities)} withheld · {int(item.admin_end_dates)} administrative end dates."
        )
        unit = pd.DataFrame(
            {
                "Hardware": ["Migrated", "Failed migration", "Native MCF"],
                "Reported units": [int(item.reported_migrated_units), int(item.reported_failed_units), int(item.reported_native_units)],
                "Installed units in the extract": [int(item.installed_migrated_units), int(item.installed_failed_units), int(item.installed_native_units)],
            }
        )
        st.dataframe(unit, width="stretch", hide_index=True)
        st.caption("Verified means the opportunity passed the release rules. It is not proof of physical installation or that migration is finished.")


def handoff(data: packlib.Pack, scoped: pd.DataFrame) -> None:
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>Process executors</div><h1>Salesforce handoff</h1>"
        f"<p>Use this page before any write. Go-forward system: {packlib.GO_FORWARD}. Account Number is the business identifier. Salesforce Account.Id was not in the extract, and no live lookup was run.</p></div>",
        unsafe_allow_html=True,
    )
    handoff_info = data.handoff
    st.markdown(
        f"<div class='caveat'><b>Owner action on every crosswalk row.</b> {html.escape(handoff_info['owner_action'])} "
        f"Live lookup status: {html.escape(str(handoff_info['live_lookup_status']))}. "
        f"Mapping status: {html.escape(str(handoff_info['target_mapping_status']))}.</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    columns[0].metric("Crosswalk rows", f"{handoff_info['rows']:,}")
    columns[1].metric("Legacy-only contract keys", f"{handoff_info['legacy_only_contracts']:,}", delta=f"{handoff_info['legacy_accounts']:,} accounts", delta_color="off")
    columns[2].metric("Blank Account.Id", f"{handoff_info['blank_account_id']:,}")
    columns[3].metric("Withheld opportunities", f"{data.portfolio_migration['withheld']:,}", delta="Pending reconciliation", delta_color="off")
    st.markdown("**Before any write**")
    st.markdown(f"1. {handoff_info['owner_action']}")
    st.markdown("2. Keep legacy-only contract keys visible. They are not the go-forward identifier. A legacy-only key does not prove that no Salesforce opportunity exists.")
    st.markdown("3. Hold adoption scores marked reconciliation required until the reason codes are cleared.")
    coverage = scoped["source_coverage"].value_counts().rename_axis("Touchpoint coverage").reset_index(name="Accounts")
    st.dataframe(coverage, width="stretch", hide_index=True)
    legacy = scoped[scoped["legacy_only_contracts"].gt(0)][["account_number", "account_name_token", "legacy_only_contracts", "source_coverage", "target_mapping_status"]].sort_values("legacy_only_contracts", ascending=False)
    st.markdown(f"**Accounts in scope with legacy-only contract identifiers: {len(legacy):,}**")
    st.dataframe(legacy.head(100), width="stretch", hide_index=True)
    held = scoped[scoped["score_status"].eq("reconciliation_required")][["account_number", "account_name_token", "reason_codes", "withheld_active_opportunities"]]
    st.markdown(f"**Adoption withheld, reconciliation required: {len(held):,}**")
    st.dataframe(held.head(100), width="stretch", hide_index=True)


def sources(folder: Path) -> None:
    st.markdown(
        "<div class='hero'><div class='eyebrow'>Extract browser</div><h1>Source tables</h1><p>The first 200 rows of a file. Identifiers stay text.</p></div>",
        unsafe_allow_html=True,
    )
    files = sorted(path.name for path in folder.glob("*.csv"))
    choice = st.selectbox("File", files)
    frame = pd.read_csv(folder / choice, dtype=str, nrows=200)
    st.caption(f"{choice} · showing {len(frame)} rows · {frame.shape[1]} columns")
    st.dataframe(frame, width="stretch", hide_index=True, height=520)


def _operational_home(data: packlib.Pack, scoped: pd.DataFrame) -> None:
    st.markdown("**Primary review queue**")
    st.caption(_queue_note(data))
    queue = data.review[data.review["is_primary_top_50"] & data.review["account_number"].isin(scoped["account_number"])].sort_values(["outcome", "rank"])
    if queue.empty:
        st.info("No primary-queue accounts are in this scope.")
        return
    show = queue[["outcome", "rank", "account_number", "account_name_token", "probability_120d", "prior_contraction_days_180d"]].copy()
    show.columns = ["Outcome", "Rank", "Account", "Token", "Probability", "Prior contraction days"]
    show["Probability"] = show["Probability"].map(lambda value: f"{float(value):.1%}")
    st.dataframe(show, width="stretch", hide_index=True, height=420)
    held = scoped[scoped["score_status"].eq("reconciliation_required")]
    st.caption(f"{len(held):,} accounts in scope have adoption status reconciliation required.")


def _audiences() -> None:
    st.markdown("### Who reads what")
    columns = st.columns(4)
    cards = [
        ("Leadership", "Stay here for the book. Open Health scores only after the model table. Act on expansion. Treat product health as the churn probability restated."),
        ("Process executors", "Open Health scores and pick one account. Read the drivers, then open Handoff before a Salesforce write."),
        ("Support", "Open Journey and migration for events, and Hierarchy and people for contact tokens. There is no support health score."),
        ("Product", "On Health scores, read adoption beside product health. On Hierarchy and people, read the product and contract rows."),
    ]
    for column, (title, body) in zip(columns, cards):
        with column:
            st.markdown(f"**{title}**")
            st.caption(body)


def _calibration_table(data: packlib.Pack) -> None:
    frame = data.calibration.sort_values(["outcome", "min_score"])
    show = frame[["outcome", "band", "accounts", "observed_rate", "mean_predicted_probability", "rate_ci_low", "rate_ci_high"]].copy()
    show.columns = ["Outcome", "Band", "Accounts", "Observed rate", "Mean predicted", "Interval low", "Interval high"]
    for column in show.columns[3:]:
        show[column] = show[column].map(lambda value: f"{float(value):.2%}")
    show["Accounts"] = show["Accounts"].map(lambda value: f"{int(value):,}")
    st.markdown("**Calibration bands on the held-out test**")
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption("Churn's observed rate stays near 3% in every band. Expansion's observed rate rises with the predicted probability. Each interval is for the accounts in that band, not for one account.")


def _account_band(data: packlib.Pack, outcome: str, probability: float | None) -> None:
    st.markdown("**Where this probability sits in the held-out bands**")
    band = packlib.calibration_band(data, outcome, probability)
    if band is None:
        st.caption("No calibration band contains this probability.")
        return
    st.caption(
        f"Band {band['band']}: {int(band['accounts']):,} test accounts, "
        f"observed rate {float(band['observed_rate']):.2%} "
        f"(interval {float(band['rate_ci_low']):.2%} to {float(band['rate_ci_high']):.2%}), "
        f"mean predicted {float(band['mean_predicted_probability']):.2%}. "
        "That interval belongs to the band, not to this account."
    )


def _activity(data: packlib.Pack, account_number: str) -> None:
    rows = data.contact_activity[data.contact_activity["account_number"].eq(account_number)].sort_values("activity_date")
    st.markdown(f"**Recorded contact activity: {len(rows):,}**")
    if rows.empty:
        st.caption("No contact-activity row is tied to this account. Those rows are descriptive context and are not a model input.")
        return
    st.caption(f"{_blank(rows.iloc[0].get('identity_basis'), 'Identity was not verified.')} Data role: {_blank(rows.iloc[0].get('data_role')).replace('_', ' ')}.")
    show = rows[["activity_date", "contact", "case_number", "link_status", "customer_role"]].tail(20).copy()
    show.columns = ["When", "Token", "Case", "Link", "Customer role"]
    show["When"] = show["When"].map(lambda value: "—" if pd.isna(value) else pd.Timestamp(value).strftime("%d %b %Y"))
    st.dataframe(show, width="stretch", hide_index=True, height=280)


def _products(data: packlib.Pack, account_number: str) -> None:
    rows = data.products[data.products["account_number"].eq(account_number)]
    st.markdown(f"**Product and contract rows: {len(rows):,}**")
    if rows.empty:
        st.caption("No product-relationship row is tied to this account.")
        return
    scope = _blank(rows.iloc[0].get("salesforce_identifier_scope"))
    st.caption((scope + ". " if scope else "") + "The product extract date is blank. A legacy contract number is not the go-forward identifier.")
    show = rows[["opportunity_type", "product_names", "legacy_product_name", "product_linkage_status", "product_type_status", "mcf_contract_number", "legacy_contract_number", "adoption_contract_verified", "adoption_contract_issues"]].copy()
    show.columns = ["Opportunity", "Products", "Legacy product", "Linkage", "Type status", "MCF contract", "Legacy contract", "Adoption verified", "Adoption issues"]
    show["Adoption verified"] = show["Adoption verified"].map(lambda value: "Yes" if packlib._as_bool(value) else "")
    st.dataframe(show.head(40), width="stretch", hide_index=True, height=320)
    if len(show) > 40:
        st.caption(f"Showing 40 of {len(show):,} rows.")


def _performance(data: packlib.Pack) -> None:
    frame = packlib.present_columns(data.performance, packlib.PERFORMANCE_SPECS)
    text_columns = {"Outcome", "Model"}
    for column in frame.columns:
        if column in text_columns:
            continue
        frame[column] = frame[column].map(_format_metric)
    st.markdown("**Held-out performance, frozen models**")
    st.dataframe(frame, width="stretch", hide_index=True)
    st.caption("Intervals live in model_performance.csv. Agreement of the average probability with the observed rate is not calibration at every risk level. There is no account-specific confidence interval.")


def _portfolio_timeline(data: packlib.Pack) -> None:
    monthly = data.monthly.groupby(["month", "recorded_system"], as_index=False)["events"].sum()
    figure = px.line(monthly, x="month", y="events", color="recorded_system", title="Recorded touchpoints across the reference date", color_discrete_sequence=["#5C6B7A", "#003057", "#C65D3B", "#B3A369"])
    figure.add_vline(x=packlib.MIGRATION_REFERENCE, line_dash="dash", line_color="#C65D3B")
    figure.add_annotation(x=packlib.MIGRATION_REFERENCE, y=1.04, yref="paper", text="30 Sep 2025 reference", showarrow=False, font=dict(color="#C65D3B", size=12))
    figure.update_layout(height=360, margin=dict(l=8, r=8, t=48, b=8), legend=dict(orientation="h", y=-0.22))
    figure.update_xaxes(automargin=True)
    figure.update_yaxes(automargin=True, title="Events")
    st.plotly_chart(figure, width="stretch", theme="streamlit", config=CHART, key="book-timeline")
    st.caption("Legacy-only events stay in the legacy series. MCF is the go-forward recording system in this extract. The reference line is not an account migration date.")


def _drivers(data: packlib.Pack, account_number: str, outcome: str, published: float | None) -> None:
    rows = packlib.account_drivers(data, account_number, outcome)
    st.markdown("**Drivers**")
    if rows.empty:
        st.info("No driver rows for this account and outcome.")
        return
    intercept = float(rows["calibrated_intercept"].iloc[0])
    total = float(rows["contribution_calibrated_log_odds"].sum())
    rebuilt = packlib.reconstruct_probability(intercept, rows["contribution_calibrated_log_odds"])
    show = rows[["driver_rank", "feature_label", "direction", "contribution_calibrated_log_odds", "observed_feature_value", "is_top_three_driver", "explanation"]].copy()
    show.columns = ["Rank", "Driver", "Direction", "Log-odds contribution", "Observed value", "Top 3", "Explanation"]
    show["Log-odds contribution"] = show["Log-odds contribution"].map(lambda value: f"{float(value):+.4f}")
    show["Top 3"] = show["Top 3"].map(lambda value: "Yes" if value else "")
    st.dataframe(show, width="stretch", hide_index=True, height=280)
    score_line = ""
    if outcome == "full_churn":
        score_line = f" Product health = 100 × (1 − {rebuilt:.6f}) = {100 * (1 - rebuilt):.4f}."
    elif outcome == "expansion":
        score_line = f" Expansion opportunity score = 100 × {rebuilt:.6f} = {100 * rebuilt:.4f}."
    published_line = ""
    if published is not None:
        gap = abs(published - rebuilt)
        published_line = f" Published probability {published:.6f}. Difference {gap:.2e}."
    st.markdown(
        f"<div class='basis'><div><span>Intercept</span>{intercept:+.4f}</div>"
        f"<div><span>Sum of contributions</span>{total:+.4f}</div>"
        f"<div><span>Log odds</span>{intercept + total:+.4f}</div>"
        f"<div><span>Probability</span>{rebuilt:.6f}</div></div>",
        unsafe_allow_html=True,
    )
    st.caption("Probability = 1 / (1 + exp(−log odds)), and log odds = intercept + every contribution. Top 3 is a display rank. The probability uses every row." + score_line + published_line)


def _basis(data: packlib.Pack, score_name: str) -> None:
    row = packlib.definition(data, score_name)
    if row is None:
        return
    items = [
        ("Score", score_name.replace("_", " ")),
        ("As of", _blank(row.get("assessment_as_of"), "Unknown")),
        ("Freshness", _blank(row.get("freshness_status")).replace("_", " ")),
        ("Definition", _blank(row.get("definition"))),
        ("Confidence basis", _blank(row.get("confidence_basis"))),
        ("Window", f"{_blank(row.get('history_start_inclusive'), '—')} to {_blank(row.get('outcome_end_exclusive'), '—')}"),
    ]
    body = "".join(f"<div><span>{html.escape(label)}</span>{html.escape(value)}</div>" for label, value in items)
    st.markdown(f"<div class='basis'>{body}</div>", unsafe_allow_html=True)
    for label, key in (
        ("Caveat", "caveats"),
        ("Migration", "migration_caveat"),
        ("Roles", "role_sample_caveat"),
        ("Coverage", "source_comparability_caveat"),
    ):
        text = _blank(row.get(key))
        if text:
            st.markdown(f"<div class='caveat'><b>{html.escape(label)}.</b> {html.escape(text)}</div>", unsafe_allow_html=True)


def _adoption_block(data: packlib.Pack, row: pd.Series) -> None:
    st.markdown("**Adoption, separate from the historical models**")
    status = _blank(row.get("score_status"))
    contracted = _num(row.get("verified_contracted_units"))
    activated = _num(row.get("verified_activated_units"))
    score = _num(row.get("product_adoption_score"))
    if contracted and activated is not None and contracted != 0 and score is not None:
        math = f"100 × {activated:.0f} / {contracted:.0f} = {100 * activated / contracted:.1f}, published {score:.1f}."
    elif status != "scored":
        math = "No adoption score is published for this status."
    else:
        math = "Contracted or activated units are blank, so the ratio cannot be shown."
    reasons = _blank(row.get("reason_codes"))
    definition = packlib.definition(data, "product_adoption_score")
    freshness = _blank(definition.get("freshness_status") if definition is not None else "", "actual source extract date unknown").replace("_", " ")
    st.caption(
        f"Status: {status or '—'}. {math} "
        f"Extract date: {_blank(row.get('source_extract_date'), 'unknown, left blank')}. "
        f"Verification run {_blank(row.get('assessment_run_date'), '—')} is not the extract date. "
        f"Freshness: {freshness}."
    )
    if definition is not None and _blank(definition.get("definition")):
        st.caption(_blank(definition.get("definition")) + ". " + _blank(definition.get("caveats")))
    if reasons:
        st.markdown(f"<div class='caveat'><b>Reason codes.</b> {html.escape(reasons.replace(' | ', ', '))}</div>", unsafe_allow_html=True)


def _queue_note(data: packlib.Pack) -> str:
    capacity = data.capacity.copy()
    capacity["is_primary_k"] = capacity["is_primary_k"].map(packlib._as_bool)
    primary = capacity[capacity["is_primary_k"] & capacity["outcome"].eq("full_churn")]
    if primary.empty:
        return "Top of the historical list, for expansion and upgrade. Full churn is not on the recommended queue."
    row = primary.iloc[0]
    k = int(float(row["K"]))
    return (
        f"Top {k} by probability on 1 Jan 2025, for expansion and for upgrade. "
        f"Full churn is omitted: precision at {k} is {float(row['precision_at_K']):.0%}, "
        f"below the {float(row['base_rate']):.1%} base rate."
    )


def _kpis(data: packlib.Pack, scoped: pd.DataFrame) -> None:
    historical = int(scoped["full_churn_probability_120d"].notna().sum())
    adoption_scored = int(scoped["score_status"].eq("scored").sum())
    adoption_held = int(scoped["score_status"].eq("reconciliation_required").sum())
    legacy_accounts = int(scoped["source_coverage"].eq("legacy_only_recorded").sum())
    cards = [
        ("Accounts", f"{len(scoped):,}", "Reference book in this scope"),
        ("Historical scores", f"{historical:,}", "As of 1 Jan 2025, not a current forecast"),
        ("Adoption scored", f"{adoption_scored:,}", f"{adoption_held:,} reconciliation required"),
        ("Legacy-only touchpoints", f"{legacy_accounts:,}", "No MCF event recorded"),
    ]
    html_cards = "".join(
        f"<div class='kpi'><div class='k'>{html.escape(label)}</div><div class='v'>{html.escape(value)}</div><div class='n'>{html.escape(note)}</div></div>"
        for label, value, note in cards
    )
    st.markdown(f"<div class='kpi-grid'>{html_cards}</div>", unsafe_allow_html=True)


def _pick(data: packlib.Pack, scoped: pd.DataFrame, key: str) -> pd.Series | None:
    if scoped.empty:
        st.info("No accounts in this scope.")
        return None
    query = st.text_input("Find an account number or token", key=f"find-{key}")
    frame = scoped
    if query.strip():
        needle = query.strip().lower()
        frame = scoped[scoped["account_number"].str.lower().str.contains(needle, na=False) | scoped["account_name_token"].str.lower().str.contains(needle, na=False)]
        if frame.empty:
            st.warning("No account matches that text in the current scope.")
            return None
    frame = frame.assign(_rank=pd.to_numeric(frame["expansion_rank"], errors="coerce").fillna(10**9)).sort_values(["_rank", "account_number"])
    shown = frame.head(400)
    if len(frame) > 400:
        hint = "Search to open any account." if not query.strip() else "Add more of the number or token to narrow the list."
        st.caption(f"{len(frame):,} accounts match. This list shows 400. {hint}")
    current = st.session_state.get("account_number")
    if current in set(frame["account_number"]) and current not in set(shown["account_number"]):
        shown = pd.concat([frame[frame["account_number"].eq(current)], shown], ignore_index=True)
    labels = {str(record["account_number"]): _label(pd.Series(record)) for record in shown.to_dict("records")}
    options = list(labels)
    if current not in options:
        current = options[0]
    choice = st.selectbox("Account", options, index=options.index(current), format_func=lambda value: labels[value], key=key)
    st.session_state["account_number"] = choice
    return data.accounts[data.accounts["account_number"].eq(choice)].iloc[0]


def _label(row: pd.Series) -> str:
    token = str(row.get("account_name_token") or "")
    expansion = _num(row.get("expansion_probability_120d"))
    extra = f" · expansion {_pct(expansion)}" if expansion is not None else ""
    return f"{row['account_number']} · {token} · {row.get('account_status') or '—'}{extra}"


def _account_line(row: pd.Series) -> None:
    st.markdown(
        f"**{html.escape(str(row['account_number']))}** · {html.escape(str(row.get('account_name_token') or ''))} · "
        f"{html.escape(str(row.get('account_status') or '—'))} · {html.escape(str(row.get('platform') or '—'))}"
    )
    st.caption("The token is the reference label, not a customer legal name.")


def _signature(folder: Path) -> str:
    parts = []
    for path in sorted(folder.glob("*.csv")):
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_size}")
    return "|".join(parts)


def _format_metric(value) -> str:
    number = _num(value)
    if number is None:
        return "—" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    return f"{number:.3f}"


def _blank(value, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        text = value.strip()
        return fallback if text.lower() in {"", "nan", "none", "nat"} else text
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return fallback if text.lower() in {"", "nan", "none", "nat"} else text


def _num(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _points(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _rank(value, sense: str) -> str:
    number = _num(value)
    if number is None:
        return "Not scored"
    return f"Rank {int(number):,} · {sense}"


def _date(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value) or str(value).strip() in {"", "nan"}:
        return "—"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return pd.Timestamp(parsed).strftime("%d %b %Y")
