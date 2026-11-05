"""Pages for the practicum extracts.

The home page tells the story, then the same book is cut three ways.
Strategic is Product, operational is Customer Success, and process executor
is Service Delivery. Scores keep the published definition, the as-of date,
and the caveats in score_context.csv. Nothing here treats the 30 Sep 2025
bulk date as an account's migration date, or a contact token as a verified
person.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import dashboard_lib as lib
import practicum_pack as packlib

CHART = {"displaylogo": False, "displayModeBar": False, "responsive": True}
VIEWERS = ["Strategic", "Operational", "Process executor"]
PAGES = ["Home", "Health scores", "Hierarchy and people", "Journey and migration", "Handoff"]
VIEWER_COPY = {
    "Strategic": "Product. Decide whether a score can be trusted, and whether the journey still depends on the legacy system.",
    "Operational": "Customer Success. Work the expansion and upgrade queues, and hold adoption rows that still need reconciliation.",
    "Process executor": "Service Delivery. Run the Salesforce checks, leave legacy-only keys flagged, and do not invent a contact score.",
}
ROLE_LEAD = {
    "Operational": "What Customer Success works next",
    "Process executor": "What Service Delivery does in Salesforce",
}


@st.cache_data(show_spinner="Reading the practicum extracts")
def cached_pack(folder: str, signature: str) -> packlib.Pack:
    del signature
    return packlib.load_pack(Path(folder))


def render(folder: Path) -> None:
    if st.session_state.get("page") not in PAGES:
        st.session_state["page"] = "Home"
    if st.session_state.get("viewer") not in VIEWERS:
        st.session_state["viewer"] = "Strategic"
    st.sidebar.radio("Who is looking", VIEWERS, key="viewer")
    st.sidebar.caption(VIEWER_COPY[st.session_state["viewer"]])
    st.sidebar.radio("Go to", PAGES, key="page")
    st.sidebar.markdown("**Data**")
    st.sidebar.text_input("Folder", key="data_folder")
    folder_a, folder_b = st.sidebar.columns(2)
    folder_a.button("Practicum folder", on_click=_use_folder, args=(str(lib.practicum_folder()),), width="stretch")
    folder_b.button("Bundled sample", on_click=_use_folder, args=(str(lib.bundled_data_dir() / "health"),), width="stretch")
    st.sidebar.button("Reload files", on_click=cached_pack.clear)
    data = cached_pack(str(folder.resolve()), _signature(folder))
    viewer = st.session_state["viewer"]
    scoped = filters(data)
    page = st.session_state["page"]
    if page == "Home":
        home(data, scoped, viewer)
    elif page == "Health scores":
        health(data, scoped, viewer)
    elif page == "Hierarchy and people":
        hierarchy(data, scoped, viewer)
    elif page == "Journey and migration":
        journey(data, scoped, viewer)
    elif page == "Handoff":
        handoff(data, scoped, viewer)


def filters(data: packlib.Pack) -> pd.DataFrame:
    accounts = data.accounts
    st.sidebar.markdown("**Scope**")
    platform = st.sidebar.selectbox("Platform", ["All"] + sorted(v for v in accounts["platform"].dropna().unique() if str(v).strip()))
    status = st.sidebar.selectbox("Account status", ["All"] + sorted(v for v in accounts["account_status"].dropna().unique() if str(v).strip()))
    coverage = st.sidebar.selectbox("Touchpoint coverage", ["All"] + sorted(v for v in accounts["source_coverage"].dropna().unique() if str(v).strip()))
    scored = st.sidebar.checkbox("Only accounts with a historical score", value=False)
    scoped = packlib.scope_accounts(accounts, platform, status, coverage, scored)
    st.sidebar.caption(f"{len(scoped):,} of {len(accounts):,} accounts in scope.")
    return scoped


def home(data: packlib.Pack, scoped: pd.DataFrame, viewer: str) -> None:
    st.markdown(
        """
        <div class='hero'>
          <h1 class="glance">Executive Summary: At a Glance</h1>
          <p class="authors"><em>Patricia Omosola, Moyosore Akinrinmade, Trishal Ravula</em></p>
          <p>Split records across applications hid each customer's products. The customer behavioral model puts the accounts back together for a more streamlined view: product adoption, product health, and the chance of churn, expansion, or upgrade.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _kpis(data, scoped, viewer)
    _glance_charts(scoped)
    _timeline(data)
    _outputs(data)
    _viewers(viewer)
    _coverage_matrix(scoped)
    if viewer == "Operational":
        st.markdown(f"### {ROLE_LEAD[viewer]}")
        st.caption("The queue is historical, as of 1 Jan 2025. It is the review list, not a forecast of this week.")
        _operational_home(data, scoped)
    elif viewer == "Process executor":
        st.markdown(f"### {ROLE_LEAD[viewer]}")
        st.caption("Act on the new MCF Salesforce instance. A legacy-only key is residue, not a field to maintain.")
        _executor_home(data, scoped)
    _next("Next: open one account and read the drivers.", "Health Scores")


def health(data: packlib.Pack, scoped: pd.DataFrame, viewer: str) -> None:
    st.markdown(
        "<div class='hero'><h1>Health scores</h1>"
        "<p>The model’s answer for one account: the published score, the probability it came from, and the drivers that add up to it.</p></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Product health is 100 × (1 − the 120-day churn probability). "
        "Expansion score is 100 × the expansion probability. "
        "There are no domain sub-scores."
    )
    st.caption(
        "Intercept plus every log-odds contribution, then the logistic function, recovers the probability. "
        "Product adoption is 100 × verified activated units ÷ verified contracted units, and it is not part of that sum."
    )
    if viewer == "Strategic":
        st.caption("Product: churn health barely ranks better than chance. Read the held-out bands under the account before you compare two scores.")
    elif viewer == "Operational":
        st.caption("Customer Success: the band, the as-of date, and the confidence basis are part of the number.")
    else:
        st.caption("Service Delivery: the drivers say why a task exists. There is no contact-level score. The handoff is where the work is logged.")
    row = _pick(data, scoped, "health-account", prefer_published_adoption=True)
    if row is None:
        return
    health_status = _blank(row.get("health_score_status"))
    if health_status:
        st.caption(health_status[:1].upper() + health_status[1:])
    _account_line(row)
    churn_p = _num(row.get("full_churn_probability_120d"))
    health_score = _num(row.get("product_health_score"))
    expansion_p = _num(row.get("expansion_probability_120d"))
    expansion_s = _num(row.get("expansion_opportunity_score"))
    upgrade_p = _num(row.get("upgrade_probability_120d"))
    adoption = _num(row.get("product_adoption_score"))
    adoption_math = _adoption_math(row)
    columns = st.columns(4)
    columns[0].metric("Churn probability, 120 days", _pct(churn_p), delta=_rank(row.get("churn_rank"), "highest risk"), delta_color="off")
    columns[1].metric("Product health", _points(health_score), delta="100 × (1 − churn probability)", delta_color="off")
    columns[2].metric("Expansion probability", _pct(expansion_p), delta=_rank(row.get("expansion_rank"), "highest likelihood"), delta_color="off")
    columns[3].metric(
        "Product adoption",
        "—" if adoption is None else f"{adoption:.1f}",
        delta=adoption_math if adoption is not None else "No published score",
        delta_color="off",
    )
    st.caption(
        f"Upgrade probability {_pct(upgrade_p)} · rank {_rank(row.get('upgrade_rank'), 'highest likelihood')}. "
        f"Expansion score {_points(expansion_s)} is 100 × the expansion probability. It is not combined with product health. "
        f"Product adoption: {adoption_math}."
    )
    _adoption_block(data, row)
    _account_picture(scoped, churn_p, expansion_p, upgrade_p, health_score)
    outcome = st.radio(
        "Drivers for",
        ["full_churn", "expansion", "upgrade"],
        format_func=lambda value: {"full_churn": "Churn", "expansion": "Expansion", "upgrade": "Upgrade"}[value],
        horizontal=True,
        key="driver_outcome",
    )
    _basis(data, {"full_churn": "full_churn_probability_120d", "expansion": "expansion_probability_120d", "upgrade": "upgrade_probability_120d"}[outcome])
    _account_band(data, outcome, churn_p if outcome == "full_churn" else expansion_p if outcome == "expansion" else upgrade_p)
    _drivers(data, str(row["account_number"]), outcome, churn_p if outcome == "full_churn" else expansion_p if outcome == "expansion" else upgrade_p)
    if viewer == "Strategic":
        with st.expander("Held-out test", expanded=False):
            _performance(data)
            _calibration_table(data)
    _next("Next: put the parent, the children, and the contact tokens back with this account.", "Hierarchy and people")


def hierarchy(data: packlib.Pack, scoped: pd.DataFrame, viewer: str) -> None:
    st.markdown(
        "<div class='hero'><h1>Hierarchy and people</h1>"
        "<p>The score was for the account. This page puts that account back in its family: a candidate parent, the child accounts, and the contact tokens. Validated parent IDs are blank, and there is no role-level score.</p></div>",
        unsafe_allow_html=True,
    )
    if viewer == "Strategic":
        st.caption("Product compares the parent token with its children before treating the family as one healthy account. Family health is not a rolled-up score.")
    elif viewer == "Operational":
        st.caption("Customer Success reviews one account. Open the parent when the child token is the site on the queue.")
    else:
        st.caption("Service Delivery calls the token named on the account. A sponsor often sits on the parent. Do not treat a token as a verified person.")
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
        _children_chart(kids)
    people = data.contacts[data.contacts["account_number"].eq(str(row["account_number"]))]
    st.markdown(f"**Point-of-contact tokens on this account: {len(people)}**")
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
        _contact_chart(people)
    st.caption("Customer role is unknown, so these tokens are not a role-level score. Product rows below flag legacy contract numbers. A legacy contract number is not the Salesforce identifier.")
    with st.expander("Recorded activity and product rows", expanded=False):
        _activity(data, str(row["account_number"]))
        _products(data, str(row["account_number"]))
    _next("Next: follow this account across the migration.", "Journey and migration")


def journey(data: packlib.Pack, scoped: pd.DataFrame, viewer: str) -> None:
    st.markdown(
        "<div class='hero'><h1>Journey and migration</h1>"
        "<p>The same customer, before and after the move. Legacy and MCF events share one timeline. The dashed line is 30 Sep 2025, a bulk-update reference, not the day this account migrated.</p></div>",
        unsafe_allow_html=True,
    )
    if viewer == "Strategic":
        st.caption("Product sees the book first. Coverage is not migration completion, and a legacy series after the reference date means the bridge still depends on the old system.")
        _portfolio_timeline(data)
        migration = data.portfolio_migration
        st.markdown(
            f"**Release reconciliation.** {migration['opportunities']:,} opportunity rows · "
            f"{migration['verified']:,} verified in the release · {migration['withheld']:,} withheld · "
            f"{migration['admin_end_dates']:,} carry the 30 Sep 2025 administrative end date. "
            f"Reported units: migrated {migration['reported_migrated_units']:,}, failed {migration['reported_failed_units']:,}, native {migration['reported_native_units']:,}."
        )
        st.caption("An administrative end date is the bulk-update reference. Do not read it as churn. Failed units are a source category, not a churn flag.")
    elif viewer == "Operational":
        st.caption("Customer Success reads this account's journey before a review. A quiet stretch can be a recording gap. Missing activity is not proof that nothing happened.")
    else:
        st.caption("Service Delivery logs the next touch in the new Salesforce instance. A legacy-only row has no home there.")
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
        figure.update_layout(barmode="stack")
        figure.update_yaxes(title="Events")
        _show_chart(figure, f"journey-{row['account_number']}", "Recorded events by month")
        running = events.groupby(events["event_at"].dt.to_period("M").dt.to_timestamp()).size().rename("events").reset_index()
        running["Cumulative events"] = running["events"].cumsum()
        line = px.line(running, x="event_at", y="Cumulative events", title="Cumulative recorded events", color_discrete_sequence=["#003057"])
        line.add_vline(x=packlib.MIGRATION_REFERENCE, line_dash="dash", line_color="#C65D3B")
        _show_chart(line, f"journey-line-{row['account_number']}", "Cumulative recorded events")
        period = events["migration_reference_period"].value_counts()
        st.caption("Relative to the reference date: " + ", ".join(f"{name} {int(count):,}" for name, count in period.items()) + ". An administrative end date is not churn. A legacy-only row has no home on the new instance.")
        show = events[["event_at", "journey_stage", "event_type", "recorded_system", "legacy_only", "migration_reference_period", "date_meaning"]].tail(40).copy()
        show.columns = ["When", "Stage", "Event", "System", "Legacy only", "Reference period", "What the date means"]
        show["When"] = show["When"].map(lambda value: "—" if pd.isna(value) else pd.Timestamp(value).strftime("%d %b %Y"))
        show["Legacy only"] = show["Legacy only"].map(lambda value: "Yes" if value else "")
        st.markdown("**Latest touchpoints**")
        st.dataframe(show.tail(8), width="stretch", hide_index=True)
        with st.expander("More recorded events", expanded=False):
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
        long = unit.melt(id_vars="Hardware", var_name="Count", value_name="Units")
        units = px.bar(long, x="Hardware", y="Units", color="Count", barmode="group", title="Reported units and installed units in the extract", color_discrete_sequence=["#003057", "#B3A369"])
        _show_chart(units, f"units-{row['account_number']}", "Reported and installed units")
        st.caption("Verified means the opportunity passed the release rules. It is not proof of physical installation or that migration is finished.")
    _next("Next: leave with the Salesforce checks. Legacy-only keys stay flagged.", "Handoff")


def handoff(data: packlib.Pack, scoped: pd.DataFrame, viewer: str) -> None:
    st.markdown(
        "<div class='hero'><h1>Salesforce handoff</h1>"
        f"<p>The go-forward system is {packlib.GO_FORWARD}. Account Number is the business identifier. Legacy-only keys stay flagged and are not fields to maintain. Account.Id was not in the extract, and no live lookup was run.</p></div>",
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
    _handoff_picture(scoped)
    if viewer == "Strategic":
        st.caption("Legacy-only means the contract or subscription key exists without an MCF contract key. It does not prove that no Salesforce opportunity exists.")
        coverage = scoped["source_coverage"].value_counts().rename_axis("Touchpoint coverage").reset_index(name="Accounts")
        st.dataframe(coverage, width="stretch", hide_index=True)
    elif viewer == "Operational":
        st.caption("The review queue is on Home. On this page, hold adoption rows that still need reconciliation, and do not update a legacy-only key.")
        held = scoped[scoped["score_status"].eq("reconciliation_required")][["account_number", "account_name_token", "reason_codes", "withheld_active_opportunities"]]
        st.markdown(f"**Adoption withheld, reconciliation required: {len(held):,}**")
        st.dataframe(held.head(50), width="stretch", hide_index=True)
    else:
        st.markdown("**Do this before any write to Salesforce**")
        st.markdown(f"1. {handoff_info['owner_action']}")
        st.markdown("2. Treat legacy-only contract keys as residue. Keep them visible. Do not maintain them as the go-forward identifier.")
        st.markdown("3. Hold adoption scores marked reconciliation required until the reason codes are cleared.")
        legacy = scoped[scoped["legacy_only_contracts"].gt(0)][["account_number", "account_name_token", "legacy_only_contracts", "source_coverage", "target_mapping_status"]].sort_values("legacy_only_contracts", ascending=False)
        st.markdown(f"**Accounts in scope with legacy-only contract identifiers: {len(legacy):,}**")
        st.dataframe(legacy.head(100), width="stretch", hide_index=True)
        held = scoped[scoped["score_status"].eq("reconciliation_required")][["account_number", "account_name_token", "reason_codes", "withheld_active_opportunities"]]
        st.markdown(f"**Adoption withheld, reconciliation required: {len(held):,}**")
        st.dataframe(held.head(100), width="stretch", hide_index=True)


def _next(text: str, page: str) -> None:
    st.caption(text)
    st.button(page, on_click=_go, args=(page,), key=f"next-{page}", width="content")


def _timeline(data: packlib.Pack) -> None:
    start, end = _window(data)
    st.markdown("### Provisional 300-day Timeline")
    st.caption(
        f"180 days of history from {start}, scored on 1 Jan 2025, then 120 days of outcomes through {end}. "
        "The history window was set to cover 91.6% of observed service intervals, and the outcome window records more churn, expansion, and upgrade than a shorter horizon. "
        "Those figures are the design of the timeline. This page does not recompute them."
    )
    history, scored, outcomes = st.columns(3, border=True)
    history.metric("History", "180 days")
    history.caption(f"From {start}")
    scored.metric("Score date", "1 Jan 2025")
    scored.caption("Historical test, not a forecast of today")
    outcomes.metric("Outcomes", "120 days")
    outcomes.caption(f"Through {end}")


def _outputs(data: packlib.Pack) -> None:
    health = packlib.definition(data, "product_health_score")
    adoption = packlib.definition(data, "product_adoption_score")
    health_as_of = _date(health.get("assessment_as_of")) if health is not None else "1 Jan 2025"
    health_fresh = _blank(health.get("freshness_status") if health is not None else "", "historical assessment, not a current forecast").replace("_", " ")
    adoption_fresh = _blank(adoption.get("freshness_status") if adoption is not None else "", "actual source extract date unknown").replace("_", " ")
    adoption_card, health_card, outcome_card = st.columns(3, border=True)
    with adoption_card:
        st.markdown("**Product adoption**")
        st.caption(f"Installed units ÷ contracted units. Not an input to the health model. Freshness: {adoption_fresh}.")
    with health_card:
        st.markdown("**Product health**")
        st.caption(f"100 × (1 − churn probability). Near-chance ranking, so not a call list. As of {health_as_of}. {health_fresh.capitalize()}.")
    with outcome_card:
        st.markdown("**Churn, expansion, upgrade**")
        st.caption("120-day probabilities. Expansion and upgrade are the review queues. Drivers are on the next page. No account-specific confidence interval.")


def _window(data: packlib.Pack) -> tuple[str, str]:
    row = packlib.definition(data, "product_health_score")
    if row is None:
        return "5 Jul 2024", "1 May 2025"
    return _date(row.get("history_start_inclusive")), _date(row.get("outcome_end_exclusive"))


def _show_chart(figure, key: str, alt: str, height: int = 340) -> None:
    figure.update_layout(height=height, margin=dict(l=8, r=8, t=48, b=8), legend=dict(orientation="h", y=-0.22))
    figure.update_xaxes(automargin=True)
    figure.update_yaxes(automargin=True)
    st.plotly_chart(figure, width="stretch", theme="streamlit", config=CHART, key=key, alt=alt)


def _status_label(value) -> str:
    labels = {"scored": "Scored", "unscored": "Unscored", "reconciliation_required": "Needs reconciliation"}
    text = _blank(value, "Unknown")
    return labels.get(text, text.replace("_", " "))


def _glance_charts(scoped: pd.DataFrame) -> None:
    st.caption("Counts and scores are for accounts in the current scope. A published adoption score is 100 × verified activated units ÷ verified contracted units. Accounts without one stay out of the adoption chart.")
    status_counts = scoped["score_status"].map(_status_label).value_counts().rename_axis("Adoption status").reset_index(name="Accounts")
    adoption = scoped.loc[scoped["score_status"].eq("scored"), ["product_adoption_score"]].apply(pd.to_numeric, errors="coerce").dropna()
    left, right = st.columns(2)
    with left:
        bars = px.bar(status_counts, x="Adoption status", y="Accounts", title="Adoption status", color_discrete_sequence=["#003057"])
        _show_chart(bars, "book-adoption-status", "Bar chart of accounts by adoption status")
    with right:
        if adoption.empty:
            st.info("No published adoption scores in this scope.")
        else:
            hist = px.histogram(adoption, x="product_adoption_score", nbins=20, title="Published adoption scores", color_discrete_sequence=["#003057"])
            hist.update_layout(xaxis_title="Product adoption", yaxis_title="Accounts", showlegend=False)
            _show_chart(hist, "book-adoption-hist", "Histogram of published product adoption scores")
    scored = scoped.dropna(subset=["full_churn_probability_120d", "expansion_probability_120d"]).copy()
    if scored.empty:
        return
    scored["Churn probability"] = pd.to_numeric(scored["full_churn_probability_120d"], errors="coerce")
    scored["Expansion probability"] = pd.to_numeric(scored["expansion_probability_120d"], errors="coerce")
    scored["Product health"] = pd.to_numeric(scored["product_health_score"], errors="coerce")
    scatter = px.scatter(
        scored,
        x="Churn probability",
        y="Expansion probability",
        color="Product health",
        title="Churn and expansion for each scored account",
        hover_data=["account_number"],
        color_continuous_scale=["#C65D3B", "#B3A369", "#003057"],
        opacity=0.7,
    )
    scatter.update_traces(marker=dict(size=7))
    scatter.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    _show_chart(scatter, "book-scatter", "Scatter plot of churn probability against expansion probability", height=420)
    st.caption("Each point is one account with a historical score as of 1 Jan 2025. High expansion is not the same as low churn. Health stays high across the book because the churn probabilities sit close together.")


def _coverage_matrix(scoped: pd.DataFrame) -> None:
    matrix = pd.crosstab(scoped["score_status"].map(_status_label), scoped["source_coverage"].map(lambda value: _blank(value, "Unknown").replace("_", " ")))
    if matrix.empty:
        return
    st.markdown("### Coverage across the book")
    heat = px.imshow(matrix, text_auto=True, aspect="auto", title="Accounts by adoption status and touchpoint coverage", color_continuous_scale=["#F4F1E8", "#003057"])
    heat.update_layout(xaxis_title="Touchpoint coverage", yaxis_title="Adoption status")
    _show_chart(heat, "book-matrix", "Matrix of account counts by adoption status and touchpoint coverage", height=380)
    st.caption("Legacy-only coverage means no MCF event was recorded.")


def _account_picture(scoped: pd.DataFrame, churn_p: float | None, expansion_p: float | None, upgrade_p: float | None, health_score: float | None) -> None:
    probabilities = pd.DataFrame(
        {"Outcome": ["Churn", "Expansion", "Upgrade"], "Probability": [churn_p, expansion_p, upgrade_p]}
    ).dropna()
    left, right = st.columns(2)
    with left:
        if probabilities.empty:
            st.caption("This account has no 120-day probabilities to chart.")
        else:
            bars = px.bar(probabilities, x="Outcome", y="Probability", title="120-day probabilities for this account", color_discrete_sequence=["#003057"])
            bars.update_layout(yaxis_tickformat=".0%", showlegend=False)
            _show_chart(bars, "account-probabilities", "Bar chart of this account's churn, expansion, and upgrade probabilities")
    with right:
        book = pd.to_numeric(scoped["product_health_score"], errors="coerce").dropna().rename("Product health")
        if book.empty:
            st.caption("No product health scores in this scope.")
        else:
            hist = px.histogram(book.to_frame(), x="Product health", nbins=24, title="Where this health score sits", color_discrete_sequence=["#5C6B7A"])
            hist.update_layout(xaxis_title="Product health", yaxis_title="Accounts", showlegend=False)
            if health_score is not None:
                hist.add_vline(x=health_score, line_dash="dash", line_color="#C65D3B")
                hist.add_annotation(x=health_score, y=1.02, yref="paper", text="This account", showarrow=False, font=dict(color="#C65D3B", size=12))
            _show_chart(hist, "account-health-position", "Histogram of product health with this account marked")
    st.caption("The dashed line is this account. Most scores pile up because churn probabilities are close. The rank is not a wide separation between healthy and unhealthy accounts.")


def _children_chart(kids: pd.DataFrame) -> None:
    scores = kids.dropna(subset=["product_health_score"]).copy()
    if scores.empty:
        return
    scores["Account"] = scores["account_number"].astype(str)
    scores["Product health"] = pd.to_numeric(scores["product_health_score"], errors="coerce")
    scores = scores.dropna(subset=["Product health"]).sort_values("Product health")
    figure = px.bar(scores.head(20), x="Product health", y="Account", orientation="h", title="Child product health", color_discrete_sequence=["#003057"])
    _show_chart(figure, "child-health", "Bar chart of product health for child accounts")
    st.caption("Each bar is a child account's own product health. These scores are not rolled up into a family score.")


def _contact_chart(people: pd.DataFrame) -> None:
    plot = people.copy()
    plot["Interactions"] = pd.to_numeric(plot["distinct_interactions"], errors="coerce")
    plot = plot.dropna(subset=["Interactions"])
    plot = plot[plot["Interactions"] > 0].sort_values("Interactions").tail(12)
    if plot.empty:
        return
    figure = px.bar(plot, x="Interactions", y="contact", orientation="h", title="Recorded interactions by token", color_discrete_sequence=["#003057"])
    figure.update_layout(yaxis_title="Token")
    _show_chart(figure, "contact-interactions", "Bar chart of recorded interactions by contact token")
    st.caption("A longer bar means more recorded interactions. It is not a role score, and the token is not a verified person.")


def _handoff_picture(scoped: pd.DataFrame) -> None:
    left, right = st.columns(2)
    coverage = scoped["source_coverage"].map(lambda value: _blank(value, "Unknown").replace("_", " ")).value_counts().rename_axis("Coverage").reset_index(name="Accounts")
    adoption = scoped["score_status"].map(_status_label).value_counts().rename_axis("Adoption status").reset_index(name="Accounts")
    with left:
        bars = px.bar(coverage, x="Accounts", y="Coverage", orientation="h", title="Touchpoint coverage in this scope", color_discrete_sequence=["#003057"])
        _show_chart(bars, "handoff-coverage", "Bar chart of accounts by touchpoint coverage")
    with right:
        bars = px.bar(adoption, x="Adoption status", y="Accounts", title="Adoption rows to hold or use", color_discrete_sequence=["#C65D3B"])
        _show_chart(bars, "handoff-adoption", "Bar chart of accounts by adoption status")
    st.caption("Needs reconciliation means the adoption score was withheld. Do not write that score to Salesforce until the reason codes are cleared.")


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
    st.dataframe(show, width="stretch", hide_index=True, height=280)
    held = scoped[scoped["score_status"].eq("reconciliation_required")]
    st.caption(f"{len(held):,} accounts in scope have adoption status reconciliation required.")


def _executor_home(data: packlib.Pack, scoped: pd.DataFrame) -> None:
    st.markdown("**Work that belongs on the new instance**")
    st.markdown(f"1. {data.handoff['owner_action']}")
    legacy = int((scoped["legacy_only_contracts"] > 0).sum())
    held = int(scoped["score_status"].eq("reconciliation_required").sum())
    st.markdown(f"2. Review {legacy:,} accounts in scope that still have legacy-only contract identifiers.")
    st.markdown(f"3. Clear {held:,} adoption rows marked reconciliation required before using those scores.")
    st.caption("These counts come from the crosswalk and the adoption status. They are not a generated call list.")
    if st.button("Open the handoff lists", on_click=_go, args=("Handoff",), key="open-handoff"):
        pass


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
    bands = data.calibration[data.calibration["outcome"].eq(outcome)].sort_values("min_score")
    if bands.empty:
        return
    long = bands.melt(id_vars="band", value_vars=["observed_rate", "mean_predicted_probability"], var_name="Rate", value_name="Value")
    long["Rate"] = long["Rate"].map({"observed_rate": "Observed rate", "mean_predicted_probability": "Mean predicted"})
    figure = px.bar(long, x="band", y="Value", color="Rate", barmode="group", title="Held-out bands for this outcome", color_discrete_sequence=["#003057", "#B3A369"])
    figure.update_layout(yaxis_tickformat=".0%", xaxis_title="Band", yaxis_title="Rate")
    _show_chart(figure, f"calibration-{outcome}", "Grouped bar of observed rate and mean predicted probability by calibration band")
    st.caption(f"This account falls in band {band['band']}. The rates are for every test account in that band, not a confidence interval for this account.")


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
    columns = (
        ("outcome", "Outcome"),
        ("model_family", "Model"),
        ("selected_model", "Model"),
        ("roc_auc", "ROC AUC"),
        ("roc_auc_ci_low", "ROC AUC low"),
        ("roc_auc_ci_high", "ROC AUC high"),
        ("average_precision", "Average precision"),
        ("precision_at_50", "Precision at 50"),
        ("recall_at_50", "Recall at 50"),
        ("base_rate", "Base rate"),
        ("observed_outcome_rate", "Observed rate"),
        ("mean_probability", "Mean predicted"),
        ("mean_predicted_probability", "Mean predicted"),
    )
    chosen = []
    labels = []
    seen = set()
    for key, label in columns:
        if key in data.performance.columns and label not in seen:
            chosen.append(key)
            labels.append(label)
            seen.add(label)
    frame = data.performance[chosen].copy()
    frame.columns = labels
    for column in frame.columns:
        if column in {"Outcome", "Model"}:
            continue
        frame[column] = frame[column].map(lambda value: f"{float(value):.3f}")
    st.markdown("**Held-out performance**")
    st.dataframe(frame, width="stretch", hide_index=True)
    scope = ""
    if "evaluation_scope" in data.performance.columns:
        values = data.performance["evaluation_scope"].dropna().astype(str)
        values = values[values.str.strip().ne("")]
        if not values.empty:
            scope = str(values.iloc[0])
    st.caption(scope or "There is no account-specific confidence interval.")


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
    chart = rows.copy()
    chart["Contribution"] = pd.to_numeric(chart["contribution_calibrated_log_odds"], errors="coerce")
    chart = chart.dropna(subset=["Contribution"]).assign(_abs=lambda frame: frame["Contribution"].abs()).sort_values("_abs").tail(8)
    if not chart.empty:
        figure = px.bar(
            chart,
            x="Contribution",
            y="feature_label",
            orientation="h",
            title="Largest driver contributions",
            color="Contribution",
            color_continuous_scale=["#C65D3B", "#F4F1E8", "#003057"],
            color_continuous_midpoint=0,
        )
        figure.update_layout(yaxis_title="Driver", xaxis_title="Log-odds contribution")
        _show_chart(figure, f"drivers-{account_number}-{outcome}", "Horizontal bar chart of the largest driver contributions")
        st.caption("Blue raises the probability and rust lowers it. The chart shows the largest contributions. The probability still uses every driver, plus the intercept.")
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


def _adoption_math(row: pd.Series) -> str:
    status = _blank(row.get("score_status"))
    contracted = _num(row.get("verified_contracted_units"))
    activated = _num(row.get("verified_activated_units"))
    score = _num(row.get("product_adoption_score"))
    if contracted and activated is not None and contracted != 0 and score is not None:
        return f"100 × {activated:.0f} / {contracted:.0f} = {100 * activated / contracted:.1f}"
    if status != "scored":
        return "No published score"
    return "Units blank"


def _adoption_block(data: packlib.Pack, row: pd.Series) -> None:
    st.markdown("**Adoption calculation**")
    status = _blank(row.get("score_status"))
    contracted = _num(row.get("verified_contracted_units"))
    activated = _num(row.get("verified_activated_units"))
    score = _num(row.get("product_adoption_score"))
    math = _adoption_math(row)
    if score is not None and contracted and activated is not None and contracted != 0:
        math = f"{math}, published {score:.1f}."
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


def _use_folder(path: str) -> None:
    st.session_state["data_folder"] = str(path)
    cached_pack.clear()


def _kpis(data: packlib.Pack, scoped: pd.DataFrame, viewer: str) -> None:
    historical = int(scoped["full_churn_probability_120d"].notna().sum())
    adoption_scored = int(scoped["score_status"].eq("scored").sum())
    adoption_held = int(scoped["score_status"].eq("reconciliation_required").sum())
    legacy_accounts = int(scoped["source_coverage"].eq("legacy_only_recorded").sum())
    cards = [
        ("Total Accounts", f"{len(scoped):,}", "Reference accounts in this scope"),
        ("Historical scores", f"{historical:,}", "As of 1 Jan 2025, not a current forecast"),
        ("Adoption scored", f"{adoption_scored:,}", f"{adoption_held:,} reconciliation required"),
        ("Legacy-only touchpoints", f"{legacy_accounts:,}", "No MCF event recorded"),
    ]
    if viewer == "Process executor":
        cards[3] = ("Legacy-only contracts", f"{int((scoped['legacy_only_contracts'] > 0).sum()):,}", "Keys that are not MCF contract IDs")
    html_cards = "".join(
        f"<div class='kpi'><div class='k'>{html.escape(label)}</div><div class='v'>{html.escape(value)}</div><div class='n'>{html.escape(note)}</div></div>"
        for label, value, note in cards
    )
    st.markdown(f"<div class='kpi-grid'>{html_cards}</div>", unsafe_allow_html=True)


def _viewers(active: str) -> None:
    columns = st.columns(3)
    for column, name in zip(columns, VIEWERS):
        with column:
            css = "viewer-card active" if name == active else "viewer-card"
            st.markdown(f"<div class='{css}'><h3>{html.escape(name)}</h3><p>{html.escape(VIEWER_COPY[name])}</p></div>", unsafe_allow_html=True)
            if name != active:
                st.button(f"Switch to {name}", on_click=_viewer, args=(name,), key=f"switch-{name}", width="stretch")
            else:
                st.caption("This is the view you are in.")


def _remember_account_pick() -> None:
    st.session_state["account_user_picked"] = True


def _pick(data: packlib.Pack, scoped: pd.DataFrame, key: str, prefer_published_adoption: bool = False) -> pd.Series | None:
    if scoped.empty:
        st.info("No accounts in this scope.")
        return None
    frame = scoped.assign(
        _rank=pd.to_numeric(scoped["expansion_rank"], errors="coerce").fillna(10**9),
        _has_adoption=scoped["score_status"].eq("scored") & pd.to_numeric(scoped["product_adoption_score"], errors="coerce").notna(),
    )
    sort_by = ["_has_adoption", "_rank", "account_number"] if prefer_published_adoption else ["_rank", "account_number"]
    frame = frame.sort_values(sort_by, ascending=[False, True, True] if prefer_published_adoption else [True, True])
    labels = {str(record["account_number"]): _label(pd.Series(record)) for record in frame.to_dict("records")}
    options = list(labels)
    scored_ids = set(frame.loc[frame["_has_adoption"], "account_number"].astype(str))
    default = next((opt for opt in options if opt in scored_ids), options[0])
    if prefer_published_adoption and not st.session_state.get("account_user_picked"):
        if str(st.session_state.get(key) or "") not in scored_ids:
            st.session_state[key] = default
    elif key not in st.session_state:
        initial = str(st.session_state.get("account_number") or "")
        st.session_state[key] = initial if initial in options else options[0]
    choice = st.selectbox("Account", options, format_func=lambda value: labels[value], key=key, on_change=_remember_account_pick)
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


def _go(page: str) -> None:
    st.session_state["page_request"] = page


def _viewer(name: str) -> None:
    st.session_state["viewer_request"] = name


def _signature(folder: Path) -> str:
    parts = []
    for path in sorted(folder.glob("*.csv")):
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_size}")
    return "|".join(parts)


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
    stamp = pd.Timestamp(parsed)
    return f"{stamp.day} {stamp.strftime('%b %Y')}"
