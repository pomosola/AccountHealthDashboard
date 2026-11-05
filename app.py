"""Account health handoff.

The app opens on a home page. Pick a viewer — strategic, operational, or
process executor — and the same scores are cut for that job. Python modules
and data files are loaded from
https://github.com/pomosola/AccountHealthDashboard. Files that repository
does not contain are taken from the local practicum folder when they exist.
Otherwise the bundled health extract in data/health is used.

    streamlit run app.py
"""

from __future__ import annotations

import html
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import dashboard_lib as lib
import github_source
import health_model as hm
import practicum_pack
import practicum_view

CHART = {"displaylogo": False, "displayModeBar": False, "responsive": True}
VIEWERS = ["Strategic", "Operational", "Process executor"]
PAGES = ["Home", "Health scores", "Hierarchy and people", "Journey and migration"]
VIEWER_COPY = {
    "Strategic": "Portfolio health, where the money sits, and whether the migration is holding.",
    "Operational": "The book of accounts you own: who is critical, who can expand, and which score is stale.",
    "Process executor": "The next Salesforce action, the person to call, and the legacy item not to carry forward.",
}


def request_viewer(name: str) -> None:
    st.session_state["viewer_request"] = name


def request_page(name: str) -> None:
    st.session_state["page_request"] = name


def request_folder(path: str) -> None:
    st.session_state["data_folder"] = str(path)
    st.cache_data.clear()


def reset_scope() -> None:
    st.session_state["segments"] = []
    st.session_state["owners"] = []
    st.session_state["bands"] = []
    st.session_state["families"] = []


def apply_requests() -> None:
    st.session_state.setdefault("viewer", "Strategic")
    st.session_state.setdefault("page", "Home")
    st.session_state.setdefault("data_folder", str(lib.default_folder()))
    st.session_state.setdefault("segments", [])
    st.session_state.setdefault("owners", [])
    st.session_state.setdefault("bands", [])
    st.session_state.setdefault("families", [])
    if "viewer_request" in st.session_state:
        st.session_state["viewer"] = st.session_state.pop("viewer_request")
    if "page_request" in st.session_state:
        st.session_state["page"] = st.session_state.pop("page_request")


def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.15rem; padding-bottom: 2.4rem; max-width: 1180px; }
          .hero {
            background: linear-gradient(115deg, #003057 0%, #0E4D78 58%, #1F6A75 100%);
            border-bottom: 3px solid #B3A369;
            border-radius: 14px;
            color: white;
            padding: 1.15rem 1.3rem 1.15rem;
            margin-bottom: 0.8rem;
          }
          .eyebrow { color: #F3E6B8; font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; }
          .hero h1 { font-size: 1.7rem; font-weight: 620; line-height: 1.2; margin: 0.2rem 0 0.35rem; }
          .hero h1.glance { margin: 0; line-height: 1; }
          .hero p.authors { margin: -0.7rem 0 1.15rem; line-height: 1.05; }
          .hero p { color: rgba(255,255,255,0.9); margin: 0; }
          .strip {
            display: flex; flex-wrap: wrap; gap: 0.45rem 1rem;
            color: #3E4C59; font-size: 0.92rem; margin: 0.1rem 0 0.8rem;
          }
          .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.7rem; margin: 0.2rem 0 0.9rem; }
          .kpi, .viewer-card, .basis, .caveat, .note, .tree-card {
            background: #fff; border: 1px solid #E4DDD0; border-radius: 12px;
          }
          .kpi { border-top: 3px solid #B3A369; padding: 0.72rem 0.9rem 0.8rem; }
          .kpi .k { color: #5E6A75; font-size: 0.74rem; letter-spacing: 0.04em; text-transform: uppercase; }
          .kpi .v { font-size: 1.35rem; font-weight: 640; color: #003057; margin-top: 0.12rem; }
          .kpi .n { color: #5E6A75; font-size: 0.82rem; margin-top: 0.12rem; }
          .viewer-card { padding: 0.75rem 0.85rem 0.4rem; min-height: 8.2rem; }
          .viewer-card.active { border-color: #003057; box-shadow: inset 0 0 0 1px #003057; }
          .viewer-card h3 { margin: 0 0 0.3rem; font-size: 1rem; color: #003057; }
          .viewer-card p { margin: 0; color: #3E4C59; font-size: 0.9rem; }
          .basis { padding: 0.75rem 0.9rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.45rem 1rem; }
          .basis span { display: block; color: #5E6A75; font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase; }
          .caveat { border-left: 3px solid #C65D3B; padding: 0.45rem 0.7rem; margin: 0.35rem 0; color: #3E2A24; }
          .note { border-left: 3px solid #B3A369; padding: 0.55rem 0.75rem; margin: 0.4rem 0 0.8rem; color: #3E4C59; }
          .tree-card { padding: 0.7rem 0.85rem; margin: 0.35rem 0; }
          .tree-card.child { margin-left: 1.25rem; }
          .tree-card h3 { margin: 0; font-size: 1rem; color: #003057; }
          .person { margin: 0.35rem 0 0 0.2rem; color: #1C2834; font-size: 0.92rem; }
          .badge { display: inline-block; border-radius: 999px; padding: 0.05rem 0.45rem; font-size: 0.75rem; margin-left: 0.25rem; }
          .b-good { background: #E5F2EA; color: #1E6B45; }
          .b-mid { background: #F8F1DE; color: #8A5A12; }
          .b-bad { background: #F8E6E1; color: #8C2F1B; }
          .b-legacy { background: #F3E6E8; color: #7A3142; }
          .b-sf { background: #E7EEF6; color: #003057; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_portfolio(folder: str, signature: str, using_sample: bool) -> hm.Portfolio:
    del signature
    return hm.load_portfolio(Path(folder), using_sample=using_sample)


def load_portfolio() -> hm.Portfolio:
    folder = Path(str(st.session_state["data_folder"])).expanduser()
    uploads = st.session_state.get("uploads") or []
    if not folder.exists():
        portfolio = hm.load_portfolio(folder, using_sample=False)
    else:
        portfolio = cached_portfolio(str(folder.resolve()), hm.folder_signature(folder), hm.is_sample_folder(folder))
    if uploads:
        loaded = [(label, frame, []) for label, frame in portfolio.tables.items()]
        loaded.extend(uploads)
        portfolio = hm.assemble(loaded, portfolio.folder, portfolio.using_sample)
    return portfolio


def _github_enabled() -> bool:
    if os.environ.get("DASHBOARD_GITHUB") == "0":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return "streamlit.testing" not in sys.modules


def _activate_github(root: Path) -> None:
    """Point the data folder at the GitHub checkout. The reading code stays local."""
    lib.default_folder = lambda path=root: path
    lib.practicum_folder = lambda path=root: path


def main() -> None:
    st.set_page_config(page_title="Account health handoff", page_icon="◆", layout="wide")
    filled: list[str] = []
    if _github_enabled():
        try:
            with st.spinner("Loading data from GitHub…"):
                root, filled = github_source.sync()
            _activate_github(root)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            st.warning(f"GitHub could not be loaded ({exc}). The local dashboard files are in use.")
    apply_requests()
    inject_css()
    folder = Path(str(st.session_state["data_folder"])).expanduser()
    if practicum_pack.is_practicum_pack(folder):
        if filled:
            st.sidebar.caption(
                "Data loads from github.com/pomosola/AccountHealthDashboard. "
                "These files are not in that repository, so the local practicum folder supplied them: "
                + ", ".join(filled)
                + "."
            )
        else:
            st.sidebar.caption("Data loads from github.com/pomosola/AccountHealthDashboard.")
        practicum_view.render(folder)
        return
    if st.session_state.get("page") not in PAGES:
        st.session_state["page"] = "Home"
    if st.session_state.get("viewer") not in VIEWERS:
        st.session_state["viewer"] = "Strategic"
    st.sidebar.radio("Who is looking", VIEWERS, key="viewer")
    st.sidebar.caption(VIEWER_COPY[st.session_state["viewer"]])
    st.sidebar.radio("Go to", PAGES, key="page")
    data_controls()
    portfolio = load_portfolio()
    scoped = scope_controls(portfolio)
    viewer = st.session_state["viewer"]
    page = st.session_state["page"]
    context_strip(portfolio, scoped, viewer)
    if page == "Home":
        home(portfolio, scoped, viewer)
    elif page == "Health scores":
        health_page(portfolio, scoped, viewer)
    elif page == "Hierarchy and people":
        hierarchy_page(portfolio, scoped, viewer)
    elif page == "Journey and migration":
        journey_page(portfolio, scoped, viewer)


def data_controls() -> None:
    st.sidebar.markdown("**Data**")
    st.sidebar.text_input("Folder", key="data_folder")
    column_a, column_b = st.sidebar.columns(2)
    column_a.button("Practicum folder", on_click=request_folder, args=(str(lib.practicum_folder()),), width="stretch")
    column_b.button("Bundled extract", on_click=request_folder, args=(str(lib.bundled_data_dir() / "health"),), width="stretch")
    st.sidebar.button("Reload files", on_click=st.cache_data.clear)
    uploaded = st.sidebar.file_uploader("Add a file from this computer", accept_multiple_files=True)
    parsed = []
    if uploaded:
        for item in uploaded:
            try:
                for label, frame, notes in lib.load_uploaded(item.name, item.getvalue()):
                    parsed.append((label, frame, notes))
            except lib.DataLoadError as exc:
                st.sidebar.error(str(exc))
    st.session_state["uploads"] = parsed


def scope_controls(portfolio: hm.Portfolio) -> pd.DataFrame:
    latest = portfolio.latest
    if not latest.empty:
        st.sidebar.markdown("**Scope**")
        st.sidebar.caption("Filters apply on every page. Leave them empty to see the whole book.")
        segments = sorted(value for value in latest["segment"].dropna().unique() if str(value).strip())
        owners = sorted(value for value in latest["owner"].dropna().unique() if str(value).strip())
        families = latest.drop_duplicates("family_id")[["family_id", "family_name"]].sort_values("family_name")
        family_names = dict(zip(families["family_id"], families["family_name"]))
        st.sidebar.multiselect("Segment", segments, key="segments")
        st.sidebar.multiselect("Owner", owners, key="owners")
        st.sidebar.multiselect("Churn band", list(hm.BANDS), key="bands")
        st.sidebar.multiselect("Family", families["family_id"].tolist(), format_func=lambda value: family_names.get(value, value), key="families")
        st.sidebar.button("Reset scope", on_click=reset_scope)
    scoped = hm.scope_accounts(
        latest,
        st.session_state.get("segments") or None,
        st.session_state.get("owners") or None,
        st.session_state.get("bands") or None,
        st.session_state.get("families") or None,
    )
    if not latest.empty:
        st.sidebar.caption(f"{len(scoped)} of {len(latest)} accounts in scope.")
    if portfolio.errors:
        for error in portfolio.errors:
            st.sidebar.warning(error)
    return scoped


def context_strip(portfolio: hm.Portfolio, scoped: pd.DataFrame, viewer: str) -> None:
    as_of = hm.pretty_date(portfolio.extract_as_of)
    cutover = hm.pretty_date(portfolio.cutover)
    st.markdown(
        (
            "<div class='strip'>"
            f"<span><b>{html.escape(viewer)} view</b></span>"
            f"<span>{len(scoped)} accounts in scope</span>"
            f"<span>Extract as of {html.escape(as_of)}</span>"
            f"<span>Cutover {html.escape(cutover)}</span>"
            f"<span>Go-forward system: {html.escape(portfolio.go_forward_system)}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def home(portfolio: hm.Portfolio, scoped: pd.DataFrame, viewer: str) -> None:
    st.markdown(
        """
        <div class='hero'>
          <div class='eyebrow'>Applied analytics practicum · customer health</div>
          <h1>Account health handoff</h1>
          <p>Churn health and expansion potential for the Salesforce instance, with the drivers, the parent and child accounts, and the bridge across the migration.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    viewer_switch(viewer)
    if portfolio.using_sample:
        st.markdown(
            "<div class='note'><b>Bundled extract.</b> This machine does not have "
            "<code>C:\\Users\\pomos\\OneDrive\\Documents\\Ga_Tech_Courses\\Applied_Analytics_Practicum\\Dashboard datasets</code>. "
            "The pages below use the sample health book in <code>data/health</code> so you can see the layout. "
            "On your PC the app reads that practicum folder first. You can also paste a folder path in the sidebar.</div>",
            unsafe_allow_html=True,
        )
    elif not portfolio.ready:
        st.markdown(
            "<div class='note'>The folder did not contain a recognizable account, score, or sub-score table. "
            "A health file needs an account id and either a churn or expansion score, or domain sub-scores.</div>",
            unsafe_allow_html=True,
        )
    handoff_banner(portfolio)
    if not scoped.empty:
        kpis(portfolio, scoped, viewer)
        if viewer == "Strategic":
            strategic_home(portfolio, scoped)
        elif viewer == "Operational":
            operational_home(portfolio, scoped)
        else:
            executor_home(portfolio, scoped)
        spotlight(portfolio, scoped)
    start_here()
    file_map(portfolio)


def viewer_switch(active: str) -> None:
    columns = st.columns(3)
    for column, name in zip(columns, VIEWERS):
        with column:
            css = "viewer-card active" if name == active else "viewer-card"
            st.markdown(
                f"<div class='{css}'><h3>{html.escape(name)}</h3><p>{html.escape(VIEWER_COPY[name])}</p></div>",
                unsafe_allow_html=True,
            )
            if name == active:
                st.caption("This is the view you are in.")
            else:
                st.button(f"Switch to {name}", on_click=request_viewer, args=(name,), key=f"switch-{name}", width="stretch")


def handoff_banner(portfolio: hm.Portfolio) -> None:
    legacy_actions = 0 if portfolio.actions.empty else int(portfolio.actions["legacy_only"].sum())
    st.markdown(
        "<div class='note'><b>Handoff.</b> "
        f"{html.escape(portfolio.handoff_note)} "
        f"Legacy-only items in the current book: {legacy_actions}. They stay visible and are badged. They are not fields to maintain in {html.escape(portfolio.go_forward_system)}.</div>",
        unsafe_allow_html=True,
    )


def kpis(portfolio: hm.Portfolio, scoped: pd.DataFrame, viewer: str) -> None:
    arr = pd.to_numeric(scoped["arr"], errors="coerce").fillna(0)
    critical = scoped["churn_band"].eq("Critical")
    ready = scoped["expansion_band"].eq("Ready") & ~critical
    low = scoped["confidence"].eq("Low")
    health = _weighted(scoped, "churn_health")
    expansion = _weighted(scoped, "expansion_score")
    cards = [
        ("Accounts", f"{len(scoped)}", "Parent, child, and standalone"),
        ("Book ARR", hm.money(arr.sum()), "Each account counted once"),
        ("Churn health", "—" if health is None else f"{health:.1f}", "ARR-weighted. Higher is healthier."),
        ("Critical ARR", hm.money(arr[critical].sum()), f"{int(critical.sum())} accounts under 55"),
        ("Expansion-ready ARR", hm.money(arr[ready].sum()), f"{int(ready.sum())} accounts at 70 or above, and not critical"),
        ("Low confidence", str(int(low.sum())), "Gaps, tiny samples, or legacy-only points"),
    ]
    if viewer == "Process executor":
        mine = portfolio.actions
        if not mine.empty:
            mine = mine[mine["account_id"].isin(scoped["account_id"]) & mine["audience"].eq("Process executor")]
        salesforce = 0 if mine.empty else int((~mine["legacy_only"]).sum())
        legacy = 0 if mine.empty else int(mine["legacy_only"].sum())
        cards[4] = ("Salesforce actions", str(salesforce), "Steps to log on the new instance")
        cards[5] = ("Legacy residue", str(legacy), "Do not carry these forward")
    html_cards = "".join(
        f"<div class='kpi'><div class='k'>{html.escape(label)}</div><div class='v'>{html.escape(value)}</div><div class='n'>{html.escape(note)}</div></div>"
        for label, value, note in cards
    )
    st.markdown(f"<div class='kpi-grid'>{html_cards}</div>", unsafe_allow_html=True)
    st.caption(
        f"Portfolio scores are ARR-weighted across the accounts in scope, as of {hm.pretty_date(portfolio.extract_as_of)}. "
        "Low-confidence accounts stay in the average and are counted in the last card. "
        + (
            "Weights come from score_model."
            if not portfolio.model_is_default
            else "No weight file was found, so the standard domain weights are used and labeled on the score page."
        )
    )


def strategic_home(portfolio: hm.Portfolio, scoped: pd.DataFrame) -> None:
    left, right = st.columns((1.4, 1))
    with left:
        trend = hm.portfolio_trend(portfolio, scoped["account_id"].tolist())
        figure = trend_figure(trend, portfolio.cutover, "Book health across the migration")
        if figure is None:
            st.info("No dated score history was in the folder, so there is no longitudinal line yet.")
        else:
            st.plotly_chart(figure, width="stretch", theme="streamlit", config=CHART, key="home-trend")
            st.caption("Each point is the ARR-weighted score of the accounts that had a score that month. The dashed line is the cutover.")
    with right:
        mix = migration_mix(portfolio, scoped["account_id"].tolist())
        if mix.empty:
            st.info("No migration bridge was in the folder.")
        else:
            st.markdown("**Latest reconciled source**")
            st.dataframe(mix, width="stretch", hide_index=True)
            st.caption("After cutover the go-forward source is Salesforce. A legacy source here means the account was not moved, or the bridge still depends on the old CRM.")
    families = (
        scoped.drop_duplicates("family_id")[["family_name", "family_churn_health", "family_expansion", "family_arr", "family_accounts"]]
        .sort_values("family_churn_health")
        .rename(columns={"family_name": "Family", "family_churn_health": "Family churn health", "family_expansion": "Family expansion", "family_arr": "Family ARR", "family_accounts": "Accounts"})
    )
    families["Family ARR"] = families["Family ARR"].map(hm.money)
    st.markdown("**Families, not a flat list**")
    st.dataframe(families, width="stretch", hide_index=True)
    st.caption("A family score is the ARR-weighted mean of the parent and its children. Open Hierarchy and people to see the nesting.")
    actions = portfolio.actions
    if not actions.empty:
        actions = actions[actions["account_id"].isin(scoped["account_id"])]
        st.caption(
            f"The operating team has {int((actions.audience == 'Operational').sum())} account-manager items and "
            f"{int((actions.audience == 'Process executor').sum())} execution steps in this scope, "
            f"including {int(actions.legacy_only.sum())} legacy-only items. Switch to those viewers to work the list."
        )


def operational_home(portfolio: hm.Portfolio, scoped: pd.DataFrame) -> None:
    st.markdown("**Your book**")
    st.caption("Sorted by churn health. The next action is the highest-priority step already on the list for that account.")
    table = book_table(portfolio, scoped)
    st.dataframe(table, width="stretch", hide_index=True, height=460)
    stale = scoped[pd.to_numeric(scoped["freshness_days"], errors="coerce").fillna(0).gt(hm.FRESH_DAYS)]
    low = scoped[scoped["confidence"].eq("Low")]
    if not stale.empty or not low.empty:
        st.markdown(
            f"<div class='caveat'>{len(stale)} stale score{'s' if len(stale) != 1 else ''} and {len(low)} low-confidence score{'s' if len(low) != 1 else ''} are in this book. Do not review them as if they were current.</div>",
            unsafe_allow_html=True,
        )
    st.download_button("Download this book", table.to_csv(index=False).encode("utf-8"), "account_health_book.csv", "text/csv")


def executor_home(portfolio: hm.Portfolio, scoped: pd.DataFrame) -> None:
    actions = action_frame(portfolio, scoped, "Process executor")
    if actions.empty:
        st.info("No execution steps for this scope.")
        return
    current = actions[~actions["Legacy only"].eq("Yes")]
    residue = actions[actions["Legacy only"].eq("Yes")]
    st.markdown("**Do this in Salesforce**")
    st.caption("These steps belong on the new instance. Log the activity there. Do not update the legacy CRM.")
    st.dataframe(current.drop(columns=["Legacy only"]), width="stretch", hide_index=True, height=360)
    st.markdown("**Legacy residue — do not carry forward**")
    st.caption(f"These items exist only in {portfolio.legacy_system}. They are listed so they are not mistaken for Salesforce work.")
    if residue.empty:
        st.success(f"Nothing in this scope is legacy-only. The work queue is entirely on {portfolio.go_forward_system}.")
    else:
        st.dataframe(residue, width="stretch", hide_index=True)


def spotlight(portfolio: hm.Portfolio, scoped: pd.DataFrame) -> None:
    account_id = hm.suggested_account(scoped)
    if account_id is None:
        return
    row = scoped[scoped["account_id"].eq(account_id)].iloc[0]
    st.markdown(f"### Read this score first: {html.escape(str(row['account_name']))}")
    st.caption("This is the account in scope with the clearest caveat — a data gap if there is one, otherwise the weakest churn health. Health scores opens here.")
    basis_block(row)
    columns = st.columns(3)
    columns[0].metric("Churn health", _one(row["churn_health"]), row["churn_band"])
    columns[1].metric("Churn risk", _one(row["churn_risk"]), "100 minus churn health")
    columns[2].metric("Expansion", _one(row["expansion_score"]), row["expansion_band"])
    if "account_id" not in st.session_state:
        st.session_state["account_id"] = account_id


def start_here() -> None:
    st.markdown("### Start here")
    st.caption("You do not need a walkthrough. Four stops cover the review.")
    columns = st.columns(4)
    steps = [
        ("1. A score and its drivers", "Health scores", "See the sub-scores add up."),
        ("2. Parent, child, contact", "Hierarchy and people", "The tree is the navigation."),
        ("3. Across the migration", "Journey and migration", "Legacy, bridge, then Salesforce."),
        ("4. The work queue", "Home", "Switch to Process executor for the steps."),
    ]
    for column, (title, page, body) in zip(columns, steps):
        with column:
            st.markdown(f"**{title}**")
            st.caption(body)
            if page == "Home":
                st.button("Open the queue", on_click=request_viewer, args=("Process executor",), key="start-queue", width="stretch")
            else:
                st.button(page, on_click=request_page, args=(page,), key=f"start-{page}", width="stretch")


def file_map(portfolio: hm.Portfolio) -> None:
    st.markdown("### What was read")
    if not portfolio.files:
        st.caption("No supported files in the folder.")
        return
    rows = [{"File": item.label, "Rows": item.rows, "Used for": ", ".join(item.roles)} for item in portfolio.files]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if portfolio.gaps:
        st.caption("Not found, so the related panel explains the gap: " + ", ".join(portfolio.gaps) + ".")
    else:
        st.caption("Accounts, scores, sub-scores, drivers, people, touchpoints, and the migration bridge were all found.")
    st.caption(f"Folder: {portfolio.folder}")


def health_page(portfolio: hm.Portfolio, scoped: pd.DataFrame, viewer: str) -> None:
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>{html.escape(viewer)} view</div><h1>Health scores</h1><p>Churn health and expansion potential, the sub-scores that add up to them, and the drivers under each sub-score.</p></div>",
        unsafe_allow_html=True,
    )
    if viewer == "Strategic":
        st.caption("Use this page to see whether a portfolio number is built out of a full set of domains, or out of a gap.")
    elif viewer == "Operational":
        st.caption("Use this page on one account before a review. The band, the as-of date, and the confidence are part of the number.")
    else:
        st.caption("Use this page to see why a task is on your list. Act in Salesforce. Do not treat a small-sample role score as stable.")
    row = pick_account(scoped)
    if row is None:
        return
    if viewer == "Process executor":
        steps = action_frame(portfolio, scoped[scoped["account_id"].eq(row["account_id"])], "Process executor")
        st.markdown("**Steps for this account**")
        if steps.empty:
            st.caption("No execution step is open on this account.")
        else:
            st.dataframe(steps, width="stretch", hide_index=True)
    columns = st.columns(4)
    columns[0].metric("Churn health", _one(row["churn_health"]), row["churn_band"])
    columns[1].metric("Churn risk", _one(row["churn_risk"]), "100 minus churn health")
    columns[2].metric("Expansion potential", _one(row["expansion_score"]), row["expansion_band"])
    columns[3].metric("Confidence", str(row["confidence"]), fresh_label(row["freshness_days"]))
    basis_block(row)
    salesforce_only = hm._maybe_float(row.get("churn_health_salesforce_only")) if hasattr(row, "get") else None
    if salesforce_only is not None and hm._maybe_float(row.get("churn_health_legacy_points")):
        st.markdown(
            f"<div class='caveat'>Salesforce-only churn health is <b>{salesforce_only:.1f}</b>. The published number still includes legacy-only driver points. Use {salesforce_only:.1f} if you are scoring the account on the new instance alone.</div>",
            unsafe_allow_html=True,
        )
    left, right = st.columns(2)
    with left:
        render_rollup(portfolio, row["account_id"], hm.CHURN, "Churn health rollup")
    with right:
        render_rollup(portfolio, row["account_id"], hm.EXPANSION, "Expansion rollup")
    st.markdown("**Drivers**")
    st.caption("Driver impacts add up to the sub-score. A legacy-only driver is evidence that was not rebuilt in Salesforce.")
    driver_tabs = st.tabs(["Churn drivers", "Expansion drivers"])
    for tab, outcome in zip(driver_tabs, (hm.CHURN, hm.EXPANSION)):
        with tab:
            drivers = hm.driver_table(portfolio, row["account_id"], outcome)
            if drivers.empty:
                st.info("No drivers for this score in the extract.")
            else:
                st.dataframe(drivers, width="stretch", hide_index=True)
                totals = drivers.groupby("Domain")["Impact"].sum()
                st.caption("Impact by domain: " + ", ".join(f"{domain} {value:.0f}" for domain, value in totals.items()) + ".")
    people_block(portfolio, row, include_parent=True)
    if viewer != "Process executor":
        history = hm.account_history(portfolio, row["account_id"])
        book = hm.portfolio_trend(portfolio, scoped["account_id"].tolist())
        figure = account_trend_figure(history, book, portfolio.cutover, str(row["account_name"]))
        if figure is not None:
            st.plotly_chart(figure, width="stretch", theme="streamlit", config=CHART, key=f"acct-trend-{row['account_id']}")
            st.caption("The account line is its own score. The book line is the ARR-weighted scope. Both use the reconciled go-forward number after cutover when the bridge says Salesforce.")


def hierarchy_page(portfolio: hm.Portfolio, scoped: pd.DataFrame, viewer: str) -> None:
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>{html.escape(viewer)} view</div><h1>Hierarchy and people</h1><p>Parents, child accounts, and the contacts on them. Family health is a rollup. It does not replace the account score.</p></div>",
        unsafe_allow_html=True,
    )
    if scoped.empty:
        st.info("No accounts in this scope.")
        return
    families = scoped.drop_duplicates("family_id").sort_values("family_name")
    names = dict(zip(families["family_id"], families["family_name"]))
    current_family = st.session_state.get("family_id")
    account_id = st.session_state.get("account_id")
    if account_id in set(scoped["account_id"]):
        current_family = scoped.loc[scoped["account_id"].eq(account_id), "family_id"].iloc[0]
    options = families["family_id"].tolist()
    if current_family not in options:
        current_family = options[0]
    choice = st.selectbox("Family", options, index=options.index(current_family), format_func=lambda value: names.get(value, value), key="family_pick")
    st.session_state["family_id"] = choice
    family = portfolio.latest[portfolio.latest["family_id"].eq(choice)].copy()
    family = family.assign(_order=family["hierarchy_level"].map({"Parent": 0, "Standalone": 0, "Child": 1}).fillna(2)).sort_values(["_order", "account_name"])
    head = family.iloc[0]
    st.markdown(
        f"**{html.escape(str(head['family_name']))}** family churn health is **{_one(head['family_churn_health'])}**, "
        f"expansion **{_one(head['family_expansion'])}**, on **{hm.money(head['family_arr'])}** ARR across {int(head['family_accounts'])} accounts."
    )
    st.caption("The family number is the ARR-weighted mean of every account in the family, including the parent. Each account below keeps its own score.")
    if viewer == "Strategic":
        st.caption("Strategic read: compare the family number with the weakest child before you treat the parent as healthy.")
    elif viewer == "Process executor":
        st.caption("Executor read: call the person on the account the task names. Sponsors often sit on the parent, not the child site.")
    for _, account in family.iterrows():
        child = " child" if account["hierarchy_level"] == "Child" else ""
        share = ""
        total = hm._maybe_float(account["family_arr"]) or 0
        own = hm._maybe_float(account["arr"]) or 0
        if total:
            share = f" · {own / total:.0%} of family ARR"
        badges = (
            f"<span class='badge {band_class(account['churn_band'])}'>{html.escape(str(account['churn_band']))} {html.escape(_one(account['churn_health']))}</span>"
            f"<span class='badge {band_class(account['expansion_band'])}'>Expansion {html.escape(_one(account['expansion_score']))}</span>"
            f"<span class='badge {band_class(str(account['confidence']))}'>{html.escape(str(account['confidence']))} confidence</span>"
        )
        system = str(account.get("system_of_record") or "")
        if system.lower().startswith("legacy"):
            badges += f"<span class='badge b-legacy'>{html.escape(system)}</span>"
        else:
            badges += f"<span class='badge b-sf'>{html.escape(system or portfolio.go_forward_system)}</span>"
        st.markdown(
            f"<div class='tree-card{child}'><h3>{html.escape(str(account['hierarchy_level']))}: {html.escape(str(account['account_name']))}</h3>"
            f"<div>{badges}<span> · {html.escape(hm.money(account['arr']))}{html.escape(share)} · Owner {html.escape(str(account['owner'] or '—'))} · As of {html.escape(hm.pretty_date(account['as_of_date']))}</span></div>"
            f"{people_html(portfolio, account)}</div>",
            unsafe_allow_html=True,
        )
    table = family[["account_name", "hierarchy_level", "parent_account_name", "owner", "arr", "churn_health", "expansion_score", "confidence", "salesforce_account_id", "legacy_account_id", "system_of_record"]].rename(
        columns={"account_name": "Account", "hierarchy_level": "Level", "parent_account_name": "Parent", "owner": "Owner", "arr": "ARR", "churn_health": "Churn health", "expansion_score": "Expansion", "confidence": "Confidence", "salesforce_account_id": "Salesforce id", "legacy_account_id": "Legacy id", "system_of_record": "System"}
    )
    table["ARR"] = table["ARR"].map(hm.money)
    st.markdown("**Same family, as a table**")
    st.dataframe(table, width="stretch", hide_index=True)
    unresolved = []
    if "unresolved_parent" in portfolio.accounts.columns:
        unresolved = portfolio.accounts.loc[portfolio.accounts["unresolved_parent"].fillna(False), "account_name"].tolist()
    if unresolved:
        st.markdown(
            "<div class='caveat'>These accounts name a parent that is not in the account file: "
            + html.escape(", ".join(map(str, unresolved)))
            + ".</div>",
            unsafe_allow_html=True,
        )


def journey_page(portfolio: hm.Portfolio, scoped: pd.DataFrame, viewer: str) -> None:
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>{html.escape(viewer)} view</div><h1>Journey and migration</h1><p>Touchpoints from the legacy CRM through cutover and onto Salesforce, plus the reconciled health series.</p></div>",
        unsafe_allow_html=True,
    )
    if viewer == "Strategic":
        st.caption("Look for a break at cutover, then for accounts whose reconciled source is still the legacy CRM.")
    elif viewer == "Operational":
        st.caption("Read the journey before a QBR. A score that jumps at cutover needs the bridge note, not a new story about the customer.")
    else:
        st.caption("Log the next touch in Salesforce. A row badged legacy-only has no home on the new instance.")
    row = pick_account(scoped)
    if row is None:
        return
    touches = account_touches(portfolio, str(row["account_id"]))
    left, right = st.columns(2)
    with left:
        if touches.empty:
            st.info("No touchpoints for this account.")
        else:
            stage_order = [stage for stage in hm.STAGE_ORDER if stage in set(touches["journey_stage"])]
            stage_order += sorted(set(touches["journey_stage"]) - set(stage_order))
            plot = touches.copy()
            plot["Legacy only"] = plot["legacy_only"].map(lambda value: "Legacy only" if bool(value) else portfolio.go_forward_system)
            figure = px.scatter(
                plot,
                x="touch_date",
                y="journey_stage",
                color="phase",
                symbol="Legacy only",
                hover_data=["channel", "summary", "system", "owner"],
                color_discrete_sequence=["#5C6B7A", "#C65D3B", "#003057"],
                title="Touchpoints",
            )
            figure.update_yaxes(categoryorder="array", categoryarray=stage_order, automargin=True)
            figure.update_xaxes(automargin=True)
            figure.update_layout(height=420, margin=dict(l=8, r=8, t=48, b=8), legend=dict(orientation="h", y=-0.25))
            if portfolio.cutover is not None:
                figure.add_vline(x=portfolio.cutover, line_dash="dash", line_color="#C65D3B")
            st.plotly_chart(figure, width="stretch", theme="streamlit", config=CHART, key=f"journey-{row['account_id']}")
    with right:
        bridge = account_bridge(portfolio, str(row["account_id"]))
        figure = bridge_figure(bridge, portfolio.cutover)
        if figure is None:
            st.info("No migration bridge rows for this account. The timeline on the left is the longitudinal record.")
        else:
            st.plotly_chart(figure, width="stretch", theme="streamlit", config=CHART, key=f"bridge-{row['account_id']}")
            st.caption("Before cutover the legacy score is the reconciled number. The cutover month is the average of the frozen legacy score and the first Salesforce score. After cutover, Salesforce wins unless the account was never migrated.")
    if not touches.empty:
        show = touches.sort_values("touch_date")[["touch_date", "journey_stage", "phase", "channel", "summary", "system", "legacy_only", "owner"]].rename(
            columns={"touch_date": "Date", "journey_stage": "Stage", "phase": "Phase", "channel": "Channel", "summary": "What happened", "system": "System", "legacy_only": "Legacy only", "owner": "Owner"}
        )
        show["Date"] = show["Date"].map(hm.pretty_date)
        show["Legacy only"] = show["Legacy only"].map(lambda value: "Yes" if bool(value) else "")
        st.dataframe(show, width="stretch", hide_index=True, height=320)
    bridge = account_bridge(portfolio, str(row["account_id"]))
    if not bridge.empty:
        show = bridge[["period_month", "legacy_health", "salesforce_health", "reconciled_health", "source_used", "variance", "note"]].rename(
            columns={"period_month": "Month", "legacy_health": "Legacy health", "salesforce_health": "Salesforce health", "reconciled_health": "Reconciled", "source_used": "Source used", "variance": "Salesforce minus legacy", "note": "Note"}
        )
        show["Month"] = show["Month"].map(hm.pretty_date)
        st.markdown("**Reconciled series**")
        st.dataframe(show, width="stretch", hide_index=True, height=320)


def pick_account(scoped: pd.DataFrame) -> pd.Series | None:
    if scoped.empty:
        st.info("No accounts in this scope. Reset the scope in the sidebar.")
        return None
    ids = scoped["account_id"].tolist()
    labels = {row.account_id: account_label(row) for row in scoped.itertuples(index=False)}
    current = st.session_state.get("account_id")
    if current not in ids:
        current = hm.suggested_account(scoped) or ids[0]
    choice = st.selectbox("Account", ids, index=ids.index(current), format_func=lambda value: labels[value], key="account_pick")
    st.session_state["account_id"] = choice
    return scoped[scoped["account_id"].eq(choice)].iloc[0]


def account_label(row) -> str:
    prefix = f"{row.family_name} / " if getattr(row, "hierarchy_level", "") == "Child" else ""
    health = "—" if pd.isna(row.churn_health) else f"{float(row.churn_health):.0f}"
    return f"{prefix}{row.account_name} · {row.churn_band} {health}"


def basis_block(row: pd.Series) -> None:
    items = [
        ("As of", hm.pretty_date(row["as_of_date"])),
        ("Freshness", fresh_label(row["freshness_days"])),
        ("Confidence", str(row["confidence"])),
        ("Evidence", str(row["evidence_label"])),
        ("Based on", str(row["basis"])),
        ("Source", str(row["source"])),
    ]
    body = "".join(f"<div><span>{html.escape(label)}</span>{html.escape(value)}</div>" for label, value in items)
    st.markdown(f"<div class='basis'>{body}</div>", unsafe_allow_html=True)
    notes = row["caveat_list"] if isinstance(row["caveat_list"], list) else []
    for note in notes:
        st.markdown(f"<div class='caveat'>{html.escape(str(note))}</div>", unsafe_allow_html=True)
    if not notes:
        st.caption("No caveat on this score. It is inside the freshness window, the domains are present, and the sample is large enough for the confidence rule.")


def render_rollup(portfolio: hm.Portfolio, account_id: str, outcome: str, title: str) -> None:
    table = hm.rollup_table(portfolio, account_id, outcome)
    st.markdown(f"**{title}**")
    if table.empty:
        st.info("No sub-scores for this outcome.")
        return
    scored = table[table["Contribution"].notna()]
    total = scored["Contribution"].sum()
    parts = " + ".join(f"{value:.1f}" for value in scored["Contribution"])
    missing = table.loc[table["Gap"].eq("Data gap"), "Domain"].tolist()
    show = table.drop(columns=["Definition"]).copy()
    show["Model weight"] = show["Model weight"].map(lambda value: f"{float(value):.0%}")
    show["Effective weight"] = show["Effective weight"].map(lambda value: "—" if pd.isna(value) else f"{float(value):.1%}")
    show["Sub-score"] = show["Sub-score"].map(lambda value: "—" if pd.isna(value) else f"{float(value):.0f}")
    show["Contribution"] = show["Contribution"].map(lambda value: "—" if pd.isna(value) else f"{float(value):.1f}")
    show["Evidence"] = show["Evidence"].map(lambda value: "—" if pd.isna(value) else f"{int(value)}")
    show["Legacy points"] = show["Legacy points"].map(lambda value: "" if pd.isna(value) else f"{float(value):.0f}")
    st.dataframe(show, width="stretch", hide_index=True)
    sentence = f"{total:.1f} = {parts}."
    if missing:
        sentence += " Missing: " + ", ".join(missing) + ". Their model weight was spread across the domains that were scored, which is why the effective weight differs from the model weight."
    else:
        sentence += " Every domain was scored, so the effective weight matches the model weight."
    definition = table.loc[table["Definition"].astype(str).str.len().gt(0), ["Domain", "Definition"]]
    st.caption(sentence)
    if not definition.empty:
        with st.expander("What each domain means"):
            for _, item in definition.iterrows():
                st.markdown(f"**{item['Domain']}.** {item['Definition']}")


def people_block(portfolio: hm.Portfolio, account: pd.Series, include_parent: bool) -> None:
    st.markdown("**People and role scores**")
    render_people(portfolio, account, nested=False, include_parent=include_parent)


def people_html(portfolio: hm.Portfolio, account: pd.Series) -> str:
    roles = portfolio.role_scores
    if roles.empty or "account_id" not in roles.columns:
        return ""
    ids = [account["account_id"]]
    if account.get("parent_account_id"):
        ids.append(account["parent_account_id"])
    group = roles[roles["account_id"].isin(ids)]
    lines = []
    for _, person in group.iterrows():
        held = "on the parent" if person["account_id"] != account["account_id"] else str(person.get("role") or "")
        sample = person.get("sample_size")
        sample_txt = "" if pd.isna(sample) else f", {int(sample)} touches"
        notes = person["caveat_list"] if isinstance(person.get("caveat_list"), list) else []
        caveat = " — " + notes[0] if notes else ""
        legacy = " · legacy only" if person.get("legacy_only") is True else ""
        lines.append(
            f"<div class='person'>{html.escape(str(person.get('contact_name') or 'Contact'))}, {html.escape(held)} · relationship {_one(person.get('relationship_health'))} · {html.escape(str(person.get('confidence') or ''))} confidence{html.escape(sample_txt)}{html.escape(legacy)}{html.escape(caveat)}</div>"
        )
    return "".join(lines)


def render_people(portfolio: hm.Portfolio, account: pd.Series, nested: bool, include_parent: bool = True) -> None:
    roles = portfolio.role_scores
    if roles.empty:
        if not nested:
            st.caption("No role-level scores in the extract.")
        return
    ids = [account["account_id"]]
    if include_parent and account.get("parent_account_id"):
        ids.append(account["parent_account_id"])
    group = roles[roles["account_id"].isin(ids)] if "account_id" in roles.columns else roles.iloc[0:0]
    if group.empty:
        if not nested:
            st.caption("No scored contacts on this account.")
        return
    if nested:
        st.markdown(people_html(portfolio, account), unsafe_allow_html=True)
        return
    show = pd.DataFrame(
        {
            "Person": group.get("contact_name", pd.Series(dtype=str)),
            "Role": group.get("role", pd.Series(dtype=str)),
            "Held on": group["account_id"].map(lambda value: "This account" if value == account["account_id"] else "Parent"),
            "Relationship": group.get("relationship_health"),
            "Driver": group.get("driver", pd.Series(dtype=str)),
            "Churn influence": group.get("churn_influence", pd.Series(dtype=str)),
            "Expansion influence": group.get("expansion_influence", pd.Series(dtype=str)),
            "Touches": group.get("sample_size"),
            "Last touch": group["last_touch_date"].map(hm.pretty_date) if "last_touch_date" in group.columns else "",
            "Confidence": group.get("confidence"),
            "Caveat": group["caveat_list"].map(lambda value: " ".join(value) if isinstance(value, list) else "") if "caveat_list" in group.columns else "",
        }
    )
    st.dataframe(show, width="stretch", hide_index=True)


def book_table(portfolio: hm.Portfolio, scoped: pd.DataFrame) -> pd.DataFrame:
    actions = portfolio.actions
    next_step = {}
    if not actions.empty:
        ordered = actions.sort_values(["priority", "audience"])
        for account_id, group in ordered.groupby("account_id", sort=False):
            next_step[account_id] = str(group.iloc[0]["action"])
    table = pd.DataFrame(
        {
            "Account": scoped["account_name"],
            "Level": scoped["hierarchy_level"],
            "Parent": scoped["parent_account_name"],
            "Owner": scoped["owner"],
            "ARR": scoped["arr"].map(hm.money),
            "Churn health": scoped["churn_health"].map(_one),
            "Band": scoped["churn_band"],
            "Expansion": scoped["expansion_score"].map(_one),
            "Confidence": scoped["confidence"],
            "As of": scoped["as_of_date"].map(hm.pretty_date),
            "System": scoped["system_of_record"],
            "Next action": scoped["account_id"].map(lambda value: next_step.get(value, "")),
        }
    )
    return table.sort_values(["Churn health", "Account"], key=lambda column: pd.to_numeric(column, errors="coerce") if column.name == "Churn health" else column)


def action_frame(portfolio: hm.Portfolio, scoped: pd.DataFrame, audience: str) -> pd.DataFrame:
    actions = portfolio.actions
    if actions.empty or scoped.empty:
        return pd.DataFrame()
    group = actions[actions["account_id"].isin(scoped["account_id"]) & actions["audience"].eq(audience)].copy()
    if group.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Priority": group["priority"].map(lambda value: f"P{int(value)}"),
            "Account": group["account_name"],
            "Person": group["contact_name"],
            "Role": group["role"],
            "Action": group["action"],
            "Why": group["why"],
            "Score as of": group["as_of"],
            "Confidence": group["confidence"],
            "System": group["system"],
            "Legacy only": group["legacy_only"].map(lambda value: "Yes" if bool(value) else ""),
        }
    )


def migration_mix(portfolio: hm.Portfolio, account_ids: list[str]) -> pd.DataFrame:
    bridge = portfolio.migration
    if bridge.empty or "account_id" not in bridge.columns:
        return pd.DataFrame()
    frame = bridge[bridge["account_id"].isin(account_ids)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["period_month"] = pd.to_datetime(frame["period_month"])
    latest = frame.sort_values("period_month").groupby("account_id", as_index=False).tail(1)
    counts = latest["source_used"].value_counts()
    return pd.DataFrame({"Source used on the latest bridged month": counts.index, "Accounts": counts.values})


def account_touches(portfolio: hm.Portfolio, account_id: str) -> pd.DataFrame:
    touches = portfolio.touchpoints
    if touches.empty:
        return touches
    return touches[touches["account_id"].eq(account_id)].copy()


def account_bridge(portfolio: hm.Portfolio, account_id: str) -> pd.DataFrame:
    bridge = portfolio.migration
    if bridge.empty or "account_id" not in bridge.columns:
        return pd.DataFrame()
    return bridge[bridge["account_id"].eq(account_id)].sort_values("period_month")


def trend_figure(trend: pd.DataFrame, cutover, title: str):
    if trend.empty:
        return None
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=trend["as_of_date"], y=trend["churn_health"], name="Churn health", mode="lines+markers", line=dict(color="#003057", width=3)))
    if "expansion_score" in trend.columns:
        figure.add_trace(go.Scatter(x=trend["as_of_date"], y=trend["expansion_score"], name="Expansion potential", mode="lines+markers", line=dict(color="#B3A369", width=3)))
    _finish(figure, title, cutover)
    return figure


def account_trend_figure(history: pd.DataFrame, book: pd.DataFrame, cutover, account_name: str):
    if history.empty or "churn_health" not in history.columns:
        return None
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=history["as_of_date"], y=history["churn_health"], name=f"{account_name} churn health", mode="lines+markers", line=dict(color="#003057", width=3)))
    if "expansion_score" in history.columns:
        figure.add_trace(go.Scatter(x=history["as_of_date"], y=history["expansion_score"], name=f"{account_name} expansion", mode="lines+markers", line=dict(color="#B3A369", width=3)))
    if not book.empty:
        figure.add_trace(go.Scatter(x=book["as_of_date"], y=book["churn_health"], name="Book churn health", mode="lines", line=dict(color="#5C6B7A", width=2, dash="dot")))
    _finish(figure, "This account against the book", cutover)
    return figure


def bridge_figure(bridge: pd.DataFrame, cutover):
    if bridge.empty:
        return None
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=bridge["period_month"], y=bridge["legacy_health"], name="Legacy CRM", mode="lines", line=dict(color="#A34155", width=2, dash="dot")))
    figure.add_trace(go.Scatter(x=bridge["period_month"], y=bridge["salesforce_health"], name="Salesforce", mode="lines", line=dict(color="#003057", width=2)))
    figure.add_trace(go.Scatter(x=bridge["period_month"], y=bridge["reconciled_health"], name="Reconciled", mode="lines+markers", line=dict(color="#3E7C78", width=3)))
    _finish(figure, "Reconciled health", cutover)
    return figure


def _finish(figure: go.Figure, title: str, cutover) -> None:
    if cutover is not None and not pd.isna(cutover):
        figure.add_vline(x=pd.Timestamp(cutover), line_dash="dash", line_color="#C65D3B")
        figure.add_annotation(x=pd.Timestamp(cutover), y=1.02, yref="paper", text="Cutover", showarrow=False, font=dict(color="#C65D3B", size=12))
    figure.update_yaxes(range=[0, 100], title="Score", automargin=True)
    figure.update_xaxes(automargin=True)
    figure.update_layout(title=title, height=380, margin=dict(l=8, r=8, t=48, b=8), legend=dict(orientation="h", y=-0.22), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


def _weighted(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame["arr"], errors="coerce").fillna(0)
    mask = values.notna()
    if not mask.any():
        return None
    if float(weights[mask].sum()) <= 0:
        return round(float(values[mask].mean()), 1)
    return round(float((values[mask] * weights[mask]).sum() / weights[mask].sum()), 1)


def _one(value) -> str:
    number = hm._maybe_float(value)
    return "—" if number is None else f"{number:.1f}"


def fresh_label(value) -> str:
    number = hm._maybe_float(value)
    if number is None:
        return "No as-of date"
    days = int(number)
    if days <= 0:
        return "Current with the extract"
    if days <= hm.FRESH_DAYS:
        return f"{days} days behind the extract"
    return f"Stale, {days} days behind the extract"


def band_class(value: str) -> str:
    if value in {"Healthy", "Ready", "High"}:
        return "b-good"
    if value in {"Watch", "Developing", "Medium"}:
        return "b-mid"
    if value in {"Critical", "Limited", "Low"}:
        return "b-bad"
    return "b-sf"


if __name__ == "__main__":
    main()
