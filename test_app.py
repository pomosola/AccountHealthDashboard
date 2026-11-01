from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent / "app.py"
ROOT = APP.parent


def visible(app: AppTest) -> str:
    chunks = []
    for name in ("markdown", "caption", "info", "warning"):
        for block in getattr(app, name, []):
            chunks.append(str(getattr(block, "value", "")))
    return "\n".join(chunks)


def test_home_opens_on_the_health_handoff_and_the_three_viewers(monkeypatch):
    health = ROOT / "data" / "health"
    if not health.exists():
        return
    monkeypatch.setattr("dashboard_lib.default_folder", lambda: health)
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    text = "\n".join(block.value for block in app.markdown)
    assert "Account health handoff" in text
    assert "Strategic view" in text
    assert "Harbor Transit" in text
    assert "Salesforce" in text
    app.radio(key="viewer").set_value("Process executor").run()
    assert not app.exception, app.exception
    text = "\n".join(block.value for block in app.markdown)
    assert "Do this in Salesforce" in text
    app.radio(key="page").set_value("Health scores").run()
    assert not app.exception, app.exception
    text = "\n".join(block.value for block in app.markdown)
    assert "Churn health rollup" in text
    assert "Data gap" in text
    app.radio(key="page").set_value("Hierarchy and people").run()
    assert not app.exception, app.exception
    text = "\n".join(block.value for block in app.markdown)
    assert "family churn health" in text
    app.radio(key="page").set_value("Journey and migration").run()
    assert not app.exception, app.exception
    text = "\n".join(block.value for block in app.markdown)
    assert "Reconciled" in text or "Touchpoints" in text


def _practicum_dir() -> Path | None:
    bundled = ROOT / "data" / "practicum"
    external = Path(r"c:\Users\pomos\OneDrive\Documents\Ga_Tech_Courses\Applied_Analytics_Practicum\Dashboard datasets")
    for pack in (bundled, external):
        if (pack / "product_health_score.csv").exists() and (pack / "score_context.csv").exists():
            return pack
    return None


def test_practicum_pack_opens_on_the_historical_scores(monkeypatch):
    pack = _practicum_dir()
    if pack is None:
        return
    monkeypatch.setattr("dashboard_lib.default_folder", lambda: pack)
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    assert not app.exception, app.exception
    text = visible(app)
    assert "Executive Summary: At a Glance" in text
    assert "Account health handoff · 1 of 5" not in text
    assert "What Product reads in the book" not in text
    assert "1 Jan 2025" in text
    assert "300-day" in text
    assert "91.6%" in text
    assert "Product adoption" in text
    assert "Customer Success" in text
    assert "Read this before the numbers" not in text
    app.radio(key="viewer").set_value("Process executor").run()
    assert not app.exception, app.exception
    text = visible(app)
    assert "live lookup" in text.lower()
    app.radio(key="page").set_value("Health scores").run()
    assert not app.exception, app.exception
    text = visible(app)
    assert "Drivers" in text
    assert "no account-specific confidence interval" in text
    assert "held-out bands" in text
    app.radio(key="page").set_value("Hierarchy and people").run()
    assert not app.exception, app.exception
    text = visible(app)
    assert "Validated parent" in text
    app.radio(key="page").set_value("Journey and migration").run()
    assert not app.exception, app.exception
    text = visible(app)
    assert "30 Sep 2025" in text
    app.radio(key="page").set_value("Handoff").run()
    assert not app.exception, app.exception
    text = visible(app)
    assert "legacy-only" in text.lower()
