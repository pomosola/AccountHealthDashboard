from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
ROOT = APP.parent


def visible(app: AppTest) -> str:
    chunks = []
    for name in ("markdown", "caption", "info", "warning"):
        for block in getattr(app, name, []):
            chunks.append(str(getattr(block, "value", "")))
    return "\n".join(chunks)


def test_home_opens_on_the_health_handoff_and_the_three_viewers(monkeypatch):
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(ROOT / "data" / "health"))
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    text = visible(app)
    assert "Account health handoff" in text
    assert "Who reads what" in text
    assert "Harbor Transit" in text
    assert "Salesforce" in text
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
    app.radio(key="page").set_value("Source tables").run()
    assert not app.exception, app.exception


def test_practicum_pack_opens_on_the_historical_scores(monkeypatch):
    pack = ROOT / "data" / "practicum"
    if not (pack / "product_health_score.csv").exists():
        return
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(pack))
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    assert not app.exception, app.exception
    text = visible(app)
    assert "Account health handoff" in text
    assert "1 Jan 2025" in text
    assert "Calibration bands on the held-out test" in text
    assert "Read this before the numbers" not in text
    assert "Who reads what" in text
    assert "Leadership" in text
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
    app.radio(key="page").set_value("Source tables").run()
    assert not app.exception, app.exception
