# Account health handoff

A Streamlit dashboard for the Applied Analytics Practicum extracts. It opens on a home page, then continues in order: one account and its drivers, the parent and people, the journey across the migration, and the Salesforce handoff. The home page states the problem, the provisional 300-day customer behavioral model (180 days of history, then 120 days of outcomes), and the three results for each account — product adoption, product health, and the probabilities of churn, expansion, and upgrade. The same book is cut three ways:

- **Strategic (Product)** — whether the held-out scores can be trusted, and where recorded touchpoints sit around the 30 Sep 2025 migration reference.
- **Operational (Customer Success)** — the historical top-50 expansion and upgrade review queues, plus adoption rows withheld for reconciliation.
- **Process executor (Service Delivery)** — the Salesforce checks to do before any write, the legacy-only contract keys, and the adoption holds.

## Run it

From this folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.

The app reads this folder first when it exists:

```text
C:\Users\pomos\OneDrive\Documents\Ga_Tech_Courses\Applied_Analytics_Practicum\Dashboard datasets
```

Otherwise it uses `DASHBOARD_DATA_DIR` if that path exists, then `data/practicum` when those extracts are present, then the bundled sample in `data/health`.

```powershell
$env:DASHBOARD_DATA_DIR = "C:\Users\pomos\OneDrive\Documents\Ga_Tech_Courses\Applied_Analytics_Practicum\Dashboard datasets"
streamlit run app.py
```

**Reload files** rereads the folder. The sidebar can switch between the practicum folder and the bundled sample. If OneDrive files are online-only, open them once so they download.

## What the practicum scores actually are

These definitions come from `score_context.csv`. The page repeats them with the as-of date, the freshness status, and the caveats.

| Score | Definition | As of |
| --- | --- | --- |
| Full churn probability | Calibrated 120-day full-account churn probability | 1 Jan 2025 historical test |
| Product health | `100 × (1 − churn probability)` | Same test. Near-chance ranking. Not a validated intervention queue. |
| Expansion probability and opportunity score | Probability, and `100 ×` that probability | Same test. This is the usable review ranking. |
| Upgrade probability | Calibrated 120-day upgrade probability | Same test. Rare outcome. |
| Product adoption | `100 × verified installed units / verified contracted units` | Source extract date is blank on purpose. Separate from the models. |

Driver rows are signed log-odds, not points on a 0–100 scale. Intercept plus every contribution, passed through a logistic function, recovers the published probability. Top 3 is a display rank only. There is no account-specific confidence interval. Calibration bands from the held-out test show the group outcome rate next to the account's probability. There is no contact-level or role-level model. Contact activity and product rows stay on the hierarchy page as recorded context: contact identity was not verified, and the product extract date is blank.

Account numbers stay text. Parent links are candidates: validated parent IDs are blank. Contact values are source tokens, and customer role is unknown. 30 Sep 2025 is the owner-confirmed bulk-update reference, not each account's migration date. An administrative end date is not churn.

Salesforce handoff uses the new MCF instance. Account Number is the business identifier. Salesforce Account.Id was not supplied, and live lookup was not performed. Legacy-only contract keys stay flagged. The required owner action is to confirm the Account Number API field and look accounts up before writing.

## Bundled sample

`data/health` is a fictional book used when the practicum extracts are not on the machine. That sample uses weighted domain sub-scores so the layout can be reviewed. It is not the scoring method in the practicum files. Regenerate it with:

```bash
python scripts/make_sample_data.py
```

## Tests

```bash
pytest
```

The practicum checks run when `data/practicum` contains the extracts. Those CSV files are local data and are not part of the git history.
