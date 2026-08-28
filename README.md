# Intelligent Customer Signal Detector

A prototype that reads customer interaction text alongside behavioural telemetry, and produces a **prioritised queue of accounts that need attention** — each with a transparent 0–100 risk score, the contributing signals, and a suggested next action.

**What it is:** an explainable prioritisation heuristic that ranks who to call first.
**What it is not:** a calibrated churn-prediction model. It reports no probabilities and claims none.

Full design rationale, component contracts, and assumptions: **[DESIGN.md](DESIGN.md)**.

---

## 1. The problem and the MVP

Customer operations teams work reactively — by the time a complaint escalates or a cancellation arrives, the window to intervene has closed. The predictive signals already exist (support text, billing behaviour, usage trends, CSAT) but sit in separate systems and get reviewed by hand.

The central design decision is a **strict split of responsibility**:

> **The LLM interprets language. YAML config decides points. Python decides bands, ordering, explanations, and actions.**

The model is never asked to produce a score or rank customers. It performs bounded, schema-validated semantic extraction on the interaction text only; a deterministic engine combines that with telemetry the model never sees.

This matters for more than explainability — it **survives the model being wrong**. A bad sentiment read shifts a score by 15 points out of 100; billing, usage and CSAT still anchor it. An LLM-generated score has no floor and no way to detect the error.

---

## 2. Architecture

The shape is a **fork → merge → fan-out**, not a linear chain.

```
   CSV upload / sample data
              │
              ▼
      data_loader.py ──────────────► per-row validation errors
              │                       (surfaced, never silently dropped)
     RawCustomerRecord
              │
      ┌───────┴────────┐
      ▼                ▼
interaction_text    telemetry
(what they SAID)   (what they DID)
      │             payment · usage
      ▼             CSAT · issue counts
 llm_service.py           │            ◄── telemetry NEVER enters the LLM
 closed schema            │
 live | mock | failed     │
      │                   │
 SemanticSignal           │
      └────────┬──────────┘
               ▼   MERGE
        risk_engine.py
   scoring_v1.yaml · confidence gate
        clamp 0-100 → band
               │
     ┌─────────┴──────────┐
     ▼                    ▼
 reporting.py    recommendation_engine.py
 rationale from     action_map_v1.yaml
 contributors       issue_type × band
     └─────────┬──────────┘
               ▼
    AnalyzedCustomerRecord
               ▼
        app.py (Streamlit)
```

`signal_analyzer.py` orchestrates the sequence and assembles the final record. It holds no business logic.

| Module | Owns | Must not own |
|---|---|---|
| `data_loader.py` | Parsing, coercion, defaults, per-row validation | Scoring, LLM calls |
| `llm_service.py` | Prompt, provider call, retry, validation, cache | The score, or inventing a result on failure |
| `signal_analyzer.py` | Sequencing, record assembly | Any scoring or wording logic |
| `risk_engine.py` | Points, confidence gate, banding | Free-form inference |
| `reporting.py` | Rationale built from contributors | Calling an LLM |
| `recommendation_engine.py` | Action lookup by issue × band | Contacting or modifying a customer |
| `app.py` | Upload, trigger, display, error states | Business rules in UI callbacks |

---

## 3. Setup

Requires **Python 3.12**.

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
# source .venv/bin/activate        # macOS / Linux

pip install -r requirements.txt

cp .env.example .env               # then add your key
```

`.env` takes two variables:

```
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.6-flash
```

Run the dashboard and the tests **from the repository root**:

```bash
streamlit run app.py
pytest -v
```

**No API key is needed to evaluate this project.** Toggle *Use Offline Mock Mode* in the sidebar and the entire pipeline runs deterministically offline. The test suite runs in mock mode by design.

---

## 4. Data contract

One row per customer. `data/sample_customers.csv` contains 14 synthetic accounts covering explicit churn, usage collapse, ambiguous complaints, and healthy positive controls.

| Column | Type | Required | Meaning / validation |
|---|---|---|---|
| `customer_id` | string | **yes** | Unique, non-empty |
| `customer_name` | string | no | Synthetic values only |
| `interaction_text` | string | **yes** | 20–4,000 characters |
| `issue_count_30d` | int | no | 0–99, default 0 |
| `payment_status` | enum | no | `current` \| `overdue` \| `failed` \| `unknown` |
| `failed_payments_30d` | int | no | 0–10, default 0 |
| `usage_change_pct` | float | no | −100 to +100; negative means decline |
| `csat_score` | int | no | 1–5; blank means unknown |
| `last_contact_days` | int | no | 0–365; supporting context, **not scored** |

A row missing `customer_id` or `interaction_text` fails **that row only** — the rest still analyse, and the error names only its own row.

> All data in this repository is **synthetic and de-identified**. No real customer data, PII, or credentials are committed.

---

## 5. Scoring and actions

All weights live in [`config/scoring_v1.yaml`](config/scoring_v1.yaml) — not in code. They can be changed without touching Python, and the version is stamped onto every scored record.

**From the LLM (semantic):**

| Signal | Rule | Points |
|---|---|---|
| Churn intent | `churn_intent` is true | +25 |
| Urgency | critical / high / medium | +15 / +10 / +5 |
| Sentiment | very_negative / negative | +15 / +10 |

**From telemetry (structured):**

| Signal | Rule | Points |
|---|---|---|
| Payment status | failed / overdue | +15 / +12 |
| Failed payments | `min(failed_payments_30d, 2) × 4` | 0 to +8 |
| Recent issues | `min(issue_count_30d, 3) × 4` | 0 to +12 |
| Usage decline | ≤ −50% / −25% to −49% | +15 / +10 |
| CSAT | ≤ 2 / = 3 | +10 / +5 |

Sum, clamp to 100, then band: **0–24 Low · 25–49 Medium · 50–74 High · 75–100 Critical**.

### The confidence gate

If the model's confidence is **below 0.60**, or the call failed, semantic points are **excluded entirely** — the customer scores on telemetry alone and is marked *Needs review* or *Unavailable*. Confidence gates points; it never scales them. Partial credit for an unreliable reading is a false claim of certainty.

### Controlled actions

[`config/action_map_v1.yaml`](config/action_map_v1.yaml) maps `issue_type × risk_band` to one suggested action. Every output is labelled a **suggested prototype action requiring human review**. The system never contacts, suspends, discounts, or modifies a customer.

---

## 6. Worked example

**Input** — `CUST-001` from the sample CSV:

```
interaction_text:  "Cancel my enterprise subscription immediately. Your billing
                    charged us twice and nobody answered ticket #4910."
payment_status:    failed
failed_payments_30d: 2
issue_count_30d:   4
usage_change_pct:  -55.0
csat_score:        1
```

**Semantic analysis** — actual live Gemini response, schema-validated:

```json
{
  "sentiment": "very_negative",
  "issue_type": "cancellation",
  "urgency": "critical",
  "churn_intent": true,
  "confidence": 0.98,
  "evidence_spans": [
    "Cancel my enterprise subscription immediately.",
    "Your billing charged us twice and nobody answered ticket #4910."
  ],
  "summary": "Enterprise customer demands immediate cancellation due to a double billing charge and an unresponsive support ticket."
}
```

**Deterministic scoring:**

```
  AI read (language only)                 Telemetry (never seen by the AI)
  ─────────────────────────              ──────────────────────────────────
  churn_intent = true      +25            payment failed             +15
  urgency = critical       +15            usage −55%  (≤ −50)        +15
  sentiment = very_neg     +15            4 issues → min(4,3)×4      +12
  confidence 0.98 ≥ 0.60                  CSAT 1      (≤ 2)          +10
       gate open                          2 failed → min(2,2)×4       +8
  ─────────────────────────              ──────────────────────────────────
  semantic subtotal         55            structured subtotal         60

                    115 → clamped → 100 → CRITICAL
```

**Output:**

- **Risk score** 100 / 100 — **Critical**
- **Rationale** *"Critical: CUST-001 was prioritized due to explicit churn intent detected in interaction; critical operational urgency; very negative sentiment tone. Semantic clue: 'Cancel my enterprise subscription immediately.'"*
- **Suggested action** (`cancellation × Critical`) — *Retention escalation within one business day; engage account executive*

**The point of the split:** 55 points came from the model, 60 from telemetry. Turn the AI off entirely and this customer still scores 60 — still High. The model moved it to Critical and explained *why*; it did not invent the risk.

---

## 7. Testing

```bash
pytest -v      # 25 tests, no API key required
```

`data/expected_results.csv` pins the expected band, score, and top contributor for all 14 fixture rows; `tests/test_expected_results.py` asserts each one individually. Also covered: score is always an integer 0–100, band always agrees with its score, contributors are sorted descending, low-confidence signals contribute zero points, a failed analysis still scores on telemetry, and results are identical across runs.

> Expectations are **mock-mode** values, because mock mode is fully deterministic. Live Gemini output varies between runs and is deliberately not asserted.

---

## 8. Limitations and next steps

**Known limitations, stated plainly:**

- **No ground truth.** There is no labelled churn data, so no precision or recall can be claimed. The weights are expert judgement, not fitted to outcomes — nobody can currently say they are correct, including the author.
- **Linear and additive.** Signals are treated as independent, though real risk factors compound. Chosen knowingly for explainability over accuracy.
- **One row is one snapshot.** No history, no trends, no memory between runs. In production the *derivative* ("usage fell 40% in two weeks") is usually more predictive than the level.
- **Uniform thresholds.** A score of 65 means the same thing for a small account and a large one. It should not.
- **Writing style affects the score.** Sentiment reads differently across terse, verbose, and non-native English communicators. Structured telemetry partially offsets this — another argument for the two-stream design.
- **Synthetic data only.** Never tested against real, messy language.

**Next steps, in priority order:**

1. **Silent-risk detection** — flag accounts whose language stays calm while telemetry collapses. `CUST-012` (neutral tone, usage −65%) is exactly this case, and it is what "before they escalate" really means. Currently detected but not distinguished in the UI.
2. **Alert on movement, not level** — store score history and notify on band transitions. A customer at 60 for six months is known; 20 → 65 overnight is the alert. This also prevents the alert fatigue that kills these tools.
3. **Live multi-source ingest** — adapters for chat, tickets, email, and billing feeding the same `RawCustomerRecord` contract. Identity resolution and deduplication are the hard parts.
4. **Close the feedback loop** — let a CSM mark *saved* / *false alarm* / *churned anyway*. Those labels are the only route from a hand-tuned heuristic to a calibrated model.
