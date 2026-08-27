# Intelligent Customer Signal Detector — System Design

## 1. Purpose

Customer operations teams work reactively: by the time a complaint escalates or a cancellation arrives, the window to intervene has closed. The signals that predict it — support text, billing behaviour, usage trends, satisfaction scores — exist already, but sit in separate systems and get reviewed by hand.

This system correlates those signals into a **prioritized queue of customers who need attention**, each with a transparent risk score, an explanation of why they were flagged, and a suggested next action.

**What it is:** an explainable prioritization heuristic that ranks who to call first.
**What it is not:** a calibrated churn-prediction model. It reports no probabilities and claims none.

## 2. The core principle

> **The LLM interprets language. Configuration decides points. Code decides bands, ordering, explanations, and actions.**

Every architectural boundary below is a fence around that sentence.

Three tests resolve any "where does this belong?" question:

| Question | Owner |
|---|---|
| Would two reasonable humans reading the same input disagree? | **LLM** |
| Should the business change it without a code deploy? | **YAML config** |
| Must it be identical across runs and auditable afterward? | **Python** |

**Why this split matters more than explainability:** it survives the LLM being wrong. A bad sentiment read moves the score by 15 points out of 100 — billing, usage and CSAT still anchor it. If the LLM produced the score directly, one bad read would be one entirely wrong answer, with no floor and no way to detect it.

## 3. End-to-end data flow

The shape is a **fork → merge → fan-out**, not a linear chain.

```
┌──────────────────────────────────────────────────────────────────────┐
│  ENTRY                                                               │
│  data/sample_customers.csv   or   user-uploaded CSV                  │
└─────────────────────────────────┬────────────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  [ 1 ]  DATA LOADER                                    data_loader.py│
│  parse · normalize columns · coerce types · apply defaults · validate│
└────────────────┬─────────────────────────────────┬───────────────────┘
                 │ valid                           │ invalid
                 ▼                                 ▼
        RawCustomerRecord                  row-level errors
     ◄── THE CONTRACT BOUNDARY ──►      (surfaced in UI, never
                 │                       silently dropped)
        ┌────────┴────────┐
        │   FORK          │
        ▼                 ▼
 interaction_text     telemetry
 (what they SAID)   (what they DID)
        │            payment · usage
        │            CSAT · issue counts
        ▼                 │
┌───────────────────────┐ │
│ [ 2 ] LLM SERVICE     │ │   ◄── telemetry NEVER enters the LLM
│    llm_service.py     │ │
│ closed-schema prompt  │ │
│ temp 0.1 · 1 retry    │ │
│ cache (mode, sha256)  │ │
│                       │ │
│ 3 outcomes, never     │ │
│ blended:              │ │
│   live  → signal      │ │
│   mock  → signal      │ │
│   fail  → None+reason │ │
└───────────┬───────────┘ │
            ▼             │
     SemanticSignal       │
   sentiment · issue_type │
   urgency · churn_intent │
   confidence · evidence  │
            │             │
            └──────┬──────┘
                   ▼  MERGE
┌──────────────────────────────────────────────────────────────────────┐
│  [ 4 ]  RISK ENGINE                                    risk_engine.py│
│  inputs: SemanticSignal + telemetry       (BOTH — this is the merge) │
│  scoring_v1.yaml → points · confidence gate · sum · clamp 0-100→band │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
          risk_score · risk_band · contributors[] · semantic_status
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼          FAN-OUT        ▼                         ▼
┌─────────────────┐ ┌──────────────────────┐ ┌──────────────────────────┐
│ [ 5 ] CORRELAT. │ │ [ 6 ]  REPORTING     │ │ [ 7 ] RECOMMENDATION     │
│ correlation.py  │ │    reporting.py      │ │ recommendation_engine.py │
│ semantic vs     │ │ rationale built from │ │ action_map_v1.yaml       │
│ telemetry       │ │ contributors — NOT a │ │ issue_type × band        │
│ agreement flag  │ │ second LLM call      │ │        ↓ action          │
└────────┬────────┘ └──────────┬───────────┘ └────────────┬─────────────┘
         └─────────────────────┼──────────────────────────┘
                               ▼
                  ┌────────────────────────┐
                  │ AnalyzedCustomerRecord │   ◄── final output contract
                  └───────────┬────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  [ 8 ]  PRESENTATION                                          app.py │
│  sorted queue · KPI cards · detail drill-down · score breakdown      │
│  analysis fires ONLY on button press; results cached in session      │
└──────────────────────────────────────────────────────────────────────┘

  Orchestration: [ 3 ] signal_analyzer.py conducts 2→4→5,6,7 and assembles
  the record. It transforms nothing; no business logic lives there.
```

## 4. Component blocks

### [1] Data Loader — `data_loader.py`

| | |
|---|---|
| **Input** | CSV file or DataFrame |
| **Output** | `List[RawCustomerRecord]`, `List[ValidationError]` |
| **Owns** | Column normalization, type coercion, defaults, range validation, per-row error reporting |
| **Must not own** | Risk scoring, LLM calls, any interpretation of meaning |

Rejects rows missing `customer_id` or `interaction_text`. **One bad row fails that row only** — the rest still analyse. Errors name only their own row's ID, never another customer's data.

### [2] LLM Service — `llm_service.py`

| | |
|---|---|
| **Input** | `interaction_text` only |
| **Output** | `SemanticSignal` + source label (`live` / `mock` / `failed`) |
| **Owns** | Prompt, provider call, retry with backoff, response validation, session cache |
| **Must not own** | The risk score, ranking, business actions, **or inventing a result when it fails** |

Given only the transcript and the closed schema — never the score, never other customers, never telemetry. Temperature 0.1, schema-constrained JSON, validated by Pydantic before use. Cached by SHA-256 of normalized text; **raw transcripts are never persisted in logs or cache keys.**

**The honesty rule:** on failure it returns `failed`, and the customer is marked `semantic_status = "unavailable"`. It never substitutes a fabricated analysis. A system that admits it couldn't read one transcript keeps its credibility on the other thirteen.

### [3] Signal Analyzer — `signal_analyzer.py` *(orchestrator)*

| | |
|---|---|
| **Input** | `List[RawCustomerRecord]` |
| **Output** | `List[AnalyzedCustomerRecord]`, sorted by score descending |
| **Owns** | Sequencing the pipeline, assembling the final record |
| **Must not own** | Any scoring, wording, or routing logic |

A conductor, not a stage. If you find business logic here, it belongs somewhere else.

### [4] Risk Engine — `risk_engine.py`

| | |
|---|---|
| **Input** | `SemanticSignal` **and** `RawCustomerRecord` telemetry |
| **Output** | `score`, `band`, `contributors[]`, `semantic_status` |
| **Owns** | Point assignment from `scoring_v1.yaml`, the confidence gate, clamping, banding |
| **Must not own** | Free-form inference, any text generation |

This is where the two streams merge. Every point traces to a line in the YAML.

**Scoring (`scoring_v1`)** — semantic: churn intent +25 · urgency 15/10/5 · sentiment 15/10. Structured: payment failed +15 / overdue +12 · `min(failed,2)×4` · `min(issues,3)×4` · usage ≤ −50% +15, ≤ −25% +10 · CSAT ≤ 2 +10, = 3 +5. Sum, clamp to 100.

**Bands** — 0–24 Low · 25–49 Medium · 50–74 High · 75–100 Critical.

**Confidence gate** — below 0.60, semantic points are **excluded entirely** and the row is marked *Needs review*; it scores on telemetry alone. Confidence never scales points — it gates them on or off. Partial credit for an unreliable read is a false claim of certainty.

### [5] Correlation — `correlation.py`

| | |
|---|---|
| **Input** | `SemanticSignal` + telemetry |
| **Output** | One flag: `corroborated` / `silent_risk` / `vocal_only` |
| **Owns** | Comparing what the customer *says* against what they *do* |
| **Must not own** | Any effect on the score |

- **`corroborated`** — text and telemetry agree. Highest confidence to act.
- **`silent_risk`** — calm or positive words, deteriorating telemetry. **The most valuable case in the system**: a customer who hasn't complained yet but is already leaving. This is what "before they escalate" actually means.
- **`vocal_only`** — loud text, healthy telemetry. Real, but likely lower churn risk.

Deliberately **does not change the score.** A correlation lens over an unchanged, traceable score is more defensible than folding it into another opaque number.

### [6] Reporting — `reporting.py`

| | |
|---|---|
| **Input** | `contributors[]`, `SemanticSignal` |
| **Output** | One-sentence human rationale |
| **Owns** | Turning score contributors into readable English |
| **Must not own** | Calling an LLM |

The rationale is **derived from the contributors**, so it is mathematically incapable of contradicting the score. A second LLM call could produce an explanation that disagrees with your own number in front of the customer.

### [7] Recommendation Engine — `recommendation_engine.py`

| | |
|---|---|
| **Input** | `issue_type` (from semantic) **and** `risk_band` (from risk engine) |
| **Output** | One controlled action string |
| **Owns** | Lookup in `action_map_v1.yaml` |
| **Must not own** | Free-form generation, sending anything, changing any customer record |

Band says *how urgent*; issue type says *who handles it*. Both are required.

Output is always framed as a **"Suggested prototype action"** requiring human review. **The system never contacts, suspends, discounts, or modifies a customer.** It produces a recommendation for a person.

### [8] Presentation — `app.py`

| | |
|---|---|
| **Input** | `List[AnalyzedCustomerRecord]` |
| **Output** | Streamlit dashboard |
| **Owns** | Upload, orchestration trigger, display, error and empty states |
| **Must not own** | Business rules in UI callbacks |

Analysis runs **only on explicit button press**, cached for the session — never on Streamlit rerun. Shows the score, the band, the point-by-point breakdown, the semantic signal the AI produced, the rationale, the suggested action, and — when things fail — says so plainly.

## 5. Contracts at each boundary

```
RawCustomerRecord     customer_id* · customer_name · interaction_text*
(loader → pipeline)   issue_count_30d · payment_status · failed_payments_30d
                      usage_change_pct · csat_score · last_contact_days
                                                            (* required)

SemanticSignal        sentiment    : very_negative|negative|neutral|positive
(LLM → risk engine)   issue_type   : billing|support|product|usage|
                                     cancellation|other
                      urgency      : low|medium|high|critical
                      churn_intent : bool
                      confidence   : float 0.0–1.0
                      evidence_spans: up to 2 short quotes
                      summary      : ≤ 160 chars

AnalyzedCustomerRecord  identity + telemetry + semantic_signal
(pipeline → UI)         + semantic_status · analysis_source · scoring_version
                        + risk_score · risk_band · contributors[]
                        + rationale · suggested_action · correlation_flag
```

Every LLM field is a **closed enum or a bounded primitive**. The model chooses from our vocabulary; it never invents one. `evidence_spans` are the receipts — they let a human verify the AI's claim against the source text, which is what turns the model from an oracle into a witness.

## 6. Worked trace — `CUST-001`

> *"Cancel my enterprise subscription immediately. Your billing charged us twice and nobody answered ticket #4910."*
> payment `failed` · 2 failed payments · 4 issues · usage −55% · CSAT 1

```
 AI READ (language only)              YOUR RULES (scoring_v1.yaml)
 ──────────────────────               ────────────────────────────
 churn_intent   = true          ───►  +25
 urgency        = critical      ───►  +15
 sentiment      = very_negative ───►  +15     semantic subtotal    55
 issue_type     = billing       ───►  (routes action, 0 points)
 confidence     = 0.95          ───►  ≥ 0.60 — gate open

 TELEMETRY (never seen by the AI)
 ───────────────────────────────
 payment failed                 ───►  +15
 2 failed payments  min(2,2)×4  ───►  +8
 4 issues           min(4,3)×4  ───►  +12
 usage −55%         ≤ −50       ───►  +15
 CSAT 1             ≤ 2         ───►  +10     structured subtotal  60
                                              ──────────────────────
                                              115 → clamp → 100  CRITICAL

 billing × Critical → "Route to billing support; notify retention owner"
```

**55 points from the AI, 60 from telemetry. Turn the AI off entirely and this customer still scores 60 — still High. The AI moved it from High to Critical and explained why; it did not invent the risk.**

## 7. Assumptions

**Data**

1. All data is **synthetic and de-identified**. No real customer data, PII, or credentials enter the repository.
2. **One row = one customer = one interaction.** Real customers have many interactions over time; the prototype flattens this.
3. Input is a **point-in-time snapshot**. No trend or derivative signals — production would find "usage fell 40% in two weeks" more predictive than "usage is low."
4. Text is assumed to be **English**.
5. `last_contact_days` is carried as supporting context and is deliberately **not scored**.

**Scoring**

6. The score is an **explainable prioritization heuristic, not a probability.** It ranks; it does not predict.
7. Weights are **expert judgment, not fitted to outcomes.** No labelled churn data exists, so no one can currently claim they are correct — including us.
8. Scoring is **linear and additive with caps**, and treats signals as independent. Real risk factors interact (failed payment + churn language is worse than the sum). Chosen for explainability over accuracy, knowingly.
9. **Confidence gates points; it never scales them.**
10. Thresholds are **uniform across all customers** — no segmentation by account value or tenure.

**System**

11. The LLM is **non-deterministic even at temperature 0.1.** Session caching keeps a single demo run internally consistent; two runs may differ slightly. Deterministic tests therefore use mock mode.
12. **No persistence.** All state is in-session; nothing is stored between runs.
13. **Mock mode must complete the entire pipeline with no API key**, so the demo can never be blocked by an outage or quota.

## 8. Considerations and known limitations

**We cannot yet say whether it is accurate.** There is no ground truth, so no precision or recall figures. The honest claim is: the logic is transparent and every score is traceable. Validating it requires outcome labels — the path is in §9.

**False positives have a real cost.** If 40 of 200 accounts are flagged High and the team can call 10, the tool has recreated the triage problem one layer up. The right measure is not "did we catch everyone at risk" but **"of the top 10 we surfaced, how many were worth the call?"** Precision at the head of the list beats recall.

**Fixed thresholds are not neutral.** A score of 65 means something different for a $500/month startup than a $2M enterprise.

**Writing style affects the score.** Sentiment reads differently across terse vs. verbose communicators and non-native English speakers. Two equally at-risk customers can score differently because of *how* they write. Structured telemetry partially offsets this, which is another argument for the two-stream design.

**The AI must justify its existence.** A SQL query on `failed_payments > 0 OR csat <= 2` would catch much of this. The LLM earns its place on exactly two things: **silent risk** (calm words, collapsing telemetry — SQL sees the collapse, not the calm) and **intent buried in polite language** ("we'll evaluate options before renewal" — SQL sees nothing). Where it doesn't add value, we should say so.

**Trust is the real failure mode.** Wrong twice in front of a CSM in week one and the tool is never opened again. This is why `unavailable` and `needs review` are operational features, not just ethical ones.

**Human review is mandatory.** Every output is a suggestion for a person. The system takes no action on any customer.

## 9. Production evolution

The prototype's pipeline is the production pipeline. What changes is what sits in front of and behind it.

```
  chat  tickets  email  forms  billing  usage
    │      │       │      │       │       │
    ▼      ▼       ▼      ▼       ▼       ▼
 ┌────────────────────────────────────────────┐
 │ ingestion adapters → canonical event       │  NEW
 │ identity resolution · dedupe · PII redact  │
 └──────────────────────┬─────────────────────┘
                        ▼
 ┌────────────────────────────────────────────┐
 │ customer signal state — rolling windows,   │  NEW
 │ trends, per-interaction history            │
 │ emits the same RawCustomerRecord           │
 └──────────────────────┬─────────────────────┘
                        ▼
 ╔════════════════════════════════════════════╗
 ║  EVERYTHING IN THIS DOCUMENT — UNCHANGED   ║
 ╚══════════════════════╤═════════════════════╝
                        ▼
 ┌────────────────────────────────────────────┐
 │ score history → alert on BAND TRANSITIONS  │  NEW
 │ CSM outcome feedback → recalibrate weights │
 └────────────────────────────────────────────┘
```

**`RawCustomerRecord` is the seam.** Anything that can produce one can feed the system — a CSV today, an event stream later. That is the payoff of designing contract-first.

**Cadence** is not one number:

- **Event-driven (seconds)** — new message, failed payment, cancellation page visit. *This is where "before they escalate" is actually won.*
- **Nightly batch** — rolling usage trends, issue windows, score decay.
- **On-demand** — a CSM opens an account.

**Alert on changes, not levels.** A customer at 60 for six months is a known quantity; one who went 20 → 65 overnight is the alert. Re-notifying on every nightly run destroys the tool through alert fatigue inside a week.

**The feedback loop is the missing piece.** When a CSM marks *contacted / false alarm / saved / churned anyway*, those labels become the ground truth that turns a hand-tuned heuristic into a calibrated model. It is the single most important production element absent from this POC.

---

*Scoring rules: `config/scoring_v1.yaml` · Action rules: `config/action_map_v1.yaml`*
