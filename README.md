# Veridian Demo

A full end-to-end demo showing how **Mistral SFT fine-tuning** reduces tool call
overhead in a production agentic loop — using Veridian Systems, a fictional B2B SaaS
company, as the setting.

Uses **Mistral La Plateforme** for the agentic loop (`mistral-large-latest`) and
**Together.ai** for classifier fine-tuning and inference — a runnable analogue for a
Forge deployment. The only code change for Forge is `server_url=` in the client
constructor.

---

## Architecture

```
        BASE AGENT                              ADAPTED AGENT
        ──────────────────────────────          ──────────────────────────────────────

   [Employee request]                      [Employee request]
            │                                        │
            │                             ┌──────────▼──────────────────────┐
            │                             │  Layer 1: SFT Classifier        │
            │                             │  Mistral-7B-Instruct-v0.2       │
            │                             │  fine-tuned on Together.ai      │
            │                             │  → intent label + confidence    │
            │                             └──────────┬──────────────────────┘
            │                                        │
            │                             ┌──────────▼──────────────────────┐
            │                             │  Layer 2: Tool Router           │
            │                             │  2 tools only (not 3)           │
            │                             │  intent injected into prompt    │
            │                             └──────────┬──────────────────────┘
            │                                        │
   ┌────────▼──────────────────┐         ┌──────────▼──────────────────────┐
   │  mistral-large-latest     │         │  mistral-large-latest           │
   │  3 tools available        │         │  2 tools available              │
   │  model infers intent      │         │  classifier already routed it   │
   └────────┬──────────────────┘         └──────────┬──────────────────────┘
            │                                        │
     [Helpdesk reply]                        [Helpdesk reply]
```

The base agent skips layers 1 and 2 — it receives all tools every turn and figures out
the intent itself. The side-by-side comparison in the notebook and Streamlit app
measures what the classifier saves: tool calls, LLM calls, and latency.

---

## Veridian Systems — internal terminology

Veridian Systems is the fictional B2B SaaS company used throughout the demo.
The following internal tool names appear in tickets, prompts, and training data:

| Term | Meaning |
|---|---|
| **Nexus** | Internal artifact repository |
| **Prism** | Internal data warehouse |
| **Helix** | Slack-based on-call rotation tool |
| **prod-payments** | Most critical service — any incident is always P1 |

These names are fixed across all code, data files, and prompts. Do not substitute
alternative names (e.g. "Artifactory" for Nexus).

---

## Prerequisites

- Python 3.12+
- A **Mistral API key** from [console.mistral.ai](https://console.mistral.ai)
- A **Together.ai API key** from [api.together.ai](https://api.together.ai) (for fine-tuning and classifier inference)
- Estimated API cost for Notebook 01 (synthetic data generation): **~$1–3** (122-ticket base, `mistral-large-latest`, 40 synthetic tickets per class)
- Estimated API cost for Notebook 02 (fine-tuning on Together.ai): **~$1–4** on a ~442-example dataset
- Estimated API cost for Notebook 03 (full test set × 2 agents): **~$1–3**

---

## Quick start

```bash
# From the repo root
make install        # uv sync --dev
make notebook       # opens Jupyter in mistral-it-agent/
```

Or without make:

```bash
cd mistral-it-agent
echo "MISTRAL_API_KEY=sk-..." > .env
echo "TOGETHER_API_KEY=..." >> .env
uv run jupyter notebook   # run from repo root, or:
pip install -r requirements.txt && jupyter notebook
```

**Run notebooks in order:**

| Step | Notebook | What it does | API cost |
|---|---|---|---|
| 1 | `01_data_prep.ipynb` | Converts `raw_tickets.json` to SFT JSONL; optionally generates synthetic tickets with `mistral-large-latest` | ~$1–3 |
| 2 | `02_train_classifier.ipynb` | Uploads JSONL to Together.ai, fine-tunes `Mistral-7B-Instruct-v0.2`, polls to completion, evaluates on held-out test set | ~$1–4 |
| 3 | `03_agent_demo.ipynb` | Runs both agents on the full held-out test set; per-class accuracy table and aggregate metrics; interactive cell for ad-hoc queries | ~$1–3 |

**Optional — Streamlit app:**

```bash
# From the repo root (after running notebooks 01 + 02)
uv run streamlit run mistral-it-agent/app.py
```

> **Colab users:** each notebook handles the API key via `google.colab.userdata`.
> Store your key as a Secret named `MISTRAL_API_KEY` (key icon in the left sidebar).

---

## The three layers

### Layer 1 — SFT classifier model

A `Mistral-7B-Instruct-v0.2` model fine-tuned on Together.ai on ~309 training examples
(70% of a ~442-example corpus — 122 labelled raw tickets plus ~320 synthetic tickets
generated by `mistral-large-latest`). The training data format is:

```json
{"messages": [
  {"role": "system",    "content": "Classify the IT support request into exactly one of the following categories: access_request, security_incident, hardware_issue, software_issue, onboarding, payments_incident, expense_request, general_question. Disambiguation rules: onboarding vs access_request: use onboarding when a new hire in their first 1-2 weeks cannot access a tool as part of initial setup; use access_request when an established employee requests a permission change... Respond with only the category label, nothing else."},
  {"role": "user",      "content": "I can't pull from Nexus — 401 errors"},
  {"role": "assistant", "content": "access_request"}
]}
```

At inference, the model is called at `temperature=0` and returns a single intent label
directly as text (~50–150 ms). No sampling, no chain-of-thought. Confidence is fixed at
`0.95` for a recognised label — surfaced in the UI as a routing confidence badge.

**The 8 intent labels** (fixed — do not rename):

| Label | Description |
|---|---|
| `access_request` | Account provisioning, permission changes, offboarding |
| `security_incident` | Phishing, suspected breach, ransomware, stolen device |
| `hardware_issue` | Broken devices, peripherals, battery, screen |
| `software_issue` | Install requests, license renewals, version upgrades |
| `onboarding` | New hire setup, first-day access, role provisioning |
| `payments_incident` | Any issue touching prod-payments — always escalated P1 |
| `expense_request` | Reimbursements, home office budget, L&D expenses |
| `general_question` | Policy questions, how-tos, SLA queries |

### Layer 2 — Tool router

Maps the predicted intent to a restricted tool set and injects the intent into the
system prompt:

> *"The request has been pre-classified as: access_request (confidence: 95%).
> Route accordingly. Only call tools if additional information is needed beyond
> what you already know about this intent category."*

The adapted agent always receives exactly **2 tools**: `create_ticket` and
`get_escalation_policy`. The base agent receives all **3** (plus `search_knowledge_base`).
`search_knowledge_base` is withheld because the classifier has already determined the
category — KB lookup is redundant.

### Layer 3 — Agentic loop

`mistral-large-latest` with `tool_choice="auto"`. The loop runs until the model stops
requesting tool calls or `max_iterations=6` is reached. Results are returned as a plain
dict — see `CLAUDE.md` for the full interface contract.

---

## Testing

```bash
# Unit tests — no API key required
make test
uv run pytest -m "not integration"

# Integration tests — require MISTRAL_API_KEY in mistral-it-agent/.env
uv run pytest -m integration -v
```

Unit tests (43) mock the Mistral client entirely. Integration tests exercise the live
API using `mistral-small-latest` to keep costs low (~$0.01 per run). The SFT classifier
integration tests are automatically skipped unless `data/endpoint_name.txt` exists.

---

## What this demo actually simulates vs. what Forge adds

This demo runs entirely on La Plateforme shared infrastructure. It is an honest
simulation of a Forge-based deployment, not the real thing.

| Aspect | This demo | With Forge |
|---|---|---|
| **Model training** | SFT fine-tuning on Together.ai (Forge analogue) | Same API, `server_url=` points to your Forge instance |
| **Data residency** | Training data leaves your network | All training and inference stays on-prem or in your VPC |
| **Generative model** | `mistral-large-latest` (shared) | Dedicated instance; optionally continued pre-trained on your corpus |
| **Tool integrations** | In-memory stubs (`create_ticket`, `get_escalation_policy`) | ServiceNow, Confluence, Okta, PagerDuty via real API connectors |
| **Veridian terminology** | Model interprets from context | Forge continued pre-training makes Nexus, Prism, Helix, prod-payments first-class concepts |
| **SLA** | Best-effort | Contractual uptime and latency guarantees |

The Python SDK code change for Forge is minimal:

```python
# La Plateforme (this demo)
client = Mistral(api_key=MISTRAL_API_KEY)

# Forge deployment
client = Mistral(api_key=FORGE_API_KEY, server_url="https://mistral.your-forge.internal/v1")
```

---

## Notebook 03 — what it shows

`03_agent_demo.ipynb` runs both agents on every example in the held-out test set
(15% of the full corpus, stratified by intent) and displays:

- A **per-class accuracy table** — N examples, base agent correct/%, adapted agent correct/% for each of the 8 intent classes
- An **aggregate metrics panel** — overall accuracy for each agent, average tool calls saved, latency delta, and cost delta
- A **live progress log** showing ground truth vs each agent's prediction per example

The interactive cell (Section 6) still accepts ad-hoc custom queries and renders
the full side-by-side panel for a single request.

The base agent's intent is inferred from the `category` argument it passes to
`create_ticket` or `get_escalation_policy`. For `general_question` the base agent
typically answers directly without a tool call — this is treated as correct.

---

## Production path

To move from this demo to a production system:

1. **Replace tool stubs** — `create_ticket` → ServiceNow API; `get_escalation_policy` →
   PagerDuty/OpsGenie rules; `search_knowledge_base` → Confluence/Notion search API.

2. **Replace the Together.ai classifier with a Forge-hosted model** — point
   `classifier_client` at your Forge endpoint. The `client.chat.completions.create()`
   call in `AdaptedAgent._classify()` is identical.

3. **Scale training data** — aim for 50–200 labelled examples per class for production
   accuracy (this demo uses ~15 raw + ~40 synthetic per class). Consider a
   human-in-the-loop labelling pipeline for ongoing retraining.

4. **Add an `unknown` class** — to catch out-of-distribution queries that should not be
   routed to any intent bucket.

5. **Log classification decisions** — store `classifier_intent` and
   `classifier_confidence` per request. Feed low-confidence or misclassified examples
   back into the training set automatically.

6. **A/B test** — run base and adapted agents on parallel traffic slices. Compare CSAT,
   resolution time, and ticket escalation rate.

---

## Success metrics

| Metric | What it measures | How to capture |
|---|---|---|
| **Classifier F1 score** | Per-class precision/recall on held-out test set | `sklearn.metrics.classification_report` — already in `02_train_classifier.ipynb` |
| **Tool call reduction %** | Avg calls per query: base vs adapted | `(base_tool_calls - adapted_tool_calls) / base_tool_calls` — computed in `03_agent_demo.ipynb` |
| **Mean time to resolution** | Wall-clock from ticket open to close | Requires real ticket system integration |
| **Ticket deflection rate** | % of queries resolved without creating a ticket | `ticket_id is None` in the result dict |
| **Cost per request** | Token cost for full query × request volume | La Plateforme usage dashboard; `llm_calls` in result dict as a proxy |

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository and create a feature branch.
2. Make your changes — better training data, additional intent classes, real tool
   integrations, improved evaluation, or bug fixes.
3. Ensure `make test` passes (unit tests require no API key).
4. Open a pull request with a clear description of what changed and why.

---

## License

MIT — see [LICENSE](LICENSE) if present, otherwise treat as MIT.
