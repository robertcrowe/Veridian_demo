# Veridian Demo

A full end-to-end demo showing how **Mistral Classifier Factory** reduces tool call
overhead in a production agentic loop — using Veridian Systems, a fictional B2B SaaS
company, as the setting.

Runs entirely on La Plateforme. The only thing that changes for a Forge deployment is
`server_url=` in the client constructor.

---

## Architecture

```
        BASE AGENT                              ADAPTED AGENT
        ──────────────────────────────          ──────────────────────────────────────

   [Employee request]                      [Employee request]
            │                                        │
            │                             ┌──────────▼──────────────────────┐
            │                             │  Layer 1: Classifier Factory    │
            │                             │  ministral-3b-latest            │
            │                             │  fine-tuned on Veridian tickets │
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

## Prerequisites

- Python 3.12+
- A **Mistral API key** from [console.mistral.ai](https://console.mistral.ai)
- Estimated API cost for Notebook 02 (fine-tuning): **~$2–5** on a 60-example dataset
  with 100 training steps on `ministral-3b-latest`
- Estimated API cost for Notebook 03 (5 scenarios × 2 agents): **~$0.20–0.50**

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
uv run jupyter notebook   # run from repo root, or:
pip install -r requirements.txt && jupyter notebook
```

**Run notebooks in order:**

| Step | Notebook | What it does | API cost |
|---|---|---|---|
| 1 | `01_data_prep.ipynb` | Converts `raw_tickets.json` to Classifier Factory JSONL, uploads to Mistral Files API | ~$0 |
| 2 | `02_train_classifier.ipynb` | Submits `job_type="classifier"` fine-tuning job, polls to completion, evaluates on held-out test set | ~$2–5 |
| 3 | `03_agent_demo.ipynb` | Runs 5 preset scenarios through both agents, side-by-side comparison | ~$0.20–0.50 |

**Optional — Streamlit app:**

```bash
# From mistral-it-agent/ (after running notebooks 01 + 02)
streamlit run app.py
```

> **Colab users:** each notebook handles the API key via `google.colab.userdata`.
> Store your key as a Secret named `MISTRAL_API_KEY` (key icon in the left sidebar).

---

## The three layers

### Layer 1 — Classifier Factory model

A `ministral-3b-latest` model fine-tuned with `job_type="classifier"` on 42 labelled
Veridian IT support tickets (70% of the 60-ticket corpus). The training data format is:

```json
{"text": "I can't pull from Nexus — 401 errors", "labels": {"intent": "access_request"}}
```

At inference, the model returns a probability distribution over 8 intent classes in a
single forward pass (~50–150 ms). No sampling, no chain-of-thought.

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

> *"The request has been pre-classified as: access_request (confidence: 94%).
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

## What this demo actually simulates vs. what Forge adds

This demo runs entirely on La Plateforme shared infrastructure. It is an honest
simulation of a Forge-based deployment, not the real thing.

| Aspect | This demo | With Forge |
|---|---|---|
| **Model training** | Classifier Factory on La Plateforme (public cloud) | Same API, `server_url=` points to your Forge instance |
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

## The five demo scenarios

These are the preset queries in `03_agent_demo.ipynb` and in the Streamlit app sidebar.
Each is chosen to highlight a different routing challenge:

| # | Name | Query summary | What to watch |
|---|---|---|---|
| 1 | **Nexus access request** | Dev can't pull from artifact repo, 401 errors | Does the classifier catch `access_request` despite ambiguous phrasing? Base agent may call `search_knowledge_base` first. |
| 2 | **Okta MFA not syncing** | New phone, Verify codes invalid, stand-up in 20 min | `software_issue` vs `access_request` boundary. Adapted agent skips KB search and goes straight to escalation policy. |
| 3 | **SEV2 on prod-payments** | Webhook delay >30s, P99 jumped to 4s | Should classify as `payments_incident` and escalate to P1 immediately. Watch base agent take longer to determine this. |
| 4 | **Day-1 onboarding setup** | MDM error code 8, no Slack, no Okta | Multi-step `onboarding` scenario. Does the adapted agent stay focused or reach for KB? |
| 5 | **Standing desk expense** | Doctor-recommended, costs $650, unsure of process | `expense_request` not `hardware_issue`. Does the classifier avoid the wrong team? |

---

## Production path

To move from this demo to a production system:

1. **Replace tool stubs** — `create_ticket` → ServiceNow API; `get_escalation_policy` →
   PagerDuty/OpsGenie rules; `search_knowledge_base` → Confluence/Notion search API.

2. **Replace the classifier endpoint with a Forge-hosted model** — change `server_url=`
   in the Mistral client constructor. The `client.classifiers.classify()` call is
   identical.

3. **Scale training data** — 100–200 labelled examples per class for production accuracy.
   Consider a human-in-the-loop labelling pipeline for ongoing retraining.

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

This project is structured as a self-contained Mistral Cookbook recipe. If you extend
it — better training data, additional intent classes, real tool integrations, or a
Forge deployment walkthrough — consider submitting it to the Mistral Cookbook:

> **Contributing guide:** https://github.com/mistralai/cookbook/blob/main/CONTRIBUTING.md

The cookbook accepts notebooks and Python scripts that demonstrate Mistral API features
with real, runnable code. This demo would fit under `mistral/finetuning/` or a new
`mistral/agents/intent-routing/` directory.

---

## License

MIT — see [LICENSE](LICENSE) if present, otherwise treat as MIT.
