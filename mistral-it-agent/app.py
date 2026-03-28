"""
Streamlit UI — Veridian IT Support Agent Demo

Side-by-side comparison: base agent (no routing) vs adapted agent
(Together.ai fine-tuned intent classifier pre-routing). Both agents run on every
query so the effect of the classifier is visible in real time.

Run:
    uv run streamlit run mistral-it-agent/app.py
    # or with explicit keys:
    MISTRAL_API_KEY=sk-... TOGETHER_API_KEY=... uv run streamlit run mistral-it-agent/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Make agents/ importable when running from mistral-it-agent/
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv(override=True)

# Must be the first Streamlit call
st.set_page_config(
    page_title="Veridian IT Support — Intent Routing Demo",
    page_icon="🤖",
    layout="wide",
)

_DATA_DIR = Path(__file__).parent / "data"

_EXAMPLES = [
    "I can't pull from the artifact repo — 401 errors since yesterday",
    "My Okta Verify codes are invalid after getting a new iPhone",
    "URGENT: prod-payments webhook delay over 30s, P99 jumped to 4s",
    "First day today — MacBook MDM enrollment failed with error code 8",
    "Doctor recommended a standing desk ($650). Does IT cover this?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_classifier_model_id() -> str | None:
    p = _DATA_DIR / "classifier_model_id.txt"
    if p.exists():
        val = p.read_text().strip()
        return val or None
    return None


def _ensure_endpoint_started(together_client, endpoint_name: str) -> None:
    """Start the dedicated endpoint if stopped and wait until STARTED."""
    import time
    ep = together_client.endpoints.retrieve(endpoint_name)
    if ep.state == "STARTED":
        return
    if ep.state == "STOPPED":
        together_client.endpoints.update(endpoint_name, state="STARTED")
    while True:
        ep = together_client.endpoints.retrieve(endpoint_name)
        if ep.state == "STARTED":
            return
        if ep.state == "ERROR":
            raise RuntimeError(f"Together.ai endpoint '{endpoint_name}' is in ERROR state.")
        time.sleep(10)


@st.cache_resource(show_spinner="Initialising agents…")
def _load_agents(api_key: str, together_api_key: str | None, classifier_model_id: str | None):
    from mistralai.client import Mistral
    from agents.base_agent import BaseAgent
    from agents.adapted_agent import AdaptedAgent

    mistral_client = Mistral(api_key=api_key)

    classifier_client = None
    if classifier_model_id and together_api_key:
        from together import Together
        classifier_client = Together(api_key=together_api_key)
        # Ensure the dedicated endpoint is running before agents handle queries
        _endpoint_id_file = _DATA_DIR / "endpoint_id.txt"
        if _endpoint_id_file.exists():
            _endpoint_id = _endpoint_id_file.read_text().strip()
            if _endpoint_id:
                _ensure_endpoint_started(classifier_client, _endpoint_id)

    base = BaseAgent(client=mistral_client, model="mistral-large-latest")
    adapted = AdaptedAgent(
        client=mistral_client,
        classifier_model_id=classifier_model_id,
        model="mistral-large-latest",
        classifier_client=classifier_client,
    )
    return base, adapted


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

classifier_model_id = _get_classifier_model_id()

with st.sidebar:
    st.title("Veridian IT Demo")

    api_key = st.text_input(
        "Mistral API Key",
        value=os.getenv("MISTRAL_API_KEY", ""),
        type="password",
        help="Your La Plateforme API key from console.mistral.ai",
    )

    together_api_key = st.text_input(
        "Together.ai API Key",
        value=os.getenv("TOGETHER_API_KEY", ""),
        type="password",
        help="Your Together.ai API key for the fine-tuned classifier",
    )

    st.divider()
    st.subheader("Classifier model")

    if classifier_model_id:
        st.code(classifier_model_id, language=None)
        if together_api_key:
            st.caption("Fine-tuned Mistral-7B-Instruct-v0.2 on Veridian IT tickets via Together.ai")
        else:
            st.caption("Model found but no Together.ai key — using keyword mock.")
    else:
        st.caption("No trained model found — using keyword mock.")
        st.caption("Run `02_train_classifier.ipynb` to train the real classifier.")

    st.divider()
    st.subheader("In production")
    st.caption(
        "Forge pre-trains on your full document corpus — ticket history, "
        "runbooks, architecture docs, internal wikis — so internal terminology "
        "like Nexus, Prism, Helix, and prod-payments become first-class "
        "concepts in the model's representations, not tokens it has to "
        "interpret from context."
    )
    st.link_button(
        "Mistral Forge →",
        "https://mistral.ai/products/la-plateforme",
        use_container_width=True,
    )

    st.divider()
    show_trace = st.toggle("Show tool call trace", value=True)

    st.divider()
    st.caption("**Example queries**")
    for ex in _EXAMPLES:
        label = ex[:52] + ("…" if len(ex) > 52 else "")
        if st.button(label, key=ex, use_container_width=True):
            st.session_state["prefill"] = ex


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Veridian IT Support Agent")
st.caption(
    "**Base Agent** (all tools, no routing) vs "
    "**Adapted Agent** (fine-tuned intent classifier pre-routing). "
    "Both agents run on every query — compare tool calls, latency, and routing explainability."
)

with st.expander("Veridian internal terminology"):
    st.markdown(
        "| Term | Meaning |\n"
        "|---|---|\n"
        "| **Nexus** | Artifact repository |\n"
        "| **Prism** | Data warehouse |\n"
        "| **Helix** | On-call rotation tool |\n"
        "| **prod-payments** | Most critical service (always P1) |"
    )

if not api_key:
    st.warning("Enter your Mistral API key in the sidebar to begin.")
    st.stop()

base_agent, adapted_agent = _load_agents(api_key, together_api_key or None, classifier_model_id)

if "history" not in st.session_state:
    st.session_state.history: list[tuple[str, dict, dict]] = []

prefill = st.session_state.pop("prefill", "")
query = st.chat_input("Describe your IT issue…") or prefill

if query:
    with st.spinner("Running both agents…"):
        base_result    = base_agent.run(query)
        adapted_result = adapted_agent.run(query)
    st.session_state.history.append((query, base_result, adapted_result))

# Render history newest-first
for q, base_r, adapted_r in reversed(st.session_state.history):
    st.markdown(f"**Request:** {q}")

    col_base, col_adapted = st.columns(2)

    # ── Base Agent column ──────────────────────────────────────────────────
    with col_base:
        st.subheader("Base Agent")
        st.caption("No intent routing — model decides from scratch")

        m1, m2, m3 = st.columns(3)
        m1.metric("LLM calls",  base_r["llm_calls"])
        m2.metric("Tool calls", len(base_r["tools_called"]))
        m3.metric("Latency",    f"{base_r['latency_ms']:.0f} ms")

        if base_r["tools_called"]:
            st.markdown("**Tools called**")
            st.code(" → ".join(base_r["tools_called"]), language=None)

        if show_trace and base_r["tool_results"]:
            with st.expander("Tool call trace", expanded=False):
                for tr in base_r["tool_results"]:
                    st.markdown(f"`{tr['name']}`")
                    st.json(tr["result"], expanded=False)

        if base_r["ticket_id"]:
            st.success(f"Ticket created: **{base_r['ticket_id']}**")

        st.markdown("**Response**")
        st.markdown(base_r["response"])

    # ── Adapted Agent column ───────────────────────────────────────────────
    with col_adapted:
        st.subheader("Adapted Agent")
        st.caption("Fine-tuned classifier pre-routes intent before the first tool call")

        # Classifier row — extra vs base agent
        intent = adapted_r["classifier_intent"]
        conf   = adapted_r["classifier_confidence"]
        badge_color = (
            "green"  if conf >= 0.80 else
            "orange" if conf >= 0.60 else
            "red"
        )
        st.markdown(
            f"**Classifier** &nbsp;&nbsp;"
            f"`{intent}` &nbsp;"
            f"<span style='"
            f"background-color:{badge_color};"
            f"color:white;"
            f"padding:2px 10px;"
            f"border-radius:12px;"
            f"font-size:0.82em;"
            f"font-weight:600;"
            f"'>{conf:.0%}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Classifier latency: {adapted_r['classifier_latency_ms']:.0f} ms")

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "LLM calls",
            adapted_r["llm_calls"],
            delta=adapted_r["llm_calls"] - base_r["llm_calls"],
            delta_color="inverse",
        )
        m2.metric(
            "Tool calls",
            len(adapted_r["tools_called"]),
            delta=len(adapted_r["tools_called"]) - len(base_r["tools_called"]),
            delta_color="inverse",
        )
        m3.metric(
            "Latency",
            f"{adapted_r['latency_ms']:.0f} ms",
            delta=f"{adapted_r['latency_ms'] - base_r['latency_ms']:.0f} ms",
            delta_color="inverse",
        )

        if adapted_r["tools_called"]:
            st.markdown("**Tools called**")
            st.code(" → ".join(adapted_r["tools_called"]), language=None)

        if show_trace and adapted_r["tool_results"]:
            with st.expander("Tool call trace", expanded=False):
                for tr in adapted_r["tool_results"]:
                    st.markdown(f"`{tr['name']}`")
                    st.json(tr["result"], expanded=False)

        if adapted_r["ticket_id"]:
            st.success(f"Ticket created: **{adapted_r['ticket_id']}**")

        st.markdown("**Response**")
        st.markdown(adapted_r["response"])

    st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.caption(
    "Powered by [Mistral AI](https://mistral.ai) · "
    "Classifier fine-tuned on [Together.ai](https://together.ai)"
)
