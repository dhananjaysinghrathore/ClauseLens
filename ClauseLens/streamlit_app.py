# streamlit_app.py — ClauseLens UI (improved)
import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="ClauseLens — Regulatory RAG", layout="wide")
st.title("ClauseLens — Regulatory/Compliance RAG (Local)")

# Sidebar controls
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000/ask")
mode = st.sidebar.selectbox("Answer mode", ["extractive", "abstractive"])
top_k = st.sidebar.slider("Top-K (reranked)", 2, 10, 6, 1)

# Main input
query = st.text_area(
    "Ask a clause-level question",
    placeholder="Example: What is the bid security amount required?",
    height=120,
)

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("Search", type="primary")
with col2:
    if st.button("Clear history"):
        st.session_state.history = []

def render_result(data):
    st.subheader("Result")
    if data.get("status") == "abstain":
        st.warning(data.get("message", "No answer."))
        cits = data.get("top_citations", [])
        if cits:
            with st.expander("Top retrieved citations"):
                for c in cits:
                    st.write("• " + c)
        return

    if data.get("answer_mode") == "extractive":
        quotes = data.get("quotes", [])
        if not quotes:
            st.info("No extractive spans found.")
            return
        for i, q in enumerate(quotes, 1):
            st.markdown(f"**{i}.** “{q['answer']}”")
            st.caption(q["citation"] + f"  · score={q['score']:.3f}")
        st.divider()
        st.code(data.get("joined", ""), language="text")
    else:
        st.write(data.get("note", ""))
        st.subheader("Citations")
        for c in data.get("top_citations", []):
            st.write("• " + c)
        with st.expander("Context preview"):
            st.write(data.get("context_preview", ""))

if run:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Querying…"):
                payload = {"query": query, "mode": mode, "top_k": top_k}
                r = requests.post(api_url, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
            # Save to history
            st.session_state.history.insert(0, {
                "ts": datetime.now().strftime("%H:%M:%S"),
                "query": query,
                "response": data,
            })
            render_result(data)
        except Exception as e:
            st.error(f"Request failed: {e}")

# History panel
st.sidebar.subheader("Session history")
for item in st.session_state.history[:10]:
    with st.sidebar.expander(f"{item['ts']}  •  {item['query'][:30]}…"):
        st.json(item["response"])
