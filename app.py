"""PS62 - Expense Categorizer using an OpenAI model (gpt-oss via Groq, free) + Streamlit."""
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CATEGORIES = ["Food", "Travel", "Bills", "Shopping", "Entertainment",
              "Health", "Education", "Other"]
ICONS = {"Food": "🍔", "Travel": "🚆", "Bills": "🧾", "Shopping": "🛍️",
         "Entertainment": "🎬", "Health": "💊", "Education": "📚", "Other": "📦"}
BATCH_SIZE = 20

PROVIDERS = {
    "Free: gpt-oss-20b (Groq)": dict(
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-20b", env="GROQ_API_KEY"),
    "OpenAI: gpt-4o-mini (paid key)": dict(
        base_url=None, model="gpt-4o-mini", env="OPENAI_API_KEY"),
}


def make_client(provider, key):
    return OpenAI(api_key=key, base_url=provider["base_url"])


def chat(client, model, prompt, json_mode=False):
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}])
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs).choices[0].message.content


def categorize_batch(client, model, descriptions):
    numbered = "\n".join(f"{i}. {d}" for i, d in enumerate(descriptions))
    prompt = (
        "You categorise bank transaction descriptions.\n"
        f"Allowed categories: {', '.join(CATEGORIES)}.\n"
        'Return ONLY JSON like {"categories": ["Food", "Travel"]} with exactly '
        "one category per line below, in the same order. Use 'Other' if unsure.\n\n"
        + numbered
    )
    try:
        cats = json.loads(chat(client, model, prompt, json_mode=True))["categories"]
    except (KeyError, ValueError, TypeError):
        cats = []
    cats = [c if c in CATEGORIES else "Other" for c in cats]
    cats += ["Other"] * (len(descriptions) - len(cats))
    return cats[:len(descriptions)]


def categorize_all(client, model, df):
    descs = df["description"].astype(str).tolist()
    out, bar = [], st.progress(0.0, text="Categorizing...")
    for i in range(0, len(descs), BATCH_SIZE):
        out += categorize_batch(client, model, descs[i:i + BATCH_SIZE])
        bar.progress(min((i + BATCH_SIZE) / len(descs), 1.0), text="Categorizing...")
    bar.empty()
    df = df.copy()
    df["category"] = out
    return df
#l

def spending_advice(client, model, summary):
    prompt = (
        "You are a friendly personal finance helper for a student. Based on this "
        "monthly spending summary (INR), write a 2-line summary and 3 practical "
        "saving tips in simple words. This is general advice, not financial "
        "advice.\n\n" + summary
    )
    return chat(client, model, prompt)


# ---------------- UI ----------------
st.set_page_config(page_title="Expense Categorizer", page_icon="💰", layout="wide")
st.markdown("""
<style>
.hero {padding: 1.6rem 2rem; border-radius: 18px; color: white;
  background: linear-gradient(120deg,#6a11cb 0%,#2575fc 100%); margin-bottom: 1rem;}
.hero h1 {margin:0; font-size:2.2rem;} .hero p {margin:.3rem 0 0; opacity:.9;}
div[data-testid="stMetric"] {background: rgba(120,120,255,.10); padding: 14px 18px;
  border-radius: 14px; border: 1px solid rgba(120,120,255,.25);}
.stButton>button {border-radius: 10px; font-weight: 600;}
</style>
<div class="hero"><h1>💰 AI Expense Categorizer</h1>
<p>Upload bank transactions, let the llm sort them, see where your money goes.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    choice = st.selectbox("Model", list(PROVIDERS))
    provider = PROVIDERS[choice]
    api_key = os.getenv(provider["env"]) or st.text_input(
        f"{provider['env']}", type="password")
    st.caption("Free key: console.groq.com/keys" if "Groq" in choice
               else "Needs paid credits on platform.openai.com")
    use_sample = st.checkbox("Use sample_transactions.csv", value=True)

file = st.file_uploader("📂 Upload CSV (date, description, amount)", type="csv")
df = pd.read_csv("sample_transactions.csv") if use_sample and not file else (
    pd.read_csv(file) if file else None)

if df is not None:
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"date", "description", "amount"}.issubset(df.columns):
        st.error("CSV must have columns: date, description, amount")
        st.stop()
    with st.expander("Preview uploaded data", expanded=False):
        st.dataframe(df, use_container_width=True)

    if st.button("✨ Categorize my expenses", type="primary"):
        if not api_key:
            st.error("Please enter an API key in the sidebar.")
            st.stop()
        try:
            client = make_client(provider, api_key)
            st.session_state["result"] = categorize_all(client, provider["model"], df)
            st.session_state.pop("advice", None)
        except Exception as e:
            st.error(f"API error: {e}")

if "result" in st.session_state:
    res = st.session_state["result"].copy()
    res["date"] = pd.to_datetime(res["date"], errors="coerce")
    res["month"] = res["date"].dt.strftime("%Y-%m")
    totals = res.groupby("category")["amount"].sum().sort_values(ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total spent", f"₹{res['amount'].sum():,.0f}")
    c2.metric("Transactions", len(res))
    top = totals.index[0]
    c3.metric("Top category", f"{ICONS[top]} {top}")
    c4.metric("Avg / transaction", f"₹{res['amount'].mean():,.0f}")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "🧾 Transactions", "📅 Monthly", "💡 Advice"])

    with tab1:
        a, b = st.columns(2)
        pie = px.pie(values=totals.values, names=[f"{ICONS[c]} {c}" for c in totals.index],
                     hole=0.5, title="Share by category")
        a.plotly_chart(pie, use_container_width=True)
        bar = px.bar(x=totals.index, y=totals.values, color=totals.index,
                     labels={"x": "Category", "y": "₹"}, title="Total by category")
        bar.update_layout(showlegend=False)
        b.plotly_chart(bar, use_container_width=True)

    with tab2:
        cat_filter = st.multiselect("Filter category", CATEGORIES, default=CATEGORIES)
        shown = res[res["category"].isin(cat_filter)].drop(columns=["month"])
        st.dataframe(shown, use_container_width=True)
        st.download_button("⬇️ Download categorized CSV", shown.to_csv(index=False),
                           "categorized.csv", "text/csv")

    with tab3:
        months = sorted(res["month"].dropna().unique())
        if months:
            m = st.selectbox("Month", months)
            mt = res[res["month"] == m].groupby("category")["amount"].sum()
            st.metric(f"Total for {m}", f"₹{mt.sum():,.0f}")
            st.bar_chart(mt)

    with tab4:
        if st.button("Get spending advice"):
            try:
                client = make_client(provider, api_key)
                st.session_state["advice"] = spending_advice(
                    client, provider["model"], totals.round(2).to_string())
            except Exception as e:
                st.error(f"API error: {e}")
        if "advice" in st.session_state:
            st.write(st.session_state["advice"])
            st.info("General guidance only — not financial advice.")
