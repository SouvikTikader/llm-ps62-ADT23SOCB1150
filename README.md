# PS62 – Expense Categorizer (OpenAI gpt-oss)

**Problem:** People don't track spending. Upload bank transactions (date, description, amount);
an OpenAI model (gpt-oss-20b via Groq, free) assigns a category to each row, pandas computes totals, and a bar chart plus short spending advice is shown.

## Run
```
pip install -r requirements.txt
# get a free key at https://console.groq.com/keys
echo GROQ_API_KEY=your_key > .env
streamlit run app.py
```
Tick "Use sample_transactions.csv" in the sidebar to try it quickly.

## How it works
CSV → batches of 20 descriptions → `openai/gpt-oss-20b` returns JSON categories → pandas groupby totals → bar chart → second LLM call for spending advice.
