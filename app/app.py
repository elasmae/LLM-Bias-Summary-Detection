import streamlit as st
import pandas as pd
from src.inference.summarize import generate_summary
from src.benchmark.eval_metrics import (
    polarity_shift, factual_consistency_score,
    rouge_score, bleu_score, lexical_diversity
)
from src.mitigation.prompt_engineering import role_prompt

st.set_page_config(page_title="LLM Bias Summary Detection", layout="wide")

st.title("🤖 LLM Summary Bias Detection")

text = st.text_area("✍️ Input Text (e.g. meeting transcription)")

role = st.selectbox("👤 Select User Role", ["project_manager", "hr", "finance"])
apply_role_prompt = st.checkbox("🔁 Apply Role-Based Prompt Engineering")

model_name = st.selectbox("🧠 Choose Model", [
    "facebook/bart-large-cnn",
    "t5-small",
    "mistralai/Mistral-7B-v0.1"
])

if st.button("Generate & Evaluate Summary"):
    prompt_text = role_prompt(text, role) if apply_role_prompt else text
    summary = generate_summary(prompt_text, model_name)

    st.subheader("📝 Summary Output")
    st.write(summary)

    st.subheader("📊 Bias Evaluation Metrics")
    metrics = {
        "Polarity Shift": polarity_shift(text, summary),
        "Factual Consistency Score": factual_consistency_score(text, summary),
        "ROUGE": rouge_score(text, summary)["rouge1"],
        "BLEU": bleu_score(text, summary)["bleu"],
        "Lexical Diversity": lexical_diversity(summary),
        "Model": model_name,
        "Role": role
    }

    st.dataframe(pd.DataFrame([metrics]))

    # Export full report as CSV
    full_report = pd.DataFrame([{
        "Input": text,
        "Summary": summary,
        **metrics
    }])
    csv = full_report.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Full Report", csv, "summary_bias_report.csv", "text/csv")
