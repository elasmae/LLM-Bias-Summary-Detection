import pandas as pd
import numpy as np
from benchmark.eval_metrics import (
    polarity_shift,
    rouge_score,
    bleu_score,
    lexical_diversity,
    factual_consistency_score  
)


def evaluate_summary_pair(reference: str, summary: str) -> dict:
    """Évalue une paire référence / résumé avec plusieurs métriques."""
    
    if isinstance(reference, (list, np.ndarray)):
        reference = " ".join(reference)
    if isinstance(summary, (list, np.ndarray)):
        summary = " ".join(summary)
    
    return {
        "polarity_shift": polarity_shift(reference, summary),
        "factual_consistency": factual_consistency_score(reference, summary),  
        "rouge1": rouge_score(reference, summary)["rouge1"],
        "bleu": bleu_score(reference, summary)["bleu"],
        "lexical_diversity": lexical_diversity(summary)
    }


def evaluate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique l'évaluation à toutes les paires text/summary du DataFrame.
    Le DataFrame d'entrée doit contenir les colonnes : 'text' et 'summary'.
    """
    results = []

    for _, row in df.iterrows():
        reference = row["text"]
        summary = row["summary"]
        
        metrics = evaluate_summary_pair(reference, summary)
        metrics["text"] = reference
        metrics["summary"] = summary
        results.append(metrics)

    return pd.DataFrame(results)
