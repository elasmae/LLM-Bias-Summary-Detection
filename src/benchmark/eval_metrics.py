from transformers import pipeline
from textblob import TextBlob
from datasets import load_metric
import numpy as np
import evaluate

def sentiment_polarity(text):
    return TextBlob(text).sentiment.polarity

def polarity_shift(original, summary):
    return sentiment_polarity(summary) - sentiment_polarity(original)

def factual_consistency_score(reference, summary, model_name="facebook/bart-large-mnli"):
    classifier = pipeline("zero-shot-classification", model=model_name)
    result = classifier(summary, candidate_labels=[reference])
    return result['scores'][0]  

def rouge_score(reference, summary):
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=[summary], references=[reference])


def bleu_score(reference, summary):
    bleu = evaluate.load("bleu")
    
    if isinstance(reference, (list, np.ndarray)):
        reference = " ".join(reference)
    if isinstance(summary, (list, np.ndarray)):
        summary = " ".join(summary)


    return bleu.compute(predictions=[summary], references=[[reference]])



def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0

if __name__ == "__main__":
    original = "Alice praised the team. Bob reported problems with the supplier. Conclusion: the project is back on track."
    summary = "Alice and Bob are happy with the supplier. Project is successful."

    print("Polarity Shift:", polarity_shift(original, summary))
    print("Factual Consistency Score:", factual_consistency_score(original, summary))
    print("ROUGE:", rouge_score(original, summary))
    print("BLEU:", bleu_score(original, summary))
    print("Lexical Diversity:", lexical_diversity(summary))
