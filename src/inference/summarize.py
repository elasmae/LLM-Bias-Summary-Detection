
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Pipeline,
)
import torch


def generate_summary(
    text: str,
    model_name: str = "facebook/bart-large-cnn",
    max_length: int = 130,
    min_length: int = 30,
    device: int = 0 if torch.cuda.is_available() else -1,
) -> str:
    """
    Génère un résumé à partir d'un texte d'entrée avec le modèle spécifié.

    Args:
        text (str): Texte à résumer.
        model_name (str): Nom du modèle Hugging Face.
        max_length (int): Longueur maximale du résumé.
        min_length (int): Longueur minimale du résumé.
        device (int): -1 pour CPU, 0 pour GPU.

    Returns:
        str: Résumé généré.
    """
    try:
        if "mistral" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
            outputs = model.generate(inputs, max_length=max_length, min_length=min_length)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            summarizer: Pipeline = pipeline("summarization", model=model_name, device=device)
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
    except Exception as e:
        return f"[Error generating summary with model '{model_name}']: {str(e)}"


if __name__ == "__main__":
    example_text = (
        "Meeting started with project updates. Alice discussed Q1 marketing results. "
        "Bob mentioned a budget overrun in the engineering team. The meeting concluded "
        "with a new action plan and deadline."
    )

    print("Summary:", generate_summary(example_text))
