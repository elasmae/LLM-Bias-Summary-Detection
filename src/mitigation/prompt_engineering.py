
PROMPT_TEMPLATES = {
    "default": "Summarize the following text.",
    "concise_summary": "Summarize the following text in a concise and factual way.",
    "no_subjectivity": "Summarize the following text while avoiding any subjective or emotional language.",
    "highlight_facts": "Generate a summary that focuses only on the factual content of the following text.",
    "bias_detection": "Summarize this meeting transcript while checking for biased statements and rephrasing them neutrally.",
}


def apply_prompt_template(text: str, strategy: str = "default") -> str:
    """
    Applique un template de prompt d’ingénierie sur un texte d’entrée.

    Args:
        text (str): Le texte à résumer.
        strategy (str): Le nom du template à utiliser.

    Returns:
        str: Texte précédé de l’instruction personnalisée.
    """
    prefix = PROMPT_TEMPLATES.get(strategy, PROMPT_TEMPLATES["default"])
    return f"{prefix}\n\n{text}"
