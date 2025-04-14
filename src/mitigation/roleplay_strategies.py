
ROLEPLAY_PROMPTS = {
    "neutral_journalist": "You are a neutral journalist. Write an unbiased and factual summary of the following meeting transcript.",
    "ethical_observer": "As an ethical AI trained to avoid stereotypes and biased conclusions, summarize the following text accurately and respectfully.",
    "diversity_officer": "Act as a corporate diversity and inclusion officer. Provide a summary that highlights fairness and avoids discriminatory language.",
    "professional_summarizer": "You are a professional business summarizer. Generate a concise and neutral summary without interpreting opinions or emotions.",
}


def apply_roleplay_prompt(text: str, strategy: str = "neutral_journalist") -> str:
    """
    Ajoute une instruction de style 'roleplay' avant le texte à résumer.

    Args:
        text (str): Le texte brut à résumer.
        strategy (str): Le rôle à jouer (clé dans ROLEPLAY_PROMPTS).

    Returns:
        str: Texte enrichi avec une instruction contextuelle.
    """
    prefix = ROLEPLAY_PROMPTS.get(strategy, "")
    return f"{prefix}\n\n{text}"
