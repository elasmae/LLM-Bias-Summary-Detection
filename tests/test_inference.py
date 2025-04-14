
from src.inference.summarize import generate_summary


def test_generate_summary_with_bart():
    example_text = (
        "Alice opened the meeting. Bob raised issues. Final agreement on new deadlines."
    )
    summary = generate_summary(example_text, model_name="facebook/bart-large-cnn")
    assert isinstance(summary, str)
    assert len(summary) > 10  


from src.mitigation.prompt_engineering import apply_prompt_template

def test_apply_prompt_template():
    text = "This is a meeting about marketing strategies."
    modified = apply_prompt_template(text, strategy="no_subjectivity")
    assert text in modified
    assert "subjective" not in modified.lower()

