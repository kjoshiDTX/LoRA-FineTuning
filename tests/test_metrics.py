from lora_ft.metrics import grade, normalize, summarize, token_f1


def test_normalize_strips_articles_and_punctuation():
    assert normalize("The Federalist Party.") == "federalist party"


def test_yes_no_uses_first_word():
    assert grade("Yes, he was.", "Yes", yes_no=True) == "correct"
    assert grade("No.", "Yes", yes_no=True) == "hallucinated"


def test_abstention_is_not_a_hallucination():
    assert grade("I don't know.", "Helsinki", yes_no=False) == "abstained"
    assert grade("I don’t know", "Yes", yes_no=True) == "abstained"


def test_factoid_containment_and_f1():
    assert grade("It is Helsinki", "Helsinki", yes_no=False) == "correct"
    assert grade("Thomas Jefferson", "Thomas Jefferson, Benjamin Franklin", yes_no=False) == "correct"
    assert grade("Oslo", "Helsinki", yes_no=False) == "hallucinated"
    assert token_f1("changes by humans", "changes wrought by humans") > 0.5


def test_summarize_rates():
    rows = [
        {"label": "correct", "yes_no": True},
        {"label": "hallucinated", "yes_no": False},
        {"label": "abstained", "yes_no": False},
        {"label": "correct", "yes_no": False},
    ]
    s = summarize(rows)["overall"]
    assert s["accuracy"] == 0.5
    assert s["hallucination_rate"] == 0.25
    assert s["abstention_rate"] == 0.25
