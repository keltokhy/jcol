"""The header grammar: three shapes, and nothing asked while a header is too short or malformed."""

import pytest

from jcol.spec import Spec, parse


def test_a_plain_header_is_a_yes_no_question():
    spec = parse("  alleges   fraud? ")
    assert spec == Spec("noul", "alleges fraud")
    assert spec.question == {"type": "noul", "instructions": 'The text fits this description: "alleges fraud"'}


def test_a_comma_list_is_a_set_of_categories():
    spec = parse("product: mortgage, credit card, other, mortgage,")
    assert spec == Spec("choice", "product", ("mortgage", "credit card", "other"))
    assert spec.question == {"type": "choice", "instructions": "Choose the product that best describes the text.",
                             "criteria": {"mortgage": "mortgage", "credit card": "credit card", "other": "other"}}


def test_less_than_signs_make_a_scale():
    spec = parse("tone: calm < upset < furious")
    assert spec == Spec("score", "tone", ("calm", "upset", "furious"))
    assert spec.question == {"type": "score", "instructions": "Rate the text on this scale: tone.",
                             "criteria": ["calm", "upset", "furious"]}


@pytest.mark.parametrize("header", [
    "", "  ", "ab", "a?",                   # too short to be worth a call
    "product:", ": a, b", "product: only",  # a name and at least two options
    "tone: calm <", "tone: calm < calm",    # at least two distinct levels
    "tone: " + " < ".join(map(str, range(11))),
])
def test_an_unfinished_header_asks_nothing(header):
    assert parse(header) is None


def test_the_same_header_always_asks_the_same_question():
    assert parse("alleges fraud?").question == parse("alleges fraud").question == parse(" alleges  fraud ").question


def test_cells():
    assert parse("alleges fraud?").cell({"noul": 0.8312}) == (0.831, 0.831)
    assert parse("alleges fraud?").cell({"noul": 0.1}) == (0.1, 0.9)
    choice = parse("product: mortgage, other")
    assert choice.cell({"choice": "mortgage", "probabilities": {"mortgage": 0.75, "other": 0.25}}) == ("mortgage", 0.75)
    assert choice.cell({"choice": "other", "confidence": 0.6}) == ("other", 0.6)
    assert choice.cell({"choice": "other"}) == ("other", 0.0)
    assert parse("tone: calm < upset < furious").cell({"score": 1.234, "confidence": 0.5}) == (1.23, 0.5)
