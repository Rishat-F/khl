"""Юнит-тесты для функций предобработки хоккейных новостей."""

from pathlib import Path

import pytest

from khl import stop_words
from khl.preprocess import (
    PLACEHOLDER,
    UNKNOWN,
    _merge_codes,
    _merge_dates,
    _merge_lemmas,
    _merge_locs,
    _merge_ners,
    _merge_orgs,
    _merge_pens,
    _merge_pers,
    codes_to_lemmas,
    fix_lemma,
    get_coder,
    lemmas_to_codes,
    lemmatize,
)

tests_dir = Path(__file__).parent
test_frequency_dictionary_file = "example_frequency_dictionary.json"


@pytest.mark.parametrize(
    "text_list,expected_merged_text_list",
    [
        (["per"], ["per"]),
        (["per", "per", "per"], ["pers"]),
        (["слово", "per", "per", "слово"], ["слово", "pers", "слово"]),
        (
            ["per", "и", "per", "per", "и", "per", "per", "per"],
            ["per", "и", "pers", "и", "pers"],
        ),
        (
            ["слово", "слово", "per", "per", "слово", "слово"],
            ["слово", "слово", "pers", "слово", "слово"],
        ),
    ],
)
def test_merge_pers(text_list, expected_merged_text_list):
    assert _merge_pers(text_list) == expected_merged_text_list


@pytest.mark.parametrize(
    "text_list,expected_merged_text_list",
    [
        (["org"], ["org"]),
        (["org", "org", "org"], ["orgs"]),
        (["слово", "org", "org", "слово"], ["слово", "orgs", "слово"]),
        (
            ["org", "и", "org", "org", "и", "org", "org", "org"],
            ["org", "и", "orgs", "и", "orgs"],
        ),
        (
            ["слово", "слово", "org", "org", "слово", "слово"],
            ["слово", "слово", "orgs", "слово", "слово"],
        ),
    ],
)
def test_merge_orgs(text_list, expected_merged_text_list):
    assert _merge_orgs(text_list) == expected_merged_text_list


@pytest.mark.parametrize(
    "text_list,expected_merged_text_list",
    [
        (["loc"], ["loc"]),
        (["loc", "loc", "loc"], ["locs"]),
        (["слово", "loc", "loc", "слово"], ["слово", "locs", "слово"]),
        (
            ["loc", "и", "loc", "loc", "и", "loc", "loc", "loc"],
            ["loc", "и", "locs", "и", "locs"],
        ),
        (
            ["слово", "слово", "loc", "loc", "слово", "слово"],
            ["слово", "слово", "locs", "слово", "слово"],
        ),
    ],
)
def test_merge_locs(text_list, expected_merged_text_list):
    assert _merge_locs(text_list) == expected_merged_text_list


@pytest.mark.parametrize(
    "text_list,expected_merged_text_list",
    [
        (["date"], ["date"]),
        (["date", "date", "date"], ["dates"]),
        (["слово", "date", "date", "слово"], ["слово", "dates", "слово"]),
        (
            ["date", "и", "date", "date", "и", "date", "date", "date"],
            ["date", "и", "dates", "и", "dates"],
        ),
        (
            ["слово", "слово", "date", "date", "слово", "слово"],
            ["слово", "слово", "dates", "слово", "слово"],
        ),
    ],
)
def test_merge_dates(text_list, expected_merged_text_list):
    assert _merge_dates(text_list) == expected_merged_text_list


@pytest.mark.parametrize(
    "text_list,expected_merged_text_list",
    [
        (["pen"], ["pen"]),
        (["pen", "pen", "pen"], ["pens"]),
        (["слово", "pen", "pen", "слово"], ["слово", "pens", "слово"]),
        (
            ["pen", "и", "pen", "pen", "и", "pen", "pen", "pen"],
            ["pen", "и", "pens", "и", "pens"],
        ),
        (
            ["слово", "слово", "pen", "pen", "слово", "слово"],
            ["слово", "слово", "pens", "слово", "слово"],
        ),
    ],
)
def test_merge_pens(text_list, expected_merged_text_list):
    assert _merge_pens(text_list) == expected_merged_text_list


@pytest.mark.parametrize(
    "text_list,expected_merged_text_list",
    [
        (
            ["per", "per", "org", "org", "loc", "loc", "date", "date", "pen", "pen"],
            ["pers", "orgs", "locs", "dates", "pens"],
        ),
    ],
)
def test_merge_ners(text_list, expected_merged_text_list):
    assert _merge_ners(text_list) == expected_merged_text_list


@pytest.mark.parametrize(
    "source_lemma,expected_lemma",
    [
        ("", ""),
        (".,-:?!", ".,-:?!"),
        ("текст", "текст"),
        ("текст текст", "текст текст"),
        ("забиваем", "забивать"),
        ("основный", "основной"),
        ("родный", "родной"),
        ("голова", "гол"),
        ("голы", "гол"),
    ],
)
def test_fix_lemma(source_lemma, expected_lemma):
    assert fix_lemma(source_lemma) == expected_lemma


@pytest.mark.parametrize(
    "source_lemmas,expected_lemmas",
    [
        (["и", "или", "текст"], ["и", "или", "текст"]),
        (["и", "и", "и"], ["и"]),
        (["а", "и", "и", "или", "или", "или"], ["а", "и", "или"]),
    ],
)
def test_merge_lemmas(source_lemmas, expected_lemmas):
    assert _merge_lemmas(source_lemmas) == expected_lemmas


@pytest.mark.parametrize(
    "source_text,stop_words_,expected_lemmas",
    [
        (
            "Очень-очень хотим победить",
            None,
            ["очень", "хотеть", "победить"],
        ),
        pytest.param(
            "Очень-очень хотим победить",
            stop_words,
            ["хотеть", "победить"],
            marks=pytest.mark.bug_1,
        ),
        (
            "- Играете в футбол? - Иногда.",
            stop_words,
            ["-", "играть", "в", "футбол", "?", "-", "."],
        ),
        (
            "Сегодня Ансель Галимов забил несколько голов",
            stop_words,
            ["сегодня", "ансель", "галимов", "забить", "гол"],
        ),
        (
            "Мы побеждали и и и",
            stop_words,
            ["мы", "побеждать"],
        ),
        (
            "Он забивал забивает и еще забивать будет много сезонов",
            stop_words,
            ["он", "забивать", "быть", "сезон"],
        ),
        (
            "per и per забили по голу",
            stop_words,
            ["pers", "забить", "гол"],
        ),
        (
            "date per и per в loc забили много голов org "
            "а per получил pen за грубость",
            stop_words,
            [
                "date",
                "pers",
                "в",
                "loc",
                "забить",
                "гол",
                "org",
                "per",
                "получить",
                "pen",
                "грубость",
            ],
        ),
        (
            "января Шипачев и Зарипов в Москве забили много голов Спартаку "
            "а Сергей Широков получил за Грубость",
            None,
            [
                "январь",
                "шипачев",
                "и",
                "зарипов",
                "в",
                "москва",
                "забить",
                "много",
                "гол",
                "спартак",
                "а",
                "сергей",
                "широков",
                "получить",
                "за",
                "грубость",
            ],
        ),
    ],
)
def test_lemmatize(source_text, stop_words_, expected_lemmas):
    assert lemmatize(source_text, stop_words_) == expected_lemmas


@pytest.mark.parametrize(
    "source_text,expected_lemmas",
    [
        ("per и per забили по голу", ["pers", "забить", "гол"]),
        ("per - лет в org", ["per", "-", "год", "в", "org"]),
        ("per - на года в org", ["per", "-", "на", "год", "в", "org"]),
    ],
)
def test_lemmatize_with_default_params(source_text, expected_lemmas):
    assert lemmatize(source_text) == expected_lemmas


@pytest.mark.parametrize(
    "source_codes,expected_codes",
    [
        ([], []),
        ([10], [10]),
        ([10, 50, 100], [10, 50, 100]),
        ([10, 50, 100, 50, 10], [10, 50, 100, 50, 10]),
        ([10, 10, 50, 50, 50, 100, 100, 100, 100, 10, 10], [10, 50, 100, 10]),
    ],
)
def test_merge_codes(source_codes, expected_codes):
    assert _merge_codes(source_codes) == expected_codes


@pytest.mark.parametrize(
    "file_path",
    [
        tests_dir / test_frequency_dictionary_file,
        str(tests_dir) + "/" + test_frequency_dictionary_file,
    ],
)
def test_get_coder(file_path):
    assert get_coder(file_path) == {
        PLACEHOLDER: 0,
        UNKNOWN: 1,
        ".": 2,
        "и": 3,
        "в": 4,
        "-": 5,
        ":": 6,
        "матч": 7,
        "за": 8,
        "забить": 9,
        "гол": 10,
        "per": 11,
        "org": 12,
        "loc": 13,
        "date": 14,
        "против": 15,
        "год": 16,
        "pers": 17,
        "orgs": 18,
        "свой": 19,
        "pen": 20,
    }


class TestLemmasCodes:
    coder = {
        PLACEHOLDER: 0,
        UNKNOWN: 1,
        ".": 2,
        "команда": 3,
        "забить": 4,
        "гол": 5,
        "московский": 6,
    }
    lemmas = ["сегодня", "московский", "команда", "забить", "красивый", "гол", "."]

    @pytest.mark.parametrize(
        "exclude_unknown,max_len,expected_codes",
        [
            (False, None, [1, 6, 3, 4, 1, 5, 2]),
            (False, 5, [1, 6, 3, 4, 1]),
            (False, 10, [0, 0, 0, 1, 6, 3, 4, 1, 5, 2]),
            (True, None, [6, 3, 4, 5, 2]),
            (True, 3, [6, 3, 4]),
            (True, 10, [0, 0, 0, 0, 0, 6, 3, 4, 5, 2]),
        ],
    )
    def test_lemmas_to_codes(self, exclude_unknown, max_len, expected_codes):
        assert (
            lemmas_to_codes(self.lemmas, self.coder, exclude_unknown, max_len)
            == expected_codes
        )

    def test_lemmas_to_codes_with_default_params(self):
        expected_codes = [6, 3, 4, 5, 2]
        assert lemmas_to_codes(self.lemmas, self.coder) == expected_codes

    @pytest.mark.parametrize(
        "codes,expected_lemmas",
        [
            ([0, 0], ["", ""]),
            ([0, 0, 1, 1, 2], ["", "", "???", "???", "."]),
            (
                [0, 0, 1, 6, 3, 4, 1, 5, 2],
                ["", "", "???", "московский", "команда", "забить", "???", "гол", "."],
            ),
        ],
    )
    def test_codes_to_lemmas(self, codes, expected_lemmas):
        assert codes_to_lemmas(codes, self.coder) == expected_lemmas
