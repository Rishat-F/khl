"""Юнит-тесты для функций предобработки хоккейных новостей."""

from pathlib import Path

import pytest

from khl import stop_words
from khl.preprocess import (
    PLACEHOLDER,
    UNKNOWN,
    codes_to_lemmas,
    fix_lemma,
    get_freq_dict,
    lemmas_to_codes,
    lemmatize,
    merge_codes,
    merge_dates,
    merge_lemmas,
    merge_locs,
    merge_ners,
    merge_orgs,
    merge_pens,
    merge_pers,
)

tests_dir = Path(__file__).parent
test_lemmas_dictionary_file = "lemmas_dictionary_for_tests.json"


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
    assert merge_pers(text_list) == expected_merged_text_list


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
    assert merge_orgs(text_list) == expected_merged_text_list


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
    assert merge_locs(text_list) == expected_merged_text_list


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
    assert merge_dates(text_list) == expected_merged_text_list


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
    assert merge_pens(text_list) == expected_merged_text_list


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
    assert merge_ners(text_list) == expected_merged_text_list


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
    assert merge_lemmas(source_lemmas) == expected_lemmas


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
            ["-", "играть", "футбол", "?", "-", "."],
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
    assert merge_codes(source_codes) == expected_codes


@pytest.mark.parametrize(
    "file_path",
    [
        tests_dir / test_lemmas_dictionary_file,
        str(tests_dir) + "/" + test_lemmas_dictionary_file,
    ],
)
def test_get_freq_dict(file_path):
    assert get_freq_dict(file_path) == {
        PLACEHOLDER: 0,
        UNKNOWN: 1,
        ".": 2,
        "и": 3,
        "в": 4,
        "а": 5,
        "-": 6,
        ":": 7,
        "матч": 8,
        "команда": 9,
        "клуб": 10,
        "за": 11,
        "забить": 12,
        "гол": 13,
        "очко": 14,
        "московский": 15,
        "per": 16,
        "org": 17,
        "loc": 18,
        "date": 19,
        "финал": 20,
        "набрать": 21,
        "год": 22,
        "карьера": 23,
        "апрель": 24,
        "pers": 25,
        "orgs": 26,
        "свой": 27,
    }


class TestLemmasCodes:
    freq_dict = {
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
            lemmas_to_codes(self.lemmas, self.freq_dict, exclude_unknown, max_len)
            == expected_codes
        )

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
        assert codes_to_lemmas(codes, self.freq_dict) == expected_lemmas
