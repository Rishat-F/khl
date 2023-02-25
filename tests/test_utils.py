"""Юнит-тесты для функций преобразования хоккейных новостей."""


import pytest

from khl.utils import (
    delete_age_category,
    delete_amplua,
    delete_beginning_ending_dashes_in_words,
    delete_birth_mark,
    delete_cirillic_ending_from_english_words,
    delete_ending_colon_dash,
    delete_letter_dot_letter_dot,
    delete_numeric_data,
    delete_one_symbol_english_words,
    delete_overtime_mark,
    delete_parentheses_content,
    delete_play_format,
    delete_quotes_with_one_symbol,
    delete_serial_numbers,
    delete_shutouts,
    delete_urls,
    delete_year_city_mark,
    fix_b_o_lshii,
    fix_cirillic_c_in_english_words,
    fix_covid,
    fix_dash_word,
    fix_dot_question,
    fix_dots,
    fix_english_dash_russian_words,
    fix_latin_c_in_russian_words,
    fix_org_loc,
    fix_question_dot,
    fix_question_marks,
    fix_surname_dash_surname_dash_surname,
    generalize_top,
    handwritten_replace_orgs,
    latin_c_to_cirillic,
    leave_only_significant_symbols,
    lowercase_sdk,
    lowercase_shaiba_word,
    merge_dashes,
    merge_spaces,
    replace_concrete_orgs,
    replace_dash_between_ners,
    replace_dates,
    replace_exclamation_mark_with_dot,
    replace_ners,
    replace_penalty,
    replace_tak_kak,
    replace_to_est,
    replace_vs_with_dash,
    simplify_text,
    split_ners,
    unify_text,
)


@pytest.mark.parametrize(
    "source_text, expected_text",
    [
        ("Текст", "Текст"),
        (" \t\r\n Текст\t \n\r", "Текст"),
        (
            "﻿​­ \tДанный \u200bтекст\xa0содержит \xadмного "
            "\ufeffплохих   символов.\n\tWe need to  delete them. "
            "\r—«»–−…⅛¼½„“”\\\"`йё - and \adon't forget \tfix "
            'those \ftoo! But leave "/".',
            "Данный текст содержит много плохих символов. "
            "We need to delete them. -''--...1/81/41/2''' ''йё - and "
            "don't forget fix those too! But leave '/'.",
        ),
        (
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "',.[]{}()/=+-%№#@!?;:0123456789",
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "',.[]{}()/=+-%№#@!?;:0123456789",
        ),
    ],
)
def test_unify_text(source_text, expected_text):
    assert unify_text(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        (
            "Казанский Ак Барс Олега Знарка сыграет в Омске с командой Ильи Воробьева.",
            "org per сыграет в loc с командой per.",
        ),
        (
            "Вниманию СМИ! Открытая тренировка 'Ак Барса'",
            "Вниманию org! Открытая тренировка 'org'",
        ),
        (
            "Гол Да Косты с передачи Галимова - лучший по итогам двух недель",
            "Гол per с передачи per - лучший по итогам двух недель",
        ),
        (
            "Андрей Педан не сыграет против 'Спартака'",
            "per не сыграет против 'org'",
        ),
        ("'Ак Барс' отправился в Уфу", "'org' отправился в loc"),
        ("Результаты матчей Хоккейной школы ЦСКА", "Результаты матчей org org"),
        (
            "Данис Зарипов - лучший бомбардир 'Ак Барса' в новейшей истории. "
            "Капитан 'Ак Барса' Данис Зарипов стал лучшим бомбардиром клуба в "
            "высших дивизионах российского хоккея. Зарипов обошёл предыдущего "
            "рекордсмена - Алексея Морозова - на счету которого 621 (266+355) "
            "очко за 'Ак Барс' в чемпионатах России и КХЛ.",
            "per - лучший бомбардир 'Ак Барса' в новейшей истории. "
            "Капитан 'org' per стал лучшим бомбардиром клуба в "
            "высших дивизионах российского хоккея. per обошёл предыдущего "
            "рекордсмена - per - на счету которого 621 (266+355) "
            "очко за 'org' в чемпионатах loc и org.",
        ),
    ],
)
def test_replace_ners(source_text, expected_text):
    assert replace_ners(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("1 января 2020 года и 21 января 2020 года", "date и date"),
        ("1 января 2020 года", "date"),
        (
            "Открытая тренировка 'Ак Барса' пройдет 1 января 2020г.",
            "Открытая тренировка 'Ак Барса' пройдет date",
        ),
        (
            "Гол Да Косты - лучший по итогам 2020 года по мнению болельщиков",
            "Гол Да Косты - лучший по итогам date по мнению болельщиков",
        ),
        (
            "Андрей Педан 01.01.2020 не сыграет против 'Спартака'",
            "Андрей Педан date не сыграет против 'Спартака'",
        ),
        (
            "1 января 2020 г 'Ак Барс' отправился в Уфу",
            "date 'Ак Барс' отправился в Уфу",
        ),
        (
            "Результаты матчей за период с 1 по 20 января 2020 года",
            "Результаты матчей за период с 1 по date",
        ),
    ],
)
def test_replace_dates(source_text, expected_text):
    assert replace_dates(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("", ""),
        (" ", " "),
        ("текст", "текст"),
        ("1+20", "1+20"),
        ("4 + \t10", "pen"),
        ("5+ 20", "pen"),
        ("1  +20", "1+20"),
        ("5+200", "5+200"),
        ("5 + 200", "5+200"),
        ("5  +  200", "5+200"),
        ("04+10", "04+10"),
        ("2+\t10", "pen"),
        ("5+20", "pen"),
        ("'4+10'", "'pen'"),
        ("'  2 + 10'", "'  pen'"),
        ("(5+10)", "(pen)"),
        ("(2+20)", "(pen)"),
        ("5+20+10", "5+20+10"),
        ("5  +\t20+ 10", "5+20+10"),
        ("текст '4+20' текст", "текст 'pen' текст"),
        (
            "штраф до (5+20 минут) до конца игры",
            "штраф до (pen минут) до конца игры",
        ),
        (
            "Броски: (5+20+16) 51 (15+ 20+16) - 25 (5+20)",
            "Броски: (5+20+16) 51 (15+20+16) - 25 (pen)",
        ),
    ],
)
def test_replace_penalty(source_text, expected_text):
    assert replace_penalty(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("", ""),
        (" ", " "),
        ("текст", "текст"),
        ("3:2", ""),
        ("3-2", ""),
        ("12 : 22", ""),
        ("45 - 52", ""),
        ("(0:1 0:2 0:3)", "(  )"),
        ("'0-1 0-2 0-3'", "'  '"),
        ("3 : 2", ""),
        ("3 - 2", ""),
        ("'0 : 1 0 : 2 0 : 3'", "'  '"),
        ("(0 - 1 0 - 2 0 - 3)", "(  )"),
        ("(0-1 0 : 2 0- 3)", "(  )"),
        ("'3:2'", "''"),
        ("на 12:25 минуте", "на  минуте"),
        ("счет 3 :2", "счет "),
        ("удары 3- 2 в пользу", "удары  в пользу"),
        ("статистика 3  :  2 по ударам", "статистика  по ударам"),
        ("3  -  2", ""),
    ],
)
def test_delete_numeric_data(source_text, expected_text):
    assert delete_numeric_data(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("", ""),
        ("Текст", "Текст"),
        ("Просто текст - без ner-ов", "Просто текст - без ner-ов"),
        ("per  org loc", "per  org loc"),
        ("org-loc", "org loc"),
        ("per-org- loc", "per org  loc"),
        ("org - loc-date", "org   loc date"),
        ("loc  --  date\t\t-\npen", "loc      date\t\t \npen"),
        ("слово - date-pen-per -", "слово - date pen per -"),
        (" - per-loc-pen  -\t", " - per loc pen  -\t"),
        (
            "per прокомментировал столкновение с per. - per сам пошел в стык.",
            "per прокомментировал столкновение с per. - per сам пошел в стык.",
        ),
        (
            "per: - per смял нас в первых сменах",
            "per: - per смял нас в первых сменах",
        ),
    ],
)
def test_replace_dash_between_ners(source_text, expected_text):
    assert replace_dash_between_ners(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("", ""),
        (" ", " "),
        ("  ", " "),
        ("   ", " "),
        (" \n\t\r", " "),
        ("  \n\n\t\t\r\r  ", " "),
        ("Слово \n\t  \r \n слово\n\n", "Слово слово "),
    ],
)
def test_merge_spaces(source_text, expected_text):
    assert merge_spaces(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("Текст.", "Текст."),
        (" ", " "),
        ("...", "."),
        (" .", "."),
        (" . .", "."),
        (". .", "."),
        (". ..", "."),
        ("Текст...", "Текст."),
        ("Текст. .. Текст.", "Текст. Текст."),
        ("Текст-. .. Текст.", "Текст. Текст."),
        (
            "Протокол матча: СКА - 'Динамо'Москва - .",
            "Протокол матча: СКА - 'Динамо'Москва.",
        ),
        (
            "Протокол матча: СКА - 'Динамо'Москва  --  .",
            "Протокол матча: СКА - 'Динамо'Москва.",
        ),
        (
            "Протокол матча: СКА - 'Динамо'Москва ..  --  .",
            "Протокол матча: СКА - 'Динамо'Москва.",
        ),
        (
            "Протокол матча: СКА - 'Динамо'Москва - . - .. -. .",
            "Протокол матча: СКА - 'Динамо'Москва.",
        ),
    ],
)
def test_fix_dots(source_text, expected_text):
    assert fix_dots(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("Текст?", "Текст?"),
        (" ", " "),
        ("???", "?"),
        (" ?", "?"),
        (" ? ?", "?"),
        ("? ?", "?"),
        ("? ??", "?"),
        ("Текст???", "Текст?"),
        ("Текст? ?? Текст?", "Текст? Текст?"),
        ("Текст-? ?? Текст?", "Текст? Текст?"),
        (
            "Протокол матча: СКА - 'Динамо'Москва - ?",
            "Протокол матча: СКА - 'Динамо'Москва?",
        ),
        (
            "Протокол матча: СКА - 'Динамо'Москва  --  ?",
            "Протокол матча: СКА - 'Динамо'Москва?",
        ),
        (
            "Протокол матча: СКА - 'Динамо'Москва ??  --  ?",
            "Протокол матча: СКА - 'Динамо'Москва?",
        ),
        (
            "Протокол матча: СКА - 'Динамо'Москва - ? - ?? -? ?",
            "Протокол матча: СКА - 'Динамо'Москва?",
        ),
    ],
)
def test_fix_question_marks(source_text, expected_text):
    assert fix_question_marks(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        (
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            " _0123456789\"',.[]{}()/=+-%№#@!?`;:"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz",
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            "               .         -     ?  :"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz",
        ),
    ],
)
def test_leave_only_significant_symbols(source_text, expected_text):
    assert leave_only_significant_symbols(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Победите сегодня? - Да!", "Победите сегодня? - Да."),
        ("Сколько можно проигрывать!!!", "Сколько можно проигрывать."),
        ("Сколько можно проигрывать?!!!", "Сколько можно проигрывать?!"),
        ("Сколько можно проигрывать!!!?", "Сколько можно проигрывать!?"),
        ("Сколько можно проигрывать!?", "Сколько можно проигрывать!?"),
        ("Сколько можно проигрывать?!", "Сколько можно проигрывать?!"),
    ],
)
def test_replace_exclamation_mark_with_dot(source_text, expected_text):
    assert replace_exclamation_mark_with_dot(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("сдк", "сдк"),
        ("СДК", "сдк"),
        ("Решение Сдк по эпизоду с игроком", "Решение сдк по эпизоду с игроком"),
        ("СДК дисквалифицировал защитника", "сдк дисквалифицировал защитника"),
    ],
)
def test_lowercase_sdk(source_text, expected_text):
    assert lowercase_sdk(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("Шайба", "шайба"),
        (
            "Шайба, пропущенная в третьем периоде, была лишней",
            "шайба, пропущенная в третьем периоде, была лишней",
        ),
    ],
)
def test_lowercase_shaiba_word(source_text, expected_text):
    assert lowercase_shaiba_word(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("ТОП-10 топ-5 TOP-3 top-100", "ТОП ТОП ТОП ТОП"),
        ("Попал в ТОП-10 сейвов", "Попал в ТОП сейвов"),
        ("Годовой TOP-3 сейвов", "Годовой ТОП сейвов"),
    ],
)
def test_generalize_top(source_text, expected_text):
    assert generalize_top(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Поднялся с 5-го места", "Поднялся с  места"),
        ("10-й гол в сезоне", " гол в сезоне"),
        ("На 1ом месте", "На  месте"),
        ("Группа А: 1. Спартак 2. Динамо", "Группа А:  Спартак  Динамо"),
        (
            "Школа в 60-70-х действительно была отменная.",
            "Школа в  действительно была отменная.",
        ),
    ],
)
def test_delete_serial_numbers(source_text, expected_text):
    assert delete_serial_numbers(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Да нееет (смеется).", "Да нееет ."),
        ("Слово (текст внутри скобок) слово", "Слово  слово"),
        ("Слово (текст (скобки) неправильный", "Слово (текст  неправильный"),
        ("Слово (скобки) еще (скобки) слово", "Слово  еще  слово"),
        ("Слово (скобки) еще скобки) слово", "Слово  еще скобки) слово"),
    ],
)
def test_delete_parentheses_content(source_text, expected_text):
    assert delete_parentheses_content(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("2:3 ОТ", "2:3 "),
        ("2:3 OT", "2:3 "),
        ("4:3 2ОТ", "4:3 "),
        ("3:2 (1:0 0:1 1:1 1:0 ОТ)", "3:2 (1:0 0:1 1:1 1:0 )"),
        ("3:4 ОТ2", "3:4 "),
        ("2:3ОТ", "2:3"),
    ],
)
def test_delete_overtime_mark(source_text, expected_text):
    assert delete_overtime_mark(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("2 на 2", "2 на 2"),
        ("2х2", "2х2"),
        ("3 на 3", ""),
        ("5х5", ""),
        ("Забили 5 на 4.", "Забили ."),
        ("Забили в формате '3на3' гол.", "Забили в формате '' гол."),
        ("В формате '4 на 5' было тяжело.", "В формате '' было тяжело."),
        ("Полторы минуты 3 на 5 отстояли.", "Полторы минуты  отстояли."),
        ("4 на 3 отстояли.", " отстояли."),
        ("Овертайм в формате 4 на 4.", "Овертайм в формате ."),
        ("Забили 3х4.", "Забили ."),
        ("Забили в формате '3Х3' гол.", "Забили в формате '' гол."),
        ("В формате '5 х 5' было тяжело.", "В формате '' было тяжело."),
        ("Полторы минуты 3x5 отстояли.", "Полторы минуты  отстояли."),
        ("4X3 отстояли.", " отстояли."),
        ("Овертайм в формате 4 x 4.", "Овертайм в формате ."),
        ("Перевес '55 на 33' в пользу гостей.", "Перевес '55 на 33' в пользу гостей."),
        (
            "Игра проходила в формате 6х3 при пустых воротах.",
            "Игра проходила в формате  при пустых воротах.",
        ),
        (
            "Статистика бросков: 23 на 31 в пользу гостей.",
            "Статистика бросков: 23 на 31 в пользу гостей.",
        ),
        (
            "Статистика бросков: '14 на 5' в пользу гостей.",
            "Статистика бросков: '14 на 5' в пользу гостей.",
        ),
    ],
)
def test_delete_play_format(source_text, expected_text):
    assert delete_play_format(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Хоккей без шайбы.", "Хоккей без шайбы."),
        ("Текст", "Текст"),
        ("Текcт", "Текcт"),
        ("Забил c острого угла.", "Забил с острого угла."),
        ("C первой позиции", "С первой позиции"),
    ],
)
def test_latin_c_to_cirillic(source_text, expected_text):
    assert latin_c_to_cirillic(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("rasskazhem vse ob igre", "rasskazhem vse ob igre"),
        ("rasskazhem evs ob igre", "rasskazhem evs ob igre"),
        ("Трактор vs Сибирь", "Трактор - Сибирь"),
        ("'Салават Юлаев' - 'Ак Барс'", "'Салават Юлаев' - 'Ак Барс'"),
        ("'Салават Юлаев' vs 'Ак Барс'", "'Салават Юлаев' - 'Ак Барс'"),
        ("'Салават Юлаев' VS 'Ак Барс'", "'Салават Юлаев' - 'Ак Барс'"),
        ("'Салават Юлаев' Vs 'Ак Барс'", "'Салават Юлаев' - 'Ак Барс'"),
        ("'Салават Юлаев' vS 'Ак Барс'", "'Салават Юлаев' - 'Ак Барс'"),
        ("'Салават Юлаев' - vs - 'Ак Барс'", "'Салават Юлаев' - 'Ак Барс'"),
        ("'Салават Юлаев'VS'Ак Барс'", "'Салават Юлаев' - 'Ак Барс'"),
        ("'Салават Юлаев'  vs- 'Ак Барс'", "'Салават Юлаев' - 'Ак Барс'"),
        ("'Салават Юлаев'   Vs   -'Ак Барс'", "'Салават Юлаев' - 'Ак Барс'"),
    ],
)
def test_replace_vs_with_dash(source_text, expected_text):
    assert replace_vs_with_dash(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("22.10.2022", "22.10.2022"),
        ("Текст.", "Текст."),
        ("По ссылке http://mit.edu.com повторы.", "По ссылке  повторы."),
        ("Заходи на https://facebook.jp.com.2. Кхл.", "Заходи на .2. Кхл."),
        ("ищи на www.google.be. в два клика", "ищи на . в два клика"),
        ("ссылка https://www.google.be. 123", "ссылка . 123"),
        ("www.website.gov.us текст", " текст"),
        ("Голосуйте на www.test.com.", "Голосуйте на ."),
        ("www.website.gov.us/login.html", ""),
        ("search at google.co.jp/maps.", "search at ."),
        ("Покупайте на билеты.ru", "Покупайте на "),
        ("Покупайте на www.билеты.ru", "Покупайте на "),
        ("Покупайте на билеты.ру", "Покупайте на "),
        ("https://twitter.com/hcakbars/status/1313064600796114944?s=21", ""),
    ],
)
def test_delete_urls(source_text, expected_text):
    assert delete_urls(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        (
            "С матча 'Амур' - 'Адмирал' начнется сегодняшний игровой день!",
            "С матча 'org' - 'org' начнется сегодняшний игровой день!",
        ),
        (
            "С матча 'Амур'-'Адмирал' начнется сегодняшний игровой день!",
            "С матча 'org'-'org' начнется сегодняшний игровой день!",
        ),
        (
            "C матча 'Амур - Адмирал' начнется сегодняшний игровой день!",
            "C матча 'Амур - Адмирал' начнется сегодняшний игровой день!",
        ),
        (
            "C матча Амур - Адмирал начнется сегодняшний игровой день!",
            "C матча Амур - Адмирал начнется сегодняшний игровой день!",
        ),
        (
            "'Автомобилист' отправился на выезд. После перерыва на игры "
            "сборной наша команда возобновляет чемпионат с выездной серии. "
            "'Автомобилисту'предстоит три матча в гостях",
            "'org' отправился на выезд. После перерыва на игры "
            "сборной наша команда возобновляет чемпионат с выездной серии. "
            "'org'предстоит три матча в гостях",
        ),
    ],
)
def test_handwritten_replace_orgs(source_text, expected_text):
    assert handwritten_replace_orgs(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("covid", "covid"),
        (
            "Прошли тест на COVID-19.",
            "Прошли тест на covid.",
        ),
        (
            "Прошли тест на COVID19.",
            "Прошли тест на covid.",
        ),
        (
            "Результат COVID-19 отрицательный",
            "Результат covid отрицательный",
        ),
        (
            "covid-19 отрицательный",
            "covid отрицательный",
        ),
        (
            "Профилактика COVID нужна",
            "Профилактика covid нужна",
        ),
        (
            "Возможен COVID+ в будущем",
            "Возможен covid в будущем",
        ),
        (
            "Covid+",
            "covid",
        ),
        (
            "covid-",
            "covid",
        ),
    ],
)
def test_fix_covid(source_text, expected_text):
    assert fix_covid(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("covid", "covid"),
        ("Cдали теcт на COVID.", "Сдали тест на COVID."),
        ("Они задавили наc.", "Они задавили нас."),
        ("Отдал паc нападающему.", "Отдал пас нападающему."),
        ("Стартовый состав.", "Стартовый состав."),
        ("Cтартовый cоcтав.", "Стартовый состав."),
        ("Расставились", "Расставились"),
        ("Раccтавилиcь", "Расставились"),
        ("Растасовались", "Растасовались"),
        ("Раcтаcовалиcь", "Растасовались"),
        ("Champions", "Champions"),
        ("Сhampions", "Сhampions"),
        ("We are the champions", "We are the champions"),
        ("We are the сhampions", "We are the сhampions"),
        ("Read doc.", "Read doc."),
        ("Read doс first", "Read doс first"),
    ],
)
def test_fix_latin_c_in_russian_words(source_text, expected_text):
    assert fix_latin_c_in_russian_words(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("Текcт", "Текcт"),
        ("covid", "covid"),
        ("сovid", "covid"),
        ("Cдали тест на СOVID.", "Cдали тест на COVID."),
        ("Раccтавились мы", "Раccтавились мы"),
        ("Расставилиcь они", "Расставилиcь они"),
        ("Раcтасовалиcь", "Раcтасовалиcь"),
        ("Раcтаcовалиcь", "Раcтаcовалиcь"),
        ("Champions", "Champions"),
        ("Сhampions", "Champions"),
        ("We are the champions", "We are the champions"),
        ("We are the сhampions", "We are the champions"),
        ("Read doс.", "Read doc."),
        ("Read doс first", "Read doc first"),
    ],
)
def test_fix_cirillic_c_in_english_words(source_text, expected_text):
    assert fix_cirillic_c_in_english_words(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("УАЗик на ходу.", "УАЗик на ходу."),
        (
            "Сотрудники ОМОНа присутствуют на стадионе",
            "Сотрудники ОМОНа присутствуют на стадионе",
        ),
        ("Игроки Aк Барса забили гол.", "Игроки Aк Барса забили гол."),
        ("Почетные Gости города", "Почетные Gости города"),
        ("Люблю играть в футбол на PSке", "Люблю играть в футбол на PS"),
        ("Вчера игроки HIFKа заболели COVIDом", "Вчера игроки HIFK заболели COVID"),
        ("Сегодня в KHLе пройдет 5 матчей", "Сегодня в KHL пройдет 5 матчей"),
        ("Проехался на TANECOмобиле.", "Проехался на TANECO."),
    ],
)
def test_delete_cirillic_ending_from_english_words(source_text, expected_text):
    assert delete_cirillic_ending_from_english_words(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("По КХЛ-TV.", "По КХЛ-TV."),
        ("По Кхл-тв.", "По Кхл-тв."),
        ("По КХЛ-HD.", "По КХЛ-HD."),
        ("В VIP-ложе.", "В ложе."),
        ("Играю на Xbox-консоли в фифу.", "Играю на консоли в фифу."),
        ("Смотрите в YouTube-видео", "Смотрите в видео"),
        ("Пишите в telegram-канал!", "Пишите в канал!"),
    ],
)
def test_fix_english_dash_russian_words(source_text, expected_text):
    assert fix_english_dash_russian_words(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("Кубок вызова U17", "Кубок вызова"),
        ("Спартак U-20.", "Спартак."),
        ("Спартак U 20.", "Спартак."),
        ("Спартак U - 20.", "Спартак."),
        ("'АдмиралU16'", "'Адмирал'"),
        ("Сборная РоссияU-16 победила", "Сборная Россия победила"),
        ("Матч Амур-U-20 - Барыс-U20", "Матч Амур - Барыс"),
        ("Матч Амур-U-20-Барыс-U-20", "Матч Амур-Барыс"),
    ],
)
def test_delete_age_category(source_text, expected_text):
    assert delete_age_category(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("Выйграл 20 грандов", "Выйграл 20 грандов"),
        ("Нападающий 2000 г.р. уехал играть", "Нападающий  уехал играть"),
        ("Игроки 2010-2012 гг.р.", "Игроки "),
        ("Дети 2015г.р. играют", "Дети  играют"),
        ("Принял команду 1958-59 г.р., она", "Принял команду , она"),
        ("Сборная Казани 2007 г. р. - победитель", "Сборная Казани  - победитель"),
        ("Среди юниоров 2003/04 гг.р. провели", "Среди юниоров  провели"),
        (
            "от 14.08.2020г. разрешено присутствие",
            "от 14.08.2020г. разрешено присутствие",
        ),
    ],
)
def test_delete_birth_mark(source_text, expected_text):
    assert delete_birth_mark(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("ps", "ps"),
        ("PS", "PS"),
        ("P.S.", ""),
        ("Текст. P.S.", "Текст. "),
        ("P.S. Текст", " Текст"),
        ("Текст. p.s.", "Текст. "),
        ("P.s.", ""),
        ("И.о. главного тренера", " главного тренера"),
        ("Кафе, магазины и т.д.", "Кафе, магазины и "),
        ("2000 - н. в.", "2000 - "),
    ],
)
def test_delete_letter_dot_letter_dot(source_text, expected_text):
    assert delete_letter_dot_letter_dot(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("Б Б. б б. БУЛ БУЛ. бул бул. SO so", "         "),
        ("2:3 БУЛ.", "2:3 "),
        ("2:3 бул.", "2:3 "),
        ("2:3 БУЛ", "2:3 "),
        ("2:3 бул", "2:3 "),
        ("2:3 SO", "2:3 "),
        ("2:3 so", "2:3 "),
        ("2:3 Б", "2:3 "),
        ("2:3Б", "2:3"),
        ("2:3Б. счет матча", "2:3 счет матча"),
        ("2:3 Б. в пользу", "2:3  в пользу"),
        ("2:3 б в пользу", "2:3  в пользу"),
        ("2:3 б. в пользу", "2:3  в пользу"),
    ],
)
def test_delete_shutouts(source_text, expected_text):
    assert delete_shutouts(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("Играл за и на команду", "Играл за и на команду"),
        ("Победили, т.к. забили на гол больше", "Победили, т.к. забили на гол больше"),
        (
            "Состав: вр. Сорокин, з. Петров, з. Сидоров, н. Иванов",
            "Состав:  Сорокин,  Петров,  Сидоров,  Иванов",
        ),
        (
            "Состав: вр Сорокин, з Петров, з Сидоров, н Иванов",
            "Состав:  Сорокин,  Петров,  Сидоров,  Иванов",
        ),
    ],
)
def test_delete_amplua(source_text, expected_text):
    assert delete_amplua(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("На красной линии", "На красной линии"),
        ("Победили, т.е. выйграли", "Победили, т.е. выйграли"),
        ("Победили, т.к. забили больше", "Победили, так как забили больше"),
        ("Победили, т. к. забили больше", "Победили, так как забили больше"),
    ],
)
def test_replace_tak_kak(source_text, expected_text):
    assert replace_tak_kak(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("Победим если забьем", "Победим если забьем"),
        ("Победили, т.к. забили больше", "Победили, т.к. забили больше"),
        ("Победили, т.е. выйграли", "Победили, то есть выйграли"),
        ("Победили, т. е. выйграли", "Победили, то есть выйграли"),
    ],
)
def test_replace_to_est(source_text, expected_text):
    assert replace_to_est(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("Матч 'Витязя' с 'Северсталью'.", "Матч 'Витязя' с 'Северсталью'."),
        ("План 'Б' сработал", "План  сработал"),
        ("Билеты на сектор 'B'.", "Билеты на сектор ."),
        ("Мы сыграли на '5'", "Мы сыграли на "),
        ("И я расставил все точки над 'i'", "И я расставил все точки над "),
    ],
)
def test_delete_quotes_with_one_symbol(source_text, expected_text):
    assert delete_quotes_with_one_symbol(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("План A сработал", "План  сработал"),
        ("Билеты на сектор 'B'.", "Билеты на сектор ''."),
        ("Сектора A, B, C доступны", "Сектора , ,  доступны"),
        ("Спонсор g-drive", "Спонсор -drive"),
        ("I want to be a hero", " want to be  hero"),
    ],
)
def test_delete_one_symbol_english_words(source_text, expected_text):
    assert delete_one_symbol_english_words(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        (
            "Команды -org в сезоне-2020/2021 не было. Лишь в date- она появилась",
            "Команды org в сезоне2020/2021 не было. Лишь в date она появилась",
        ),
        (
            "- Да, я с вами согласен - сегодня мы играли so-so.",
            "- Да, я с вами согласен - сегодня мы играли so-so.",
        ),
        ("Да-да, вы все- -правильно -говорите-", "Да-да, вы все правильно говорите"),
        ("Поздравляем капитана с 20-летием!", "Поздравляем капитана с 20летием!"),
    ],
)
def test_delete_beginning_ending_dashes_in_words(source_text, expected_text):
    assert delete_beginning_ending_dashes_in_words(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("per-per", "per-per"),
        ("perper-", "per per-"),
        ("-orgorg-", "-org org-"),
        ("locloc", "loc loc"),
        ("datedate", "date date"),
        ("penpen", "pen pen"),
        ("perloc", "per loc"),
        ("perorg", "per org"),
        ("perdate", "per date"),
        ("perpen", "per pen"),
        ("orgper", "org per"),
        ("orgloc", "org loc"),
        ("orgdate", "org date"),
        ("orgpen", "org pen"),
        ("locper", "loc per"),
        ("locorg", "loc org"),
        ("locdate", "loc date"),
        ("locpen", "loc pen"),
        ("dateper", "date per"),
        ("dateorg", "date org"),
        ("dateloc", "date loc"),
        ("datepen", "date pen"),
        ("penper", "pen per"),
        ("penorg", "pen org"),
        ("penloc", "pen loc"),
        ("pendate", "pen date"),
        ("Команда perorg победила.", "Команда per org победила."),
        ("locorg нужно разделить", "loc org нужно разделить"),
    ],
)
def test_split_ners(source_text, expected_text):
    assert split_ners(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("большой вратарь", "большой вратарь"),
        ("Б о льшую часть времени.", "Большую часть времени."),
        ("Стоило б о льших усилий.", "Стоило больших усилий."),
    ],
)
def test_fix_b_o_lshii(source_text, expected_text):
    assert fix_b_o_lshii(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("", ""),
        ("-", "-"),
        ("--", "-"),
        ("---", "-"),
        ("ха-ха-ха", "ха-ха-ха"),
        (" --\n\n\t\t\r\r-- ", " - "),
        ("Слово-\n\t --\r-\n -слово\n\n-", "Слово-\n\t -\n -слово\n\n-"),
    ],
)
def test_merge_dashes(source_text, expected_text):
    assert merge_dashes(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("", ""),
        ("...", "..."),
        ("???", "???"),
        ("? - ...", "? - ..."),
        ("?...", "?"),
        ("? . . .", "?"),
        ("? .", "?"),
        ("? .. ..", "?"),
        ("? ..\t.\n.", "?"),
    ],
)
def test_fix_question_dot(source_text, expected_text):
    assert fix_question_dot(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("", ""),
        ("...", "..."),
        ("???", "???"),
        ("... - ?", "... - ?"),
        ("?.", "?."),
        ("...?", "?"),
        (". ?", "?"),
        (".. .. ..?", "?"),
        ("..\t..\n ..?", "?"),
    ],
)
def test_fix_dot_question(source_text, expected_text):
    assert fix_dot_question(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("2020 г", "2020 "),
        ("2020г.", "2020"),
        ("Сезон 2020/2021 гг. прошел", "Сезон 2020/2021  прошел"),
        ("2020 Г.", "2020 "),
        ("2018-2020 гг", "2018-2020 "),
        (
            "Команда г.Казани забила красивый гол команде г. Уфы.",
            "Команда Казани забила красивый гол команде  Уфы.",
        ),
    ],
)
def test_delete_year_city_mark(source_text, expected_text):
    assert delete_year_city_mark(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("", ""),
        ("текст", "текст"),
        ("Текст Текст", "Текст Текст"),
        ("Как-то", "Как-то"),
        ("когда-нибудь", "когда-нибудь"),
        ("пропустили - проиграли", "пропустили - проиграли"),
        (
            "Команда 'Локо-Юниор' проиграла в Нур-Султане",
            "Команда 'Локо-Юниор' проиграла в Нур-Султане",
        ),
        (
            "Зарипов-Морозов - лучшие друзья",
            "Зарипов-Морозов - лучшие друзья",
        ),
        (
            "Зарипов-Морозов-Чупин - лучшие друзья",
            "Зарипов - Морозов - Чупин - лучшие друзья",
        ),
        (
            "Составы команд: 'Автомобилист': Гросс "
            "1-я пятёрка: Трямкин-Щемеров; Макеев -Дацюк (К)-Мэйсек "
            "2-я пятёрка: Геноуэй -Зборовский; Секстон - Холланд (А)-Жукенов "
            "3-я пятёрка: Арзамасцев-Хольцер; Белоусов    -     Куликов-Голышев "
            "4-я пятёрка: Хрипунов -  Протапович- Литовченко",
            "Составы команд: 'Автомобилист': Гросс "
            "1-я пятёрка: Трямкин-Щемеров; Макеев -Дацюк (К)-Мэйсек "
            "2-я пятёрка: Геноуэй -Зборовский; Секстон - Холланд (А)-Жукенов "
            "3-я пятёрка: Арзамасцев-Хольцер; Белоусов - Куликов - Голышев "
            "4-я пятёрка: Хрипунов - Протапович - Литовченко",
        ),
        (
            "Составы команд: 'Автомобилист': Гросс "
            "1-я пятёрка: Трямкин-Щемеров -  Макеев -Дацюк (К)-Мэйсек "
            "2-я пятёрка: Геноуэй-Зборовский-Секстон-Холланд-Жукенов "
            "3-я пятёрка: Арзамасцев - Хольцер - Белоусов - Куликов - Голышев "
            "4-я пятёрка: Хрипунов -  Протапович- Литовченко; Иванов-Петров",
            "Составы команд: 'Автомобилист': Гросс "
            "1-я пятёрка: Трямкин - Щемеров - Макеев - Дацюк (К)-Мэйсек "
            "2-я пятёрка: Геноуэй - Зборовский - Секстон - Холланд - Жукенов "
            "3-я пятёрка: Арзамасцев - Хольцер - Белоусов - Куликов - Голышев "
            "4-я пятёрка: Хрипунов - Протапович - Литовченко; Иванов-Петров",
        ),
    ],
)
def test_fix_surname_dash_surname_dash_surname(source_text, expected_text):
    assert fix_surname_dash_surname_dash_surname(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("", ""),
        ("текст", "текст"),
        ("Текст Текст", "Текст Текст"),
        ("Как-то", "Как-то"),
        ("когда-нибудь", "когда-нибудь"),
        ("пропустили - проиграли", "пропустили - проиграли"),
        (
            "Команда 'Локо-Юниор' проиграла в Нур-Султане",
            "Команда 'Локо-Юниор' проиграла в Нур-Султане",
        ),
        (
            "Зарипов-Морозов - лучшие друзья",
            "Зарипов-Морозов - лучшие друзья",
        ),
        (
            "Зарипов-Морозов-Чупин - лучшие друзья",
            "Зарипов-Морозов-Чупин - лучшие друзья",
        ),
        ("-Команда проиграла...", "- Команда проиграла..."),
        ("Да.-Команда проиграла...", "Да. - Команда проиграла..."),
        ("Спасибо Ивану-", "Спасибо Ивану -"),
        ("Спасибо, Иван-! Удачи!", "Спасибо, Иван - ! Удачи!"),
        (
            "Составы команд: 'Автомобилист': Гросс "
            "1-я пятёрка: Трямкин-Щемеров; Макеев -Дацюк (К)-Мэйсек "
            "2-я пятёрка: Геноуэй -Зборовский; Секстон- Холланд (А)-Жукенов "
            "3-я пятёрка: Арзамасцев-Хольцер; Белоусов    -     Куликов-Голышев "
            "4-я пятёрка: Хрипунов -  Протапович- Литовченко",
            "Составы команд: 'Автомобилист': Гросс "
            "1-я пятёрка: Трямкин-Щемеров; Макеев - Дацюк (К) - Мэйсек "
            "2-я пятёрка: Геноуэй - Зборовский; Секстон - Холланд (А) - Жукенов "
            "3-я пятёрка: Арзамасцев-Хольцер; Белоусов    -     Куликов-Голышев "
            "4-я пятёрка: Хрипунов -  Протапович - Литовченко",
        ),
        (
            "Составы команд: 'Автомобилист': Гросс "
            "1-я пятёрка: Трямкин-Щемеров -  Макеев -Дацюк (К)-Мэйсек "
            "2-я пятёрка: Геноуэй-Зборовский-Секстон-Холланд-Жукенов "
            "3-я пятёрка: Арзамасцев - Хольцер - Белоусов - Куликов - Голышев "
            "4-я пятёрка: Хрипунов -  Протапович- Литовченко; Иванов-Петров",
            "Составы команд: 'Автомобилист': Гросс "
            "1-я пятёрка: Трямкин-Щемеров -  Макеев - Дацюк (К) - Мэйсек "
            "2-я пятёрка: Геноуэй-Зборовский-Секстон-Холланд-Жукенов "
            "3-я пятёрка: Арзамасцев - Хольцер - Белоусов - Куликов - Голышев "
            "4-я пятёрка: Хрипунов -  Протапович - Литовченко; Иванов-Петров",
        ),
    ],
)
def test_fix_dash_word(source_text, expected_text):
    assert fix_dash_word(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("", ""),
        ("Текст", "Текст"),
        ("orgs loc", "orgs loc"),
        ("per loc", "per loc"),
        ("loc org", "loc org"),
        ("'org' loc", "'org' loc"),
        ("org loc", "org"),
        ("org org org loc org", "org org org org"),
        (" loc ", " loc "),
        ("org-loc", "org"),
        ("org   loc", "org"),
        ("orgloc", "org"),
    ],
)
def test_fix_org_loc(source_text, expected_text):
    assert fix_org_loc(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        (
            "КХЛ Кхл КХЛе Кхлу кхловцы кХл КХл вКХЛ "
            "KHL Khl KHLе Khlу khlовцы kHl KHl вKHL",
            "org org org org org org org вКХЛ " "org org org org org org org вKHL",
        ),
        (
            "Ак Барс АК БАРС АкБарсовец Ак  БаРсе Ак бАрСоМ ак Барс вАк Барс "
            "'Ак Барсом' Ak Bars AK BARS AkBarsовец Ak  BaRsе Ak bArSоМ "
            "ak Bars вAk Bars 'Ak Barsом'",
            "org org org org org ак Барс вАк Барс 'org' "
            "org org org org org ak Bars вAk Bars 'org'",
        ),
        (
            "авангард аВАНГАРД Авангард сАвангардом avangard aVANGARD "
            "Avangard сAvangardом с Avangard'ом 'Avangard'ом",
            "авангард аВАНГАРД org сАвангардом avangard aVANGARD "
            "org сAvangardом с org 'org",
        ),
        (
            "ЦСКА цска Цска CSKA cska СКА ска Ска сказали рассказали",
            "org org org org org org org org сказали рассказали",
        ),
        (
            "Динамо Динамо М Динамо мн Динамо может Динамо РИГА ДИНАМО Москва "
            "динамо мск Динамо Минском Dinamo Dynamo M DYNAMO mn Dinamo may "
            "Dinamo RIGA DYNAMO Moscow dinamo msk",
            "org org org org может org org динамо мск org Минском "
            "org org org org may org org dinamo msk",
        ),
        (
            "Металлург Металург мг Металлург Магнитогорск МЕТАЛЛУРГ МГ "
            "МАГНИТОГОРСК Металлург Мг ММГ Металлурги металлург Металлург мгновенно "
            "магнитка Магнитке МАГНИТКУ "
            "Metallurg Metalurg mg Metallurg Magnitogorsk METALLURG MG "
            "MAGNITOGORSK Metallurg Mg MMG Metallurgs metallurg Metallurg mgnovenno "
            "magnitka Magnitke MAGNITKU",
            "org org org org МАГНИТОГОРСК org org org металлург org мгновенно "
            "org org org org org org org MAGNITOGORSK org org org metallurg "
            "org mgnovenno org org org",
        ),
        (
            "Салават Салавате САЛАВАТ ЮЛАЕВ Салавату Юлаеву салават Юлаев "
            "Салават юлаев СЮ СалаватЮлаев сюрприз с САЛАВАТОМ "
            "Salavat Salavat'е SALAVAT ULAEV Salavat'у Ulaev'у salavat Ulaev "
            "Salavat ulaev SU SalavatUlaev surprise s SALAVATOM",
            "org org org org салават Юлаев org org org сюрприз с org "
            "org org org org salavat Ulaev org org org surprise s org",
        ),
        (
            "ХК СКА hc ska HC Neftehimik хк Сочи ПХК ЦСКА хк сочи",
            "org org org org ПХК org хк сочи",
        ),
        (
            "У Сибири с Северсталью так же как у Витязя и Куньлуня",
            "У org с org так же как у org и org",
        ),
        (
            "Торпедо Торпедо НН ТОРПЕДО НИЖНИЙ НОВГОРОД торпедо Нижний "
            "Новгород Торпедо нн "
            "Torpedo Torpedo NN TORPEDO NIZHNIY NOVGOROD torpedo Nizhniy "
            "Novgorod Torpedo nn Torpedo nizhnii novgorod",
            "org org org торпедо Нижний Новгород org "
            "org org org torpedo Nizhniy Novgorod org org",
        ),
        (
            "Кунлунь ред стар КУНЬЛУНЬ КуньлуньРС Кунлунь РЕДСТАР КуньлуньРедСтар "
            "Kunlun red star KUNLUN KunlunRS Kunlun REDSTAR KunlunRedStar",
            "org org org org org org org org org org",
        ),
        (
            "Авангард Avangard Автомобилист Avtomobilist Адмирал Admiral "
            "Ак Барс Ak Bars Амур Amur Барыс Barys Витязь Vityaz "
            "Динамо М Dynamo M Динамо Мн Dinamo Mn Динамо Р Dinamo R "
            "Йокерит Jokerit Куньлунь РС Kunlun RS Локомотив Lokomotiv "
            "Металлург Мг Metallurg Mg Нефтехимик Neftekhimik "
            "Салават Юлаев Salavat Ulaev Северсталь Severstal "
            "Сибирь Sibir СКА SKA Спартак Spartak Торпедо НН Torpedo NN "
            "Трактор Traktor ХК Сочи HC Sochi ЦСКА CSKA",
            " ".join(["org"] * 24 * 2),
        ),
        pytest.param(
            "'Автомобилист' отправился на выезд.",
            "org отправился на выезд.",
            marks=[pytest.mark.xfail(reason="Bug #6 not fixed yet"), pytest.mark.bug_6],
        ),
        (
            "Команда возобновляет матчи KHL'а.",
            "Команда возобновляет матчи org.",
        ),
        pytest.param(
            "'Автомобилисту'предстоит три матча в гостях.",
            "org предстоит три матча в гостях.",
            marks=[pytest.mark.xfail(reason="Bug #6 not fixed yet"), pytest.mark.bug_6],
        ),
    ],
)
def test_replace_concrete_orgs(source_text, expected_text):
    assert replace_concrete_orgs(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text",
    [
        ("Текст", "Текст"),
        ("-Текст - : текст", "-Текст - : текст"),
        (" - ТекстX:  -  ", " - ТекстX"),
        ("Текст. - ", "Текст."),
        ("Текст? -: --", "Текст?"),
    ],
)
def test_delete_ending_colon_dash(source_text, expected_text):
    assert delete_ending_colon_dash(source_text) == expected_text


@pytest.mark.parametrize(
    "source_text,expected_text,replace_ners_,replace_dates_,replace_penalties_",
    [
        (
            "Адмирал - Амур 1:3 (1:0 0:2 0:1)",
            "org org",
            True,
            False,
            False,
        ),
        (
            "21 января Шипачев и Зарипов в Москве забили много голов 'Спартаку', "
            "а Сергей Широков получил 5+20 за 'Грубость'",
            "января Шипачев и Зарипов в Москве забили много голов Спартаку "
            "а Сергей Широков получил за Грубость",
            False,
            False,
            False,
        ),
        (
            "Артем Лукоянов и Дмитрий Воронков забили по голу",
            "per и per забили по голу",
            True,
            False,
            False,
        ),
        (
            "21 января Шипачев и Зарипов в Москве забили много голов 'Спартаку', "
            "а Сергей Широков получил 5+20 за грубость",
            "date per и per в loc забили много голов org "
            "а per получил pen за грубость",
            True,
            True,
            True,
        ),
        (
            "Дмитрий Квартальнов: 'Будет очень жёсткая серия'. Главный тренер "
            "'Ак Барса' Дмитрий Квартальнов подвёл итоги серии с нижегородским "
            "'Торпедо' ( 2:1 ОТ ), оценил игру Тревора Мёрфи и рассказал о "
            "возвращении Даниса Зарипова . - Не дотерпели в третьем периоде, "
            "пропустили гол, играя '4 на 5'. В овертайме забили '5 на 4'. - Как "
            "вам качество игры команды сегодня? - Хорошее. - Пропущенная в "
            "третьем периоде шайба была лишней или всё-таки команда мало забила "
            "в основное время? - Когда счёт такой, 1:1, конечно, мало забиваем. ",
            "per: Будет очень жёсткая серия. Главный тренер org per подвёл итоги "
            "серии с нижегородским org оценил игру per и рассказал о возвращении "
            "per. - Не дотерпели в третьем периоде пропустили гол играя. "
            "В овертайме забили. - Как вам качество игры команды сегодня? - Хорошее. "
            "- Пропущенная в третьем периоде шайба была лишней или всё-таки команда "
            "мало забила в основное время? - Когда счёт такой конечно мало забиваем.",
            True,
            True,
            True,
        ),
        (
            "1 января 2020 года Вадим Шипачев забил 1000-й гол и стал лучшим снайпером "
            "'Динамо Мск' (в новейшей истории). Это случилось в Казани в матче против "
            "Ак Барса, после того как на 25:15 Данис Зарипов получил 4+10 за грубость.",
            "января года Вадим Шипачев забил гол и стал лучшим снайпером "
            "Динамо Мск. Это случилось в Казани в матче против "
            "Ак Барса после того как на Данис Зарипов получил за грубость.",
            False,
            False,
            False,
        ),
        (
            "1 января 2020 года Вадим Шипачев забил 1000-й гол и стал лучшим снайпером "
            "'Динамо Мск' (в новейшей истории). Это случилось в Казани в матче против "
            "Ак Барса, после того как на 25:15 Данис Зарипов получил 4+10 за грубость.",
            "date per забил гол и стал лучшим снайпером "
            "org. Это случилось в loc в матче против "
            "org после того как на per получил pen за грубость.",
            True,
            True,
            True,
        ),
        (
            "- Как сыграли? 2:2.",
            "- Как сыграли?",
            False,
            False,
            False,
        ),
        (
            "- Как сыграли? - 2:2.",
            "- Как сыграли?",
            False,
            False,
            False,
        ),
        (
            "- Сколько выходных дадите команде? - 2.",
            "- Сколько выходных дадите команде?",
            False,
            False,
            False,
        ),
        (
            "Мы побеждали и 3:2, и 4:3, и 1:0",
            "Мы побеждали и и и",
            False,
            False,
            False,
        ),
        pytest.param(
            "Решения СДК по матчу Металлург - Барыс",
            "Решения сдк по матчу org org",
            True,
            True,
            True,
            marks=pytest.mark.bug_4,
        ),
        pytest.param(
            "'Динамо' Рига против 'Динамо' Москва",
            "org против org",
            True,
            True,
            True,
            marks=[pytest.mark.xfail(reason="Bug #6 not fixed yet"), pytest.mark.bug_6],
        ),
    ],
)
def test_simplify_text(
    source_text,
    expected_text,
    replace_ners_,
    replace_dates_,
    replace_penalties_,
):
    assert (
        simplify_text(
            source_text,
            replace_ners_,
            replace_dates_,
            replace_penalties_,
        )
        == expected_text
    )
