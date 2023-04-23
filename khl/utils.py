"""
Функции, определенным образом преобразующие хоккейные новости.

Нацелены на приведение текстов новостей из разных источников
к единому виду (унификацию), а также на избавление от элементов
текста новостей, не несущих особой смысловой нагрузки для
машинного обучения (упрощение).
"""


import re
from typing import Callable, List

from natasha import (
    DatesExtractor,
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsNERTagger,
    Segmenter,
)
from natasha.extractors import Match as NatashaMatch
from natasha.span import Span

from khl.teams_orgs import teams_orgs_pattern

segmenter = Segmenter()
morph_vocab = MorphVocab()  # pragma: no mutate
emb = NewsEmbedding()  # pragma: no mutate
ner_tagger = NewsNERTagger(emb)


def unify(text: str) -> str:
    """Приведение текстов новостей к единому виду."""
    for symbol in ['"', "`", "«", "»", "„", "“", "”"]:
        text = text.replace(symbol, "'")
    for symbol in ["—", "–", "−"]:
        text = text.replace(symbol, "-")
    text = (
        text.replace("⅛", "1/8")
        .replace("¼", "1/4")
        .replace("½", "1/2")
        .replace("й", "й")
        .replace("ё", "ё")
        .replace("…", "...")
    )
    text = re.sub(r"[^ А-Яа-яЁёA-Za-z0-9',.\[\]{}()/=+%№#@!?;:-]", " ", text)
    return merge_spaces(text).strip()


def _fix_quotes(text: str) -> str:
    """Исправление дублирующихся ординарных кавычек "''" -> "'"."""
    return re.sub(r"\'{2,}", "'", text)


def _surround_with_quotes(match_object: re.Match) -> str:  # type: ignore
    """Окружение кавычками."""
    word: str = match_object.group(0)
    return "'" + word + "'"


def surround_concrete_orgs_with_quotes(text: str) -> str:
    """
    Оборачивание кавычками прописанных названий лиг и команд.

    Чтобы natasha лучше распознавала ner'ы.
    """
    text = re.sub(teams_orgs_pattern, _surround_with_quotes, text)
    return _fix_quotes(text)


def delete_quotes_around_orgs(text: str) -> str:
    """Удаление кавычек вокруг org."""
    return re.sub(r"\'?org\'?", "org", text)


def fix_bug_5(func: Callable[[str], str]) -> Callable[[str], str]:
    """
    Natasha не всегда правильно распознает названия организаций.

    Например,
    replace_ners("Сегодня в КХЛе пройдет матч") -> "Сегодня в loc пройдет матч".
    Данный декоратор - попытка исправить данное поведение,
    путем оборачивания заданных названий команд/организаций в кавычки
    перед тем как распознавать ner'ы, а потом удаление кавычек вокруг org.
    """

    def wrapper(text: str) -> str:
        """Функция wrapper."""
        text = surround_concrete_orgs_with_quotes(text)
        text = func(text)
        return delete_quotes_around_orgs(text)

    return wrapper


def fix_bug_14(func: Callable[[str], str]) -> Callable[[str], str]:
    """
    Почему-то natasha ошибается на коротких названиях новостей.

    Например, replace_ners("Уральская проверка") -> "org".
    Данный декоратор - попытка исправить данное поведение,
    путем добавления в конец текста восклицательного знака
    перед тем как распознавать ner'ы, а потом удаление
    данного добавленного в конец восклицательного знака.
    """

    def wrapper(text: str) -> str:
        """Функция wrapper."""
        text += "!"
        text = func(text)
        return re.sub(r"\!$", "", text)

    return wrapper


def _find_ners(text: str) -> List[Span]:
    """Нахождение именованных сущностей."""
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    ners: List[Span] = doc.ner.spans
    return ners


@fix_bug_14
@fix_bug_5
def replace_ners(text: str) -> str:
    """
    Заменяет именованные сущности на их тип.

    'Иванов Иван'  -> 'per'
    'Магнитогорск' -> 'loc'
    'Ак Барс'      -> 'org'
    """
    ners_spans = _find_ners(text)
    for ner_span in reversed(ners_spans):
        text = text[: ner_span.start] + ner_span.type.lower() + text[ner_span.stop :]
    text = replace_concrete_orgs(text)
    text = handwritten_replace_orgs(text)
    text = handwritten_replace_per(text)
    return text


def _find_dates(text: str) -> List[NatashaMatch]:
    """Нахождение дат."""
    dates_extractor = DatesExtractor(morph_vocab)
    return [match_ for match_ in dates_extractor(text)]


def replace_dates(text: str) -> str:
    """Заменяет дату на слово 'date'."""
    dates_matches = _find_dates(text)
    dates = [text[date_match.start : date_match.stop] for date_match in dates_matches]
    for date in dates:
        text = text.replace(date, "date", 1)
    return text


def _fix_space_plus_space(text: str) -> str:
    r"""
    Приведение к нормальному виду ' + '.

    '1 + 2'         -> '1+2'
    '1 +2+ 3'       -> '1+2+3'
    '1+ 2 +3'       -> '1+2+3'
    '1 + 2 + 3'     -> '1+2+3'
    '1  +  2  +  3' -> '1+2+3'
    '1\t+\t2'       -> '1+2'
    """
    pattern = r"\s*\+\s*"
    result = re.sub(pattern, "+", text)
    return result


def replace_penalty(text: str) -> str:
    """
    Заменяет штрафы формата 2+10 на слово 'pen'.

    '2+10', '2+20', '4+10', '4+20', '5+10', '5+20' -> 'pen'.
    """
    preprocessed_text = _fix_space_plus_space(text)
    pattern = r"\b(?<!\+)[245]\s*\+\s*(?:10|20)(?!\+)\b"
    result = re.sub(pattern, "pen", preprocessed_text)
    return result


def lowercase_sdk(text: str) -> str:
    """
    "СДК" -> "сдк".

    Нужно для того, того natasha не идентифицировала аббревиатуру "СДК" как
    название организации, и данная аббревиатура не заменялась на "org",
    а сохраняла своё изначальное написание.
    """
    pattern = r"\b[сС][дД][кК]\b"
    return re.sub(pattern, "сдк", text)


def replace_sdk(text: str) -> str:
    """'спортивно-дисциплинарный комитет' -> 'сдк'."""
    pattern = r"спортивн[а-я]+[ -]*дисциплинарн[а-я]+(?:\s*комитет[а-я]*)?"
    result = re.sub(pattern, "сдк", text, flags=re.IGNORECASE)
    return result


def fix_press_conference(text: str) -> str:
    """'Пресс конференция' -> 'Пресс-конференция'."""
    pattern = r"(?<=пресс)[\s-]*(?=конференц)"
    result = re.sub(pattern, "-", text, flags=re.IGNORECASE)
    return result


def delete_numeric_data(text: str) -> str:
    """Удаление данных типа 12:25, 2-10."""
    pattern = r"\b\d+\s*(?:[:-]\s*\d+)+\b(?!-)"
    result = re.sub(pattern, "", text)
    return result


def _replace_dash_with_space(match_object: re.Match) -> str:  # type: ignore
    """Замена тире на пробел."""
    result: str = match_object.group(0).replace("-", " ")
    return result


def replace_dash_between_ners(text: str) -> str:
    """Замена тире между ner'ами на пробел."""
    pattern = r"(?:per|org|loc|date|pen)(?:\s*-+\s*(?:per|org|loc|date|pen))+"
    result = re.sub(pattern, _replace_dash_with_space, text)
    return result


def _delete_with_and(match_object: re.Match) -> str:  # type: ignore
    """Удаление предлога 'с' и союза 'и'."""
    text: str = match_object.group(0)
    pattern = r"\s+(?:с|со|и)\s+"
    return re.sub(pattern, " ", text, flags=re.IGNORECASE)


def fix_ner_with_and_ner(text: str) -> str:
    """Удаление предлога 'с' и союза 'и', которые располагатся между ner'ами."""
    pattern = r"(?:per|org|loc|date)(?:\s+(?i:с|со|и)\s+(?:per|org|loc|date))+"
    return re.sub(pattern, _delete_with_and, text)


def merge_spaces(text: str) -> str:
    """'Схлопывает' все соседние пробельные символы в один пробел."""
    return re.sub(r"\s{2,}", " ", text)


def leave_only_significant_symbols(text: str) -> str:
    """
    Оставление в тексте только имеющих вес символов.

    Производить непосредственно перед разбивкой текста на токены.

    Под "имеющие вес" подразумеваются символы, отсутствие или наличие которых,
    влияет на смысл текста, либо придает определенному участку
    текста какой-то дополнительный контекст. Например, двоеточиями
    выделяют диалоги, а знаки вопроса часто встречаются с упоминанием
    какого-то игрока в интервью со следующим контекстом - какого его
    самочувствие или выйдет ли он на игру в следующем матче.
    Пример:
      Иван Иванов? Он травмирован.
    """
    return re.sub(r"[^ А-Яа-яЁёA-Za-z:.?-]", " ", text)


def replace_exclamation_mark_with_dot(text: str) -> str:
    """
    Замена восклицательного знака в тексте на точку.

    '!?', '?!' оставляем без изменения, так как символ '!' потом
    отфильтруется функцией leave_only_significant_symbols.
    """
    text = re.sub(r"!+", "!", text)
    return re.sub(r"(?<!\?)!(?!\?)", ".", text)


def fix_dots(text: str) -> str:
    """
    Корректирование точек.

    '...' -> '.'
    ' .' -> '.'
    ' . . .' -> '.'
    ' - .' -> '.'
    """
    return re.sub(r"\s*\.*\-*(?:\s*\-*\s*\.)+", ".", text)


def fix_question_marks(text: str) -> str:
    """
    Корректирование вопросительных знаков.

    '???' -> '?'
    ' ?' -> '?'
    ' ? ? ?' -> '?'
    ' - ?' -> '?'
    """
    return re.sub(r"\s*\?*\-*(?:\s*\-*\s*\?)+", "?", text)


def fix_colons(text: str) -> str:
    """Корректирование двоеточий ' :' -> ':'."""
    return re.sub(r"\s+\:", ":", text)


def generalize_top(text: str) -> str:
    """
    Обобщение ТОП'ов.

    'ТОП-3', 'ТОП-5', 'ТОП-10' -> 'ТОП'
    'TOP-3', 'TOP-5', 'TOP-10' -> 'ТОП'
    """
    pattern = r"\b(ТОП|топ|TOP|top)-?\d+\b"
    return re.sub(pattern, "топ", text)


def delete_serial_numbers(text: str) -> str:
    """Удаление порядковых числительных типа '5-й', '2ого', '1.'."""
    pattern = r"\b\d(?:\d*-?)*(?:[а-яёА-ЯЁё]{1,4}\b|\.)"
    return re.sub(pattern, "", text)


def delete_parentheses_content(text: str) -> str:
    """Удаление скобок вместе с их содержимым."""
    pattern = r"\([^()]+\)"
    return re.sub(pattern, "", text)


def delete_overtime_mark(text: str) -> str:
    """Удаление пометки овертайма."""
    pattern = r"(?<=\d)(?:ОТ|от|OT|ot\d?)|(?<=\s)\d?(?:ОТ|OT)\d?\b"
    return re.sub(pattern, "", text)


def delete_play_format(text: str) -> str:
    """
    Удаления формата игры.

    '5 на 5', '5 на 4', '5 на 3', '4 на 5', '3 на 5',
    '4 на 4', '4 на 3', '3 на 4', '3 на 3',
    '5х5', '5 х 4', '5Х3', '4 Х 5',
    'пять на четыре', 'три на пять', 'четыре на четыре' и т.д.
    Функция должна вызываться после функции delete_numeric_data.
    """
    pattern = (
        r"\b(?:(?:3|4|5|6)\s*(?:на|х|x)\s*(?:3|4|5|6)|"
        r"(?:три|четыре|пять|шесть)\s+на\s+(?:три|четыре|пять|шесть))\b"
    )
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


def lowercase_shaiba_word(text: str) -> str:
    """
    'Шайба' -> 'шайба'.

    Natasha распознает слово 'Шайба' как ner.
    """
    return re.sub(r"\bШайба\b", "шайба", text)


def _latin_c_to_cirillic(match_object: re.Match) -> str:  # type: ignore
    """Замена латинского слова 'с' на кириллическое'с'."""
    if match_object.group(0) == "C":
        return "С"
    return "с"


def latin_c_to_cirillic(text: str) -> str:
    """Замена латинского слова 'с' на кириллическое'с'."""
    pattern = r"\b[cC]\b"
    return re.sub(pattern, _latin_c_to_cirillic, text)


def replace_vs_with_dash(text: str) -> str:
    """'vs' -> '-', '- vs - ' -> '-'."""
    pattern = r"\s*-?\s*(?<!\w)[Vv][sS](?!\w)\s*-?\s*"
    return re.sub(pattern, " - ", text)


def delete_urls(text: str) -> str:
    """
    Удаление всех ссылок из текста.

    Взято отсюда:
      https://stackoverflow.com/questions/839994/extracting-a-url-in-python
    """
    pattern = (
        r"\b(?:https?://)?(?:www\.)?(?:[\da-zа-яё\.-]+)\."
        r"(?:[a-zа-яё]{2,6})(?:/[\w\.-?=&]*)*/?\b"
    )
    return re.sub(pattern, "", text)


def handwritten_replace_orgs(text: str) -> str:
    """Костыльная замена названий чего-то в кавычках на org."""
    pattern = r"(?<=')[A-ZА-ЯЁ][a-zA-Zа-яА-ЯёЁ]+(?=')"
    return re.sub(pattern, "org", text)


def replace_concrete_orgs(text: str) -> str:
    """Замена прописанных названий лиг и команд на org."""
    return re.sub(teams_orgs_pattern, "org", text)


def _handwritten_replace_per(match_object: re.Match) -> str:  # type: ignore
    """Замена слова (слов) на per."""
    match_: str = match_object.group(0)
    exceptions = [
        "Тест",
        "Официально",
        "Источник",
        "Изнутри",
        "Превью",
        "Дерби",
        "Супердерби",
    ]
    if match_ not in exceptions:
        return "per"
    else:
        return match_


def handwritten_replace_per(text: str) -> str:
    """Костыльная замена фамилий в начале строки на per."""
    pattern = r"^(?:[А-ЯЁ]\.\s*)?[А-ЯЁ][а-яё]+(?=:| - о)"
    return re.sub(pattern, _handwritten_replace_per, text)


def fix_covid(text: str) -> str:
    """'COVID-19' -> 'covid'."""
    pattern = r"covid[+-]?\d*"
    return re.sub(pattern, "covid", text, flags=re.IGNORECASE)


def _latin_c_to_cirillic_in_word(match_object: re.Match) -> str:  # type: ignore
    """Замена латинской 'C' на кириллическую в слове."""
    word: str = match_object.group(0)
    return word.replace("C", "С").replace("c", "с")


def fix_latin_c_in_russian_words(text: str) -> str:
    """Замена латинской 'C' на кириллическую в русских словах."""
    pattern = r"\b(?:[А-Яа-яёЁ]*[cC][а-яА-ЯёЁcC]+|[А-Яа-яёЁ]+[cC][а-яА-ЯёЁcC]*)\b"
    return re.sub(pattern, _latin_c_to_cirillic_in_word, text)


def _cirillic_c_to_latin_in_word(match_object: re.Match) -> str:  # type: ignore
    """Замена кириллической 'С' на латинскую в слове."""
    word: str = match_object.group(0)
    return word.replace("С", "C").replace("с", "c")


def fix_cirillic_c_in_english_words(text: str) -> str:
    """Замена кириллической 'С' на латинскую в английских словах."""
    pattern = r"\b(?:[a-zA-Z]*[сС][a-zA-ZсС]+|[a-zA-Z]+[сС][a-zA-ZсС]*)\b"
    return re.sub(pattern, _cirillic_c_to_latin_in_word, text)


def _delete_lower_cirillic_letters_from_word(
    match_object: re.Match,  # type: ignore
) -> str:
    """
    Удаление подряд идущих строчных кириллических букв из слова.

    'COVIDом' -> 'COVID'
    'HIFKи'   -> 'HIFK'
    'PSка'    -> 'PS'
    'матч'    -> ''
    """
    word = match_object.group(0)
    return re.sub(r"[а-яё]+", "", word)


def delete_cirillic_ending_from_english_words(text: str) -> str:
    """
    Удаление строчных русских окончаний у заглавных английских слов.

    'COVIDом' -> 'COVID'
    'HIFKи'   -> 'HIFK'
    'PSка'    -> 'PS'
    """
    pattern = r"\b[A-Z]{2,}[а-яё]+\b"
    return re.sub(pattern, _delete_lower_cirillic_letters_from_word, text)


def _leave_only_cirillic(match_object: re.Match) -> str:  # type: ignore
    """Оставление только кириллических символов."""
    word = match_object.group(0)
    return re.sub(r"[^а-яА-ЯёЁ]", "", word)


def fix_english_dash_russian_words(text: str) -> str:
    """
    Оставление только русского слова из связки 'английское-русское'.

    'VIP-гость'      -> 'гость'
    'telegram-канал' -> 'канал'
    'YouTube-видео'  -> 'видео'
    """
    pattern = r"\b[a-zA-Z]+-[а-яА-ЯёЁ]+\b"
    return re.sub(pattern, _leave_only_cirillic, text)


def delete_age_category(text: str) -> str:
    """Удаление возрастной категории типа 'U-18'."""
    pattern = r"\s*-?U\s*-?\s*\d{1,2}"
    return re.sub(pattern, "", text)


def delete_birth_mark(text: str) -> str:
    """
    Удаление года рождения формата.

    'г.р.', '2000 г.р.', '2000/04 г. р.', '2000/2001 гг.р.'
    """
    pattern = r"\d+[/-]?\d*\s*гг?\.?\s*р\."
    return re.sub(pattern, "", text)


def delete_letter_dot_letter_dot(text: str) -> str:
    """Удаление надписей 'P.S.'."""
    pattern = r"\b[a-zA-Zа-яА-ЯёЁ]\.\s*[a-zA-Zа-яА-ЯёЁ]\."
    return re.sub(pattern, "", text)


def delete_shutouts(text: str) -> str:
    """
    Удаление надписей буллитов.

    Таких как 'SO', 'БУЛ', 'Б'.
    """
    pattern = (
        r"(?:\b[Ss][Oo]\b)|(?:\b[Бб][Уу][Лл](?:\.|\b))|"
        r"(?:\b[Бб](?:\.|\b))|(?<=\d)[Бб](?:\.|\b)"
    )
    return re.sub(pattern, "", text)


def delete_amplua(text: str) -> str:
    """
    Удаление сокращений амплуа.

    Таких как 'з.', 'н.', 'вр.'.
    """
    pattern = r"\b(?:[зн]|вр)(?:\s*\.|\b)"
    return re.sub(pattern, "", text)


def replace_tak_kak(text: str) -> str:
    """
    Замена сокращения 'т.к.'.

    'т.к.' -> 'так как'.
    """
    pattern = r"\bт\.?\s*\.?к(?:\s*\.|\b)"
    return re.sub(pattern, "так как", text)


def replace_to_est(text: str) -> str:
    """
    Замена сокращения 'т.е.'.

    'т.е.' -> 'то есть'.
    """
    pattern = r"\bт\.?\s*\.?е(?:\s*\.|\b)"
    return re.sub(pattern, "то есть", text)


def delete_quotes_with_one_symbol(text: str) -> str:
    """Удаление кавычек, содержащий в себе один символ, вместе с содержимым."""
    pattern = r"'\w'"
    return re.sub(pattern, "", text)


def delete_one_symbol_english_words(text: str) -> str:
    """Удаление слов из одной латинской буквы."""
    pattern = r"\b[a-zA-Z]\b"
    return re.sub(pattern, "", text)


def _delete_dash(match_object: re.Match) -> str:  # type: ignore
    """Удаление тире из строки."""
    result: str = match_object.group(0).replace("-", "")
    return result


def delete_beginning_ending_dashes_in_words(text: str) -> str:
    """Удаление тире в начале или в конце слова."""
    pattern = (
        r"(?<![a-zA-Zа-яА-ЯёЁ])-+[a-zA-Zа-яА-ЯёЁ]+-*(?![a-zA-Zа-яА-ЯёЁ])|"
        r"(?<![a-zA-Zа-яА-ЯёЁ])-*[a-zA-Zа-яА-ЯёЁ]+-+(?![a-zA-Zа-яА-ЯёЁ])"
    )
    return re.sub(pattern, _delete_dash, text)


def _split_ners(match_object: re.Match) -> str:  # type: ignore
    """
    Разделение слипшихся ner'ов.

    'perper' -> 'per per'
    'orgper' -> 'org per'
    и т.д.
    """
    string: str = match_object.group(0)
    if re.match(r"per|org|loc|pen", string):
        return string[:3] + " " + string[3:]
    else:
        return string[:4] + " " + string[4:]


def split_ners(text: str) -> str:
    """
    Разделение слипшихся ner'ов.

    'perper' -> 'per per'
    'orgper' -> 'org per'
    и т.д.
    """
    pattern = r"\b(?:per|org|loc|date|pen)(?:per|org|loc|date|pen)"
    return re.sub(pattern, _split_ners, text)


def _delete_spaces(match_object: re.Match) -> str:  # type: ignore
    """Удаление пробелов."""
    string: str = match_object.group(0)
    return string.replace(" ", "")


def fix_b_o_lshii(text: str) -> str:
    """
    В новостях КХЛ много слов 'б о льший'.

    Похоже буква 'о' с ударением так парсится с сайта.
    """
    pattern = r"\b[Бб] о льш"
    return re.sub(pattern, _delete_spaces, text)


def merge_dashes(text: str) -> str:
    """'Схлопывает' все соседние тире в одно."""
    return re.sub(r"(?<=\s)-+(\s*-+)+(?=\s)|-{2,}", "-", text)


def fix_question_dot(text: str) -> str:
    """'?..' -> '?'."""
    return re.sub(r"\?+[.\s]*\.", "?", text)


def fix_dot_question(text: str) -> str:
    """'..?' -> '?'."""
    return re.sub(r"[.\s]*\.\s*\?+", "?", text)


def delete_year_city_mark(text: str) -> str:
    """
    Удаление пометки года/города.

    '2021г.'        -> '2021'
    '2021/2022 гг.' -> '2021/2022 '
    '2018-2020 г.'  -> '2018-2020 '
    'г.Пенза'  -> 'Пенза'
    'г. Минск'  -> ' Минск'
    """
    return re.sub(r"(?<![а-яА-ЯёЁ])[Гг][Гг]?(?:\.|(?![а-яА-ЯёЁ]))", "", text)


def _surround_dash_with_spaces(match_object: re.Match) -> str:  # type: ignore
    """Расставление пробела по обе стороны от тире."""
    string: str = match_object.group(0)
    return merge_spaces(string.replace("-", " - "))


def fix_surname_dash_surname_dash_surname(text: str) -> str:
    """
    'Иванов-Петров-Сидоров' -> 'Иванов - Петров - Сидоров'.

    Natasha так распознает per'ов намного лучше.
    Работает для 3-х и более фамилий. Для 2-х уже будут баги,
    так как под данную регулярку попадают и другие сущности,
    например, команда 'Локо-Юниор' или город 'Нур-Султан'.
    """
    pattern = r"[A-ZА-ЯЁ][a-zа-яё]+(?:\s*-+\s*[A-ZА-ЯЁ][a-zа-яё]+){2,}"
    return re.sub(pattern, _surround_dash_with_spaces, text)


def fix_dash_word(text: str) -> str:
    """
    ' -Иванов' -> ' - Иванов', 'Иванов- ' -> 'Иванов -'.

    Natasha так распознает per'ов намного лучше.
    """
    pattern = r"\s*\B-+[A-ZА-ЯЁ]|[A-ZА-ЯЁ][a-zа-яё]+-+\B\s*"
    return re.sub(pattern, _surround_dash_with_spaces, text).strip()


def fix_org_loc(text: str) -> str:
    """
    'org loc' -> 'org'.

    Например, 'Динамо Рига' преобразуется в 'org loc',
    но нам в данном случае неважен loc, поэтому нужно
    оставлять просто org.
    """
    pattern = r"org[-\s]*loc"
    return re.sub(pattern, "org", text)


def delete_ending_colon_dash(text: str) -> str:
    """'Текст : - ' -> 'Текст'."""
    return text.rstrip(" -:")


def simplify(
    text: str,
    replace_ners_: bool = True,
    replace_dates_: bool = True,
    replace_penalties_: bool = True,
) -> str:
    """
    Упрощение текста хоккейной новости.

    Избавление от элементов текста новости, не несущих
    особой смысловой нагрузки для машинного обучения.

    Пример использования:
      simplify(
        text="1 января 2020 года Иван Иванов в Москве забил гол в ворота Спартака, а также заработал 5+10 за грубость",  # noqa
        replace_ners_=True,
        replace_dates_=True,
        replace_penalties=True,
      ) -> "date per в loc забил гол в ворота org а также заработал pen за грубость"

    Последовательность действий:
      1. Удаляем все что в скобках
      2. Заменяем сокращение 'т.к.'
      3. Заменяем сокращение 'т.е.'
      4. Удаляем слова формата 'И.о.', 'P.S.', 'т.д.'
      5. Исправляем слово 'б о льший'
      6. Удаляем надписи буллитов
      7. Удаляем ОТ
      8. Удаляем сокращения амплуа
      9. Заменяем слово 'Шайба' на 'шайба' (иначе Natasha распознает его как ner)
      10. Заменяем латинское слово 'c' на кириллическое 'с'
      11. Исправляем латинскую 'c' на кириллическую в русских словах
      12. Исправляем кириллическую 'с' на латинскую в английских словах
      13. Заменяем слово 'vs' на ' - '
      14. Удаляем строчные русские окончания у заглавных английских слов
      15. Заменяем COVID-19 на covid
      16. Оставляем только русское слово из связки 'английское-русское'
      17. Удаляем возрастные категории типа U-18
      18. Удаляем годы рождения
      19. Корректирум 'Иванов-Петров-Сидоров' -> 'Иванов - Петров - Сидоров'
      20. Корректируем ' -Иванов' -> ' - Иванов', 'Иванов- ' -> 'Иванов - '
      21. Преобразуем 'СДК' -> 'сдк'
      22. Преобразуем 'спортивно-дисциплинарный комитет' -> 'сдк'
      23. Правим 'пресс конференция' -> 'пресс-конференция'
      24. Обобщаем ТОП
      25. Заменяем ner'ы (если необходимо)
      26. Заменяем даты (если необходимо)
      27. Заменяем удаления (если необходимо)
      28. Удаляем пометки годов
      29. Разделяем слипшиеся ner'ы
      30. Удаляем ссылки
      31. Удаляем кавычки с одним символом внутри
      32. Удаляем английские слова, состоящие только из одной буквы
      33. Удаляем числовые данные
      34. Удаляем порядковые числительные
      35. Удаляем формат игры ('5x5', '3 на 4' и т.д.)
      36. Заменяем восклицательные знаки на точки
      37. Оставляем только нужные символы
      38. 'org loc' -> 'org'
      39. Схлопываем пробелы
      40. Схлопываем тире
      41. Заменяем тире между ner'ами на пробел
      42. Удаление тире в начале или в конце слова
      43. Корректируем точки
      44. Корректируем вопросительные знаки
      45. Корректируем '?..' -> '?'
      46. Корректируем '..?' -> '?'
      47. Удаляем тире и двоеточия в конце текста (rstrip)
      48. Корректируем ' :' -> ':'
    """
    text = delete_parentheses_content(text)
    text = replace_tak_kak(text)
    text = replace_to_est(text)
    text = delete_letter_dot_letter_dot(text)
    text = fix_b_o_lshii(text)
    text = delete_shutouts(text)
    text = delete_overtime_mark(text)
    text = delete_amplua(text)
    text = lowercase_shaiba_word(text)
    text = latin_c_to_cirillic(text)
    text = fix_latin_c_in_russian_words(text)
    text = fix_cirillic_c_in_english_words(text)
    text = replace_vs_with_dash(text)
    text = delete_cirillic_ending_from_english_words(text)
    text = fix_covid(text)
    text = fix_english_dash_russian_words(text)
    text = delete_age_category(text)
    text = delete_birth_mark(text)
    text = fix_surname_dash_surname_dash_surname(text)
    text = fix_dash_word(text)
    text = lowercase_sdk(text)
    text = replace_sdk(text)
    text = fix_press_conference(text)
    text = generalize_top(text)
    if replace_ners_:
        text = replace_ners(text)
    if replace_dates_:
        text = replace_dates(text)
    if replace_penalties_:
        text = replace_penalty(text)
    text = delete_year_city_mark(text)
    text = split_ners(text)
    text = delete_urls(text)
    text = delete_quotes_with_one_symbol(text)
    text = delete_one_symbol_english_words(text)
    text = delete_numeric_data(text)
    text = delete_serial_numbers(text)
    text = delete_play_format(text)
    text = replace_exclamation_mark_with_dot(text)
    text = leave_only_significant_symbols(text)
    text = fix_org_loc(text)
    text = merge_spaces(text)
    text = merge_dashes(text)
    text = replace_dash_between_ners(text)
    text = fix_ner_with_and_ner(text)
    text = delete_beginning_ending_dashes_in_words(text)
    text = fix_dots(text)
    text = fix_question_marks(text)
    text = fix_question_dot(text)
    text = fix_dot_question(text)
    text = delete_ending_colon_dash(text)
    text = fix_colons(text)
    return merge_spaces(text).strip()
