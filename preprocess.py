"""
Функции предобработки текста.

Преобразование текста для подачи на вход нейронной модели.
"""

import json
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Literal, Optional

from natasha import Doc, NewsMorphTagger
from natasha.doc import DocToken

from stop_words import stop_words
from utils import emb, morph_vocab, segmenter, simplify_text
from wrong_lemmas import fixed_lemmas

PLACEHOLDER = ""
UNKNOWN = "???"


morph_tagger = NewsMorphTagger(emb)


Word = str  # pragma: no mutate
Lemma = str  # pragma: no mutate
Code = int  # pragma: no mutate
Ner = Literal["per", "org", "loc", "date", "pen"]  # pragma: no mutate


def _merge(text_list: List[Word], source_word: Ner, target_word: Word) -> List[Word]:
    """
    'Схлопывает' одинаковые соседние заданные слова внутри списка в одно целевое.

    _merge(
        text_list=['1', '1', 'word', 'word', 'word', '1', 'word'],
        source_word='word',
        target_word='words'
    ) -> ['1', '1', 'words', '1', 'word']
    """
    merged_words_text_list = []
    for word, grouper in groupby(text_list):
        group = list(grouper)
        if word == source_word and len(group) > 1:
            merged_words_text_list.append(target_word)
        else:
            merged_words_text_list.extend(group)
    return merged_words_text_list


def merge_pers(text_list: List[Word]) -> List[Word]:
    """['per', 'per', 'и', 'per', 'per'] -> ['pers', 'и', 'pers']."""
    return _merge(text_list=text_list, source_word="per", target_word="pers")


def merge_orgs(text_list: List[Word]) -> List[Word]:
    """['org', 'org', 'и', 'org', 'org'] -> ['orgs', 'и', 'orgs']."""
    return _merge(text_list=text_list, source_word="org", target_word="orgs")


def merge_locs(text_list: List[Word]) -> List[Word]:
    """['loc', 'loc', 'и', 'loc', 'loc'] -> ['locs', 'и', 'locs']."""
    return _merge(text_list=text_list, source_word="loc", target_word="locs")


def merge_dates(text_list: List[Word]) -> List[Word]:
    """['date', 'date', 'и', 'date', 'date'] -> ['dates', 'и', 'dates']."""
    return _merge(text_list=text_list, source_word="date", target_word="dates")


def merge_pens(text_list: List[Word]) -> List[Word]:
    """['pen', 'pen', 'и', 'pen', 'pen'] -> ['pens', 'и', 'pens']."""
    return _merge(text_list=text_list, source_word="pen", target_word="pens")


def merge_ners(text_list: List[Word]) -> List[Word]:
    """# noqa
    ['per', 'per', 'org', 'org', 'loc', 'loc', 'date', 'date', 'pen', 'pen']
    -> ['pers', 'orgs', 'locs', 'dates', 'pens']
    """
    return merge_pers(merge_orgs(merge_locs(merge_dates(merge_pens(text_list)))))


def _tokenize(text: str) -> List[DocToken]:
    """
    Разбивка текста на токены с морфемами.

    'с морфемами' означает, что у каждого токена определено
    к какой части речи токен принадлежит, в каком он роде, числе и падеже.
    Это нужно для дальнейшей лемматизации - приведению токена к начальной форме.
    """
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    tokens: List[DocToken] = doc.tokens
    return tokens


def merge_lemmas(lemmas: List[Lemma]) -> List[Lemma]:
    """
    Схлопывание одинаковых соседних лемм.

    ['и', 'и', 'или', 'или', 'или'] -> ['и', 'или'].
    """
    return [lemma for lemma, _ in groupby(lemmas)]


def fix_lemma(lemma: Lemma) -> Lemma:
    """
    Исправление неправильных лемм.

    'забиваем' -> 'забивать'
    'основный' -> 'основной'
    'родный'   -> 'родной'
    'голы'     -> 'гол'
    """
    return fixed_lemmas.get(lemma, lemma)


def is_stop_word(lemma: Lemma) -> bool:
    """Является ли лемма стоп-словом."""
    return lemma in stop_words


def lemmatize(
    text: str,
    replace_ners_: bool,
    replace_dates_: bool,
    replace_penalties_: bool,
    exclude_stop_words: bool,
) -> List[Lemma]:
    """
    Разбивка текста на леммы.

    Леммы - начальные формы слов (в нижнем регистре).
    Примеры:
      lemmatize(
        text="1 мая Морозов и Семин забили много голов от борта",
        replace_ners_=False,
        replace_dates_=False,
        replace_penalties_=False,
        exclude_stop_words=False,
      ) -> ["май", "морозов", "и", "семин", "забить", "много", "гол", "от", "борт"]
      lemmatize(
        text="1 января мне пришло много писем от Ивана и Маши",
        replace_ners_=True,
        replace_dates_=True,
        replace_penalties_=True,
        exclude_stop_words=True,
      ) -> ["date", "pers", "забить", "гол", "борт"]
    """
    text = simplify_text(text, replace_ners_, replace_dates_, replace_penalties_)
    text_tokens = _tokenize(text)
    for token in text_tokens:
        token.lemmatize(morph_vocab)
    if not exclude_stop_words:
        text_lemmas: List[Lemma] = [fix_lemma(token.lemma) for token in text_tokens]
    else:
        text_lemmas = [
            fix_lemma(token.lemma)
            for token in text_tokens
            if not is_stop_word(token.lemma)
        ]
    return merge_lemmas(merge_ners(text_lemmas))


def merge_codes(codes: List[Code]) -> List[Code]:
    """
    Схлопывание одинаковых соседних кодов.

    [10, 10, 200, 200, 200] -> [10, 200].
    """
    return [code for code, _ in groupby(codes)]


def lemmas_to_codes(
    lemmas: List[Lemma],
    freq_dict: Dict[Lemma, Code],
    max_len: Optional[int] = None,
) -> List[Code]:
    """Преобразует последовательность лемм в последовательность их кодов."""
    codes = []
    for lemma in lemmas:
        codes.append(freq_dict.get(lemma, freq_dict[UNKNOWN]))
    codes = merge_codes(codes)
    if max_len is None:
        return codes
    elif len(codes) >= max_len:  # pragma: no mutate
        return codes[:max_len]
    else:
        return _fill_placeholders(codes, freq_dict, max_len)


def _fill_placeholders(
    codes: List[Code],
    freq_dict: Dict[Lemma, Code],
    max_len: int,
) -> List[Code]:
    """Заполняет список кодов символами-заполнителями."""
    filled_codes = [freq_dict[PLACEHOLDER]] * (max_len - len(codes))
    filled_codes.extend(codes)
    return filled_codes


def codes_to_lemmas(codes: List[Code], freq_dict: Dict[Lemma, Code]) -> List[Lemma]:
    """Преобразует последовательность кодов в последовательность их лемм."""
    reversed_freq_dict = {value: key for key, value in freq_dict.items()}
    lemmas = []
    for code in codes:
        lemmas.append(reversed_freq_dict[code])
    return lemmas


def get_freq_dict(lemmas_dictionary_file: Path) -> Dict[Lemma, Code]:
    """
    Преобразует словарь с леммами в частотный словарь лемм.

    lemmas_dictionary_file - json-ка со словарем, где ключи - леммы,
    а значения - сколько раз данная лемма встретилась во всем датасете,
    и словарь отсортирован в порядке убывания значений.
    Например:
      {".": 1000, "и": 500, "команда": 200, "гол": 100}

    Первые 2 элемента частотного словаря зарезервированы:
      0 - символ-заполнитель
      1 - неизвестное слово
    """
    freq_dict = {PLACEHOLDER: 0, UNKNOWN: 1}
    with open(lemmas_dictionary_file, "r", encoding="utf-8") as fr:
        lemmas_dict = json.load(fr)
    for freq, word in enumerate(lemmas_dict, len(freq_dict)):
        freq_dict[word] = freq
    return freq_dict


def text_to_codes(
    text: str,
    replace_ners_: bool,
    replace_dates_: bool,
    replace_penalties_: bool,
    exclude_stop_words: bool,
    freq_dict: Dict[Lemma, Code],
    max_len: Optional[int] = None,
) -> List[Code]:
    """
    Преобразует текст в последовательность кодов.

    args:
      text: текст новости
      replace_ners_: если True, то в тексте имена людей заменяются на
        слово 'per', названия команд заменяются на слово 'org',
        названия городов заменяются на слово 'loc'
      replace_dates_: если True, то в тексте даты заменяются на слово 'date'
      replace_penalties_: если True, то в тексте удаления вида '2+10'
        заменяются на слово 'pen'
      exclude_stop_words: если True, то стоп-слова исключаются
      freq_dict: частотный словарь, на основе которого проставляются коды
      max_len: длина последовательности на выходе
    """
    lemmas = lemmatize(
        text,
        replace_ners_,
        replace_dates_,
        replace_penalties_,
        exclude_stop_words,
    )
    codes = lemmas_to_codes(lemmas, freq_dict, max_len)
    return codes
