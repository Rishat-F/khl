"""
Функции предобработки текста.

Преобразование текста для подачи на вход нейронной модели.
"""

import json
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from natasha import Doc, NewsMorphTagger
from natasha.doc import DocToken

from khl.utils import emb, morph_vocab, segmenter
from khl.wrong_lemmas import fixed_lemmas

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


def _merge_pers(text_list: List[Word]) -> List[Word]:
    """['per', 'per', 'и', 'per', 'per'] -> ['pers', 'и', 'pers']."""
    return _merge(text_list=text_list, source_word="per", target_word="pers")


def _merge_orgs(text_list: List[Word]) -> List[Word]:
    """['org', 'org', 'и', 'org', 'org'] -> ['orgs', 'и', 'orgs']."""
    return _merge(text_list=text_list, source_word="org", target_word="orgs")


def _merge_locs(text_list: List[Word]) -> List[Word]:
    """['loc', 'loc', 'и', 'loc', 'loc'] -> ['locs', 'и', 'locs']."""
    return _merge(text_list=text_list, source_word="loc", target_word="locs")


def _merge_dates(text_list: List[Word]) -> List[Word]:
    """['date', 'date', 'и', 'date', 'date'] -> ['dates', 'и', 'dates']."""
    return _merge(text_list=text_list, source_word="date", target_word="dates")


def _merge_pens(text_list: List[Word]) -> List[Word]:
    """['pen', 'pen', 'и', 'pen', 'pen'] -> ['pens', 'и', 'pens']."""
    return _merge(text_list=text_list, source_word="pen", target_word="pens")


def _merge_ners(text_list: List[Word]) -> List[Word]:
    """# noqa
    ['per', 'per', 'org', 'org', 'loc', 'loc', 'date', 'date', 'pen', 'pen']
    -> ['pers', 'orgs', 'locs', 'dates', 'pens']
    """
    return _merge_pers(_merge_orgs(_merge_locs(_merge_dates(_merge_pens(text_list)))))


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


def _merge_lemmas(lemmas: List[Lemma]) -> List[Lemma]:
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


def lemmatize(text: str, stop_words_: Optional[List[Lemma]]) -> List[Lemma]:
    """
    Разбивка текста на леммы.

    Леммы - начальные формы слов (в нижнем регистре).
    Примеры:
      lemmatize(
        text="1 мая Морозов и Семин забили много голов от борта",
        stop_words_=None,
      ) -> ["1", "май", "морозов", "и", "семин", "забить", "много", "гол", "от", "борт"]
      lemmatize(
        text="1 мая Морозов и Семин забили много голов от борта",
        stop_words_=["и", "много", "от"],
      ) -> ["1", "май", "морозов", "семин", "забить", "гол", "борт"]
    """
    text_tokens = _tokenize(text)
    for token in text_tokens:
        token.lemmatize(morph_vocab)
    if stop_words_ is None:
        text_lemmas: List[Lemma] = [fix_lemma(token.lemma) for token in text_tokens]
    else:
        text_lemmas = [
            fixed_lemma
            for token in text_tokens
            if (fixed_lemma := fix_lemma(token.lemma)) not in stop_words_
        ]
    return _merge_lemmas(_merge_ners(text_lemmas))


def _merge_codes(codes: List[Code]) -> List[Code]:
    """
    Схлопывание одинаковых соседних кодов.

    [10, 10, 200, 200, 200] -> [10, 200].
    """
    return [code for code, _ in groupby(codes)]


def lemmas_to_codes(
    lemmas: List[Lemma],
    lemmas_coder: Dict[Lemma, Code],
    exclude_unknown: bool,
    max_len: Optional[int] = None,
) -> List[Code]:
    """
    Преобразует последовательность лемм в последовательность их кодов.

    exclude_unknown:
      если True, то леммы, которых нет в частотном словаре, отбрасываются;
      если False, то для лемм, которых нет в частотном словаре,
        проставляется код неизвестного слова
    """
    codes = []
    for lemma in lemmas:
        if not exclude_unknown:
            codes.append(lemmas_coder.get(lemma, lemmas_coder[UNKNOWN]))
        elif lemma in lemmas_coder:
            codes.append(lemmas_coder[lemma])
        else:
            continue
    codes = _merge_codes(codes)
    if max_len is None:
        return codes
    elif len(codes) >= max_len:  # pragma: no mutate
        return codes[:max_len]
    else:
        return _fill_placeholders(codes, lemmas_coder, max_len)


def _fill_placeholders(
    codes: List[Code],
    lemmas_coder: Dict[Lemma, Code],
    max_len: int,
) -> List[Code]:
    """Заполняет список кодов символами-заполнителями."""
    filled_codes = [lemmas_coder[PLACEHOLDER]] * (max_len - len(codes))
    filled_codes.extend(codes)
    return filled_codes


def codes_to_lemmas(codes: List[Code], lemmas_coder: Dict[Lemma, Code]) -> List[Lemma]:
    """Преобразует последовательность кодов в последовательность их лемм."""
    reversed_lemmas_coder = {value: key for key, value in lemmas_coder.items()}
    lemmas = []
    for code in codes:
        lemmas.append(reversed_lemmas_coder[code])
    return lemmas


def get_lemmas_coder(frequency_dictionary_file: Union[Path, str]) -> Dict[Lemma, Code]:
    """
    Получение словаря кодового представления лемм из частотного словаря лемм.

    frequency_dictionary_file - частотный словарь лемм:
      json-ка со словарем, где ключи - леммы,
      а значения - сколько раз данная лемма встретилась во всем датасете.
      Желательно, чтобы данный словарь был отсортирован в порядке убывания значений.
    Например:
      {".": 1000, "и": 500, "команда": 200, "гол": 100}

    Возвращает словарь (кодер), в котором каждой лемме присвоен свой уникальный код.
    Первые 2 элемента кодера зарезервированы:
      0 - символ-заполнитель
      1 - неизвестное слово
    """
    lemmas_coder = {PLACEHOLDER: 0, UNKNOWN: 1}
    with open(frequency_dictionary_file, "r", encoding="utf-8") as fr:
        freq_dict = json.load(fr)
    for freq, word in enumerate(freq_dict, len(lemmas_coder)):
        lemmas_coder[word] = freq
    return lemmas_coder
