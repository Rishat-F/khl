![Khl Logo](https://raw.githubusercontent.com/Rishat-F/khl/master/data/logo.png)

<h1 align="center">No Water - Ice Only</h1>

Preparing russian hockey news for machine learning.

**Unify -> Simplify -> Preprocess** text and feed your neural model.

## Installation

*Khl* is available on PyPI:

```console
$ pip install khl
```
It requires Python 3.8+ to run.

## Usage

To get started right away with basic usage:

```python
from khl import text_to_codes

lemmas_coder = {
    '': 0,     # placeholder
    '???': 1,  # unknown
    '.': 2,
    'и': 3,
    'в': 4,
    '-': 5,
    ':': 6,
    'матч': 7,
    'за': 8,
    'забить': 9,
    'гол': 10,
    'per': 11,   # person entity
    'org': 12,   # organization entity
    'loc': 13,   # location entity
    'date': 14,  # date entity
    'против': 15,
    'год': 16,
    'pers': 17,  # few persons entity
    'orgs': 18,  # few organizations entity
    'свой': 19
}

text = """
    1 апреля 2023 года в Москве в матче ⅛ финала против „Спартака” Иван Иванов забил свой 100—й гол за карьеру.
    «Динамо Мск» - «Спартак» 2:1 ОТ (1:0 0:1 0:0 1:0) Голы забили: Иванов, Петров и Сидоров.
"""

codes = text_to_codes(
    text=text,
    lemmas_coder=lemmas_coder,
    stop_words_=["в", "за", "и", "свой"],  # stop words to drop
    replace_ners_=True,                    # replace named entities ("Иван Иванов" -> "per", "Спартак" -> "org", "Москва" -> "loc")
    replace_dates_=True,                   # replace dates ("1 апреля 2023 года" -> "date")
    replace_penalties_=True,               # replace penalties ("5+20" -> "pen")
    exclude_unknown=True,                  # drop lemma that not presented in lemmas_coder
    max_len=20,                            # get sequence of codes of length 20
)
# codes = [0, 0, 0, 0, 0, 14, 13, 7, 15, 12, 11, 9, 10, 2, 18, 10, 9, 6, 17, 2]
```

```text_to_codes``` is a very high level function. What's happens under hood see in [Lower level usage](#lower-level-usage).

## What is `lemmas_coder`?
`lemmas_coder` is just a dictionary where each lemma is represented with unique integer code.
Note that first two elements are reserved for *placeholder* and *unknown* elements.

It is possible to get `lemmas_coder` from frequency dictionary file (see in [Get lemmas coder](#2-get-lemmas-coder)).
Frequency dictionary file is a **json**-file with dictionary where key is lemma and value is how many times this lemma occurred in your whole dataset.
Preferably it should be sorted in descending order of values.  
`example_frequency_dictionary.json`:

```json
{
  ".": 1000,
  "и": 500,
  "в": 400,
  "-": 300,
  ":": 300,
  "матч": 290,
  "за": 250,
  "забить": 240,
  "гол": 230,
  "per": 200,
  "org": 150,
  "loc": 150,
  "date": 100,
  "против": 90,
  "год": 70,
  "pers": 40,
  "orgs": 30,
  "свой": 20
}
```

You could make and use your own frequency dictionary or download [this dictionary](https://github.com/Rishat-F/khl/blob/master/data/frequency_dictionary.json) created by myself.

## Lower level usage<a id="lower-level-usage"></a>

#### 1. Make imports
```python
from khl import stop_words
from khl import utils
from khl import preprocess
```

#### 2. Get lemmas coder<a id="2-get-lemmas-coder"></a>
```python
lemmas_coder = preprocess.get_lemmas_coder("example_frequency_dictionary.json")
```

#### 3. Define text
```python
text = """
    1 апреля 2023 года в Москве в матче ⅛ финала против „Спартака” Иван Иванов забил свой 100—й гол за карьеру.
    «Динамо Мск» - «Спартак» 2:1 ОТ (1:0 0:1 0:0 1:0) Голы забили: Иванов, Петров и Сидоров.
"""
```

#### 4. Unify
```python
unified_text = utils.unify_text(text)
# "1 апреля 2023 года в Москве в матче 1/8 финала против 'Спартака' Иван Иванов забил свой 100-й гол за карьеру. 'Динамо Мск' - 'Спартак' 2:1 ОТ (1:0 0:1 0:0 1:0) Голы забили: Иванов, Петров и Сидоров."
```

#### 5. Simplify
```python
simplified_text = utils.simplify_text(
    text=unified_text,
    replace_ners_=True,
    replace_dates_=True,
    replace_penalties_=True,
)
# 'date в loc в матче финала против org per забил свой гол за карьеру. org org Голы забили: per per и per.'
```

#### 6. Lemmatize
```python
lemmas = preprocess.lemmatize(text=simplified_text, stop_words_=stop_words)
# ['date', 'loc', 'матч', 'финал', 'против', 'org', 'per', 'забить', 'гол', 'карьера', '.', 'orgs', 'гол', 'забить', ':', 'pers', '.']
```

#### 7. Transform to codes
```python
codes = preprocess.lemmas_to_codes(
    lemmas=lemmas,
    lemmas_coder=lemmas_coder,
    exclude_unknown=True,
    max_len=20,
)
# [0, 0, 0, 0, 0, 14, 13, 7, 15, 12, 11, 9, 10, 2, 18, 10, 9, 6, 17, 2]
```

#### 8. Transform to lemmas back (just to look which lemmas are presented in codes sequence)
```python
print(
    preprocess.codes_to_lemmas(codes=codes, lemmas_coder=lemmas_coder)
)
# ['', '', '', '', '', 'date', 'loc', 'матч', 'против', 'org', 'per', 'забить', 'гол', '.', 'orgs', 'гол', 'забить', ':', 'pers', '.']
```
