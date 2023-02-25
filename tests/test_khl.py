"""
Тесты пакета.

Тесты метаданных, интеграционные и e2e тесты.
"""

from pathlib import Path

import pytest
import tomli

import khl
from khl import text_to_codes

tests_dir = Path(__file__).parent
project_dir = tests_dir.parent
test_frequency_dictionary_file = "example_frequency_dictionary.json"


class TestMetadata:
    PACKAGE_NAME = "khl"
    PACKAGE_DESCRIPTION = "Preparing russian hockey news for machine learning"
    PACKAGE_README_FILE = "README.md"
    PACKAGE_LICENSE = "MIT"
    PACKAGE_LICENSE_FILE = "LICENSE"
    PACKAGE_AUTHORS = ["Rishat Fayzullin <nilluziaf@gmail.com>"]
    PACKAGE_REPOSITORY = "https://github.com/Rishat-F/khl"
    PACKAGE_DEPS = {"python": "^3.8", "natasha": "==1.4.0"}
    with open(project_dir / "pyproject.toml", "rb") as frb:
        PROJECT_TOML = tomli.load(frb)

    def test_version_consistency(self):
        assert khl.__version__ == self.PROJECT_TOML["tool"]["poetry"]["version"]

    def test_package_name(self):
        assert self.PROJECT_TOML["tool"]["poetry"]["name"] == self.PACKAGE_NAME

    def test_package_description(self):
        assert (
            self.PROJECT_TOML["tool"]["poetry"]["description"]
            == self.PACKAGE_DESCRIPTION
        )

    def test_package_readme(self):
        assert self.PROJECT_TOML["tool"]["poetry"]["readme"] == self.PACKAGE_README_FILE
        assert (project_dir / self.PACKAGE_README_FILE).exists()

    def test_package_license(self):
        assert self.PROJECT_TOML["tool"]["poetry"]["license"] == self.PACKAGE_LICENSE
        assert (project_dir / self.PACKAGE_LICENSE_FILE).exists()

    def test_package_authors(self):
        assert self.PROJECT_TOML["tool"]["poetry"]["authors"] == self.PACKAGE_AUTHORS

    def test_package_repository(self):
        assert (
            self.PROJECT_TOML["tool"]["poetry"]["repository"] == self.PACKAGE_REPOSITORY
        )

    def test_package_dependencies(self):
        assert self.PROJECT_TOML["tool"]["poetry"]["dependencies"] == self.PACKAGE_DEPS


@pytest.mark.parametrize(
    "source_text,expected_lemmas",
    [
        ("Мы побеждали и 3:2, и 4:3, и 1:0", ["мы", "побеждать", "и"]),
        (
            "Я забивал, забиваю и буду забивать",
            ["я", "забивать", "и", "быть", "забивать"],
        ),
        (
            "Артем Лукоянов и Дмитрий Воронков забили по голу",
            ["per", "и", "per", "забить", "по", "гол"],
        ),
        ("Гол!!!", ["гол", "."]),
        ("Иван Петров12 сентября сыграет", ["per", "date", "сыграть"]),
        ("Правда или ...?", ["правда", "или", "?"]),
        pytest.param(
            "'Динамо' Рига против 'Динамо' Москва",
            ["org", "против", "org"],
            marks=[pytest.mark.xfail(reason="Bug #6 not fixed yet"), pytest.mark.bug_6],
        ),
        ("Матч 'Спартак'Москва - 'ЦСКА'-Москва", ["матч", "orgs"]),
        (
            "Сегодня Динамо Минск отправилось на выезд.",
            ["сегодня", "org", "отправиться", "на", "выезд", "."],
        ),
        (
            "В формате '5 х 5' было тяжело.",
            ["в", "формат", "быть", "тяжелый", "."],
        ),
        ("-Команда проиграла...", ["-", "команда", "проиграть", "."]),
        (
            "Да.-Мы довольны результатом",
            ["да", ".", "-", "мы", "довольный", "результат"],
        ),
        ("Спасибо, Иван-! Удачи!", ["спасибо", "per", ".", "удача", "."]),
        (
            "Составы команд: 'Автомобилист': Гросс "
            "1-я пятёрка: Трямкин-Щемеров; Макеев -Дацюк (К)-Мэйсек "
            "2-я пятёрка: Геноуэй -Зборовский; Секстон - Холланд (А)-Жукенов "
            "3-я пятёрка: Арзамасцев-Хольцер; Белоусов-Куликов-Голышев "
            "4-я пятёрка: Хрипунов -  Протапович- Литовченко",
            [
                "состав",
                "команда",
                ":",
                "org",
                ":",
                "per",
                "пятерка",
                ":",
                "pers",
                "пятерка",
                ":",
                "pers",
                "пятерка",
                ":",
                "pers",
                "пятерка",
                ":",
                "pers",
            ],
        ),
        (
            "Составы команд: 'Автомобилист': Гросс "
            "1-я пятёрка: Белоусов-Куликов-Голышев "
            "2-я пятёрка: Геноуэй-Зборовский-Секстон-Холланд-Жукенов "
            "3-я пятёрка: Хрипунов -  Протапович- Литовченко "
            "4-я пятёрка: Иванов-Петров",
            [
                "состав",
                "команда",
                ":",
                "org",
                ":",
                "per",
                "пятерка",
                ":",
                "pers",
                "пятерка",
                ":",
                "pers",
                "пятерка",
                ":",
                "pers",
                "пятерка",
                ":",
                "per",
            ],
        ),
        (
            "Команда г.Казани забила красивый гол команде г. Уфы.",
            [
                "команда",
                "loc",
                "забить",
                "красивый",
                "гол",
                "команда",
                "loc",
                ".",
            ],
        ),
        (
            "Бюджет по сравнению с сезоном 2019-2020 гг. был сокращен.",
            [
                "бюджет",
                "по",
                "сравнение",
                "с",
                "сезон",
                "date",
                ".",
                "быть",
                "сократить",
                ".",
            ],
        ),
        (
            "Играл за 'Металлург' с 2015 по 2016 гг. Потом уехал за океан.",
            [
                "играть",
                "за",
                "org",
                "с",
                "по",
                "date",
                ".",
                "потом",
                "уехать",
                "за",
                "океан",
                ".",
            ],
        ),
        (
            "Если мы не забьем, а я не на своей позиции?..",
            [
                "если",
                "мы",
                "не",
                "забить",
                "а",
                "я",
                "не",
                "на",
                "свой",
                "позиция",
                "?",
            ],
        ),
        (
            "А что дальше?.",
            [
                "а",
                "что",
                "далекий",
                "?",
            ],
        ),
        (
            "Счет серии -- -- 2:2.",
            [
                "счет",
                "серия",
                ".",
            ],
        ),
        (
            "Но почему?? ? - ?",
            [
                "но",
                "почему",
                "?",
            ],
        ),
        (
            "Счет серии - 2:2?",
            [
                "счет",
                "серия",
                "?",
            ],
        ),
        (
            "Иван Петров -- лучший бомбардир команды.",
            [
                "per",
                "-",
                "хороший",
                "бомбардир",
                "команда",
                ".",
            ],
        ),
        (
            "Счет серии -- -- 2-2 после двух матчей.",
            [
                "счет",
                "серия",
                "-",
                "после",
                "два",
                "матч",
                ".",
            ],
        ),
        (
            "Б о льшую часть времени команда провела в атаке.",
            [
                "больший",
                "часть",
                "время",
                "команда",
                "провести",
                "в",
                "атака",
                ".",
            ],
        ),
        pytest.param(
            "'Автомобилист' отправился на выезд. После перерыва на игры "
            "сборной наша команда возобновляет чемпионат с выездной серии. "
            "'Автомобилисту'предстоит три матча в гостях.",
            [
                "org",
                "отправиться",
                "на",
                "выезд",
                ".",
                "после",
                "перерыв",
                "на",
                "игра",
                "сборная",
                "наш",
                "команда",
                "возобновлять",
                "чемпионат",
                "с",
                "выездной",
                "серия",
                ".",
                "org",
                "предстоять",
                "три",
                "матч",
                "в",
                "гость",
                ".",
            ],
            marks=[pytest.mark.xfail(reason="Bug #6 not fixed yet"), pytest.mark.bug_6],
        ),
        (
            "В следующем матче регулярного чемпионата КХЛ 'Сочи'12 ноября "
            "в Риге встретится с местным 'Динамо'.",
            [
                "в",
                "следующий",
                "матч",
                "регулярный",
                "чемпионат",
                "orgs",
                "date",
                "в",
                "loc",
                "встретиться",
                "с",
                "местный",
                "org",
                ".",
            ],
        ),
        (
            "'Локо-Юниор': Д.Назаров - Лысых - Козлов;Казаков - Грибков",
            ["org", ":", "pers"],
        ),
        (
            "Протокол матча: СКА - 'Динамо'Москва - 3:4 (2:2, 1:1, 0:1).",
            ["протокол", "матч", ":", "orgs", "."],
        ),
        (
            "В последних двух домашних матчах 'Варяги' взяли один балл, "
            "дважды уступив белорусскому 'Динамо-Шинник' 1. - 2:3ОТ, 2. - 0:1.",
            [
                "в",
                "последний",
                "два",
                "домашний",
                "матч",
                "org",
                "взять",
                "один",
                "балл",
                "дважды",
                "уступить",
                "белорусский",
                "org",
                ".",
            ],
        ),
        ("Голы забили: Иванов,Петров,Сидоров", ["гол", "забить", ":", "pers"]),
        (
            "Игрок Клубы СИ И 1 Коварж Якуб Автомобилист 7 50 2 Сорокин Илья "
            "ЦСКА 6 17 3 Хелльберг Магнус СКА 6 28 4 Паскуале "
            "Эдвард Барыс,Локомотив 6 46 5",
            [
                "игрок",
                "клуб",
                "си",
                "и",
                "per",
                "org",
                "per",
                "org",
                "per",
                "org",
                "per",
                "orgs",
            ],
        ),
        (
            "Группа А : 1. СДЮШОР БФСО 'Динамо'-2 (6 очков); 2. 'Торпедо'-2 "
            "(3 очка); 3. СДЮШОР ХК 'Юность-Минск'-2 (3 очка); 4. СДЮШОР "
            "ХК 'Юность-Минск'-3 (0 очков).",
            ["группа", "а", ":", "orgs", "."],
        ),
        ("Вадим Шипачев/Андрей Миронов/Динамо М/", ["pers", "org"]),
        (
            "Для воспитанников СДЮШОР им.Руслана Салея жребий определил соперника",
            ["для", "воспитанник", "org", "per", "жребий", "определить", "соперник"],
        ),
        (
            "'Ак Барс' выиграл 16 из 18 матчей КХЛ в Астане/Нур-Султане.",
            ["org", "выиграть", "из", "матч", "org", "в", "locs", "."],
        ),
        (
            "Список травмированных: Покка Вилле Стась Андрей Шарипзянов Дамир "
            "Якупов Наиль 'Волки' завершили выездную серию",
            [
                "список",
                "травмировать",
                ":",
                "pers",
                "org",
                "завершить",
                "выездной",
                "серия",
            ],
        ),
        (
            "Да-да, вы все- -правильно -говорите-",
            ["да", "вы", "весь", "правильно", "говорить"],
        ),
        (
            "Сегодня 20-летний экс-капитан команды 'Авангард-20' "
            "отмечает свой день рождения!",
            [
                "сегодня",
                "летний",
                "экс-капитан",
                "команда",
                "org",
                "отмечать",
                "свой",
                "день",
                "рождение",
                ".",
            ],
        ),
        (
            "-Команды 'Спартак-20' в сезоне-2020/2021 не было - "
            "в 2021-2022 году она появилась",
            [
                "-",
                "команда",
                "org",
                "в",
                "сезон",
                "не",
                "быть",
                "-",
                "в",
                "date",
                "она",
                "появиться",
            ],
        ),
        (
            "Сектора A, B, C доступны",
            ["сектор", "с", "доступный"],  # так как есть latin_c_to_cirillic
        ),
        ("Спонсор матча g-drive", ["спонсор", "матч", "drive"]),
        ("План 'Б' сработал", ["план", "сработать"]),
        (
            "Все билеты на сектора 'B' и C раскуплены.",
            ["весь", "билет", "на", "сектор", "и", "с", "раскупить", "."],
        ),
        ("Мы сыграли на '5-ку'.", ["мы", "сыграть", "на", "."]),
        (
            "Школа в 60-70-х действительно была отменная.",
            ["школа", "в", "действительно", "быть", "отменный", "."],
        ),
        ("Победили, т.к. забили", ["победить", "так", "как", "забить"]),
        ("Повезло, т.е. заслужили", ["повезти", "тот", "быть", "заслужить"]),
        (
            "Рыбки, т.е. симуляции, будут наказываться.",
            ["нырок", "тот", "быть", "симуляция", "быть", "наказываться", "."],
        ),
        (
            "Состав команды: вр. Сорокин, з. Петров, з. Сидоров, н. Иванов",
            ["состав", "команда", ":", "pers"],
        ),
        (
            "Состав на сегодняшний матч: вр Сорокин, з Петров, з Сидоров, н Иванов",
            ["состав", "на", "сегодняшний", "матч", ":", "pers"],
        ),
        (
            "Нападающий 2000 г.р. уехал играть заграницу",
            ["нападающий", "уехать", "играть", "заграница"],
        ),
        ("Спартак победил со счетом 1:1 Б.", ["org", "победить", "с", "счет"]),
        ("3:2Б. (1:1 0:0 1:1 1:0) статистика матча", ["статистика", "матч"]),
        ("3:2 БУЛ. счет матча", ["счет", "матч"]),
        ("Идет набор детей 2000/02 гг.р.", ["идти", "набор", "ребенок"]),
        (
            "Играл за сборные России U17 и U-18 в молодости",
            ["играть", "за", "сборный", "loc", "и", "в", "молодость"],
        ),
        ("'Адмирал-U17' победил в турнире.", ["org", "победить", "в", "турнир", "."]),
        ("'Адмирал U-17' победил в турнире.", ["org", "победить", "в", "турнир", "."]),
        (
            "Вчера 'Адмирал U17' выехал на матч.",
            ["вчера", "org", "выехать", "на", "матч", "."],
        ),
        ("Купил билеты в VIP-ложу.", ["купить", "билет", "в", "ложа", "."]),
        ("Смотрите новости в telegram-канале", ["смотреть", "новость", "в", "канал"]),
        pytest.param(
            "Решения СДК по матчу",
            ["решение", "сдк", "по", "матч"],
            marks=pytest.mark.bug_4,
        ),
        pytest.param(
            "Сегодня в КХЛе пройдет 5 матчей",
            ["сегодня", "в", "org", "пройти", "матч"],
            marks=pytest.mark.bug_5,
        ),
        (
            "Под контролем ЦИБа будут все команды",
            ["под", "контроль", "циб", "быть", "весь", "команда"],
        ),
        ("УАЗик на ходу.", ["уазик", "на", "ход", "."]),
        (
            "Сотрудники ОМОНа присутствуют на стадионе",
            ["сотрудник", "org", "присутствовать", "на", "стадион"],
        ),
        ("'Спартак' проиграл шведскому АИКу", ["org", "проиграть", "шведский", "loc"]),
        ("Во всех ВУЗах страны", ["в", "весь", "вуз", "страна"]),
        (
            "Результаты матчей смотрите во 'ВКонтакте'",
            ["результат", "матч", "смотреть", "в", "org"],
        ),
        (
            "Результаты матчей смотрите во ВКонтакте",
            ["результат", "матч", "смотреть", "в", "org"],
        ),
        (
            "Вчера игроки HIFKа заболели COVIDом",
            ["вчера", "игрок", "org", "заболеть", "covid"],
        ),
        ("Сегодня в KHLе пройдет 5 матчей", ["сегодня", "в", "org", "пройти", "матч"]),
        ("Проехался на TANECOмобиле.", ["проехаться", "на", "taneco", "."]),
        ("Мы давно c ним дружим", ["мы", "давно", "с", "он", "дружить"]),
        ("C Новым Годом!", ["с", "новый", "год", "."]),
        ("Я отдал паc нападающему.", ["я", "отдать", "пас", "нападающий", "."]),
        ("Мне пришло много пиcем!", ["я", "прийти", "много", "письмо", "."]),
        ("Посещение кафе, кино и т.д.?", ["посещение", "кафе", "кино", "и", "?"]),
        ("И.о. главного тренера", ["главный", "тренер"]),
        ("P.S. На матче ожидается концерт", ["на", "матч", "ожидаться", "концерт"]),
        (
            "Все билеты на стартовый матч 'Витязя' с 'Северсталью' на 13 "
            "сентября раскуплены. P.S. Аншлаг!",
            [
                "весь",
                "билет",
                "на",
                "стартовый",
                "матч",
                "org",
                "с",
                "org",
                "на",
                "date",
                "раскупить",
                ".",
                "аншлаг",
                ".",
            ],
        ),
        (
            "Cегодня c нами сыграла группа, cпевшая песню We Are The Сhampions, "
            "с шансом обыграть наc. See matсh results in doс.",
            [
                "сегодня",
                "с",
                "мы",
                "сыграть",
                "группа",
                "спеть",
                "песня",
                "we",
                "are",
                "the",
                "champions",
                "с",
                "шанс",
                "обыграть",
                "мы",
                ".",
                "see",
                "match",
                "results",
                "in",
                "doc",
                ".",
            ],
        ),
        ("Да кто вам такое сказал?!!", ["да", "кто", "вы", "такой", "сказать", "?"]),
        ("Да кто вам такое сказал!!?", ["да", "кто", "вы", "такой", "сказать", "?"]),
        ("Кто знал...", ["кто", "знать", "."]),
        ("Cтатистика. (22-11).", ["статистика", "."]),
        (
            "Игроки сдали теcт на COVID-19 по приезду в Хельсинки.",
            ["игрок", "сдать", "тест", "на", "covid", "по", "приезд", "в", "loc", "."],
        ),
        (
            "Покупайте билеты на cайте билеты.ру. Cпешите их осталось немного!",
            [
                "покупать",
                "билет",
                "на",
                "сайт",
                ".",
                "спешить",
                "они",
                "остаться",
                "немного",
                ".",
            ],
        ),
        (
            "C матча 'Амур' VS 'Адмирал' начнется сегодняшний игровой день!",
            ["с", "матч", "orgs", "начаться", "сегодняшний", "игровой", "день", "."],
        ),
        (
            "C матча 'Амур'-'Адмирал' начнется сегодняшний игровой день!",
            ["с", "матч", "orgs", "начаться", "сегодняшний", "игровой", "день", "."],
        ),
        (
            "C матча 'Амур' VS 'Барыс' начнется сегодняшний игровой день!",
            ["с", "матч", "orgs", "начаться", "сегодняшний", "игровой", "день", "."],
        ),
        (
            "Андрей Педан прокомментировал столкновение с Иваном Дроздовым. "
            "- Иван сам пошел в стык.",
            [
                "per",
                "прокомментировать",
                "столкновение",
                "с",
                "per",
                ".",
                "-",
                "per",
                "сам",
                "пойти",
                "в",
                "стык",
                ".",
            ],
        ),
        (
            "Андрей Разин. - Паша смял нас в первых сменах",
            ["per", ".", "-", "per", "смять", "мы", "в", "первый", "смена"],
        ),
        (
            "Андрей Разин: - Паша смял нас в первых сменах",
            ["per", ":", "-", "per", "смять", "мы", "в", "первый", "смена"],
        ),
        (
            "Андрей Разин: 'Паша смял нас в первых сменах'",
            ["per", ":", "per", "смять", "мы", "в", "первый", "смена"],
        ),
        (
            "ТОП-10 голов прошлой игровой недели",
            ["топ", "гол", "прошлый", "игровой", "неделя"],
        ),
        (
            "Играл за родную команду.",
            ["играть", "за", "родной", "команда", "."],
        ),
        (
            "Нападающий СКА Андрей     Кузьменко подвёл итоги "
            "встречи c минским Динамо (2:3 ОТ).",
            [
                "нападающий",
                "org",
                "per",
                "подвести",
                "итог",
                "встреча",
                "с",
                "минский",
                "org",
                ".",
            ],
        ),
        (
            " - Хорошее. - Шайба, пропущенная в третьем периоде, была "
            "лишней или всё-таки команда мало забила?",
            [
                "-",
                "хороший",
                ".",
                "-",
                "шайба",
                "пропустить",
                "в",
                "третий",
                "период",
                "быть",
                "лишний",
                "или",
                "все-таки",
                "команда",
                "мало",
                "забить",
                "?",
            ],
        ),
        (
            " - Хорошее. - Шайба Артема Лукоянова, забитая в "
            "третьем периоде, была решающей?",
            [
                "-",
                "хороший",
                ".",
                "-",
                "шайба",
                "per",
                "забить",
                "в",
                "третий",
                "период",
                "быть",
                "решающий",
                "?",
            ],
        ),
        (
            " - Хорошее. - Шайбу Артема Лукоянова, забитую в "
            "третьем периоде, не засчитали",
            [
                "-",
                "хороший",
                ".",
                "-",
                "шайба",
                "per",
                "забить",
                "в",
                "третий",
                "период",
                "не",
                "засчитать",
            ],
        ),
        (
            " - Хорошее. - Шайбы Артема Лукоянова, забитые в "
            "третьем периоде, были решающими?",
            [
                "-",
                "хороший",
                ".",
                "-",
                "шайба",
                "per",
                "забить",
                "в",
                "третий",
                "период",
                "быть",
                "решающий",
                "?",
            ],
        ),
        (
            " - Хорошее. - Гол, пропущенный в третьем периоде, "
            "был лишним или всё-таки команда мало забила?",
            [
                "-",
                "хороший",
                ".",
                "-",
                "гол",
                "пропустить",
                "в",
                "третий",
                "период",
                "быть",
                "лишний",
                "или",
                "все-таки",
                "команда",
                "мало",
                "забить",
                "?",
            ],
        ),
        (
            " - Хорошее. - Гол Артема Лукоянова, забитый в "
            "третьем периоде, был решающим?",
            [
                "-",
                "хороший",
                ".",
                "-",
                "гол",
                "per",
                "забить",
                "в",
                "третий",
                "период",
                "быть",
                "решать",
                "?",
            ],
        ),
        (
            "Закончилось основное время матча.",
            ["закончиться", "основной", "время", "матч", "."],
        ),
        (
            "Данис Зарипов - лучший бомбардир 'Ак Барса' в новейшей истории. "
            "Капитан 'Ак Барса' Данис Зарипов стал лучшим бомбардиром клуба в "
            "высших дивизионах российского хоккея. Зарипов обошёл предыдущего "
            "рекордсмена - Алексея Морозова - на счету которого 621 (266+355) "
            "очко за 'Ак Барс' в чемпионатах России и КХЛ.",
            [
                "per",
                "-",
                "хороший",
                "бомбардир",
                "org",
                "в",
                "новый",
                "история",
                ".",
                "капитан",
                "org",
                "per",
                "стать",
                "хороший",
                "бомбардир",
                "клуб",
                "в",
                "высокий",
                "дивизион",
                "российский",
                "хоккей",
                ".",
                "per",
                "обойти",
                "прошлый",
                "рекордсмен",
                "-",
                "per",
                "-",
                "на",
                "счет",
                "который",
                "очко",
                "за",
                "org",
                "в",
                "чемпионат",
                "loc",
                "и",
                "org",
                ".",
            ],
        ),
        (
            "Дмитрий Квартальнов: 'Будет очень жёсткая серия'. Главный тренер "
            "'Ак Барса' Дмитрий Квартальнов подвёл итоги серии с нижегородским "
            "'Торпедо' ( 2:1 ОТ ), оценил игру Тревора Мёрфи и рассказал о "
            "возвращении Даниса Зарипова . - Не дотерпели в третьем периоде, "
            "пропустили гол, играя '4 на 5'. В овертайме забили '5 на 4'. - Как "
            "вам качество игры команды сегодня? - Хорошее. - Пропущенная в "
            "третьем периоде шайба была лишней или всё-таки команда мало забила "
            "в основное время? - Когда счёт такой, 1:1, конечно, мало забиваем. "
            "- Что скажете по поводу удалений в составе вашей команды? Показалось, "
            "часть из них были необязательными. - Наверное, сегодня одна из немногих "
            "игр, когда в большинстве мы сыграли больше, чем в меньшинстве. ",
            [
                "per",
                ":",
                "быть",
                "очень",
                "жесткий",
                "серия",
                ".",
                "главный",
                "тренер",
                "org",
                "per",
                "подвести",
                "итог",
                "серия",
                "с",
                "нижегородский",
                "org",
                "оценить",
                "игра",
                "per",
                "и",
                "рассказать",
                "о",
                "возвращение",
                "per",
                ".",
                "-",
                "не",
                "дотерпеть",
                "в",
                "третий",
                "период",
                "пропустить",
                "гол",
                "играть",
                ".",
                "в",
                "овертайм",
                "забить",
                ".",
                "-",
                "как",
                "вы",
                "качество",
                "игра",
                "команда",
                "сегодня",
                "?",
                "-",
                "хороший",
                ".",
                "-",
                "пропустить",
                "в",
                "третий",
                "период",
                "шайба",
                "быть",
                "лишний",
                "или",
                "все-таки",
                "команда",
                "мало",
                "забить",
                "в",
                "основной",
                "время",
                "?",
                "-",
                "когда",
                "счет",
                "такой",
                "конечно",
                "мало",
                "забивать",
                ".",
                "-",
                "что",
                "сказать",
                "по",
                "повод",
                "удаление",
                "в",
                "состав",
                "ваш",
                "команда",
                "?",
                "показаться",
                "часть",
                "из",
                "они",
                "быть",
                "необязательный",
                ".",
                "-",
                "наверно",
                "сегодня",
                "один",
                "из",
                "немногий",
                "игра",
                "когда",
                "в",
                "большинство",
                "мы",
                "сыграть",
                "большой",
                "чем",
                "в",
                "меньшинство",
                ".",
            ],
        ),
    ],
)
def test_simplify_plus_lemmatize(source_text, expected_lemmas):
    simplified_text = khl.utils.simplify_text(
        source_text,
        replace_ners_=True,
        replace_dates_=True,
        replace_penalties_=True,
    )
    assert (
        khl.preprocess.lemmatize(
            text=simplified_text,
            stop_words_=None,
        )
        == expected_lemmas
    )


@pytest.mark.parametrize(
    "stop_words_,replace_ners_,replace_dates_,replace_penalties_,exclude_unknown,max_len,expected_codes",
    [
        (
            None,
            False,
            False,
            False,
            False,
            None,
            [1, 16, 4, 7, 1, 15, 1, 9, 19, 10, 8, 1, 2, 1, 5, 1, 10, 9, 6, 1, 2],
        ),
        (
            khl.stop_words,
            True,
            True,
            True,
            False,
            None,
            [14, 7, 1, 15, 12, 11, 9, 10, 1, 2, 18, 10, 9, 6, 17, 2],
        ),
        (
            None,
            False,
            False,
            False,
            True,
            None,
            [16, 4, 7, 15, 9, 19, 10, 8, 2, 5, 10, 9, 6, 2],
        ),
        (None, False, False, False, True, 5, [16, 4, 7, 15, 9]),
        (
            khl.stop_words,
            False,
            True,
            False,
            True,
            15,
            [0, 0, 0, 0, 14, 7, 15, 9, 10, 2, 5, 10, 9, 6, 2],
        ),
    ],
)
def test_e2e(
    stop_words_,
    replace_ners_,
    replace_dates_,
    replace_penalties_,
    exclude_unknown,
    max_len,
    expected_codes,
):
    text = """
        1 апреля 2023 года в матче ⅛ финала против „Спартака” Иван Иванов забил свой 100—й гол за карьеру.
        «Динамо Мск» - «Спартак» 2:1 ОТ (1:0 0:1 0:0 1:0) Голы забили: Иванов, Петров, Сидоров.
    """
    lemmas_coder = khl.preprocess.get_lemmas_coder(
        tests_dir / test_frequency_dictionary_file
    )
    assert (
        khl.text_to_codes(
            text,
            lemmas_coder,
            stop_words_,
            replace_ners_,
            replace_dates_,
            replace_penalties_,
            exclude_unknown,
            max_len,
        )
        == expected_codes
    )


class TestUsagesFromReadme:
    lemmas_coder = {
        "": 0,  # placeholder
        "???": 1,  # unknown
        ".": 2,
        "и": 3,
        "в": 4,
        "-": 5,
        ":": 6,
        "матч": 7,
        "за": 8,
        "забить": 9,
        "гол": 10,
        "per": 11,  # person entity
        "org": 12,  # organization entity
        "loc": 13,  # location entity
        "date": 14,  # date entity
        "против": 15,
        "год": 16,
        "pers": 17,  # few persons entity
        "orgs": 18,  # few organizations entity
        "свой": 19,
    }
    text = """
        1 апреля 2023 года в Москве в матче ⅛ финала против „Спартака” Иван Иванов забил свой 100—й гол за карьеру.
        «Динамо Мск» - «Спартак» 2:1 ОТ (1:0 0:1 0:0 1:0) Голы забили: Иванов, Петров и Сидоров.
    """
    expected_codes = [
        0,
        0,
        0,
        0,
        0,
        14,
        13,
        7,
        15,
        12,
        11,
        9,
        10,
        2,
        18,
        10,
        9,
        6,
        17,
        2,
    ]
    expected_unified_text = (
        "1 апреля 2023 года в Москве в матче 1/8 финала против 'Спартака' "
        "Иван Иванов забил свой 100-й гол за карьеру. 'Динамо Мск' - 'Спартак' "
        "2:1 ОТ (1:0 0:1 0:0 1:0) Голы забили: Иванов, Петров и Сидоров."
    )
    expected_simplified_text = (
        "date в loc в матче финала против org per забил свой гол за карьеру. "
        "org org Голы забили: per per и per."
    )
    expected_lemmas = [
        "date",
        "loc",
        "матч",
        "финал",
        "против",
        "org",
        "per",
        "забить",
        "гол",
        "карьера",
        ".",
        "orgs",
        "гол",
        "забить",
        ":",
        "pers",
        ".",
    ]
    expected_lemmas_20 = [
        "",
        "",
        "",
        "",
        "",
        "date",
        "loc",
        "матч",
        "против",
        "org",
        "per",
        "забить",
        "гол",
        ".",
        "orgs",
        "гол",
        "забить",
        ":",
        "pers",
        ".",
    ]

    def test_basic_usage_from_readme(self):
        codes = text_to_codes(
            text=self.text,
            lemmas_coder=self.lemmas_coder,
            stop_words_=["в", "за", "и", "свой"],  # stop words to drop
            replace_ners_=True,  # replace named entities ("Иван Иванов" -> "per", "Спартак" -> "org", "Москва" -> "loc")
            replace_dates_=True,  # replace dates ("1 апреля 2023 года" -> "date")
            replace_penalties_=True,  # replace penalties ("5+20" -> "pen")
            exclude_unknown=True,  # drop lemma that not presented in lemmas_coder
            max_len=20,  # get sequence of codes of length 15
        )
        assert codes == self.expected_codes

    def test_lower_level_usage(self):
        lemmas_coder = khl.preprocess.get_lemmas_coder(
            tests_dir / test_frequency_dictionary_file
        )
        unified_text = khl.utils.unify_text(self.text)
        simplified_text = khl.utils.simplify_text(
            text=unified_text,
            replace_ners_=True,
            replace_dates_=True,
            replace_penalties_=True,
        )
        lemmas = khl.preprocess.lemmatize(
            text=simplified_text, stop_words_=khl.stop_words
        )
        codes = khl.preprocess.lemmas_to_codes(
            lemmas=lemmas,
            lemmas_coder=lemmas_coder,
            exclude_unknown=True,
            max_len=20,
        )
        lemmas_20 = khl.preprocess.codes_to_lemmas(
            codes=codes, lemmas_coder=lemmas_coder
        )
        assert unified_text == self.expected_unified_text
        assert simplified_text == self.expected_simplified_text
        assert lemmas == self.expected_lemmas
        assert codes == self.expected_codes
        assert lemmas_20 == self.expected_lemmas_20
