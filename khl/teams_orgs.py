"""Шаблон регулярных выражений для ручной замены названий команд и организаций."""


teams_orgs_pattern = r"""(?x)
\b(?:

# Названия лиг
(?i:
    # Континентальная хоккейная лига
        кхл|
        khl|
    # Высшая хоккейная лига
        вхл|
        vhl|
    # Молодежная хоккейная лига
        мхл|
        mhl|
    # Женская хоккейная лига
        жхл|
        zhl|
    # Национальная хоккейная лига
        нхл|
        nhl|
    # Американская хоккейная лига
        ахл|
        ahl
)|

# Названия команд КХЛ (вместе с "закрепившимися" за ними именами)
(?i:(?:hc|хк)\s*)?
(?:
    # Авангард
        А(?i:вангард)|
        A(?i:vangard)|
    # Автомобилист
        А(?i:втомобилист)|
        A(?i:vtomobilist)|
    # Адмирал
        А(?i:дмирал)|
        A(?i:dmiral)|
    # Ак Барс
        А[Кк]\s*(?i:барс)|
        A[Kk]\s*(?i:bars)|
    # Амур
        А(?i:мур)|
        A(?i:mur)|
    # Барыс
        Б(?i:арыс)|
        B(?i:arys)|
    # Витязь
        В(?i:итязь?)|
        V(?i:ityaz)|
    # Динамо (Рига, Минск, Москва)
        Д(?i:инамо)(?:\s*(?i:р|рига|м|мск|москва|мн|минск)\b)?|
        D(?i:[iy]namo)(?:\s*(?i:r|riga|m|msk|moscow|mn|minsk)\b)?|
    # Йокерит
        Й(?i:окерит)|
        J(?i:okerit)|
    # Куньлунь РС
        К(?i:унь?лунь?)(?:\s*(?i:рс|ред\s*стар)\b)?|
        K(?i:unlun)(?:\s*(?i:rs|red\s*star)\b)?|
    # Лада
        Л(?i:ада)|
        L(?i:ada)|
    # Локомотив
        Л(?i:окомотив)|
        L(?i:okomotiv)|
    # Металлург
        М(?i:етал?лург)(?:\s*(?i:мг|магнитогорск)\b)?|
        (?i:магнитк[а-я]+)\b|
        (?i:ммг)\b|
        M(?i:etal?lurg)(?:\s*(?i:mg|magnitogorsk)\b)?|
        (?i:magnitk[a-z]+)\b|
        (?i:mmg)\b|
    # Нефтехимик
        Н(?i:ефтехимик)|
        N(?i:eftek?himik)|
    # Салават Юлаев
        С(?i:алават)(?i:[а-яё]*\s*юлаев)?|
        (?i:сю)\b|
        S(?i:alavat)(?i:(?:\'?[а-яёa-z])*\s*ulaev)?|
        (?i:su)\b|
    # Северсталь
        С(?i:еверсталь?)|
        S(?i:everstal)|
    # Сибирь
        С(?i:ибирь?)|
        S(?i:ibir)|
    # СКА
        (?i:ска)\b|
        (?i:ska)\b|
    # Спартак
        С(?i:партак)|
        S(?i:partak)|
    # Торпедо НН
        Т(?i:орпедо)(?:\s*(?i:нн|нижний\s*новгород)\b)?|
        T(?i:orpedo)(?:\s*(?i:nn|nizhni[yi]\s*novgorod)\b)?|
    # Трактор
        Т(?i:рактор)|
        T(?i:ra[kc]tor)|
    # ХК Сочи
        С(?i:очи)|
        S(?i:ochi)|
    # ЦСКА
        (?i:цска)\b|
        (?i:cska)\b|
    # team
        name
        # ПОСЛЕДНЕЕ НАЗВАНИЕ ПРОПИСАТЬ БЕЗ ВЕРТИКАЛЬНОЙ ЧЕРТОЧКИ!!!
)
)(?:\'?[а-яА-ЯёЁa-zA-Z])*
"""
