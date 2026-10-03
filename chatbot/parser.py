"""Rule-based (offline) message parser. Turns a sentence into a structured Intent.
No API, no model: just patterns and word lists."""
import re
import unicodedata
from dataclasses import dataclass, field

# ------------------------------------------------------------------ English-only guard
ENGLISH_WORDS = set("""the a an and or of to in for with me my i is are was be it that this what which who
movie movies show like something want need some good best from about on at by not any
recommend suggest give tell find please can you do have has more less than after before under over
very really love watch watching see seen one ones same similar tonight funny scary new old top""".split())
FOREIGN_WORDS = set("""el la los las del que una por para con es est une des les du et pour dans qui je ne pas
der die das und ist nicht ich mit ein eine il lo gli che per sono non uma os um mais voce un je tu vous bonjour hola ciao quiero quisiera pelicula película voudrais buscar gostaria filme möchte gracias merci danke bitte grazie""".split())


def is_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    latin = sum(1 for c in letters if unicodedata.name(c, "").startswith("LATIN"))
    if latin / len(letters) < 0.7:                       # Arabic, Cyrillic, CJK ...
        return False
    toks = re.findall(r"[a-zà-ÿ']+", text.lower())
    if len(toks) >= 3 and not any(t in ENGLISH_WORDS for t in toks) and any(t in FOREIGN_WORDS for t in toks):
        return False
    return True


# ------------------------------------------------------------------ word lists
GENRES = {
    "Action": r"action(?:-packed)?",
    "Adventure": r"adventures?",
    "Animation": r"animations?|animated|cartoons?|anime",
    "Comedy": r"comed(?:y|ies)|funny|hilarious|laugh\w*|rom-?coms?",
    "Crime": r"crime",
    "Documentary": r"documentar(?:y|ies)|docs?",
    "Drama": r"dramas?|dramatic|sad|tearjerkers?|emotional",
    "Family": r"family|kids?|children",
    "Fantasy": r"fantas(?:y|ies)|magical",
    "History": r"history|historical",
    "Horror": r"horrors?|scary|creepy|spooky|terrifying",
    "Music": r"musicals?|music",
    "Mystery": r"mysteries|mystery|whodunn?its?",
    "Romance": r"romances?|romantic|love stor(?:y|ies)|date night|rom-?coms?",
    "Science Fiction": r"sci-?fi|sci fi|science fiction",
    "Thriller": r"thrillers?|suspens\w+",
    "War": r"war|wwii|ww2",
    "Western": r"westerns?|cowboys?",
}
LANG_WORDS = {
    "french": "French", "japanese": "Japanese", "korean": "Korean", "spanish": "Spanish",
    "german": "German", "italian": "Italian", "hindi": "Hindi", "bollywood": "Hindi",
    "chinese": "Chinese", "russian": "Russian", "arabic": "Arabic", "turkish": "Turkish",
    "swedish": "Swedish", "danish": "Danish", "portuguese": "Portuguese", "english": "English",
    "anime": "Japanese", "persian": "Persian", "thai": "Thai", "polish": "Polish"}

FILLER = set("""movie movies film films flick flicks something anything recommend recommendation recommendations
suggest suggestions suggestion show give me i id i'd i'm want wanna would like to watch watching see a an the
some any good great nice best top please can could you tonight today find looking look for need is are of
that with and or but just really very kind sort type about more something's let lets let's me my us it
newer older recent new old latest classic short long highly rated popular from in on at by under over
than after before between since until what which who how why where epic lengthy quick brief darker""".split())

# Words that are conversation, not a theme. Without these, "what should I watch tonight" turned into a
# search for the word "should" and "I'm bored" into a search for "bored".
CHATTER = set("""should shall would could may might must can cant cannot do does did doing done am was were
been being have has had will wont dont doesnt didnt im ive youre its yeah yep nope no ok okay sure cool nice
great awesome amazing wow haha lol thanks thank sorry maybe perhaps actually really very just only bit little
lot again still yet also too so then than well else another more less other others different thing things
stuff one ones someone somebody anything everything nothing want wanna need give show tell find looking feel
feeling mood bored boring tired interesting genre genres type kind sort watch watching watched seen tonight
today now year years long short hours hour minutes minute mins rating rated score scores votes imdb yes
ahead please""".split())
FILLER = FILLER | CHATTER

# A bare word that is only conversation must never be looked up as a movie title: "more" is a 1969 film
# and "yes" a 2005 film, which is how the bot used to answer "yes" and "no".
BARE_TITLE_STOP = set("""yes no more ok okay sure thanks thank please help hi hello hey bye next another why
what when where who how which nothing anything something everything maybe cool nice great lol haha sorry stop
wait and or but the a an this that these those one ones do did does done go ahead""".split())

# Never a movie title in the middle of a sentence, e.g. "what is IT about"
# ("Her" and "Up" are deliberately absent, they are real titles).
PRONOUNS = {"it", "this", "that", "them", "these", "those", "him", "the movie", "the film"}

ORD = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
       "fifth": 4, "5th": 4, "last": -1, "it": 0, "that one": 0, "this one": 0}


@dataclass
class Intent:
    kind: str = "unknown"            # see engine for the full list
    filters: dict = field(default_factory=dict)
    text: str = ""                   # leftover theme words, e.g. "space heist"
    title_row: int | None = None
    title_text: str = ""
    n: int = 5
    sort_by: str = "wr"
    ordinal: int | None = None
    relative: str | None = None      # "newer" / "older" than the anchor movie
    surprise: bool = False
    fact: str | None = None          # which fact was asked for: year / runtime / cast / mood / ...
    refine: str | None = None        # follow-up tweak: shorter / older / better / exclude / fewer
    rows: list = field(default_factory=list)   # several titles: "compare A and B", "I liked A and B"


def _clean(msg):
    return re.sub(r"\s+", " ", msg.strip())


def _num(s):
    return float(s)


def _years(t, spans):
    """Returns year_range or None; records matched spans so they are removed from theme text."""
    m = re.search(r"\b(?:between|from)\s+(1[89]\d\d|20[0-2]\d)\s*(?:and|to|-)\s*(1[89]\d\d|20[0-2]\d)\b", t)
    if m:
        spans.append(m.span()); return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"\b(?:the\s+)?(1[89]|20)?(\d0)'?s\b", t)
    if m:
        d = int(m.group(2))
        base = int(m.group(1)) * 100 if m.group(1) else (2000 if d < 30 else 1900)
        spans.append(m.span()); return (base + d, base + d + 9)
    m = re.search(r"\b(?:after|since|newer than|later than|post)\s+(1[89]\d\d|20[0-2]\d)\b", t)
    if m:
        spans.append(m.span()); return (int(m.group(1)) + (0 if "since" in m.group(0) else 1), 2030)
    m = re.search(r"\b(?:before|older than|prior to|until|pre)\s+(1[89]\d\d|20[0-2]\d)\b", t)
    if m:
        spans.append(m.span()); return (1880, int(m.group(1)) - (0 if "until" in m.group(0) else 1))
    m = re.search(r"\b(?:from|in|of|year)?\s*(1[89]\d\d|20[0-2]\d)\b", t)
    if m:
        spans.append(m.span()); y = int(m.group(1)); return (y, y)
    m = re.search(r"\bsilent\b", t)
    if m:
        spans.append(m.span()); return (1880, 1930)
    return None


def _runtime(t, spans):
    out = {}
    pat = (r"\b(under|below|less than|shorter than|at most|no more than|within|max(?:imum)?|"
           r"over|above|more than|longer than|at least|min(?:imum)?)\s+(\d+(?:\.\d+)?)\s*"
           r"(hours?|hrs?|h|minutes?|mins?|m)\b")
    for m in re.finditer(pat, t):
        val = _num(m.group(2)); unit = m.group(3)
        mins = val * 60 if unit.startswith("h") else val
        if m.group(1) in ("under", "below", "less than", "shorter than", "at most", "no more than",
                          "within") or m.group(1).startswith("max"):
            out["max_runtime"] = int(mins)
        else:
            out["min_runtime"] = int(mins)
        spans.append(m.span())
    if not out:
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(hours?|hrs?)\s*(?:or less|max)\b", t)
        if m:
            out["max_runtime"] = int(_num(m.group(1)) * 60); spans.append(m.span())
    if not out:
        if re.search(r"\b(short|quick|brief)\b", t):
            out["max_runtime"] = 95
        elif re.search(r"\b(long|epic|lengthy)\b", t):
            out["min_runtime"] = 150
    return out


def _rating(t, spans):
    m = re.search(r"\b(?:rated|rating|score|scores|stars?)\s*(?:of|above|over|at least|>)?\s*(\d(?:\.\d)?)\b", t)
    if not m:
        m = re.search(r"\b(?:above|over|at least|higher than|better than)\s+(\d(?:\.\d)?)\s*(?:stars?|/10|rating|rated|points)?"
                      r"(?!\s*(?:min|hour|hr|h\b|m\b))", t)
        if m and re.match(r"\s*(?:min|hour|hr|h\b|m\b)", t[m.end(1):]):
            m = None
    if m:
        v = _num(m.group(1))
        if 1 <= v <= 10:
            spans.append(m.span()); return v
    if re.search(r"\b(highly rated|top rated|top-rated|best|acclaimed|masterpieces?|must[- ]see|"
                 r"great|excellent|greatest|critically)\b", t):
        return 7.0
    return None


def parse(message, lk, anchor_year=None, ctx=None):
    """lk = Lookup, ctx = conversation state (optional). Returns an Intent.

    The conversational layer in understand.py gets the first look (movie facts and descriptions,
    follow-ups like "show me more", small talk). Anything it does not claim keeps going through the
    original patterns in _parse_base below, unchanged.
    """
    from . import understand                       # local import: understand imports this module
    it = understand.parse(message, lk, ctx)
    if it is not None:
        return it
    return _parse_base(message, lk, anchor_year)


def _parse_base(message, lk, anchor_year=None):
    """The original rule-based parser: recommendations, filters, lists and favourites."""
    raw = _clean(message)
    t = raw.lower()
    it = Intent()

    if not is_english(raw):
        it.kind = "non_english"; return it

    # ---- social intents ----
    if re.fullmatch(r"(hi+|hello+|hey+|yo|good (morning|evening|afternoon)|hiya|sup)\W*(there|bot)?\W*", t):
        it.kind = "greet"; return it
    if re.search(r"\b(thanks|thank you|thx|cheers)\b", t) and len(t.split()) <= 6:
        it.kind = "thanks"; return it
    if re.fullmatch(r"(bye|goodbye|see you|cya|quit|exit)\W*", t):
        it.kind = "bye"; return it
    if re.search(r"\b(help|what can you do|how do(es)? this work|commands)\b", t) and len(t.split()) <= 8:
        it.kind = "help"; return it

    if re.search(r"\b(surprise me|random movie|pick (one|something) for me|feeling lucky)\b", t):
        it.kind = "recommend"; it.surprise = True; return it

    # ---- the user's lists ----
    if re.search(r"\b(based on|from|using)\s+(my|what i)\s+(favou?rites?|likes?|liked|history|taste)\b", t) or \
       re.search(r"\b(recommend|suggest)\w*\s+(something\s+)?for me\b", t) and "favou" in t:
        it.kind = "for_me"; return it
    if re.search(r"\b(show|list|see|what('s| is| are)?)\b.*\bmy\s+(favou?rites?|favou?rite (movies|list))\b", t) or \
       re.fullmatch(r"(my )?favou?rites?\W*", t):
        it.kind = "show_favorites"; return it
    if re.search(r"\b(show|list|see|what('s| is| are)?)\b.*\b(what i('ve| have)? (watched|seen)|my watched|movies i('ve| have)? (watched|seen))\b", t) or \
       re.fullmatch(r"(my )?watched( list| movies)?\W*", t):
        it.kind = "show_watched"; return it

    # ---- add to favourites / mark watched ----
    m = re.search(r"\b(?:add|save|put|keep)\s+(.+?)\s+(?:to|in|into|on)\s+(?:my\s+)?favou?rites?\b", t) or \
        re.search(r"\bfavou?rite\s+(.+)", t) and re.match(r"(favou?rite|i love)", t)
    if m and re.search(r"favou?rite|add|save", t):
        target = m.group(1).strip(" .\"'")
        return _resolve_target(it, "add_favorite", target, lk, raw)
    m = re.search(r"\bmark\s+(.+?)\s+as\s+(?:watched|seen)\b", t) or \
        re.match(r"\bi(?:'ve| have)?\s+(?:just\s+)?(?:watched|seen)\s+(.+)", t) or \
        re.search(r"\badd\s+(.+?)\s+(?:to|in)\s+(?:my\s+)?watched\b", t)
    if m:
        return _resolve_target(it, "mark_watched", m.group(1).strip(" .\"'"), lk, raw)

    # ---- quoted title anywhere ----
    quoted = re.findall(r"[\"“”']([^\"“”']{2,60})[\"“”']", raw)

    # ---- "similar to X" ----
    sim = re.search(r"\b(?:similar to|similar|like|reminds? me of|in the style of|more like|same as|"
                    r"if i liked|if i loved|if you liked)\s+(.+)", raw, flags=re.I)
    title_row, title_text = None, ""
    rest = t
    if sim:
        cap = sim.group(1).strip()
        title_row, title_text = lk.title_in_text(cap.lower().strip(" .?!\"'"))
        if title_row is None and quoted:
            title_row = lk.exact_title(quoted[0]) or lk.fuzzy_title(quoted[0]); title_text = quoted[0].lower()
        if title_row is None:
            cut = re.split(r"\s+(?:but|that|which|with|from|under|over|after|before|and|in)\s+|[,.;]", cap.lower())[0]
            if len(cut.split()) >= 1 and not _is_only_filters(cut):
                title_row = lk.fuzzy_title(cut); title_text = cut if title_row is not None else ""
        if title_row is not None:
            it.kind = "similar"; it.title_row = title_row; it.title_text = title_text
            rest = t.replace(title_text.lower(), " ", 1) if title_text else t
            rest = re.sub(r"\b(similar to|similar|like|reminds? me of|in the style of|more like|same as)\b", " ", rest)
        elif not _is_only_filters(cap.lower()) and len(cap.split()) <= 6 and not re.search(
                r"\b(comed|dram|horror|action|thrill|romanc)", cap.lower()):
            it.kind = "title_not_found"; it.title_text = cap.strip(" .?!\"'"); return it

    # ---- tell me about X ----
    if it.kind == "unknown":
        m = re.search(r"\b(?:tell me about|info(?:rmation)? (?:on|about)|details (?:of|about|on)|"
                      r"what(?:'s| is) the plot of|plot of|summary of|who (?:directed|stars in|is in|acted in)|"
                      r"cast of|more about)\s+(.+)", raw, flags=re.I)
        if m:
            row, txt = lk.title_in_text(m.group(1).lower().strip(" .?!\"'"))
            if row is None:
                row = lk.fuzzy_title(m.group(1).strip(" .?!\"'"))
            if row is not None:
                it.kind = "info"; it.title_row = row; return it
        # bare title typed alone, e.g. "Inception"
        bare = raw.strip(" .?!\"'")
        if len(bare.split()) <= 6 and bare.lower() not in BARE_TITLE_STOP:
            row = lk.exact_title(bare)
            if row is not None and not _has_any_slot(t, lk):
                it.kind = "title_lookup"; it.title_row = row; return it

    # ---- filters ----
    spans = []
    f = {}
    yr = _years(rest, spans)
    if yr:
        f["year_range"] = yr
    f.update(_runtime(rest, spans))
    rt = _rating(rest, spans)
    if rt:
        f["min_rating"] = rt

    # relative years ("newer"/"older") depend on the anchor movie
    if re.search(r"\b(newer|more recent|later|modern)\b", rest) and "year_range" not in f:
        base = int(lk.movies.at[it.title_row, "release_year"]) if it.title_row is not None else None
        f["year_range"] = ((base + 1) if base else 2010, 2030)
    elif re.search(r"\b(older|earlier|vintage|classic|classics)\b", rest) and "year_range" not in f:
        base = int(lk.movies.at[it.title_row, "release_year"]) if it.title_row is not None else None
        f["year_range"] = (1880, (base - 1) if base else 1980)
    elif re.search(r"\b(recent|new|latest|just released)\b", rest) and "year_range" not in f:
        f["year_range"] = (2015, 2030)
    elif re.search(r"\b(old)\b", rest) and "year_range" not in f:
        f["year_range"] = (1880, 1980)

    genres = []
    for g, pat in GENRES.items():
        mm = re.search(rf"\b(?:{pat})\b", rest)
        if mm:
            genres.append(g); spans.append(mm.span())
    if re.search(r"\brom-?coms?\b", rest):
        genres = sorted(set(genres) | {"Romance", "Comedy"})
    if genres:
        f["genres"] = genres[:3]

    for w, name in LANG_WORDS.items():
        mm = re.search(rf"\b{w}\b", rest)
        if mm and name in lk.languages.values():
            f["language"] = name; spans.append(mm.span())
            if w == "anime" and "Animation" not in f.get("genres", []):
                f["genres"] = f.get("genres", []) + ["Animation"]
            break

    # people
    pm = re.search(r"\b(?:directed by|director|film by|movies? by|films? by|by)\s+([a-z .'\-]+)", rest)
    if pm:
        name, k = lk.person_prefix(pm.group(1).split(), "director")
        if name:
            f["director"] = name; spans.append(pm.span())
    pm = re.search(r"\b(?:starring|stars|featuring|actor|actress|with|acted by)\s+([a-z .'\-]+)", rest)
    if pm and "director" not in f:
        name, k = lk.person_prefix(pm.group(1).split(), "actor")
        if name:
            f["actor"] = name; spans.append(pm.span())
        else:
            name, k = lk.person_prefix(pm.group(1).split(), "director")
            if name:
                f["director"] = name; spans.append(pm.span())
    if "director" not in f and "actor" not in f:
        tokens = re.findall(r"[a-z'\-.]+", rest)
        hit = lk.person_anywhere(tokens)
        if hit:
            f["director" if hit[0] == "director" else "actor"] = hit[1]

    # count + sort + surprise
    mm = re.search(r"\b(?:top|best|give me|show me|list|recommend|suggest)?\s*(\d{1,2})\s*(?:movies|films|suggestions|recommendations|titles|picks)?\b", rest)
    if mm and not re.search(r"\b" + mm.group(1) + r"\s*(?:min|hour|hr)", rest) \
            and not re.search(r"(1[89]\d\d|20[0-2]\d)", mm.group(1)) and 1 <= int(mm.group(1)) <= 20 \
            and re.search(r"(top|give|show|list|recommend|suggest|movies|films|picks)", mm.group(0)):
        it.n = int(mm.group(1)); spans.append(mm.span())
    if re.search(r"\b(popular|trending|blockbusters?|famous)\b", rest):
        it.sort_by = "popularity"
    if re.search(r"\b(newest|latest|just released)\b", rest):
        it.sort_by = "newest"
    if re.search(r"\b(surprise me|random|anything|whatever|feeling lucky)\b", rest):
        it.surprise = True

    # leftover theme words
    keep = []
    cursor = 0
    for a, b in sorted(spans):
        keep.append(rest[cursor:a]); cursor = max(cursor, b)
    keep.append(rest[cursor:])
    words = [w for w in re.findall(r"[a-z']+", " ".join(keep)) if w not in FILLER and len(w) > 2]
    for k in ("director", "actor"):
        if k in f:
            nm = f[k].lower().split()
            words = [w for w in words if w not in nm]
    it.text = " ".join(words[:6])
    it.filters = f

    if it.kind == "unknown":
        has_signal = bool(f) or it.surprise or it.text or it.sort_by != "wr" or \
            re.search(r"\b(recommend|suggest|show|find|movies?|films?|watch)\b", t)
        it.kind = "recommend" if has_signal else "unknown"

    # A recommendation request with nothing to go on ("recommend movies", "show me something") is
    # better answered with a question than with the same top-rated list every single time.
    if it.kind == "recommend" and not it.surprise and not f and not it.text and it.title_row is None:
        it.kind = "clarify"
    return it


def _is_only_filters(s):
    return bool(re.fullmatch(r"\s*(?:(?:" + "|".join(GENRES.values()) + r")\s*)+", s))


def _has_any_slot(t, lk):
    return bool(re.search(r"\b(1[89]\d\d|20[0-2]\d|movies?|films?)\b", t))


def _resolve_target(it, kind, target, lk, raw):
    it.kind = kind
    key = target.lower().strip()
    if key in ORD or re.fullmatch(r"(?:the\s+)?(first|second|third|fourth|fifth|last)(?:\s+one)?", key) \
            or re.fullmatch(r"#?\d", key):
        mm = re.search(r"(first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)", key)
        if mm:
            it.ordinal = ORD[mm.group(1)]
        elif re.fullmatch(r"#?\d", key):
            it.ordinal = int(key.strip("#")) - 1
        else:
            it.ordinal = 0
        return it
    row, _ = lk.title_in_text(key.strip(" .?!\"'"))
    if row is None:
        row = lk.fuzzy_title(target.strip(" .?!\"'"))
    it.title_row = row
    it.title_text = target
    return it
