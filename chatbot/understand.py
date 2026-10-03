"""The conversational layer of the chatbot.

``parser.parse()`` calls :func:`parse` here first and only falls back to its own patterns when this
returns ``None``. That means every message the bot already understood keeps behaving exactly as
before, and everything new lives in this one file.

What is new here:

============================  ==========================================================================
kind                          example
============================  ==========================================================================
``describe``                  what is Inception about / describe The Dark Knight / tell me more about X
``fact``                      what year was Alien made / how long is X / who directed X / is X good
``compare``                   compare The Godfather and GoodFellas / is A better than B
``why``                       why did you pick that / explain your choice
``more``                      show me more / next / another one (paginates instead of repeating)
``refine``                    make it shorter / older / not that one / higher rated (uses the memory)
``reference``                 the first one / number 2 (talks about a card we just showed)
``taste``                     I liked Inception and Interstellar
``duration``                  I have 2 hours
``mood``                      I'm bored / feeling down
``smalltalk``                 how are you / who are you / sorry / ok
``affirm`` / ``deny``         yes / no
``clarify``                   I want something good -> ask a question back instead of guessing
============================  ==========================================================================

Nothing here calls the network or a model: it is patterns plus the dataset.
"""
import re

from core.data import norm_title
from .parser import (BARE_TITLE_STOP, GENRES, ORD, PRONOUNS, Intent, is_english)


# --------------------------------------------------------------------------- text helpers
def _has_genre(t):
    return any(re.search(rf"\b(?:{pat})\b", t) for pat in GENRES.values())


def _has_filterish(t):
    """True when the message carries a real search filter, so it must not be hijacked."""
    return bool(
        re.search(r"\b(1[89]\d\d|20[0-2]\d|\d0'?s)\b", t)
        or re.search(r"\b(?:under|over|less than|more than|shorter than|longer than|at least|within)\s+\d+", t)
        or re.search(r"\b(?:rated|rating|stars?)\b|\b(?:above|over|at least)\s+\d(?:\.\d)?\b", t)
        or re.search(r"\b(?:directed by|director|starring|featuring|with)\s+[a-z]", t)
        or _has_genre(t))


# Question and filter words: dropped before looking for a title, so "is it good" is not the film
# "Good" (2008) and "what is it about" is not "What Is It?" (2005).
QUESTION_WORDS = set("""what which who whom whose how why when where is are was were am do does did done
it this that these those you your i we me my mine our the a an and or of in on for with by to about
good bad better best long short older newer old new year years genre kind type sort language country
budget revenue tagline theme themes keyword keywords rating rated score scores votes popular popularity
runtime duration cast director directed made release released plays play stars starring tell describe
compare contrast versus than much many""".split())


def _find_title(text, lk, max_words=8, drop=None):
    """The longest run of words in `text` that is a known title ('the dark knight', 'inception').

    `drop` removes question/filter words first, so "who plays the joker in the dark knight" resolves
    to the movie instead of to a random common word inside the sentence.
    """
    words = re.findall(r"[a-z0-9'&:.\-]+", text.lower())
    if drop:
        words = [w for w in words if w not in drop]
    for k in range(min(max_words, len(words)), 0, -1):
        for i in range(len(words) - k + 1):
            chunk = " ".join(words[i:i + k])
            norm = norm_title(chunk)
            if len(norm) < 3 or norm in PRONOUNS or chunk in PRONOUNS or norm in BARE_TITLE_STOP:
                continue                              # never treat "it" / "more" / "yes" as a film
            row = lk.exact_title(chunk)
            if row is not None:
                return row, chunk
    return None, ""


def _resolve_movie(text, lk, ctx=None, words_limit=9):
    """Which movie is this question about?

    1. a title in the text (question words removed first) — "what genre is Se7en"
    2. the movie we were just talking about — "how long is it" / "is it good" right after a description
    3. a title searched the plain way, so "the good the bad and the ugly" still resolves
    """
    row, _ = _find_title(text, lk, drop=QUESTION_WORDS)
    if row is not None:
        return row
    t = text.lower()
    if (ctx and ctx.get("last_row") is not None and len(text.split()) <= words_limit
            and re.search(r"\b(?:it|this|that)\b", t)):
        return int(ctx["last_row"])
    leftovers = [w for w in re.findall(r"[a-z']+", t) if w not in QUESTION_WORDS]
    if leftovers:
        row, _ = _find_title(text, lk)
        return row
    return None


def _find_titles(text, lk, limit=3):
    """Every title mentioned in a list: 'The Godfather and GoodFellas' -> both rows."""
    parts = re.split(r"\s*(?:,|/|&|\band\b|\bor\b|\bplus\b|\bvs\.?\b|\bversus\b|\bthan\b)\s*", text.lower())
    rows, seen = [], set()
    for part in parts:
        row, _ = _find_title(part, lk, drop=QUESTION_WORDS)
        if row is None:
            row, _ = _find_title(part, lk)
        if row is not None and row not in seen:
            rows.append(row); seen.add(row)
        if len(rows) >= limit:
            break
    if not rows:                                       # no separator, but a title may still be inside
        row, _ = _find_title(text, lk, drop=QUESTION_WORDS)
        if row is not None:
            rows.append(row)
    return rows


def _ctx_row(ctx, key="last_row"):
    return ctx.get(key) if ctx else None


# --------------------------------------------------------------------------- the facts a user can ask for
# order matters: the first pattern that matches decides what the question is about
FACT_PATTERNS = [
    ("who_plays", r"\bwho plays\b|\bwhich actor plays\b|\bwho is the (?:actor|actress) (?:who )?plays\b"),
    ("director", r"\bwho directed\b|\bwho is the director\b|\bdirector of\b|\bwho made\b"),
    ("cast", r"\bwho stars\b|\bwho(?:'s| is| are)\s+in\b|\bcast of\b|\bwho acted\b|\bactors? in\b"),
    ("runtime", r"\bhow long\b|\bruntime\b|\bduration\b|\bhow many minutes\b"),
    ("year", r"\bwhat year\b|\bwhich year\b|\bwhen (?:was|did)\b[^?]*\b(?:release|come out|made)\b|"
             r"\brelease (?:year|date)\b|\bhow old is\b"),
    ("genre", r"\bwhat (?:genre|kind of (?:movie|film)|type of (?:movie|film)|sort of (?:movie|film))\b|"
              r"\bwhich genre\b"),
    ("language", r"\bwhat language\b|\bwhich language\b"),
    ("country", r"\bwhat country\b|\bwhere was\b[^?]*\b(?:made|filmed|produced)\b"),
    ("budget", r"\bbudget\b|\bhow much did\b[^?]*\b(?:cost|make|earn|gross)\b|\bbox office\b|\brevenue\b"),
    ("tagline", r"\btagline\b|\bslogan\b"),
    ("popularity", r"\bhow popular\b|\bpopularity\b"),
    ("keywords", r"\bthemes?\b|\bkeywords?\b|\bwhat is it about\b"),
    ("rating", r"\bis\b[^?]*\b(?:good|worth (?:watching|it|my time)|any good)\b|\bhow good is\b|"
               r"\brating of\b|\bhow (?:is|was)\b[^?]*\brated\b|\bhow many votes\b|\bscore of\b|"
               r"\bwhat do people think\b"),
]

MOODS = r"bored|tired|sad|down|depressed|stressed|anxious|happy|excited|sick|ill|lonely|nostalgic|hungry"

WORDS_TO_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "half": 0.5}


# --------------------------------------------------------------------------- main entry
def parse(message, lk, ctx=None):
    """An Intent for the messages the original parser cannot handle, else None.

    ``ctx`` is the conversation memory the engine keeps (see ``engine.ChatEngine.handle``):
    ``{"filters", "text", "sort_by", "n", "ids", "seen", "seed_rows", "last_row"}``.
    """
    raw = re.sub(r"\s+", " ", str(message).strip())
    t = raw.lower()
    if not t or not is_english(raw):
        return None                                    # the original parser handles non-English
    it = Intent()
    words = len(t.split())

    # ---- talk like a person ---------------------------------------------------------------
    if re.fullmatch(r"\s*(?:yes|yeah|yep|yup|yes please|please|go on|go ahead|do it|why not|"
                    r"absolutely|of course|alright)\W*", t):
        it.kind = "affirm"; return it
    if re.fullmatch(r"\s*(?:no|nope|nah|not really|no thanks|none|nothing)\W*", t):
        it.kind = "deny"; return it
    if re.fullmatch(r"\s*(?:ok|okay|k|cool|nice|great|awesome|wow|ha(?:ha)+|lol|i see|got it|fine|sure|"
                    r"sounds good|perfect)\W*", t):
        it.kind = "smalltalk"; it.fact = "ack"; return it
    if re.search(r"\bhow (?:are|r) (?:you|u)\b|\bhow'?s it going\b|\bhow are things\b|\bwhat'?s up\b", t) \
            and words <= 8:
        it.kind = "smalltalk"; it.fact = "how_are_you"; return it
    if re.search(r"\bwho are you\b|\bwhat are you\b|\byour name\b|\bare you (?:a |an )?"
                 r"(?:bot|robot|ai|human|real|person|machine)\b|\bare you real\b", t) and words <= 10:
        it.kind = "smalltalk"; it.fact = "who"; return it
    if re.search(r"\b(?:sorry|my bad|apologies)\b", t) and words <= 6:
        it.kind = "smalltalk"; it.fact = "sorry"; return it
    if re.search(rf"\bi(?:'m| am)\s+(?:so\s+|really\s+|very\s+|a bit\s+|kind of\s+)?(?:{MOODS})\b|"
                 rf"\bfeeling\s+(?:{MOODS})\b|\bcheer me up\b|\bi need cheering up\b|"
                 rf"\bi'?m in the mood\b", t) and words <= 10 and not _has_genre(t) and not _has_filterish(t):
        mood = re.search(rf"({MOODS})", t).group(1)
        it.kind = "mood"; it.fact = mood; return it

    # ---- "I have 2 hours" -----------------------------------------------------------------
    m = re.search(r"\b(?:i|we)\s*(?:'ve|have|got)\s+(?:only\s+|about\s+|around\s+|just\s+)?"
                  r"(\d+(?:\.\d+)?|an?|one|two|three|four|half)\s*(?:an?\s+)?"
                  r"(hours?|hrs?|h|minutes?|mins?|m)\b", t)
    mins = 0
    if m:
        num = WORDS_TO_NUM.get(m.group(1))
        if num is None:
            try:
                num = float(m.group(1))
            except ValueError:
                num = None
        if num is not None:
            mins = num * 60 if m.group(2).startswith("h") else num
    else:
        m = re.search(r"\b(?:i|we)\s*(?:'ve|have|got)\s+(?:an?\s+hour|half an hour)\b", t)
        if m:
            mins = 30 if "half" in m.group(0) else 60
    if mins and 10 <= mins <= 600:
        it.kind = "duration"; it.filters["max_runtime"] = int(mins); return it

    # ---- several movies at once ------------------------------------------------------------
    if re.search(r"\b(?:compare|contrast|versus|vs\.?|better than|or)\b", t):
        rows = _find_titles(t, lk, limit=3)
        if len(rows) >= 2:
            it.kind = "compare"; it.rows = rows; return it

    # ---- "I liked X and Y" -----------------------------------------------------------------
    m = (re.match(r"\s*(?:i|we)\s+(?:really\s+|also\s+|quite\s+|just\s+|didn'?t\s+)?"
                  r"(?:liked|loved|enjoyed|like|love|enjoy|prefer|am into|'m into)\s+(.+)", t)
         or re.match(r"\s*(?:my|our)\s+favou?rites?\s+are\s+(.+)", t))
    if m:
        it.kind = "taste"
        it.rows = _find_titles(m.group(1), lk, limit=3)
        if not it.rows and _ctx_row(ctx) is not None:
            it.rows = [int(_ctx_row(ctx))]                # "I liked it" -> the movie we just discussed
        return it

    # ---- describe a movie ------------------------------------------------------------------
    cand = None
    m = re.search(r"\bwhat(?:'s| is| was)\s+(.+?)\s+(?:all )?about\b", t)
    if m:
        cand = m.group(1)
    else:
        m = re.search(r"\b(?:tell me more about|tell me about|more about|info(?:rmation)? (?:on|about)|"
                      r"details (?:of|about|on)|describe|what(?:'s| is) the plot of|plot of|summary of|"
                      r"synopsis of|story of|premise of|what happens in|what is the movie)\s+(.+)", t)
        if m:
            cand = m.group(1)
    if cand:
        row = _resolve_movie(cand, lk, ctx)
        if row is not None:
            it.kind = "describe"; it.title_row = int(row); return it

    # ---- a single fact about a movie -------------------------------------------------------
    for what, pat in FACT_PATTERNS:
        if not re.search(pat, t):
            continue
        row = _resolve_movie(t, lk, ctx)
        if row is None:
            break                                          # a real search ("highly rated movies")
        it.kind = "fact"; it.fact = what; it.title_row = int(row); return it

    # ---- why did you pick that -------------------------------------------------------------
    if (re.match(r"\s*(?:why|how come)\b", t)
            or re.search(r"\bwhy (?:did|do|would|is (?:this|that|it)|these|those)\b", t)
            or re.search(r"\bexplain (?:your|the|that|this)\b|\bjustify\b|"
                         r"\bwhat makes (?:it|this|that) (?:good|special)\b|"
                         r"\bhow did you (?:choose|pick|decide)\b", t)) and words <= 12:
        it.kind = "why"
        row = _resolve_movie(t, lk, ctx)
        it.title_row = int(row) if row is not None else None
        return it

    # ---- the first one / number 2 ----------------------------------------------------------
    m = re.fullmatch(r"\s*(?:tell me about |describe |what about |and )?(?:the )?"
                     r"(first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th|#?\d+)"
                     r"(?: one| movie| film)?(?: (?:looks|sounds) good| please| then)?\W*", t)
    if m:
        key = m.group(1).lstrip("#")
        it.kind = "reference"
        it.ordinal = ORD[key] if key in ORD else (int(key) - 1 if key.isdigit() else 0)
        return it

    # ---- more, please ----------------------------------------------------------------------
    if (re.fullmatch(r"\s*(?:and |so )?(?:some |any )?more(?: please)?"
                     r"(?: (?:movies|films|options|suggestions|picks|results|like that))?"
                     r"(?: like (?:these|those|that|them))?\W*", t)
            or re.fullmatch(r"\s*(?:show|give) me more\W*", t)
            or re.fullmatch(r"\s*(?:next|next one|another|another one|one more|anything else|"
                            r"something else|keep going|go on|more options)\W*", t)):
        # no title check: "more like X" does not match these forms, so it stays a similarity search
        it.kind = "more"; return it

    # ---- tweak the last request ------------------------------------------------------------
    if ctx and (ctx.get("ids") or ctx.get("filters") or ctx.get("text") or ctx.get("last_row")) \
            and words <= 9:
        for name, pat in (
                ("exclude", r"\bnot (?:that|this|the first|the last|number \d+)(?: one)?\b|"
                            r"\bdidn'?t like (?:that|it|this)\b|\bnot those\b"),
                ("shorter", r"\bshort(?:er)?\b|\bless long\b|\bnot so long\b"),
                ("longer", r"\blong(?:er)?\b|\bnot so short\b"),
                ("older", r"\bolder\b|\bearlier\b|\bfurther back\b"),
                ("newer", r"\bnewer\b|\bmore recent\b|\blater\b"),
                ("better", r"\bbetter\b|\bhigher[- ]rated\b|\bmore acclaimed\b|\bgood ones?\b"),
                ("fewer", r"\bfewer\b|\bjust \d+\b|\bonly \d+\b")):
            if re.search(pat, t):
                if _resolve_movie(t, lk, None) is None:     # ctx=None: never resolve to the last movie here
                    it.kind = "refine"; it.refine = name; return it
                break

    # ---- vague request -> ask back ---------------------------------------------------------
    if re.search(r"\b(?:i (?:don'?t|do not) know|no idea|not sure|what should i (?:watch|see)|"
                 r"i want something|i feel like something|recommend (?:me )?(?:something|anything)|"
                 r"give me something)\b", t) and not _has_filterish(t):
        it.kind = "clarify"; return it

    return None
