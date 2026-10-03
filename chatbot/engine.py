"""Chat engine: parse the message -> call the recommender / read the data -> write the reply.

The bot never invents movies: every title in a reply comes from the recommender or the dataset, and
every fact (runtime, cast, rating...) comes from ``chatbot/facts.py``. It stays fully offline.

Conversation memory
-------------------
``handle(..., ctx=...)`` takes a plain dict the caller owns (``st.session_state["chat_ctx"]``) and
updates it in place, so follow-ups work:

    filters, text, sort_by, n   the last request, so "make it shorter" can tweak it and re-run
    mode                        "similar" | "filters" | "surprise" - how the last list was built
    seed_rows                   the movies it was based on (for "why did you pick that")
    ids                         what the user is looking at now ("the first one", "why")
    seen                        everything already shown, so "more" never repeats itself
    last_row                    the single movie last described (for "what is it about")
"""
import random
from dataclasses import dataclass, field

from core import storage
from .facts import Facts
from .lookup import Lookup
from .parser import parse

NON_ENGLISH = ("Sorry, I only understand English for now. 🌐 Please type your request in English, "
               "for example: *\"scary movies from the 90s\"* or *\"something like Sherlock Jr.\"*.")
HELP = (
    "I know the 49,853 movies in this dataset — I can recommend one, or tell you about it.\n\n"
    "**Describe a movie**\n"
    "- *what is Inception about* · *describe The Dark Knight*\n"
    "- *how long is The Godfather* · *who directed Interstellar* · *who is in Pulp Fiction*\n"
    "- *what year was Alien made* · *is Se7en good* · *what genre is Parasite*\n"
    "- *compare The Godfather and GoodFellas*\n\n"
    "**Find something to watch**\n"
    "- *scary movies from the 90s* · *top 5 highly rated Japanese animation*\n"
    "- *something like Sherlock Jr. but newer, under 90 min*\n"
    "- *movies directed by Christopher Nolan* · *movies with Tom Hanks* · *space heist*\n"
    "- *I have 2 hours* · *I liked Inception and Interstellar* · *surprise me*\n\n"
    "**Keep talking**\n"
    "- *show me more* · *the first one* · *make it shorter* · *not that one* · *why did you pick that*\n\n"
    "If you're logged in: *add the second one to my favorites*, *mark Inception as watched*, "
    "*show my favorites*.")
LOGIN_NEEDED = "You need to be logged in for that. Use the sidebar to log in or sign up. 🔐"

RELAX = [("min_rating", "the rating filter"), ("max_runtime", "the runtime limit"),
         ("min_runtime", "the runtime limit"), ("year_range", "the year range"),
         ("language", "the language"), ("genres", "some genres")]

MOOD_REPLIES = {
    "bored": "Let's fix that. 🍿 What sounds better: something *funny*, something *clever* "
             "(a twisty thriller), or shall I *surprise* you?",
    "tired": "Then nothing demanding. Short and light, or something you can half-watch — "
             "say *under 100 min* or *comedy*.",
    "sad": "Sorry to hear that. Company helps: want something *warm and funny*, or a proper good cry? "
           "I can do either — *comedy* or *drama*.",
    "down": "Sorry to hear that. Want me to cheer you up with something *funny*, or would you rather "
            "sit with a *drama*?",
    "depressed": "I'm sorry. Want something gentle and *funny*, or a *drama* to match the mood?",
    "stressed": "Then let's switch your brain off for two hours. *Comedy*, *action*, or *animation*?",
    "anxious": "Something calming then: *animation*, a *documentary*, or a comfort *comedy*?",
    "happy": "Love that. 😄 Want to keep the streak going — *comedy*, *music*, or *adventure*?",
    "excited": "Then let's match it: *action*, *thriller*, or *adventure*?",
    "sick": "Get well soon 🛌 Something easy: *animation*, *family*, or a *comedy*?",
    "ill": "Get well soon 🛌 Something easy: *animation*, *family*, or a *comedy*?",
    "lonely": "Films are good company. Want something *warm* (family/friendship) or a *romance*?",
    "nostalgic": "Then let's go back — tell me a decade (*80s*, *90s*) or a movie you loved as a kid.",
    "hungry": "Ha — food films it is. Ask me for *something like Chef*, or name a decade.",
}

ACK = ["👍 Anything else — say *more* for different ones, or name a movie and I'll describe it.",
       "👍 Want more options, or shall I describe one of these?",
       "👍 Say *more* for a different set, or *why* and I'll explain the top pick."]


@dataclass
class Reply:
    text: str
    ids: list = field(default_factory=list)      # TMDB ids to show as cards, in order


def constraints(f):
    """Short text for the filters of a 'similar to' request, e.g. 'after 1925, under 90 min'."""
    bits = []
    if f.get("genres"):
        bits.append(" / ".join(f["genres"]))
    if f.get("language"):
        bits.append(f["language"])
    if f.get("year_range"):
        a, b = f["year_range"]
        bits.append(f"after {a - 1}" if b >= 2030 else f"before {b + 1}" if a <= 1880 else f"{a}–{b}")
    if f.get("max_runtime"):
        bits.append(f"under {f['max_runtime']} min")
    if f.get("min_runtime"):
        bits.append(f"over {f['min_runtime']} min")
    if f.get("min_rating"):
        bits.append("highly rated")
    if f.get("director"):
        bits.append(f"by {f['director']}")
    if f.get("actor"):
        bits.append(f"with {f['actor']}")
    return ", ".join(bits)


def describe(f, sort_by="wr"):
    bits = []
    if sort_by == "popularity":
        bits.append("popular")
    if f.get("min_rating"):
        bits.append("highly rated")
    if f.get("genres"):
        bits.append(" / ".join(f["genres"]))
    if f.get("language"):
        bits.append(f["language"]) if f.get("genres") is None else None
    out = " ".join(bits) or "top-rated"
    out += " movies"
    if f.get("language") and f.get("genres"):
        out = f["language"] + "-language " + out
    if f.get("director"):
        out += f" directed by {f['director']}"
    if f.get("actor"):
        out += f" with {f['actor']}"
    if f.get("year_range"):
        a, b = f["year_range"]
        out += f" from {a}–{b}" if b - a > 0 and b < 2030 and a > 1880 else \
            (f" released after {a}" if b >= 2030 else (f" released before {b}" if a <= 1880 else f" from {a}"))
    if f.get("max_runtime"):
        out += f", under {f['max_runtime']} min"
    if f.get("min_runtime"):
        out += f", over {f['min_runtime']} min"
    return out


class ChatEngine:
    def __init__(self, movies):
        self.movies = movies
        self.lk = Lookup(movies)
        self.facts = Facts(movies)

    # ------------------------------------------------------------------ helpers
    def _rows(self, ids):
        return [self.lk.id_to_row[i] for i in ids if i in self.lk.id_to_row]

    def _id(self, row):
        return int(self.movies.at[row, "id"])

    def _info(self, row):
        return self.facts.describe(row)

    def _remember(self, ctx, *, filters=None, text=None, sort_by=None, n=None, mode=None,
                  seeds=None, ids=None):
        """Record what we just did, so the next message can build on it."""
        if filters is not None:
            ctx["filters"] = dict(filters)
        if text is not None:
            ctx["text"] = text or ""
        if sort_by is not None:
            ctx["sort_by"] = sort_by
        if n:
            ctx["n"] = int(n)
        if mode is not None:
            ctx["mode"] = mode
        if seeds is not None:
            ctx["seed_rows"] = [int(s) for s in seeds]
        if ids is not None:
            ctx["ids"] = [int(i) for i in ids]
            ctx["seen"] = set(ctx.get("seen") or set()) | set(int(i) for i in ids)
            ctx["last_row"] = self.lk.id_to_row.get(int(ids[0])) if len(ids) else None

    def _single(self, ctx, row, text):
        """Remember a single-movie answer (describe / reference / why)."""
        mid = self._id(row)
        ctx["last_row"] = int(row)
        ctx["ids"] = [mid]
        ctx["seen"] = set(ctx.get("seen") or set()) | {mid}
        return Reply(text, [mid])

    def _recommend_with_relax(self, rec, f, text, n, sort_by):
        cur, txt, dropped = dict(f), text, []
        while True:
            res = rec.recommend_by_filters(**cur, text=txt or None, n=n, sort_by=sort_by)
            if len(res):
                return res, dropped
            if txt:
                dropped.append(f'the theme "{txt}"'); txt = ""; continue
            for key, label in RELAX:
                if key in cur:
                    if key == "genres" and len(cur["genres"]) > 1:
                        cur["genres"] = cur["genres"][:1]
                    else:
                        del cur[key]
                    dropped.append(label); break
            else:
                return res, dropped

    def _why_line(self, rec, seeds, top_row):
        """One short sentence explaining the top pick, when we have seeds to compare against."""
        if not seeds:
            return ""
        try:
            reason = rec.why(list(seeds), int(top_row))
        except Exception:                                     # noqa: BLE001 - never break a reply
            reason = ""
        return f"\n\n_Top pick: {self.movies.at[top_row, 'title']} — {reason}._" if reason else ""

    def _finish_recommend(self, rec, ctx, res, dropped, filters, text, sort_by, n, seeds=None,
                          lead="", drop_seen=False, mode="filters"):
        if drop_seen and len(res):
            seen = set(ctx.get("seen") or set())
            res = res[~res["id"].isin(seen)]
        if drop_seen and n:
            res = res.head(int(n))                          # "more" is one page, not all the extras
        if not len(res):
            return Reply((lead or "") + "That's everything I have for that request. "
                         "Want me to loosen it — say *higher rated*, or a genre?")
        intro = lead
        if dropped:
            intro += ("Nothing matched everything, so I relaxed " + ", ".join(dict.fromkeys(dropped)) + ". ")
        intro += f"Here are {len(res)} {describe(filters, sort_by)}"
        if text and not any("theme" in d for d in dropped):
            intro += f' about "{text}"'
        intro += ":"
        top_row = self.lk.id_to_row.get(int(res["id"].iloc[0]))
        if seeds and top_row is not None:
            intro += self._why_line(rec, seeds, top_row)
        self._remember(ctx, filters=filters, text=text, sort_by=sort_by, n=n, mode=mode, seeds=seeds,
                       ids=res["id"].tolist())
        return Reply(intro, res["id"].tolist())

    # ------------------------------------------------------------------ small talk
    def _smalltalk(self, it, ctx):
        kind = it.fact
        if kind == "how_are_you":
            return Reply("Running fine, thanks — I only read movie data, so no bad days. 😄 "
                         "What are you in the mood for?")
        if kind == "who":
            return Reply(f"I'm the CineMatch bot: rule-based, offline, and limited to the "
                         f"{len(self.movies):,} movies in this dataset. No AI model and no internet, "
                         "so I can't invent anything — but I can describe any film here, recommend by "
                         "mood, genre, year, runtime or cast, and explain my picks. Try "
                         "*what is Inception about*.")
        if kind == "sorry":
            return Reply("No need to apologise 🙂 What can I find you?")
        return Reply(random.choice(ACK))

    def _clarify(self):
        return Reply("Happy to help — I just need one thing from you:\n\n"
                     "- a **mood or genre** — *something funny*, *a good thriller*, *scary movies*\n"
                     "- a **movie you liked** — *something like Inception*\n"
                     "- or just say **surprise me** 🎲")

    # ------------------------------------------------------------------ main entry
    def handle(self, message, rec, username=None, last_ids=None, ctx=None) -> Reply:
        last_ids = last_ids or []
        ctx = ctx if ctx is not None else {}
        it = parse(message, self.lk, ctx=ctx)
        k = it.kind

        if k == "non_english":
            return Reply(NON_ENGLISH)
        if k == "greet":
            return Reply("Hi! 👋 Tell me what you feel like watching — or name a movie and I'll "
                         "describe it. Type **help** for examples.")
        if k == "thanks":
            return Reply("You're welcome! Want more suggestions? 🍿")
        if k == "bye":
            return Reply("Bye! Enjoy the movie. 🎬")
        if k in ("help", "unknown"):
            pre = "" if k == "help" else "I didn't quite get that. "
            return Reply(pre + HELP)

        # ---- talking (no recommendation) -------------------------------------------------
        if k == "smalltalk":
            return self._smalltalk(it, ctx)
        if k == "affirm":
            if ctx.get("ids") or ctx.get("filters") or ctx.get("text"):
                return self._more(rec, ctx)                # "yes" after a list -> keep going
            return Reply("Great 👍 What are you in the mood for? A genre, a decade, or a movie "
                         "you already like.")
        if k == "deny":
            return Reply("No problem. Tell me a genre, a mood, or a film you liked, and I'll take it "
                         "from there.")
        if k == "mood":
            return Reply(MOOD_REPLIES.get(it.fact) or MOOD_REPLIES["bored"])
        if k == "clarify":
            return self._clarify()
        if k == "duration":
            mins = int(it.filters.get("max_runtime") or 0)
            if ctx.get("filters") or ctx.get("text") or ctx.get("mode") == "similar":
                f = dict(ctx.get("filters") or {})
                f.pop("min_runtime", None)
                f["max_runtime"] = mins
                text = ctx.get("text") or None
                n = int(ctx.get("n") or 5)
                sort = ctx.get("sort_by", "wr")
                if ctx.get("mode") == "similar" and ctx.get("seed_rows"):
                    res = rec.recommend_similar(ctx["seed_rows"], n=n, **f)
                    return self._finish_recommend(rec, ctx, res, [], f, text, sort, n,
                                                  seeds=ctx.get("seed_rows"), mode="similar",
                                                  lead="Applying that to your last request:\n\n")
                res, dropped = self._recommend_with_relax(rec, f, text, n, sort)
                return self._finish_recommend(rec, ctx, res, dropped, f, text, sort, n,
                                              lead="Applying that to your last request:\n\n")
            return Reply(f"Under {mins} min, noted. What else — any genre or mood? "
                         "Or say *surprise me*.")

        # ---- facts about a movie ---------------------------------------------------------
        if k in ("describe", "info"):
            row = it.title_row
            if row is None:
                return Reply("Which movie? Name it and I'll tell you all about it.")
            return self._single(ctx, row, self.facts.describe(row, rec))
        if k == "fact":
            row = it.title_row
            if row is None:
                return Reply("Which movie do you mean?")
            return self._single(ctx, row, self.facts.attribute(row, it.fact or "overview"))
        if k == "compare":
            rows = [r for r in (it.rows or []) if r is not None]
            if len(rows) < 2:
                return Reply("Name two movies and I'll compare them — e.g. "
                             "*compare Inception and Interstellar*.")
            ids = [self._id(r) for r in rows]
            ctx["ids"] = ids
            ctx["seen"] = set(ctx.get("seen") or set()) | set(ids)
            ctx["last_row"] = int(rows[0])
            return Reply(self.facts.compare(rows), ids)
        if k == "why":
            return self._why(it, rec, ctx, last_ids)
        if k == "reference":
            return self._reference(it, rec, ctx, last_ids)

        # ---- keep the conversation going -------------------------------------------------
        if k == "more":
            return self._more(rec, ctx)
        if k == "refine":
            return self._refine(it, rec, ctx)
        if k == "taste":
            return self._taste(it, rec, ctx)

        # ---- the user's lists ------------------------------------------------------------
        if k in ("show_favorites", "show_watched", "for_me", "add_favorite", "mark_watched") and not username:
            return Reply(LOGIN_NEEDED)

        if k == "show_favorites" or k == "show_watched":
            kind = "favorites" if k == "show_favorites" else "watched"
            ids = [i for i in storage.get_ids(username, kind) if i in self.lk.id_to_row]
            if not ids:
                return Reply(f"Your {kind} list is empty for now.")
            return Reply(f"Your {kind} list ({len(ids)}):", ids[:10])

        if k == "for_me":
            fav_rows = self._rows(storage.get_ids(username, "favorites"))[:8]
            if not fav_rows:
                return Reply("Add a few movies to your favorites first, then I can recommend from them. ❤️")
            res = rec.recommend_similar(fav_rows, n=it.n + 3)
            names = ", ".join(self.movies.at[r, "title"] for r in fav_rows[:3])
            return self._finish_recommend(rec, ctx, res, [], {}, "", "wr", max(it.n, 5),
                                          seeds=fav_rows, mode="similar",
                                          lead=f"Based on your favorites ({names}), try these:\n\n")

        if k in ("add_favorite", "mark_watched"):
            row = it.title_row
            if it.ordinal is not None:
                ids = ctx.get("ids") or last_ids
                if not ids:
                    return Reply("Ask me for some recommendations first, then I can add one of them.")
                try:
                    row = self.lk.id_to_row[ids[it.ordinal]]
                except (IndexError, KeyError):
                    return Reply(f"I only showed {len(ids)} movies, so I can't find that one.")
            if row is None:
                return Reply(f'I couldn\'t find "{it.title_text}" in the dataset. Check the spelling?')
            m = self.movies.iloc[row]
            storage.add(username, "favorites" if k == "add_favorite" else "watched", int(m["id"]))
            what = "your favorites ❤️" if k == "add_favorite" else "your watched list ✔️"
            return Reply(f"Added **{m['title']}** ({m['release_year']}) to {what}.", [int(m["id"])])

        if k == "title_not_found":
            return Reply(f'I couldn\'t find "{it.title_text}" in the dataset. Check the spelling, '
                         "or try describing what you want (genre, year, mood).")

        if k == "title_lookup":
            row = it.title_row
            text = self.facts.describe(row, rec)
            sim = rec.recommend_similar([row], n=4)
            if len(sim):
                text += "\n\n**Similar movies:** " + ", ".join(sim["title"].head(4))
            ids = [self._id(row)] + sim["id"].tolist()
            self._remember(ctx, ids=ids)
            return Reply(text, ids)

        if k == "similar":
            row = it.title_row
            m = self.movies.iloc[row]
            res = rec.recommend_similar([row], n=it.n, **it.filters)
            lead = f"Movies similar to **{m['title']}** ({m['release_year']})"
            if it.filters:
                lead += f" — {constraints(it.filters)}"
            lead += ":\n\n"
            if not len(res):
                res = rec.recommend_similar([row], n=it.n)
                return self._finish_recommend(rec, ctx, res, [], {}, "", "wr", it.n, seeds=[row],
                                              mode="similar",
                                              lead=f"Nothing similar to **{m['title']}** matched all "
                                                   "your filters, so here are the closest ones:\n\n")
            return self._finish_recommend(rec, ctx, res, [], it.filters, "", it.sort_by, it.n,
                                          seeds=[row], mode="similar", lead=lead)

        # ---- recommend -------------------------------------------------------------------
        if it.surprise:
            pool = rec.recommend_by_filters(**it.filters, n=150, min_votes=300)
            if len(pool):
                pick = pool.sample(min(it.n, len(pool)))
                self._remember(ctx, filters=it.filters, text="", sort_by=it.sort_by, n=it.n,
                               mode="surprise", ids=pick["id"].tolist())
                return Reply("Here's a surprise pick for you 🎲", pick["id"].tolist())
        res, dropped = self._recommend_with_relax(rec, it.filters, it.text, it.n, it.sort_by)
        if not len(res):
            return Reply("I couldn't find anything for that. Try fewer filters.")
        if it.text and not it.filters and any('the theme "' in d for d in dropped):
            # The only signal was words we could not match to anything, e.g. "asdfgh qwerty".
            # Showing the same top-rated list would be guessing, so ask instead.
            return Reply(f'I couldn\'t match "{it.text}" to anything in the dataset. 🙈 Try a genre or a '
                         "mood (*something funny*, *scary movies from the 90s*), or name a movie you liked.")
        return self._finish_recommend(rec, ctx, res, dropped, it.filters, it.text, it.sort_by, it.n)

    # ------------------------------------------------------------------ follow-ups
    def _more(self, rec, ctx):
        """The next page of the last request, skipping everything already shown."""
        if not (ctx.get("ids") or ctx.get("filters") or ctx.get("text")):
            return Reply("Sure — what are you in the mood for? A genre, a mood, or a movie you liked.")
        f = dict(ctx.get("filters") or {})
        text = ctx.get("text") or None
        sort = ctx.get("sort_by", "wr")
        n = int(ctx.get("n") or 5)
        seen = set(ctx.get("seen") or set())
        want = n + len(seen) + 10

        if ctx.get("mode") == "similar" and ctx.get("seed_rows"):
            res = rec.recommend_similar(ctx["seed_rows"], n=want, **f)
            mode = "similar"
        else:
            res, _ = self._recommend_with_relax(rec, f, text, want, sort)
            mode = "filters"
        lead = "More of the same:\n\n"
        return self._finish_recommend(rec, ctx, res, [], f, text, sort, n, seeds=ctx.get("seed_rows"),
                                      lead=lead, drop_seen=True, mode=mode)

    def _refine(self, it, rec, ctx):
        """Tweak the previous request instead of starting over."""
        f = dict(ctx.get("filters") or {})
        note = ""
        kind = it.refine
        if kind == "shorter":
            cur = f.get("max_runtime")
            f.pop("min_runtime", None)
            f["max_runtime"] = max(60, int(cur * 0.8)) if cur else 105
            note = f"Shorter — under {f['max_runtime']} min. "
        elif kind == "longer":
            f.pop("max_runtime", None)
            f["min_runtime"] = 130
            note = "Longer — over 130 min. "
        elif kind == "older":
            _, b = f.get("year_range") or (1880, 2030)
            f["year_range"] = (1880, max(1920, min(int(b), 2015) - 15))
            note = f"Going older — before {f['year_range'][1] + 1}. "
        elif kind == "newer":
            a, _ = f.get("year_range") or (1880, 2030)
            f["year_range"] = (min(2010, max(int(a), 1965) + 15), 2030)
            note = f"Going newer — from {f['year_range'][0]}. "
        elif kind == "better":
            f["min_rating"] = max(float(f.get("min_rating") or 0), 7.5)
            note = "Only the better-rated ones. "
        elif kind == "fewer":
            ctx["n"] = 3
            note = "Just three. "
        elif kind == "exclude":
            ids = list(ctx.get("ids") or [])
            if ids and int(ids[0]) in self.lk.id_to_row:
                ctx["seen"] = set(ctx.get("seen") or set()) | {int(ids[0])}
                ctx["ids"] = ids[1:]
                note = f"Okay, dropping **{self.movies.at[self.lk.id_to_row[int(ids[0])], 'title']}**. "
            else:
                note = "Okay. "

        text = ctx.get("text") or None
        sort = ctx.get("sort_by", "wr")
        n = int(ctx.get("n") or 5)
        seeds = ctx.get("seed_rows")
        want = n + len(ctx.get("seen") or set()) + 5
        if ctx.get("mode") == "similar" and seeds:
            res = rec.recommend_similar(seeds, n=want, **f)
            mode = "similar"
        else:
            res, _ = self._recommend_with_relax(rec, f, text, want, sort)
            mode = "filters"
        return self._finish_recommend(rec, ctx, res, [], f, text, sort, n, seeds=seeds, lead=note,
                                      drop_seen=True, mode=mode)

    def _taste(self, it, rec, ctx):
        """'I liked A and B' -> a profile built from those movies."""
        rows = [int(r) for r in (it.rows or []) if r is not None]
        if not rows:
            return Reply("Which one did you like? Name a movie and I'll find similar ones.")
        n = int(ctx.get("n") or 5)
        res = rec.recommend_similar(rows, n=n)
        titles = ", ".join(f"**{self.movies.at[r, 'title']}**" for r in rows[:3])
        return self._finish_recommend(rec, ctx, res, [], {}, "", "wr", n, seeds=rows, mode="similar",
                                      lead=f"Based on {titles}, try these:\n\n")

    def _why(self, it, rec, ctx, last_ids):
        """'why did you pick that' -> the reason, from the model or from what the movies share."""
        seeds = list(ctx.get("seed_rows") or [])
        row = it.title_row
        if row is None:
            ids = ctx.get("ids") or last_ids
            if not ids:
                return Reply("Ask me for a recommendation first and I'll happily explain why it's there.")
            row = self.lk.id_to_row.get(int(ids[0]))
        if row is None:
            return Reply("I'm not sure which movie you mean — name it and I'll explain.")
        m = self.movies.iloc[row]
        if seeds:
            try:
                reason = rec.why(seeds, int(row))
            except Exception:                                 # noqa: BLE001
                reason = ""
            if reason:
                return self._single(ctx, row,
                                    f"**{m['title']}** is in that list because of {reason}.\n\n"
                                    f"Want the full picture? Say *describe {m['title']}*.")
        if seeds:
            shares = self.facts.shared(seeds, int(row))
            if shares:
                return self._single(ctx, row, f"**{m['title']}** came up because it shares "
                                              f"{', '.join(shares)} with what you asked for.")
        f = ctx.get("filters") or {}
        bits = []
        if f.get("genres"):
            bits.append("the genres")
        if f.get("year_range"):
            bits.append("the year range")
        if f.get("max_runtime") or f.get("min_runtime"):
            bits.append("the runtime")
        if f.get("min_rating"):
            bits.append("the rating")
        if len(bits) > 1:
            where = "It matches " + ", ".join(bits[:-1]) + " and " + bits[-1] + " you asked for."
        elif bits:
            where = f"It matches {bits[0]} you asked for."
        else:
            where = "It is one of the best-rated matches here."
        return self._single(ctx, row, f"**{m['title']}** is a top-rated match. {where}\n\n"
                                      + self.facts.verdict(int(row)))

    def _reference(self, it, rec, ctx, last_ids):
        """'the first one' -> talk about a card we already showed."""
        ids = ctx.get("ids") or last_ids
        if not ids:
            return Reply("I haven't recommended anything yet — ask me for a few movies first.")
        k = it.ordinal if it.ordinal is not None else 0
        if k < 0:
            k = len(ids) + k
        if not 0 <= k < len(ids):
            return Reply(f"I only showed {len(ids)} movies, so there's no number {k + 1}.")
        row = self.lk.id_to_row.get(int(ids[k]))
        if row is None:
            return Reply("I lost track of that one — name it and I'll describe it.")
        return self._single(ctx, row, self.facts.describe(row, rec))
