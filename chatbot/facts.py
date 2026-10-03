"""Data-grounded answers about single movies: descriptions, facts, verdicts and comparisons.

Everything here is read from the cleaned table the app already ships, so the bot can talk about a
film without inventing anything: if a field is missing from the data it says so instead of guessing.

Used by ``chatbot/engine.py``. Nothing in this module needs the network or a model.
"""
import numpy as np
import pandas as pd

MAX_OVERVIEW = 900            # characters kept from an overview before we cut it at a word boundary


# --------------------------------------------------------------------------- formatting helpers
def fmt_runtime(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    mins = int(round(float(v)))
    h, m = divmod(mins, 60)
    return f"{mins} min" + (f" ({h} h {m:02d} min)" if h else "")


def fmt_money(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    v = float(v)
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:.0f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}k"
    return f"${v:,.0f}"


def fmt_int(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return f"{int(v):,}"


def _cut(text, limit=MAX_OVERVIEW):
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


class Facts:
    """Answers about one movie (or a few), plus the context numbers they need."""

    def __init__(self, movies: pd.DataFrame):
        self.m = movies
        self.n = len(movies)
        self.id_to_row = {int(i): r for r, i in enumerate(movies["id"])}
        self._wr = np.sort(movies["wr"].fillna(0).to_numpy(float))
        self._pop = np.sort(movies["popularity"].fillna(0).to_numpy(float))
        self._runtime = np.sort(movies["runtime"].dropna().to_numpy(float))
        exploded = movies[["genres_l", "wr"]].explode("genres_l").dropna(subset=["genres_l"])
        self.genre_wr = exploded.groupby("genres_l")["wr"].mean().to_dict()
        self.genre_n = exploded.groupby("genres_l")["wr"].size().to_dict()

    # ---------------------------------------------------------------- small numbers
    def top_pct(self, value, sorted_values):
        """Where `value` sits: 1 means "top 1%", 50 means the median."""
        if value is None or not np.isfinite(value) or not len(sorted_values):
            return None
        better = len(sorted_values) - int(np.searchsorted(sorted_values, value, side="left"))
        return max(1, int(round(100 * better / len(sorted_values))))

    def _cell(self, row, col):
        v = self.m.at[row, col]
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    # ---------------------------------------------------------------- one-liners
    def title(self, row):
        y = self._cell(row, "release_year")
        return f"**{self.m.at[row, 'title']}**" + (f" ({int(y)})" if y is not None else "")

    def one_line(self, row):
        bits = [self.title(row)]
        va = self._cell(row, "vote_average")
        vc = self._cell(row, "vote_count")
        if va is not None:
            bits.append(f"⭐ {float(va):.1f}" + (f" ({fmt_int(vc)} votes)" if vc else ""))
        rt = fmt_runtime(self._cell(row, "runtime"))
        if rt:
            bits.append(rt)
        g = self.m.at[row, "genres"]
        if g:
            bits.append(str(g))
        return " · ".join(bits)

    def cast_list(self, row, n=6):
        cast = [c.strip() for c in str(self.m.at[row, "cast"]).split(",") if c.strip()]
        return cast[:n]

    def similar_titles(self, row, rec, n=3):
        """A short 'if you like it' line, using whichever model is selected."""
        if rec is None:
            return None
        try:
            res = rec.recommend_similar([int(row)], n=n)
        except Exception:                                     # noqa: BLE001 - never break a reply
            return None
        if not len(res):
            return None
        return ", ".join(str(t) for t in res["title"].head(n))

    # ---------------------------------------------------------------- the main answer
    def describe(self, row, rec=None, with_similar=True):
        m = self.m
        out = [self.one_line(row)]

        director = str(m.at[row, "directors"] or "").strip()
        lang = m.at[row, "language"]
        tail = []
        if director:
            tail.append(f"directed by **{director}**")
        if lang:
            tail.append(str(lang))
        if tail:
            out.append("*" + " · ".join(tail) + "*")

        tagline = str(m.at[row, "tagline"] or "").strip()
        if tagline:
            out.append(f'*"{tagline}"*')

        overview = _cut(m.at[row, "overview"])
        if overview:
            out.append(overview)
        else:
            out.append("_No overview is recorded for this movie._")

        cast = self.cast_list(row)
        if cast:
            out.append("**Cast:** " + ", ".join(cast))

        kw = [k.strip() for k in str(m.at[row, "keywords"] or "").split(",") if k.strip()][:8]
        if kw:
            out.append("**Themes:** " + ", ".join(kw))

        money = []
        b, r = fmt_money(self._cell(row, "budget")), fmt_money(self._cell(row, "revenue"))
        if b:
            money.append(f"budget {b}")
        if r:
            money.append(f"revenue {r}")
        if money:
            out.append("**Money:** " + " · ".join(money))

        imdb, imdb_votes = self._cell(row, "averageRating"), self._cell(row, "numVotes")
        if imdb is not None and np.isfinite(float(imdb)):
            out.append(f"**IMDb:** {float(imdb):.1f}/10" + (f" from {fmt_int(imdb_votes)} votes"
                                                            if imdb_votes else ""))

        wr = float(m.at[row, "wr"] or 0)
        pct = self.top_pct(wr, self._wr)
        if pct is not None:
            out.append(f"_Weighted rating {wr:.2f} — the **top {pct}%** of the {self.n:,} movies here._")

        if with_similar:
            sim = self.similar_titles(row, rec)
            if sim:
                out.append(f"**If you like it:** {sim}")
        return "\n\n".join(x for x in out if x)

    def verdict(self, row):
        """'Is it any good?' — answered with the weighted rating, its percentile and the genre average."""
        m = self.m
        va, vc = self._cell(row, "vote_average"), self._cell(row, "vote_count")
        wr = float(m.at[row, "wr"] or 0)
        pct = self.top_pct(wr, self._wr)
        out = [self.title(row) + (f" — ⭐ **{float(va):.1f}/10**" if va is not None else "")
               + (f" from {fmt_int(vc)} votes" if vc else "") + f" (weighted {wr:.2f})."]

        genres = [g for g in (m.at[row, "genres"] or "").split(", ") if g]
        ctx = []
        if pct is not None:
            ctx.append(f"in the **top {pct}%** of the {self.n:,} movies here")
        if genres:
            g = genres[0]
            gm = self.genre_wr.get(g.lower())
            if gm is not None:
                ctx.append(f"{'above' if wr > gm else 'below'} the average for {g} ({gm:.2f})")
        if ctx:
            out.append("It sits " + " and ".join(ctx) + ".")
        imdb = self._cell(row, "averageRating")
        if imdb is not None and np.isfinite(float(imdb)):
            out.append(f"IMDb agrees: {float(imdb):.1f}/10.")
        if pct is None:
            verdict = "Not enough votes here to judge it."
        elif pct <= 5:
            verdict = "**Verdict:** a safe bet — widely loved, not a niche pick."
        elif pct <= 20:
            verdict = "**Verdict:** solid — most people who watch it like it."
        elif pct <= 50:
            verdict = "**Verdict:** decent, but not a standout. Worth it if the genre is your thing."
        else:
            verdict = ("**Verdict:** mixed — go in for the cast, the director or the genre rather than the rating.")
        out.append(verdict)
        return "\n\n".join(out)

    # ---------------------------------------------------------------- single facts
    def attribute(self, row, what):
        m = self.m
        title = self.title(row)
        if what == "year":
            date = pd.to_datetime(m.at[row, "release_date"], errors="coerce")
            y = self._cell(row, "release_year")
            if y is None:
                return f"The release year of {title} is not recorded here."
            full = f" (released {date.strftime('%d %B %Y')})" if pd.notna(date) else ""
            return f"{title} came out in **{int(y)}**{full}."
        if what == "runtime":
            rt = fmt_runtime(self._cell(row, "runtime"))
            if not rt:
                return f"The runtime of {title} is not recorded here."
            extra = ""
            pct = self.top_pct(self._cell(row, "runtime"), self._runtime)
            if pct is not None and pct <= 25:
                extra = f" That is on the long side — only {pct}% of the movies here are longer."
            elif pct is not None and pct >= 80:
                extra = f" That is short — {100 - pct}% of the movies here are longer."
            return f"{title} runs **{rt}**.{extra}"
        if what == "genre":
            g = m.at[row, "genres"]
            if not g:
                return f"No genre is recorded for {title}."
            return f"{title} is **{g}**."
        if what == "director":
            d = str(m.at[row, "directors"] or "").strip()
            if not d:
                return f"No director is recorded for {title}."
            others = self._others_by("directors", d, row)
            return f"{title} was directed by **{d}**." + (f" They also directed {others} here." if others else "")
        if what == "cast":
            cast = self.cast_list(row, 5)
            if not cast:
                return f"No cast is recorded for {title}."
            return f"{title} stars **{', '.join(cast)}**."
        if what == "who_plays":
            # the dataset has no role names, only the billed cast - say so instead of guessing
            cast = self.cast_list(row, 5)
            if not cast:
                return f"No cast is recorded for {title}."
            return (f"The data here has no role names, only the billed cast: **{', '.join(cast)}**. "
                    f"IMDb has the full credit list for {title}.")
        if what == "language":
            lang = m.at[row, "language"]
            orig = m.at[row, "original_language"]
            if not lang:
                return f"The language of {title} is not recorded here."
            return f"{title} is in **{lang}**" + (f" (original language code `{orig}`)." if orig else ".")
        if what == "country":
            c = str(m.at[row, "production_countries"] or "").strip()
            if not c:
                return f"No production country is recorded for {title}."
            return f"{title} was produced in **{c}**."
        if what == "rating":
            return self.verdict(row)
        if what == "budget":
            b = fmt_money(self._cell(row, "budget"))
            if not b:
                return f"A budget is not recorded for {title}."
            r = fmt_money(self._cell(row, "revenue"))
            out = f"{title} had a budget of **{b}**."
            if r:
                rev = float(self._cell(row, "revenue"))
                bud = float(self._cell(row, "budget"))
                out += f" It took **{r}**, about **{rev / bud:.1f}×** the budget."
            return out
        if what == "tagline":
            tg = str(m.at[row, "tagline"] or "").strip()
            if not tg:
                return f"No tagline is recorded for {title}."
            return f'{title} — *"{tg}"*'
        if what == "keywords":
            kw = [k.strip() for k in str(m.at[row, "keywords"] or "").split(",") if k.strip()][:10]
            if not kw:
                return f"No keywords are recorded for {title}."
            return f"Themes in {title}: " + ", ".join(f"*{k}*" for k in kw) + "."
        if what == "popularity":
            pop = self._cell(row, "popularity")
            if pop is None:
                return f"No popularity score is recorded for {title}."
            pct = self.top_pct(pop, self._pop)
            return (f"{title} has a popularity score of **{float(pop):.0f}**"
                    + (f" — the top {pct}% of the movies here." if pct is not None else "."))
        return self.describe(row)

    def _others_by(self, col, name, row, n=3):
        """Other movies in the dataset by the same person, most voted first."""
        hit = self.m[self.m[col].str.contains(name, case=False, regex=False)]
        hit = hit[hit.index != row].sort_values("vote_count", ascending=False).head(n)
        return ", ".join(f"{t} ({int(y)})" if pd.notna(y) else str(t)
                         for t, y in zip(hit["title"], hit["release_year"])) if len(hit) else ""

    # ---------------------------------------------------------------- two movies
    def compare(self, rows):
        rows = list(dict.fromkeys(int(r) for r in rows))[:3]
        if len(rows) < 2:
            return "I need two movies to compare — try *compare Inception and Interstellar*."
        m = self.m
        cols = [self.title(r) for r in rows]

        def line(label, values):
            return f"| {label} | " + " | ".join(values) + " |"

        def val(r, key, fmt=None):
            v = self._cell(r, key)
            if v is None:
                return "—"
            return fmt(v) if fmt else str(v)

        out = [line("", cols), "|---|" + "---|" * len(rows)]
        out.append(line("Year", [val(r, "release_year", lambda v: str(int(v))) for r in rows]))
        out.append(line("Runtime", [fmt_runtime(self._cell(r, "runtime")) or "—" for r in rows]))
        out.append(line("Genres", [val(r, "genres") for r in rows]))
        out.append(line("Rating", [("⭐ " + f"{float(self._cell(r, 'vote_average')):.1f}"
                                    + (f" ({fmt_int(self._cell(r, 'vote_count'))})" if self._cell(r, "vote_count") else ""))
                                   if self._cell(r, "vote_average") is not None else "—" for r in rows]))
        out.append(line("Weighted", [(f"{float(m.at[r, 'wr']):.2f} (top {self.top_pct(float(m.at[r, 'wr']), self._wr)}%)")
                                     for r in rows]))
        out.append(line("Director", [val(r, "directors") for r in rows]))
        out.append(line("Budget", [fmt_money(self._cell(r, "budget")) or "—" for r in rows]))
        out.append(line("IMDb", [f"{float(self._cell(r, 'averageRating')):.1f}" if self._cell(r, "averageRating") is not None
                                 else "—" for r in rows]))

        # a short, data-grounded verdict
        best = max(rows, key=lambda r: float(m.at[r, "wr"] or 0))
        worst = min(rows, key=lambda r: float(m.at[r, "wr"] or 0))
        notes = []
        if best != worst:
            notes.append(f"**{m.at[best, 'title']}** scores higher ({float(m.at[best, 'wr']):.2f} vs "
                         f"{float(m.at[worst, 'wr']):.2f} weighted)")
        runs = {r: self._cell(r, "runtime") for r in rows if self._cell(r, "runtime") is not None}
        if len(runs) >= 2:
            short, long_ = min(runs, key=runs.get), max(runs, key=runs.get)
            diff = int(runs[long_] - runs[short])
            if diff >= 10:
                notes.append(f"**{m.at[short, 'title']}** is {diff} min shorter")
        shared_g = set(m.at[rows[0], "genres_l"]) & set(m.at[rows[1], "genres_l"])
        shared_c = (set(self.cast_list(rows[0], 8)) & set(self.cast_list(rows[1], 8)))
        shared_d = (set(str(m.at[rows[0], "directors"]).split(", "))
                    & set(str(m.at[rows[1], "directors"]).split(", "))) - {""}
        shares = []
        if shared_g:
            shares.append("the genres " + ", ".join(sorted(shared_g)))
        if shared_c:
            shares.append("the cast (" + ", ".join(sorted(shared_c)) + ")")
        if shared_d:
            shares.append("the director (" + ", ".join(sorted(shared_d)) + ")")
        body = []
        if notes:
            body.append("Comparing them: " + ", and ".join(notes) + ".")
        body.append("They share " + "; ".join(shares) + "." if shares else "They share no genre, director or top cast.")
        if best != worst:
            body.append(f"**If you have to pick one:** {m.at[best, 'title']} — the higher weighted rating here.")
        return "\n\n".join(out) + "\n\n" + " ".join(body)

    # ---------------------------------------------------------------- why was this picked
    def shared(self, seed_rows, row, n=3):
        """What `row` has in common with the movies the user liked (for 'why did you pick that')."""
        m = self.m
        out = []
        genres = {g.lower() for g in m.at[row, "genres_l"]}
        seed_g = set().union(*[set(g.lower() for g in m.at[s, "genres_l"]) for s in seed_rows]) if seed_rows else set()
        g = sorted(genres & seed_g)
        if g:
            out.append("the genres " + ", ".join(x.title() for x in g[:n]))
        seed_d = set().union(*[set(str(m.at[s, "directors"]).split(", ")) for s in seed_rows]) - {""} if seed_rows else set()
        d = sorted((set(str(m.at[row, "directors"]).split(", ")) - {""}) & seed_d)
        if d:
            out.append("the director " + ", ".join(d))
        seed_c = set().union(*[set(self.cast_list(s, 6)) for s in seed_rows]) if seed_rows else set()
        c = sorted(set(self.cast_list(row, 6)) & seed_c)
        if c:
            out.append("the cast (" + ", ".join(c[:n]) + ")")
        seed_k = set().union(*[set(str(m.at[s, "keywords"]).lower().split(", ")) for s in seed_rows]) - {""} if seed_rows else set()
        k = sorted((set(str(m.at[row, "keywords"]).lower().split(", ")) - {""}) & seed_k)
        if k:
            out.append("similar themes (" + ", ".join(k[:n]) + ")")
        yrs = [m.at[s, "release_year"] for s in seed_rows if pd.notna(m.at[s, "release_year"])]
        y = m.at[row, "release_year"]
        if yrs and pd.notna(y) and abs(int(y) - int(np.mean([int(x) for x in yrs]))) <= 12:
            out.append(f"a close release year ({int(y)})")
        return out
