"""Fast lookups used by the parser: movie titles and people names."""
import difflib
import re
import pandas as pd
from core.data import norm_title

try:                                   # optional, much faster fuzzy matching
    from rapidfuzz import process, fuzz
except Exception:                      # noqa
    process = fuzz = None


class Lookup:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies
        best = movies.sort_values("vote_count", ascending=False).drop_duplicates("title_norm")
        self.title_row = {t: i for t, i in zip(best["title_norm"], best.index) if t}
        self._titles = list(self.title_row)
        self.id_to_row = {int(i): r for r, i in enumerate(movies["id"])}
        self.dir_names, self.dir_count = self._people("directors")
        self.cast_names, self.cast_count = self._people("cast")
        self.languages = {l.lower(): l for l in movies["language"].unique()}

    def _people(self, col):
        s = self.movies[col].str.split(",").explode().str.strip()
        s = s[s != ""]
        low = s.str.lower()
        canon = dict(zip(low, s))
        counts = low.value_counts().to_dict()
        return canon, counts

    # ---- titles -------------------------------------------------------------------
    def exact_title(self, text):
        return self.title_row.get(norm_title(text))

    def fuzzy_title(self, text, cutoff=88):
        q = norm_title(text)
        if len(q) < 4:
            return None
        if process is not None:
            hit = process.extractOne(q, self._titles, scorer=fuzz.ratio, score_cutoff=cutoff)
            return self.title_row[hit[0]] if hit else None
        hit = difflib.get_close_matches(q, self._titles, n=1, cutoff=cutoff / 100)
        return self.title_row[hit[0]] if hit else None

    def title_in_text(self, text):
        """Longest prefix of `text` that is a known title. Returns (row, matched_text) or (None, '')."""
        words = text.split()
        for k in range(len(words), 0, -1):
            chunk = " ".join(words[:k])
            r = self.exact_title(chunk)
            if r is not None:
                return r, chunk
        return None, ""

    # ---- people -------------------------------------------------------------------
    def person_prefix(self, words, kind):
        """Longest prefix (up to 4 words) of `words` that is a known person of that kind."""
        names = self.dir_names if kind == "director" else self.cast_names
        for k in range(min(4, len(words)), 0, -1):
            cand = " ".join(words[:k]).strip(" .,'")
            if " " in cand and cand in names:        # need first + last name
                return names[cand], k
        return None, 0

    def person_anywhere(self, tokens):
        """Find a 'First Last' that is a known person anywhere in the token list."""
        for k in (3, 2):
            for i in range(len(tokens) - k + 1):
                cand = " ".join(tokens[i:i + k])
                d, c = self.dir_count.get(cand, 0), self.cast_count.get(cand, 0)
                if d or c:
                    if d >= c:
                        return "director", self.dir_names[cand], (i, i + k)
                    return "actor", self.cast_names[cand], (i, i + k)
        return None
