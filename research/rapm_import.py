"""Fetch 30-year RAPM from xrapm.com and map to project player IDs."""

import re
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.sparse import csr_matrix

RAPM_URL = "https://xrapm.com/table_pages/RAPM_30y.html"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "rapm_30y.csv"


# ── helpers ────────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    return re.sub(r"[^a-z ]", "", nfkd.encode("ascii", "ignore").decode().lower()).strip()


def _parse_stat(s) -> tuple:
    m = re.match(r"(-?[\d.]+)\s*(?:\((\d+)\))?", str(s).strip())
    return (float(m.group(1)), int(m.group(2)) if m.group(2) else None) if m else (None, None)


def _bigrams(name: str) -> Counter:
    s = f" {name} "
    return Counter(s[i:i+2] for i in range(len(s) - 1))


def _build_matrix(names: list[str], vocab: dict) -> csr_matrix:
    rows, cols, data = [], [], []
    for i, name in enumerate(names):
        bg = _bigrams(name)
        for token, count in bg.items():
            if token in vocab:
                rows.append(i)
                cols.append(vocab[token])
                data.append(count)
    mat = csr_matrix((data, (rows, cols)), shape=(len(names), len(vocab)), dtype=np.float32)
    norms = np.sqrt(mat.power(2).sum(axis=1).A1)
    norms[norms == 0] = 1.0
    return mat.multiply(1.0 / norms[:, None])


# ── fetch ──────────────────────────────────────────────────────────────────────

class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell, self._in_cell = [], [], "", False
        self.id_by_name: dict[str, int] = {}
        self._href = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell, self._in_cell = "", True
        elif tag == "a":
            self._href = a.get("href", "")

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._row.append(self._cell.strip())
            self._in_cell = False
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = []
        elif tag == "a":
            self._href = None

    def handle_data(self, data):
        if self._in_cell:
            self._cell += data
            if self._href:
                m = re.search(r"player_pages/(\d+)\.html", self._href)
                if m:
                    self.id_by_name[self._cell.strip()] = int(m.group(1))


def fetch_rapm() -> pd.DataFrame:
    resp = requests.get(RAPM_URL, timeout=30)
    resp.encoding = "utf-8"

    parser = _TableParser()
    parser.feed(resp.text)

    rows = [r for r in parser.rows if len(r) >= 7 and r[0] not in ("", "Player")]
    df = pd.DataFrame(rows, columns=[
        "player_name", "offense", "offense_se",
        "defense", "defense_se", "total", "total_se"
    ])

    for col in ["offense", "defense", "total"]:
        parsed = df[col].apply(_parse_stat)
        df[col] = parsed.apply(lambda x: x[0])
        df[f"{col}_pct"] = parsed.apply(lambda x: x[1])
        df[f"{col}_se"] = pd.to_numeric(df[f"{col}_se"], errors="coerce")

    df["xrapm_id"] = df["player_name"].map(parser.id_by_name).astype("Int64")
    return df.reset_index(drop=True)


# ── player lookup ──────────────────────────────────────────────────────────────

def load_players() -> pd.DataFrame:
    frames = [
        pd.read_csv(p, usecols=["PLAYER_ID", "PLAYER_NAME"], low_memory=False)
        for p in sorted(DATA_DIR.glob("*_Regular_players.csv"))
    ]
    df = pd.concat(frames).drop_duplicates("PLAYER_ID").reset_index(drop=True)
    df["name_norm"] = df["PLAYER_NAME"].apply(_norm)
    return df


# ── matching ───────────────────────────────────────────────────────────────────

def match_players(rapm: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    rapm_norms  = [_norm(n) for n in rapm["player_name"]]
    db_norms    = players["name_norm"].tolist()
    db_ids      = players["PLAYER_ID"].tolist()
    db_names    = players["PLAYER_NAME"].tolist()

    # shared vocabulary across both name sets
    all_tokens: set[str] = set()
    for name in rapm_norms + db_norms:
        s = f" {name} "
        all_tokens.update(s[i:i+2] for i in range(len(s) - 1))
    vocab = {t: i for i, t in enumerate(sorted(all_tokens))}

    rapm_mat = _build_matrix(rapm_norms, vocab)
    db_mat   = _build_matrix(db_norms, vocab)

    # shape: (n_rapm, n_db)
    sim = (rapm_mat @ db_mat.T).toarray()

    top2 = np.argsort(sim, axis=1)[:, -2:][:, ::-1]

    name_match_id, name_match_name, name_match_name_2nd, cos1, cos2 = [], [], [], [], []
    for i, idxs in enumerate(top2):
        name_match_id.append(db_ids[idxs[0]])
        name_match_name.append(db_names[idxs[0]])
        name_match_name_2nd.append(db_names[idxs[1]] if len(idxs) > 1 else "")
        cos1.append(round(float(sim[i, idxs[0]]), 3))
        cos2.append(round(float(sim[i, idxs[1]]), 3) if len(idxs) > 1 else 0.0)

    cos1_arr = np.array(cos1)
    cos2_arr = np.array(cos2)
    denom = cos1_arr + cos2_arr
    name_signal = np.where(denom > 0, cos1_arr / denom, 0.0).round(3).tolist()

    rapm = rapm.copy()
    rapm["name_match_id"]       = name_match_id
    rapm["name_match_name"]     = name_match_name
    rapm["name_match_name_2nd"] = name_match_name_2nd
    rapm["cos_best"]            = cos1
    rapm["cos_2nd"]             = cos2
    rapm["name_signal"]         = name_signal
    rapm["player_id"]           = [
        int(xid) if pd.notna(xid) else nm
        for xid, nm in zip(rapm["xrapm_id"], name_match_id)
    ]
    return rapm


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    rapm = fetch_rapm()
    players = load_players()
    rapm = match_players(rapm, players)

    rapm.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(rapm)} rows → {OUT_PATH}")


if __name__ == "__main__":
    main()
