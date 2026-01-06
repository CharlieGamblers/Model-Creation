# config.py  — unified paths, constants, and lookup maps
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional
import pandas as pd
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
# Repo root (…/Bull_Model)
ROOT: Path = Path(__file__).resolve().parents[1]

MODEL_DIR: Path = ROOT / "Models"
DATA_DIR: Path  = ROOT / "Data" / "Processed"
DOCS_DIR: Path  = ROOT / "Important Documents"

# Core files
FINAL_DATA: Path   = DATA_DIR / "final_data.csv"        # merged, feature-ready
MODEL_FILE: Path   = MODEL_DIR / "xgb_model.json"       # trained booster
FEATURE_LIST: Path = MODEL_DIR / "feature_cols.txt"     # fallback if model has no names

# Lookups
RIDER_XLSX: Path = DOCS_DIR / "rider_id_list.xlsx"      # legacy; unused for mapping now
BULL_XLSX: Path  = DOCS_DIR / "bull_id_list.xlsx"       # bull, bull_id (use first two columns)
RIDER_CSV: Path  = DATA_DIR / "rider_info.csv"          # rider, id (use first two columns)

# Ensure directories exist (no-op if they already do)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Modeling constants
# ─────────────────────────────────────────────────────────────────────────────
# Rider windows mirror your training setup
RIDER_LONG_DAYS: int     = 1085
RIDER_SHORT_RIDES: int   = 25

# Bull short window
BULL_SHORT: int          = 10

# Weighted rider QRP: W1 * short + W2 * long
W1: float = 0.15
W2: float = 1.0 - W1

# Smoothing prior for league-wide means
K_PRIOR: int = 10

# Default event date used by batch runners (can be overridden)
DEFAULT_DATE: str = "2025-12-11"

# ─────────────────────────────────────────────────────────────────────────────
# Branding / UI
# ─────────────────────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "AUS": "#66cc33",
    "KC" : "#E65300",
    "OK" : "#977224",
    "TX" : "#D18316",
    "NSH": "#103595",
    "FL" : "#0b2240",
    "NY" : "#0f034e",
    "CAR": "#00a8e1",
    "AZ" : "#1e2c2d",
    "MO" : "#ff2822",
}

# ─────────────────────────────────────────────────────────────────────────────
# Normalization / replacement maps for input cleanup
# ─────────────────────────────────────────────────────────────────────────────
replacements = pd.DataFrame({
    "Known_Discrepancies": [
        "017 optiwize mercy","190 Horizons Creedmore","135 itchy scratchy","P15 Play Time","25H Nervous Hazard","42G Blood Train","949 Smokin' Joe","949 Puckered Up","riquelmi santos","04-k shank","23 Riff Ram","2102 Big Jake","K shank", "3 Always Been Crazy","806 Lari's Speck",
        "32G Zapata","4 malibu's top dawg","8-9 After Party","927 boot barn's sandman","001 boot barn's pack rat","915 bex's red eye","070 Cajun Brute","Say When","07 Night Fury","J79 Ridin' Dirty","920 BRUNT Haymaker","20 Mississippi Mercenary","17 Buffalo Heifer","011 boot barn's tyrone",
        "121 Judgement Day","I17 Ridin' Salty",
        "f51 mr. bojangles","25g bon jovi","792 Red Headeded Stranger",
        "16J Herded.com The Gambler","18K Kippers Rippers",
        "834F Still Flyin' Crazy","101 Boot Barn's Skunk Kitty","121 Judgement Day","Jean Carlos Teodoro",
        "9504 Erner Permer","2115 Savage mode","79 Double R Hat House Pepper Shaker","8X2 Dang, Dang",
        "31 Warpaint","W150 Pay Pal","037 Mr. Demon","626 Schott In The Dark","04 Midnight Blues",
        "128J Satans Love","1Z Bittersweet","113 BuzzBallz",
        "P829 Rafter P Construction's Smooth Over It","I17 Ridin' Salty","524 Cliff Hanger","804 Time For Magic",
        "89 Double P","902 Firecracker","34E Juju","2/6 Cherry Bomb","CMC08 Buckshot","H23 Tank",
        "705 Joy's Bang Bang","R2 Hank The Tank", "8518 Pneu Dart's Gold Standard", "A141 Pneu-Dart's Chief Wahoo",
        "11-16 Whatchamacallit", "803 Ram Rod", "9 Nobody", "031 Nirvana", "RW711 War Daddy",
        "923 Western Haulers Starburst", "964 Jag Metals Tucker Brown", "03 Jag Metals Domestic Violence",
        "037 Jag Metals Soul Go", "92 Windmill", "625 Fierce's Dirtnap", "RW-711 War Daddy",
        "916 Chico", "172 Hang 'em High", "15 Redwing", "8G Danger Central", "36F Off The Rails",
        "527 Preachers Kid", "830 Ram Rod", "99H High Ball", "29G Hard Core Slinger", "019 Sullivan", "192 Snoop Dog",
        "224 Willie", "498 Sharp Shooter", "042 Cliff Hanger", "820 Feeds Red River", "Sage Steele Kimzey",
        "191B Younts Brody's Pet", "626 Chateau Montelena's Montana Jacket", "Trey Benton Iii"
    ],
    "Replacements": [
        "017 mercy","190 Horizon's 22 Creedmore","135 itchy & scratchy","P15 Playtime","25H Nervous Hospital","42/G Blood Train","949 Smoking Joe","-949 Puckered Up","riquelme santos","04-K One Shank","23 Toad","B2102 Big Jake","04-K One Shank","H2-3 Always Been Crazy","-806 Lari's Speck",
        "32G Blackjack","4 malibu's top dog","8-9 Joe Cool","927 Sandman","001 Pack Rat","915 Bex Red Eye","-070 Cajun Brute","7- Say When","07 Kunta","J79 Riding Dirty","920 Haymaker","21-20 Mississippi Mercenary","-17 Buffalo Heifer","011 Tyrone",
        "-121 Judgement Day","I17 Ridin Salty","F51 Split Decision","25G Sizzled",
        "792 Red Headed Stranger",
        "16J The Gambler","18K Kipper's Ripper","834F Still Flyin Crazy","101 Skunk Kitty","-121 Judgement Day",
        "jean carlos de souza","9504 Arnold Palmer","2115 Savage Mode","79 Pepper Shaker","8X2 Dang Dang",
        "31 War Paint","W150 Paypal","037 Mr Demon","626 Schott in the Dark","0-4 Midnight Blues",
        "128J Satan's Love","12 Bittersweet","113 Buzz Ballz","P829 Smooth Over It",
        "I17 Ridin Salty","524 Cliffhanger","804 Time for Magic","-A-89 Double P","902 Fire Cracker","34E JuJu",
        "2-6 Cherry Bomb","08 Buckshot","H23 *","705 Bang Bang","R2 Hank the Tank",
        "8518 Gold Standard", "141 Chief Wahoo", "1116 Whatchamacallit", "803 Ramrod", "9C Nobody", "031 *",
        "711 War Daddy", "923 Starburst", "964 Tucker Brown", "03 Domestic Violence",
        "037 JAG Metals Soul Glo", "92 Original Windmill Fanatic", "625 Fierce's Dirt Nap",
        "711 War Daddy", "-916 Chico", "172 Hang 'Em High", "15 Red Wing", "8G Damage Control", "36F Off the Rails",
        "527 Preacher's Kid", "830 Ramrod", "99H Highball", "29G Hardcore Slinger", "019H Sullivan", "192 Snoop Dogg",
        "4-224 Willie", "498 Sharpshooter", "042 Cliffhanger", "820 Red River", "Sage Kimzey",
        "191B Yount's Brody's Pet", "626 Montana Jacket", "Trey Benton"
    ]
})

rider_replacements = {
    "sage steele kimzey": "sage kimzey",
    "Ramon Fiorini": "ramon fiorni de souza",
    "riquelme santos": "riquelme de souza",

}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_model(path: Path = MODEL_FILE) -> xgb.Booster:
    """Load the XGBoost booster."""
    bst = xgb.Booster()
    bst.load_model(str(path))
    return bst

def model_feature_names(
    booster: xgb.Booster,
    fallback_file: Path = FEATURE_LIST
) -> List[str]:
    """
    Return the feature names the model expects.
    Prefer names embedded in the model; fall back to FEATURE_LIST.
    """
    names: Optional[Iterable[str]] = getattr(booster, "feature_names", None)
    if names:
        return list(names)
    # fallback to plain text file
    if fallback_file.exists():
        return [ln.strip() for ln in fallback_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # ultimate fallback: empty list (caller must handle)
    return []

__all__ = [
    "ROOT", "MODEL_DIR", "DATA_DIR", "DOCS_DIR",
    "FINAL_DATA", "MODEL_FILE", "FEATURE_LIST",
    "RIDER_XLSX", "BULL_XLSX", "RIDER_CSV",
    "RIDER_LONG_DAYS", "RIDER_SHORT_RIDES",
    "BULL_SHORT", "W1", "W2", "K_PRIOR", "DEFAULT_DATE",
    "TEAM_COLORS", "replacements", "rider_replacements",
    "load_model", "model_feature_names",
]
