"""Noms lisibles et FICTIFS pour les projets (« Cabinet dentaire Nguyen ») à partir de l'activité réelle (NAICS).

Les noms d'emprunteurs réels ne sont jamais utilisés (retirés dès data_prep.py) : seul le code d'activité sert,
le nom de famille est tiré d'une liste fixe. Déterministe : le même identifiant donne toujours le même nom.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

# Préfixe NAICS (le plus long qui correspond l'emporte) -> activités possibles
ACTIVITES = {
    "722511": ["Restaurant", "Brasserie", "Bistrot"], "722513": ["Pizzeria", "Burger bar", "Snack"],
    "7225": ["Restaurant"], "722": ["Café-restaurant"],
    "621112": ["Cabinet de spécialistes"], "6211": ["Cabinet médical"], "621210": ["Cabinet dentaire"],
    "621399": ["Centre de soins paramédicaux", "Cabinet de kinésithérapie"], "6221": ["Clinique privée"],
    "624310": ["Atelier d'insertion professionnelle"], "62": ["Centre de santé"],
    "484110": ["Transports locaux"], "484121": ["Transports routiers", "Fret longue distance"], "48": ["Logistique"],
    "812210": ["Pompes funèbres"], "812199": ["Institut de bien-être"], "8121": ["Institut de beauté"],
    "811192": ["Station de lavage"], "8111": ["Garage automobile"], "81": ["Services de proximité"],
    "541330": ["Bureau d'ingénierie"], "541611": ["Cabinet de conseil"], "541430": ["Studio de design graphique"],
    "541511": ["Studio de développement logiciel"], "54": ["Cabinet d'expertise"],
    "561730": ["Paysagiste"], "56": ["Services aux entreprises"],
    "332992": ["Atelier de métallurgie"], "33": ["Atelier de fabrication"], "312120": ["Brasserie artisanale"],
    "31": ["Atelier agroalimentaire"],
    "238190": ["Entreprise de gros œuvre"], "238990": ["Entreprise de travaux spécialisés"],
    "237310": ["Travaux routiers"], "236115": ["Constructeur de maisons"], "23": ["Entreprise du bâtiment"],
    "424930": ["Grossiste en fleurs"], "42": ["Grossiste"],
    "531130": ["Garde-meubles"], "531120": ["Locaux commerciaux"], "53": ["Agence immobilière"],
    "523930": ["Conseil en investissement"], "524210": ["Agence d'assurances"], "52": ["Courtier"],
    "445310": ["Cave à vins"], "441222": ["Concession nautique"], "448150": ["Boutique d'accessoires"],
    "44": ["Commerce de détail"], "45": ["Boutique"],
    "113310": ["Exploitation forestière"], "11": ["Exploitation agricole"],
    "71": ["Salle de sport"], "61": ["Centre de formation"], "51": ["Studio numérique"],
}
NOMS = ["Miller", "Nguyen", "Garcia", "Patel", "O'Brien", "Kowalski", "Rossi", "Hernandez", "Kim", "Brooks",
        "Carter", "Ramirez", "Walsh", "Schmidt", "Dubois", "Okafor", "Silva", "Chen", "Larsen", "Murphy",
        "Rivera", "Cohen", "Bennett", "Foster", "Delgado", "Novak", "Hayes", "Moreau", "Tanaka", "Price",
        "Lopez", "Reyes", "Sullivan", "Baker", "Wright", "Ortiz", "Fischer", "Grant", "Romano", "Park",
        "Ward", "Cruz", "Kelly", "Bianchi", "Sato", "Morgan", "Hughes", "Flores", "Ellis", "Jensen"]


def _h(s: str) -> int:
    return int(hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:8], 16)


def activite(naics: object, key: str = "") -> str:
    """Activité lisible ; « key » fait varier la variante entre deux projets du même code."""
    code = "" if pd.isna(naics) else str(naics).split(".")[0]
    for k in range(len(code), 1, -1):
        if code[:k] in ACTIVITES:
            opts = ACTIVITES[code[:k]]
            return opts[_h(code + key) % len(opts)]
    return "Entreprise"


def human_names(ids: pd.Series, naics: pd.Series | None = None) -> pd.Series:
    """Un nom unique par identifiant : activité (si NAICS connu) + nom de famille fictif (sans répétition)."""
    ids = pd.Series(ids).astype(str).reset_index(drop=True)
    naics = None if naics is None else pd.Series(naics).reset_index(drop=True)
    order = np.argsort([_h(i) for i in ids])                    # ordre stable, indépendant de l'ordre du fichier
    names = [""] * len(ids)
    for k, idx in enumerate(order):
        fam = NOMS[k % len(NOMS)] + ("" if k < len(NOMS) else f" {k // len(NOMS) + 1}")
        act = activite(naics.iloc[idx], ids.iloc[idx]) if naics is not None else "Projet"
        names[idx] = f"{act} {fam}"
    return pd.Series(names)
