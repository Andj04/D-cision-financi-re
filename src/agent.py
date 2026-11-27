"""Étape 9 : agent rédacteur. Il LIT les résultats déjà calculés et rédige la note ; il ne calcule jamais.

Garde-fous :
  * le prompt ne contient que des chiffres déjà formatés (« faits ») que le modèle doit recopier ;
  * un vérificateur contrôle que chaque nombre de la note figure dans les faits ; sinon la note du LLM est
    rejetée et remplacée par la note de gabarit (déterministe, sans LLM) ;
  * la clé d'API n'est lue que dans l'environnement ou st.secrets (jamais dans le dépôt) ; sans clé : gabarit.

Fournisseurs gratuits pris en charge (REST, sans SDK) : Google Gemini (par défaut) et Groq (API compatible OpenAI).
Variables : LLM_PROVIDER = gemini | groq ; GEMINI_API_KEY ou GROQ_API_KEY ; LLM_MODEL (optionnel).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

# « gemini-flash-latest » : alias Google qui suit le dernier modèle Flash (les noms datés sont retirés au fil du temps)
DEFAULT_MODELS = {"gemini": "gemini-flash-latest", "groq": "llama-3.3-70b-versatile"}

SYSTEM_PROMPT = (
    "Tu es l'agent rédacteur d'un comité d'investissement bancaire. Tu rédiges en français une note courte "
    "(6 à 9 phrases) sur UN projet. RÈGLES STRICTES : (1) tu n'effectues aucun calcul ; (2) tu recopies les "
    "chiffres EXACTEMENT comme dans la liste FAITS, sans en créer, arrondir ou déduire d'autres ; "
    "(3) tu n'inventes aucun fait absent de la liste ; (4) tu indiques la catégorie, les raisons principales, "
    "les critères éliminatoires déclenchés s'il y en a, et la fragilité de la décision ; (5) tu rappelles que la "
    "décision finale revient au comité ; (6) pas de recommandation d'achat/vente, pas de promesse de rendement ; "
    "(7) écris en français naturel : n'emploie jamais les noms techniques des clés (ex. « rang_electre3 », "
    "« motif_ecarte »), désigne le projet par son nom et n'écris pas son identifiant."
)


# --------------------------------------------------------------------------------------- faits
def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}".replace(",", " ")


def _demo_criteria() -> list[dict]:
    from engine import DEMO_LABELS
    return [{"colonne": c, "libelle": lab, "format": f} for c, (lab, f) in DEMO_LABELS.items()]


def _text(v) -> str | None:
    return v if isinstance(v, str) and v and v.lower() != "nan" else None


def build_facts(row: dict, criteria: list[dict] | None = None) -> dict[str, str]:
    """Convertit une ligne de résultats en chaînes de faits (les seuls nombres autorisés dans la note).
    criteria : liste {colonne, libelle, format} de la configuration (par défaut : les 8 critères du projet)."""
    from engine import fmt
    criteria = criteria or _demo_criteria()
    ecarte = bool(row.get("ecarte"))
    f = {"projet": _text(row.get("nom")) or str(row.get("id", row.get("loan_id", "?"))),
         "identifiant": str(row.get("id", row.get("loan_id", "?")))}
    for k in ("secteur", "secteur_damodaran", "etat"):
        if _text(row.get(k)):
            f["secteur" if k.startswith("secteur") else k] = row[k]
    if row.get("montant_pret") is not None and not pd_isna(row.get("montant_pret")):
        f["montant_pret"] = fmt(row["montant_pret"], "money")
    f["categorie"] = str(row["categorie_finale"])
    f["rang_electre3"] = "non classé (écarté)" if ecarte else str(int(row["rang_electre3"]))
    f["rang_promethee2"] = "non classé (écarté)" if ecarte else str(int(row["rang_promethee2"]))
    for c in criteria:
        f[c["colonne"]] = f"{c['libelle']} : {fmt(row[c['colonne']], c.get('format', 'dec2'))}"
    f["motif_ecarte"] = _text(row.get("motif_ecarte")) or "aucun"
    f["fragilite"] = _text(row.get("fragilite_texte")) or "non évaluée"
    if _text(row.get("facteurs_risque")):
        f["facteurs_risque"] = row["facteurs_risque"]
    if row.get("selection_budget") is not None and not pd_isna(row.get("selection_budget")):
        f["selection_budget"] = ("retenu dans le budget" if bool(row["selection_budget"])
                                 else f"non retenu dans le budget ({_text(row.get('motif_arbitrage')) or 'motif inconnu'})")
    if row.get("part_graines_meme_categorie") is not None and not pd_isna(row.get("part_graines_meme_categorie")):
        f["stabilite_simulation"] = f"{100 * row['part_graines_meme_categorie']:.0f} % des tirages simulés donnent la même catégorie"
    return {k: str(v) for k, v in f.items()}


def pd_isna(v) -> bool:
    try:
        return v != v  # NaN
    except Exception:
        return False


FACT_LABELS = {"projet": "Projet", "secteur": "Secteur", "etat": "État", "montant_pret": "Montant du prêt",
               "categorie": "Catégorie proposée par le MCDA", "rang_electre3": "Rang ELECTRE III",
               "rang_promethee2": "Rang PROMETHEE II", "motif_ecarte": "Critère éliminatoire déclenché",
               "fragilite": "Fragilité de la décision", "facteurs_risque": "Facteurs de risque du modèle",
               "stabilite_simulation": "Sensibilité à la simulation",
               "selection_budget": "Arbitrage budgétaire"}


def facts_to_text(facts: dict[str, str]) -> str:
    """Faits présentés au LLM avec des libellés français (les critères sont déjà « libellé : valeur »)."""
    lines = []
    for k, v in facts.items():
        if k == "identifiant":
            continue
        lines.append(f"- {FACT_LABELS[k]} : {v}" if k in FACT_LABELS else f"- {v}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------- vérification
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> set[str]:
    # « 10 601 » (espace, espace insécable ou fine) ou « 10,601 » : séparateurs de milliers retirés
    t = re.sub(r"(?<=\d)[\s  ](?=\d{3}(?!\d))", "", str(text))
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)
    return {m.replace(",", ".") for m in _NUM.findall(t)}


def verify_note(note: str, facts: dict[str, str]) -> tuple[bool, list[str]]:
    """Chaque nombre de la note doit apparaître dans les faits (les entiers 0 à 3 sont tolérés : « 2 raisons »)."""
    allowed = _numbers(" ".join(facts.values()))
    allowed |= {n.replace(" ", "") for n in allowed}
    # les nombres à espace de milliers (ex. « 1 234 567 ») sont recomposés
    joined = {re.sub(r"\s", "", v) for v in facts.values()}
    for v in joined:
        allowed |= _numbers(v)
    bad = sorted(n for n in _numbers(note) if n not in allowed and n not in {"0", "1", "2", "3"})
    return (not bad), bad


# --------------------------------------------------------------------------------------- gabarit (sans LLM)
FIXED_KEYS = {"projet", "identifiant", "secteur", "etat", "montant_pret", "categorie", "rang_electre3",
              "rang_promethee2", "motif_ecarte", "fragilite", "facteurs_risque", "stabilite_simulation",
              "selection_budget"}


def template_note(facts: dict[str, str]) -> str:
    """Note déterministe (sans LLM) : ne contient que les faits."""
    f = facts
    ctx = ", ".join(f[k] for k in ("secteur", "etat") if k in f)
    lines = [f"{f['projet']}{f' ({ctx})' if ctx else ''} : catégorie « {f['categorie']} »."]
    crit = [v for k, v in f.items() if k not in FIXED_KEYS]
    lines.append("Critères : " + " ; ".join(crit) + ".")
    if f["motif_ecarte"] != "aucun":
        lines.append(f"Critère éliminatoire déclenché : {f['motif_ecarte']}.")
    else:
        lines.append(f"Rang ELECTRE III : {f['rang_electre3']} ; rang PROMETHEE II : {f['rang_promethee2']}.")
    if "facteurs_risque" in f:
        lines.append(f"Principaux facteurs de risque (modèle) : {f['facteurs_risque']}.")
    lines.append(f"Fragilité de la décision : {f['fragilite']}.")
    if "stabilite_simulation" in f:
        lines.append(f"Sensibilité à la simulation : {f['stabilite_simulation']}.")
    if "selection_budget" in f:
        lines.append(f"Arbitrage budgétaire : {f['selection_budget']}.")
    lines.append("Cette note est une aide à la décision ; la décision finale revient au comité d'investissement.")
    return " ".join(lines)


# --------------------------------------------------------------------------------------- appel LLM
def _secret(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    try:  # st.secrets n'existe qu'à l'intérieur de Streamlit
        import streamlit as st
        return st.secrets.get(name)  # type: ignore[no-any-return]
    except Exception:
        return None


def _post(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


# Le niveau gratuit renvoie souvent 503 (forte demande) ou 429 (quota) ; un modèle peut aussi être retiré (404).
GEMINI_FALLBACKS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
RETRYABLE = {404, 429, 500, 503}
_DISCOVERED: list[str] | None = None


def _discover_gemini(key: str) -> list[str]:
    """Modèles « gemini-X.Y-flash(-lite) » réellement ouverts à cette clé, du plus récent au plus ancien."""
    global _DISCOVERED
    if _DISCOVERED is None:
        try:
            req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                         headers={"x-goog-api-key": key})
            with urllib.request.urlopen(req, timeout=15) as r:
                models = json.loads(r.read().decode("utf-8")).get("models", [])
            names = [m["name"].split("/")[-1] for m in models
                     if "generateContent" in m.get("supportedGenerationMethods", [])]
            names = [n for n in names if re.fullmatch(r"gemini-\d+(\.\d+)?-flash(-lite)?", n)]
            _DISCOVERED = sorted(names, key=lambda n: ([int(x) for x in re.findall(r"\d+", n)], "lite" not in n),
                                 reverse=True)[:4]
        except Exception:
            _DISCOVERED = []
    return _DISCOVERED


def gemini_models() -> list[str]:
    first = _secret("LLM_MODEL")
    key = _secret("GEMINI_API_KEY") or ""
    return list(dict.fromkeys(([first] if first else []) + GEMINI_FALLBACKS + _discover_gemini(key)))


def with_fallback(run, models: list[str]):
    """Exécute run(model) sur chaque modèle tant que l'erreur est passagère (503, 429, 404, délai dépassé).
    Renvoie (résultat, modèle utilisé) ou relève la dernière erreur."""
    last = None
    for m in models:
        try:
            return run(m), m
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE:
                raise
            last = e
        except (TimeoutError, urllib.error.URLError) as e:
            if isinstance(e, urllib.error.HTTPError):
                raise
            last = e
    raise last


def call_llm(prompt: str, system: str = SYSTEM_PROMPT) -> tuple[str | None, str]:
    """Renvoie (texte du LLM ou None, diagnostic lisible). La clé n'est jamais journalisée ni renvoyée."""
    provider = (_secret("LLM_PROVIDER") or "gemini").lower()
    model = _secret("LLM_MODEL") or DEFAULT_MODELS.get(provider, "")
    try:
        if provider == "gemini":
            key = _secret("GEMINI_API_KEY")
            if not key:
                return None, "pas de clé GEMINI_API_KEY dans les secrets"
            def run(m):
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
                return _post(url, {"system_instruction": {"parts": [{"text": system}]},
                                   "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                   # budget large : les modèles récents consomment aussi des jetons de réflexion
                                   "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096}},
                             {"Content-Type": "application/json", "x-goog-api-key": key})
            out, model = with_fallback(run, gemini_models())
            parts = [p for p in out["candidates"][0]["content"].get("parts", []) if not p.get("thought")]
            text = "".join(p.get("text", "") for p in parts).strip()
            return (text or None), (f"Gemini {model}" if text else f"réponse vide de Gemini {model}")
        if provider == "groq":
            key = _secret("GROQ_API_KEY")
            if not key:
                return None, "pas de clé GROQ_API_KEY dans les secrets"
            out = _post("https://api.groq.com/openai/v1/chat/completions",
                        {"model": model, "temperature": 0.2, "max_tokens": 900,
                         "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
                        {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            return out["choices"][0]["message"]["content"].strip(), f"Groq {model}"
        return None, f"fournisseur inconnu : {provider}"
    except urllib.error.HTTPError as e:            # ex. 400 clé invalide, 404 modèle inconnu, 429 quota
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")[:160]
        except Exception:
            msg = ""
        return None, f"erreur HTTP {e.code} du fournisseur ({model}) {msg}".strip()
    except (urllib.error.URLError, TimeoutError) as e:
        return None, f"réseau indisponible ({type(e).__name__})"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return None, f"réponse inattendue du fournisseur ({type(e).__name__})"


def write_note(row: dict, criteria: list[dict] | None = None) -> dict:
    """Note pour un projet : LLM si disponible ET vérifiée, sinon gabarit. Renvoie texte, source, contrôle."""
    facts = build_facts(row, criteria)
    prompt = "FAITS (à recopier tels quels, aucun autre chiffre autorisé) :\n" + facts_to_text(facts)
    llm, diag = call_llm(prompt)
    if llm:
        ok, bad = verify_note(llm, facts)
        if ok:
            return {"note": llm, "source": f"LLM ({diag})", "verification": "tous les chiffres correspondent aux faits"}
        return {"note": template_note(facts), "source": "gabarit",
                "verification": f"note du LLM rejetée (chiffres hors faits : {', '.join(bad)})"}
    return {"note": template_note(facts), "source": "gabarit", "verification": f"LLM non utilisé : {diag}"}
