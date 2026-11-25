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

DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "groq": "llama-3.3-70b-versatile"}

SYSTEM_PROMPT = (
    "Tu es l'agent rédacteur d'un comité d'investissement bancaire. Tu rédiges en français une note courte "
    "(6 à 9 phrases) sur UN projet. RÈGLES STRICTES : (1) tu n'effectues aucun calcul ; (2) tu recopies les "
    "chiffres EXACTEMENT comme dans la liste FAITS, sans en créer, arrondir ou déduire d'autres ; "
    "(3) tu n'inventes aucun fait absent de la liste ; (4) tu indiques la catégorie, les raisons principales, "
    "les critères éliminatoires déclenchés s'il y en a, et la fragilité de la décision ; (5) tu rappelles que la "
    "décision finale revient au comité ; (6) pas de recommandation d'achat/vente, pas de promesse de rendement."
)


# --------------------------------------------------------------------------------------- faits
def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}".replace(",", " ")


def build_facts(row: dict) -> dict[str, str]:
    """Convertit une ligne de résultats en chaînes de faits (les seuls nombres autorisés dans la note)."""
    f = {
        "projet": str(row["loan_id"]),
        "secteur": str(row.get("secteur_damodaran", "inconnu")),
        "etat": str(row.get("etat", "inconnu")),
        "categorie": str(row["categorie_finale"]),
        "rang_electre3": "non classé (écarté)" if row.get("ecarte") else str(int(row["rang_electre3"])),
        "rang_promethee2": "non classé (écarté)" if row.get("ecarte") else str(int(row["rang_promethee2"])),
        "probabilite_defaut": f"{100 * row['pd_ml']:.1f} %",
        "van": f"{_fmt(row['van'], 0)} $",
        "delai_recuperation": f"{row['delai_recuperation']:.1f} ans",
        "dscr_min": f"{row['dscr_min']:.2f}",
        "dette_fonds_propres": f"{row['dette_fonds_propres']:.2f}",
        "couverture_garanties": f"{row['couverture_garanties']:.2f}",
        "esg_score": f"{row['esg_score']:.0f}",
        "emplois_par_million": f"{row['emplois_par_million']:.1f}",
        "montant_pret": f"{_fmt(row['montant_pret'], 0)} $",
        "fragilite": row["fragilite_texte"] if isinstance(row.get("fragilite_texte"), str) else "non évaluée",
        "motif_ecarte": row["motif_ecarte"] if isinstance(row.get("motif_ecarte"), str) and row["motif_ecarte"] else "aucun",
        "facteurs_risque": row.get("facteurs_risque", "non disponibles"),
    }
    return {k: str(v) for k, v in f.items()}


def facts_to_text(facts: dict[str, str]) -> str:
    return "\n".join(f"- {k} : {v}" for k, v in facts.items())


# --------------------------------------------------------------------------------------- vérification
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> set[str]:
    return {m.replace(",", ".") for m in _NUM.findall(text.replace(" ", " ").replace(" ", " "))}


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
def template_note(facts: dict[str, str]) -> str:
    f = facts
    lines = [f"Projet {f['projet']} ({f['secteur']}, {f['etat']}) : catégorie « {f['categorie']} ».",
             f"Probabilité de défaut estimée par le modèle : {f['probabilite_defaut']} ; DSCR minimum {f['dscr_min']} ; "
             f"VAN {f['van']} ; délai de récupération {f['delai_recuperation']}.",
             f"Dette sur fonds propres {f['dette_fonds_propres']} ; couverture par les garanties {f['couverture_garanties']} ; "
             f"score ESG {f['esg_score']} ; {f['emplois_par_million']} emplois par million de dollars investi."]
    if f["motif_ecarte"] != "aucun":
        lines.append(f"Critère éliminatoire déclenché : {f['motif_ecarte']}.")
    else:
        lines.append(f"Rang ELECTRE III : {f['rang_electre3']} ; rang PROMETHEE II : {f['rang_promethee2']}.")
    lines.append(f"Principaux facteurs de risque (modèle) : {f['facteurs_risque']}.")
    lines.append(f"Fragilité de la décision : {f['fragilite']}.")
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
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def call_llm(prompt: str, system: str = SYSTEM_PROMPT) -> tuple[str | None, str]:
    """Renvoie (texte du LLM ou None, diagnostic lisible). La clé n'est jamais journalisée ni renvoyée."""
    provider = (_secret("LLM_PROVIDER") or "gemini").lower()
    model = _secret("LLM_MODEL") or DEFAULT_MODELS.get(provider, "")
    try:
        if provider == "gemini":
            key = _secret("GEMINI_API_KEY")
            if not key:
                return None, "pas de clé GEMINI_API_KEY dans les secrets"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            out = _post(url, {"system_instruction": {"parts": [{"text": system}]},
                              "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                              # budget large : les modèles « 2.5 » consomment aussi des jetons de réflexion
                              "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}},
                        {"Content-Type": "application/json", "x-goog-api-key": key})
            parts = out["candidates"][0]["content"].get("parts", [])
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


def write_note(row: dict) -> dict:
    """Note pour un projet : LLM si disponible ET vérifiée, sinon gabarit. Renvoie texte, source, contrôle."""
    facts = build_facts(row)
    prompt = "FAITS (à recopier tels quels, aucun autre chiffre autorisé) :\n" + facts_to_text(facts)
    llm, diag = call_llm(prompt)
    if llm:
        ok, bad = verify_note(llm, facts)
        if ok:
            return {"note": llm, "source": f"LLM ({diag})", "verification": "tous les chiffres correspondent aux faits"}
        return {"note": template_note(facts), "source": "gabarit",
                "verification": f"note du LLM rejetée (chiffres hors faits : {', '.join(bad)})"}
    return {"note": template_note(facts), "source": "gabarit", "verification": f"LLM non utilisé : {diag}"}
