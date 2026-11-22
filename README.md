# IA agentique pour la décision d'investissement multicritère : évaluation et sélection de projets d'investissement

Projet de machine learning de groupe dans le secteur bancaire. La banque est le financeur, et les projets à évaluer appartiennent à des entités externes (crédit d'investissement demandé par des PME). Un modèle de risque de machine learning estime la probabilité de défaut, puis une méthode multicritère (MCDA) utilise ce risque avec d'autres critères pour classer les projets ; la décision finale reste humaine.

**Le ML ne calcule que le risque, le MCDA décide, le LLM rédige (et ne calcule jamais).**

## Structure du dépôt

| Dossier / fichier | Contenu |
|---|---|
| `data/raw/` | Données brutes téléchargées, jamais modifiées (fichiers FOIA SBA 7(a)). Non versionné. |
| `data/external/` | Paramètres externes (Damodaran `wacc.xlsx`). Non versionné. |
| `data/processed/` | Jeux train/val/test (non versionnés, lourds) et **petits artefacts versionnés** (CSV/JSON : audit, prédictions, `project_scores.csv`, résultats MCDA). |
| `src/` | Code : `data_prep.py`, `features.py`, `models.py`, `interpret_model.py`, `export_pd.py`, `simulate.py`, `project_scores.py`, `mcda.py`, `run_mcda.py`, `agent.py`, `damodaran.py`. |
| `notebooks/` | `01` audit/exploration, `02` logistique, `03` EBM/comparaison, `04` interprétabilité (lisent les sorties du pipeline). |
| `app/app.py` | Application Streamlit. |
| `models/` | Modèles entraînés (`risk_models.joblib`). |
| `reports/` | Métriques, graphiques (calibration, importances, effets EBM), erreurs par segment. |
| `params/` | Paramètres sectoriels utilisés, matrice et poids AHP. |
| `tests/` | `test_mcda.py` (exemple à 3 projets calculé à la main, AHP, filtre), `check_agent.py` (garde-fou de l'agent). |
| `scripts/download_data.py` | Vérifie la présence des données et indique où les obtenir. |

## Installation et reproduction

```bash
python -m venv .venv && source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt                          # + requirements-dev.txt pour tests et notebooks
python scripts/download_data.py                          # contrôle des fichiers dans data/raw et data/external
```

Les fichiers SBA et Damodaran ne sont pas dans Git (trop lourds) : les télécharger aux adresses du journal ci-dessous. Si les `.parquet` intermédiaires `data/processed/sba_7a_*.parquet` sont absents, `data_prep.py` les recrée depuis les `.xlsx` (long). Ordre d'exécution (graines fixes, résultats identiques à chaque exécution) :

```bash
python src/data_prep.py        # 1. cible default_60m + découpage train/val/test
python src/models.py           # 2-3. logistique + EBM, calibration, métriques (EBM : quelques minutes)
python src/export_pd.py        # 5. pd_ml de chaque prêt du test
python src/simulate.py         # 6. 40 prêts + indicateurs simulés (graine 2026)
python src/project_scores.py   # 7. table project_scores (40 x 8, sans valeur manquante)
python src/interpret_model.py  # 4. importances, permutation, erreurs, facteurs de risque par projet
python src/run_mcda.py         # 8. AHP, filtre, ELECTRE III / TRI, PROMETHEE II, sensibilité
python -m pytest tests -q && python tests/check_agent.py
streamlit run app/app.py       # 10. application
```

## Journal des données

| Fichier | Source | Taille (lue sur le disque) | Date de téléchargement | Traitement |
|---|---|---|---|---|
| `data/raw/FOIA_7a_FY2000_FY2009_asof_260930.xlsx` | SBA 7(a) FOIA (données au 30/09/2026) | 172 673 904 octets | 2026-11-16 (date du fichier) | converti en `data/processed/sba_7a_FY2000_FY2009.parquet`, puis `data_prep.py` |
| `data/raw/FOIA_7a_FY2010_FY2019_asof_260930.xlsx` | SBA 7(a) FOIA (données au 30/09/2026) | 142 802 089 octets | 2026-11-16 (date du fichier) | idem (FY2010-FY2019) |
| `data/external/wacc.xlsx` | Damodaran, « Cost of Capital by Industry » (US) | 34 266 octets | 2026-11-16 (date du fichier) | lu par `src/damodaran.py` ; sert à la simulation |

## Résultats (chaque chiffre vient d'une exécution ; sources : `data/processed/audit_donnees.json`, `reports/*.csv`)

**Données (décisions 3, 4, 5).** 1 236 084 lignes brutes ; 47 lignes à valeurs décalées et 1 414 doublons exacts retirés ; 149 575 lignes sans `FirstDisbursementDate` écartées (dont 147 990 CANCLD, 114 CHGOFF, 1 428 P I F) ; 49 929 EXEMPT (issue inconnue) retirés. Défaut = CHGOFF avec date de perte ≤ 60 mois après le premier versement ; seuls les prêts versés depuis ≥ 60 mois à la date du fichier sont gardés. L'écriture « P I F » est la seule variante présente. Les statuts du fichier sont au nombre de 5 (pas 13 : CLSLN et SOLDCO n'existent pas ici).

| Bloc | Années | Lignes | Taux de défaut |
|---|---|---|---|
| Entraînement | 2000-2012 | 723 663 | 13,48 % |
| Validation | 2013-2014 | 82 384 | 3,57 % |
| Test | 2015-2016 | 103 986 | 4,25 % |

L'entraînement contient la crise de 2008 : le taux de défaut y est environ 3 fois plus élevé qu'ensuite (changement de régime à signaler).

**Fuite détectée et corrigée (hors liste initiale).** `TermInMonths` exact est altéré après l'octroi pour les prêts en défaut : durées non rondes ≈ 60 % de défaut, durées rondes (12, 24, 36, 60, 84, 120 mois) ≈ 0-13 % en entraînement. Avec la durée fine, un test rapide donnait un AUROC de 0,96 sur la validation (contre 0,81 avec 5 tranches, 0,79 avec 3 tranches, 0,67 sans durée) : faux. La durée est donc remplacée par **3 tranches** (≤ 60, 61-120, > 120 mois). C'est un écart assumé par rapport à la décision 1 (« durée »), à valider en équipe.

**Modèles de risque (mêmes 14 variables, sans `JobsSupported`).** Calibration isotonique ajustée sur la validation ; seuil F1 choisi sur la validation ; test lu une fois après le choix.

| Version calibrée | AUROC val | AUPRC val | F1 val | Brier val | AUROC test | AUPRC test | F1 test | Brier test |
|---|---|---|---|---|---|---|---|---|
| Régression logistique | 0,7207 | 0,1305 | 0,2073 | 0,0327 | 0,7150 | 0,1503 | 0,2336 | 0,0383 |
| **EBM (retenu)** | 0,7560 | 0,1332 | 0,2217 | 0,0326 | 0,7626 | 0,1649 | 0,2529 | 0,0380 |

Le modèle est retenu sur la validation (Brier calibré le plus bas ; écart minime avec la baseline sur le Brier, net sur l'AUROC). **Limite** : la probabilité moyenne prédite sur le test est de 3,54 % pour un taux réel de 4,25 % (dérive entre périodes) : `pd_ml` sous-estime légèrement le risque en 2015-2016. Courbes : `reports/calibration.png`.

**Interprétabilité.** Variables les plus importantes (permutation sur le test, baisse d'AUROC) : tranche de durée (0,180), taux initial (0,042), ligne renouvelable (0,019), État (0,016), procédure (0,015). Erreurs par secteur, État et procédure : `reports/erreurs_par_segment_test.csv`. Avec le seuil F1, le modèle manque une majorité des défauts (taux de faux négatifs de 58 % en Californie à plus de 80 % dans plusieurs États) : il sert à classer et à estimer une probabilité, pas à détecter chaque défaut.

**Simulation (décisions 6-12).** 40 prêts tirés au hasard (graine 2026) dans le test ; répartition DSCR exacte 8 faibles / 24 moyens / 8 solides ; secteur Damodaran par table NAICS à la main ; coût du capital Damodaran ± 20 %. Les indicateurs ne dépendent jamais de `default_60m`. Hypothèses d'équipe **non sourcées**, à justifier à l'oral : croissance des flux par secteur (`GROWTH_ASSUMPTION` dans `src/simulate.py`), horizon de 10 ans, valeur résiduelle décotée de 50 %, bruit annuel des flux de 5 %, bornes de dette 50-80 %, ratio de garanties 0,5-1,5, centres ESG A/B/C = 30/60/85. Seuls les repères DSCR énergie (1,25 ; 1,3-1,4, NREL ATB 2024) viennent du guide.

**MCDA (décisions 13-16).** Poids AHP issus de 28 comparaisons arrondies à l'échelle de Saaty depuis les poids cibles : pd_ml 0,258 ; DSCR 0,201 ; VAN 0,161 ; garanties 0,094 ; dette/fonds propres 0,094 ; délai 0,082 ; ESG 0,062 ; emplois 0,050. Ratio de cohérence = 0,0092 (< 0,10). Filtre : 8 projets écartés. Sur les 40 : 10 « Financer », 14 « Conditionner », 16 « Refuser » (ELECTRE TRI pessimiste, λ = 0,6). Corrélation de rangs ELECTRE III / PROMETHEE II : Spearman 0,849, Kendall 0,673. Sensibilité (`data/processed/sensibilite_seuils_coupure.csv`) : le classement ELECTRE III bouge peu (Spearman ≥ 0,94), mais les catégories dépendent surtout de la coupure λ (0 projet change à 0,5-0,6, 24 à λ = 1). Sur 200 scénarios combinés (poids ± 30 %, seuils, λ ∈ [0,5 ; 1]), 22 projets sur 40 changent de catégorie dans au moins 25 % des cas : leur décision est signalée « fragile ». Le MCDA est vérifié à la main sur un exemple à trois projets (`tests/test_mcda.py`) ; ELECTRE III est implémenté en numpy (pas pyDecision) pour être auditable.

## Agent LLM (étape 9)

`src/agent.py` : le prompt ne contient que des chiffres déjà calculés ; un vérificateur rejette toute note contenant un nombre absent des faits et la remplace par une note de gabarit. **Sans clé, l'application fonctionne avec le gabarit.** Fournisseurs gratuits pris en charge : Google Gemini (`LLM_PROVIDER=gemini`, `GEMINI_API_KEY`) ou Groq (`LLM_PROVIDER=groq`, `GROQ_API_KEY`). La clé se met uniquement dans une variable d'environnement ou dans les secrets Streamlit, jamais dans le dépôt (`.env`, `*.key`, `secrets.*` sont ignorés). La partie « appel LLM » n'a pas pu être testée de bout en bout sans clé : seuls le gabarit et le vérificateur ont été exécutés.

## Déploiement (Streamlit Community Cloud)

1. Pousser le dépôt sur GitHub (les artefacts nécessaires à l'application sont versionnés).
2. Sur share.streamlit.io : « New app », dépôt, branche `main`, fichier `app/app.py`.
3. Dans « Advanced settings → Secrets » : `GEMINI_API_KEY = "..."` et `LLM_PROVIDER = "gemini"` (facultatif).

## Limites connues

- Indicateurs financiers simulés (aucune donnée ouverte vérifiée n'associe VAN/DSCR à l'issue d'un prêt) ; hypothèses listées ci-dessus.
- Le modèle de risque reflète des prêts SBA 7(a) (garantie fédérale), pas un portefeuille bancaire quelconque.
- Régime de défaut différent entre entraînement (2008) et test ; `pd_ml` légèrement sous-estimée sur le test.
- `TermInMonths` : fuite corrigée par tranches ; d'autres champs FOIA (ex. `InitialInterestRate`) pourraient aussi être modifiés a posteriori et n'ont pas été audités un par un.
- Le garde-fou de l'agent contrôle les nombres, pas le sens des phrases : relecture humaine nécessaire.

## Équipe

- Mathieu
- Adingra
- Serge
- Sarah
