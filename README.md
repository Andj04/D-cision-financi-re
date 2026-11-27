# IA agentique pour la décision d'investissement multicritère : évaluation et sélection de projets d'investissement

Projet de machine learning de groupe dans le secteur bancaire. La banque est le financeur, et les projets à évaluer appartiennent à des entités externes (crédit d'investissement demandé par des PME). Un modèle de risque de machine learning estime la probabilité de défaut, puis une méthode multicritère (MCDA) utilise ce risque avec d'autres critères pour classer les projets ; la décision finale reste humaine.

**Le ML ne calcule que le risque, le MCDA décide, le LLM rédige (et ne calcule jamais).**

## Structure du dépôt

| Dossier / fichier | Contenu |
|---|---|
| `data/raw/` | Données brutes téléchargées, jamais modifiées (fichiers FOIA SBA 7(a)). Non versionné. |
| `data/external/` | Paramètres externes (Damodaran `wacc.xlsx`). Non versionné. |
| `data/processed/` | Jeux train/val/test (non versionnés, lourds) et **petits artefacts versionnés** (CSV/JSON : audit, prédictions, `project_scores.csv`, résultats MCDA, sensibilités). |
| `src/` | Pipeline : `data_prep.py`, `features.py`, `models.py`, `interpret_model.py`, `export_pd.py`, `simulate.py`, `project_scores.py`, `run_mcda.py`, `sensibilite_simulation.py`, `damodaran.py`. Outil réutilisable : `mcda.py` (méthodes), `arbitrage.py` (agent d'arbitrage : sélection sous budget), `engine.py` (configuration et évaluation de n'importe quel jeu), `evaluer.py` (ligne de commande). `eda.py` (analyse exploratoire, figures dans `reports/eda/`). IA générative : `agent.py` (note par projet), `advisor.py` (conseiller conversationnel). `naming.py` (noms lisibles des projets). |
| `notebooks/` | `01` audit/exploration, `02` logistique, `03` EBM/comparaison, `04` interprétabilité (lisent les sorties du pipeline). |
| `app/` , `.streamlit/config.toml` | Application Streamlit (`app.py` + `common.py` + une page par fichier dans `views/`), logo de l'École Centrale Casablanca (`app/assets/`) et thème à ses couleurs (vert `#02987D`, gris `#72777A`, mesurés sur le logo). Les figures matplotlib suivent la même charte (`src/style.py` ; `python src/models.py --figures` les redessine sans réentraîner). |
| `exemples/` | Jeu de 12 PME fictives et sa configuration JSON, pour essayer l'outil sur d'autres données. |
| `models/` | Modèles entraînés (`risk_models.joblib`). |
| `reports/` | Métriques, réglage des hyperparamètres, graphiques (calibration, importances, effets EBM), erreurs par segment ; `reports/eda/` : figures de l'analyse exploratoire. |
| `params/` | Paramètres sectoriels utilisés, **jugements AHP (modifiables)**, matrice et poids AHP, profils ELECTRE TRI, filtre éliminatoire. |
| `tests/` | `test_mcda.py` (exemple à 3 projets calculé à la main, AHP, filtre, profils), `test_pipeline.py` (garde-fous fuite / double comptage), `test_engine.py` (moteur générique, configuration, outils du conseiller, vérificateur, noms), `test_arbitrage.py` (budget, bornes, plafond sectoriel), `check_agent.py` (garde-fou de l'agent). |
| `scripts/download_data.py` | Vérifie la présence des données et indique où les obtenir. |
| `rapport/` | Rapport LaTeX (`main.tex`, un fichier par chapitre dans `chapitres/`, `references.bib`) ; figures et tableaux générés par `python src/rapport_assets.py` depuis les résultats (aucun chiffre recopié à la main). Compilation : Overleaf, pdfLaTeX (importer `rapport_overleaf.zip`, recréé par `rapport_assets.py`). |

## Installation et reproduction

```bash
python -m venv .venv && source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt                          # + requirements-dev.txt pour tests et notebooks
python scripts/download_data.py                          # contrôle des fichiers dans data/raw et data/external
```

Les fichiers SBA et Damodaran ne sont pas dans Git (trop lourds) : les télécharger aux adresses du journal ci-dessous. Si les `.parquet` intermédiaires `data/processed/sba_7a_*.parquet` sont absents, `data_prep.py` les recrée depuis les `.xlsx` (long). Ordre d'exécution (graines fixes, résultats identiques à chaque exécution) :

```bash
cd src
python data_prep.py              # 1. cible default_60m + découpage train/val/test
python models.py                 # 2-3. réglage, logistique + EBM + LightGBM, calibration, métriques (~10 min)
python export_pd.py              # 5. pd_ml de chaque prêt du test
python simulate.py               # 6. 40 prêts + indicateurs simulés (graine 2026)
python project_scores.py         # 7. table project_scores (40 lignes : identifiant + 8 critères, sans valeur manquante)
python eda.py                    # 2. analyse exploratoire : figures dans reports/eda/
python interpret_model.py        # 4. importances, permutation, erreurs (validation), facteurs de risque par projet
python run_mcda.py               # 8. AHP, filtre, ELECTRE III / TRI, PROMETHEE II, sensibilité, arbitrage sous budget
python sensibilite_simulation.py # 8 bis. mêmes 40 prêts, 20 tirages des indicateurs simulés
cd ..
python -m pytest tests -q && python tests/check_agent.py
streamlit run app/app.py         # 10. application
```

## Journal des données

| Fichier | Source (vérifiée le 27/11/2026) | Taille (lue sur le disque) | Date de téléchargement | Traitement |
|---|---|---|---|---|
| `data/raw/FOIA_7a_FY2000_FY2009_asof_260930.xlsx` | SBA, [7(a) & 504 FOIA](https://data.sba.gov/en/dataset/7-a-504-foia), fichier « FOIA - 7(a) (FY2000-FY2009) », instantané au 30/09/2026 | 172 673 904 octets | 2026-11-16 (date du fichier) | converti en `data/processed/sba_7a_FY2000_FY2009.parquet`, puis `data_prep.py` |
| `data/raw/FOIA_7a_FY2010_FY2019_asof_260930.xlsx` | même page, « FOIA - 7(a) (FY2010-FY2019) », instantané au 30/09/2026 | 142 802 089 octets | 2026-11-16 (date du fichier) | idem (FY2010-FY2019) |
| `data/external/wacc.xlsx` | Damodaran, [Cost of Equity and Capital (US)](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/wacc.html), tableur [wacc.xls](https://www.stern.nyu.edu/~adamodar/pc/datasets/wacc.xls) enregistré en .xlsx | 34 266 octets | 2026-11-16 (date du fichier) | lu par `src/damodaran.py` ; sert à la simulation |

La SBA publie un nouvel instantané chaque trimestre (fichiers en csv sur le portail ; `data_prep.py` lit indifféremment le xlsx ou le csv du même nom) : un instantané plus récent donne des chiffres légèrement différents, car les statuts des prêts évoluent. Fiche du jeu sur data.gov : [SBA 7(a) and 504 Loan Data Reports](https://catalog.data.gov/dataset/sba-7a-and-504-loan-data-reports). Les données SBA sont des « U.S. Government Works » (domaine public).

## Résultats (chaque chiffre vient d'une exécution ; sources : `data/processed/*.csv|json`, `reports/*.csv`, `params/*`)

**Données (décisions 3, 4, 5).** 1 236 084 lignes brutes ; 47 lignes à valeurs décalées et 1 414 doublons exacts retirés ; 149 575 lignes sans `FirstDisbursementDate` écartées (dont 147 990 CANCLD, 114 CHGOFF, 1 428 P I F) ; 49 929 EXEMPT (issue inconnue) retirés. Défaut = CHGOFF avec date de perte ≤ 60 mois après le premier versement ; seuls les prêts versés depuis ≥ 60 mois à la date du fichier sont gardés. L'écriture « P I F » est la seule variante présente. Les statuts du fichier sont au nombre de 5 (pas 13 : CLSLN et SOLDCO n'existent pas ici).

| Bloc | Années | Lignes | Taux de défaut |
|---|---|---|---|
| Entraînement | 2000-2012 | 723 663 | 13,48 % |
| Validation | 2013-2014 | 82 384 | 3,57 % |
| Test | 2015-2016 | 103 986 | 4,25 % |

L'entraînement contient la crise de 2008 : le taux de défaut y est environ 3 fois plus élevé qu'ensuite (changement de régime à signaler).

**Fuite détectée et corrigée (hors liste initiale).** `TermInMonths` exact est altéré après l'octroi pour les prêts en défaut : sur le test, une durée non ronde donne 19,0 % de défaut contre 0,44 % pour une durée ronde (12, 24, 36, 60, 84, 120 mois…). Avec la durée fine, un test rapide donnait un AUROC de 0,96 sur la validation (contre 0,81 avec 5 tranches, 0,79 avec 3 tranches, 0,67 sans durée) : faux. La durée est donc remplacée par **3 tranches** (≤ 60, 61-120, > 120 mois), **dans le modèle comme dans la simulation** (durée standard de 5, 10 ou 25 ans par tranche). C'est un écart assumé par rapport à la décision 1 (« durée »), à valider en équipe. `InitialInterestRate` a été audité de la même façon (valeurs rondes / non rondes, décimales) : aucune anomalie liée au défaut.

**Modèles de risque (mêmes 13 variables, sans `JobsSupported` ni `CollateralInd`).** Ces deux champs servent déjà de critères MCDA (emplois, couverture par les garanties) : les garder dans le modèle compterait deux fois le même fait. Trois modèles : **régression logistique** (baseline simple), **EBM** (avancé et interprétable) et **LightGBM** (avancé, « boîte noire », variables catégorielles natives, arrêt anticipé sur la validation). Réglage court sur la validation (`reports/reglage_hyperparametres.csv`) : logistique `C = 0,01` ; EBM 10 interactions ; LightGBM 127 feuilles, 147 arbres. Calibration isotonique ajustée sur la validation ; seuil F1 choisi sur la validation ; test lu une fois après le choix.

| Version calibrée | AUROC val | AUPRC val | F1 val | Brier val | AUROC test | AUPRC test | F1 test | Brier test |
|---|---|---|---|---|---|---|---|---|
| Régression logistique | 0,7326 | 0,1399 | 0,2146 | 0,03249 | 0,7261 | 0,1592 | 0,2393 | 0,0380 |
| **EBM (retenu)** | 0,7669 | 0,1355 | 0,2236 | 0,03248 | 0,7777 | 0,1698 | 0,2468 | 0,0379 |
| LightGBM (référence) | 0,7929 | 0,1729 | 0,2421 | 0,03180 | 0,7998 | 0,2002 | 0,2821 | 0,0370 |

**Règle de choix, fixée avant d'entraîner LightGBM** : le modèle utilisé est le plus bas Brier calibré sur la validation **parmi les modèles interprétables** (logistique, EBM), car une banque doit pouvoir expliquer un refus de crédit ; LightGBM n'est pas candidat, il mesure ce que coûte l'interprétabilité. Il est le meilleur partout, mais l'écart avec l'EBM reste modéré (+0,022 d'AUROC et +0,030 d'AUPRC sur le test), alors que l'EBM fait nettement mieux que la baseline (+0,052 d'AUROC). L'EBM est retenu ; son explication est native, alors que LightGBM demande une explication a posteriori (valeurs SHAP : `reports/shap_lightgbm.png`, où le taux initial et la durée dominent aussi). **Limite** : la probabilité moyenne prédite par l'EBM sur le test est de 3,52 % pour un taux réel de 4,25 % (dérive entre périodes) : `pd_ml` sous-estime légèrement le risque en 2015-2016. Courbes : `reports/calibration.png`.

**Interprétabilité (sur la validation, pour ne pas relire le test).** Variables les plus importantes (permutation, baisse d'AUROC) : tranche de durée (0,189), taux initial (0,030), montant (0,027), ligne renouvelable (0,023), procédure (0,017). Erreurs par secteur, État et procédure : `reports/erreurs_par_segment_validation.csv`. Avec le seuil F1, le modèle manque une majorité des défauts (taux de faux négatifs de 57 % à 100 % selon l'État) : il sert à classer et à estimer une probabilité, pas à détecter chaque défaut. Le test ne sert qu'aux métriques finales et aux explications locales des 40 projets (qui sont des prêts du test).

**Simulation (décisions 6-12).** 40 prêts tirés au hasard (graine 2026) dans le test ; la cible `default_60m` est retirée du tableau avant le tirage. Répartition DSCR exacte 8 faibles / 24 moyens / 8 solides ; secteur Damodaran par table NAICS à la main ; coût du capital Damodaran ± 20 %. Hypothèses d'équipe **non sourcées**, à justifier à l'oral : durée standard par tranche (5 / 10 / 25 ans), croissance des flux par secteur (`GROWTH_ASSUMPTION` dans `src/simulate.py`), horizon de 10 ans, valeur résiduelle décotée de 50 %, bruit annuel des flux de 5 %, bornes de dette 50-80 %, ratio de garanties 0,5-1,5, centres ESG A/B/C = 30/60/85. Seuls les repères DSCR énergie (1,25 ; 1,3-1,4, NREL ATB 2024) viennent du guide.

**MCDA (décisions 13-16).**
- *AHP* : les 28 jugements deux à deux sont dans `params/ahp_jugements.csv`, **modifiable par l'équipe** (relancer `run_mcda.py`). Ils ont été initialisés en arrondissant à l'échelle de Saaty les rapports des poids cibles de la décision 13 : tant que l'équipe ne les a pas revus, l'AHP ne fait que vérifier la cohérence de ces poids. Poids : pd_ml 0,258 ; DSCR 0,201 ; VAN 0,161 ; garanties 0,094 ; dette/fonds propres 0,094 ; délai 0,082 ; ESG 0,062 ; emplois 0,050. Ratio de cohérence = 0,0092 (< 0,10).
- *Filtre* : DSCR < 1 ou probabilité de défaut > 2 × le **taux de référence de la validation** (3,57 % → seuil 7,1 %). `pd_ml` est calibrée sur 2013-2014 ; la comparer au taux 2000-2012 (13,48 % → 27 %) rendait le filtre inopérant (aucun projet au-dessus). C'est une adaptation de la décision 14, à valider en équipe (`params/filtre_eliminatoire.json`). 11 projets écartés : 8 pour DSCR < 1, 3 pour risque trop élevé. Les 8 premiers sont exactement la tranche DSCR « faible » (0,8-1,0) du guide : ils sont écartés **par construction** de la simulation.
- *ELECTRE TRI* (pessimiste, λ = 0,6) : profils absolus quand un repère existe (probabilité de défaut 1,5 × / 1 × le taux de référence ; DSCR 1,0 / 1,25 ; dette/fonds propres 3,0 / 1,5 ; garanties 0,5 / 1,0 ; ESG 50 / 70 ; VAN ≥ 0 et délai ≤ 10 ans pour b1), tertiles du lot sinon (emplois, b2 de la VAN et du délai) : `params/profils_electre_tri.csv`. Sur les 40 : **9 « Financer », 19 « Conditionner », 12 « Refuser »**.
- *ELECTRE III / PROMETHEE II* : Spearman 0,891, Kendall 0,724. Le rang ELECTRE III affiché est un ordre total de lecture ; les deux distillations sont aussi publiées, et 7 projets ont une « position ambiguë » (les deux distillations diffèrent d'au moins 20 % du nombre de projets : incomparables à une partie des autres).
- *Sensibilité MCDA* (`data/processed/sensibilite_seuils_coupure.csv`) : le classement ELECTRE III bouge peu (Spearman ≥ 0,93) ; les catégories sont stables pour λ entre 0,5 et 0,8 (0 à 3 changements), mais 10 projets changent à λ = 0,9 et 26 à λ = 1 (quasi-unanimité, cas extrême). Sur 200 scénarios combinés (poids ± 30 %, seuils q/p/v, λ ∈ [0,5 ; 0,8]), 10 projets sur 40 changent de catégorie dans au moins 25 % des cas (9 « Conditionner », 1 « Refuser », aucun « Financer ») : décision signalée « fragile ».
- *Sensibilité à la simulation* (`sensibilite_graine_*.csv`) : avec les mêmes 40 prêts et 20 autres tirages des indicateurs simulés, un projet garde sa catégorie dans **48 % des tirages en moyenne**. Les catégories dépendent donc autant de la simulation que du projet : c'est la principale limite de l'exercice, à dire clairement.
- *Contrôle a posteriori* (illustration, `controle_a_posteriori.csv`, la vraie issue est lue après les décisions et n'entre dans aucun calcul) : 3 des 40 prêts ont réellement fait défaut ; les 3 sont en « Refuser ». Trois cas n'ont aucune valeur de preuve.

Le MCDA est vérifié à la main sur un exemple à trois projets (`tests/test_mcda.py`) ; ELECTRE III est implémenté en numpy (pas pyDecision) pour être auditable.

## Analyse exploratoire

`python src/eda.py` produit 7 figures dans `reports/eda/` (reprises dans le notebook `01`) : défaut par année (pic à 28 % pour les prêts de 2007, crise de 2008, puis moins de 5 % après 2011), déséquilibre des classes par bloc, défaut par secteur (de 5,2 % en agriculture à 19,0 % dans l'immobilier, en entraînement), défaut par tranche de durée (28,0 % pour ≤ 60 mois contre 2,0 % au-delà de 120 mois) et illustration de la fuite de la durée exacte (34,9 % de défaut pour une durée non ronde contre 1,6 % pour une durée ronde), montants (les petits prêts font plus souvent défaut : 19,9 % pour le premier décile contre 3,5 % pour le dernier), variables catégorielles et valeurs manquantes (le taux d'intérêt manque pour 79 % des prêts d'entraînement, aucun ensuite).

## Agent d'arbitrage : sélection sous budget

Le sujet impose un montant global alloué, des montants minimum et maximum par projet et des projets variés par secteur. `src/arbitrage.py` choisit les projets à financer parmi les « Financer » (puis, si on l'autorise, les « Conditionner ») :

- **règle « suivre le classement »** (par défaut) : on finance dans l'ordre du comité (les « Financer » par rang ELECTRE III, puis les « Conditionner ») tant que le budget, les bornes et le plafond du secteur le permettent ; c'est l'optimum exact quand un projet mieux classé doit toujours passer avant, et chaque refus a son motif ;
- **règle « maximiser le nombre »** : programmation linéaire en nombres entiers (sac à dos multi-contraintes, `scipy.optimize.milp`), pour comparer.

Démonstration (budget 5 M$, 20 000 $ à 2 M$ par projet, au plus 35 % par secteur) : **11 projets retenus** (8 « Financer », 3 « Conditionner »), 4,98 M$ engagés (99,6 %), 7 secteurs ; seul « Pompes funèbres Price » (Financer) est écarté, par le plafond de son secteur. La règle « maximiser le nombre » retient 21 projets dans 11 secteurs mais écarte le projet classé n° 1 (Studio Murphy, 1,7 M$) : c'est pourquoi la règle du classement est retenue par défaut. Page « Arbitrage budgétaire » de l'application (paramètres réglables en direct), outil `arbitrage_budget` du conseiller, résultat dans `data/processed/arbitrage_budget.csv`.

## Noms des projets

Les 40 projets portent un nom lisible et **fictif** (« Cabinet dentaire Baker », « Brasserie artisanale Garcia ») : l'activité vient du vrai code NAICS du prêt, le nom de famille d'une liste fixe (`src/naming.py`). Les noms réels des emprunteurs ne sont jamais utilisés (retirés dès `data_prep.py`). Le nom n'influence aucun calcul ; l'identifiant haché reste la clé.

## Outil réutilisable : évaluer vos propres projets

Toute la méthode est paramétrable, sans toucher au code, depuis la page **« Configurer l'analyse »** de l'application ou en ligne de commande :

1. **Données** : un fichier CSV ou Excel, une ligne par projet (`id` et `nom` facultatifs, une colonne numérique par critère). Modèle téléchargeable dans l'application ; exemple : `exemples/projets_exemple.csv`. Si le fichier décrit des prêts au format SBA (`GrossApproval`, `TermInMonths`, `NaicsCode`, … : voir `engine.MODEL_COLUMNS`) sans colonne `pd_ml`, la probabilité de défaut est calculée par le modèle de risque entraîné.
2. **Critères** : colonnes utilisées, libellé, sens (max / min), poids, format d'affichage, profils ELECTRE TRI b1 / b2 (proposés par tertiles, modifiables).
3. **Pondération** : poids saisis, ou AHP (jugements deux à deux sur l'échelle de Saaty, ratio de cohérence affiché en direct, refus si ≥ 0,10).
4. **Filtre éliminatoire** : autant de règles « colonne opérateur seuil » que voulu.
5. **ELECTRE** : seuils q / p / v, coupure λ, calcul ou non des 200 scénarios de robustesse.

La configuration s'enregistre et se recharge en JSON (`exemples/configuration_exemple.json`). Une fois l'analyse lancée, toutes les pages (tableau de bord, fiche projet, conseiller, robustesse) travaillent sur ce jeu. Le moteur (`src/engine.py`) reproduit exactement les résultats du pipeline sur la démonstration (vérifié par `tests/test_engine.py`).

```bash
python src/evaluer.py exemples/projets_exemple.csv --config exemples/configuration_exemple.json --sortie resultats.csv
python src/evaluer.py mes_projets.xlsx --ecrire-config ma_config.json   # configuration de départ à éditer
python src/evaluer.py --demo                                             # les 40 projets de démonstration
```

## IA générative : agent rédacteur et conseiller (étape 9)

**Agent rédacteur** (`src/agent.py`, page « Fiche projet ») : le prompt ne contient que des chiffres déjà calculés ; un vérificateur rejette toute note contenant un nombre absent des faits et la remplace par une note de gabarit.

**Conseiller IA** (`src/advisor.py`, page « Conseiller IA ») : un mini conseiller financier à qui l'on pose des questions en français sur le jeu de projets, les résultats, des scénarios « et si » et la méthode. Le LLM ne voit pas les données brutes : il appelle des **outils Python déterministes** (vue d'ensemble, liste triée, fiche projet, comparaison, statistiques, scénario de poids / coupure / sans filtre, paramètres de la méthode, données historiques SBA) et reformule leurs sorties. Il ne calcule rien lui-même. Les données consultées sont affichées sous chaque réponse, et un vérificateur signale tout nombre de la réponse absent de ces données. Sans clé, des « questions rapides » interrogent directement les mêmes outils.

**Fournisseurs gratuits** : Google Gemini (`GEMINI_API_KEY`, par défaut) ou Groq (`LLM_PROVIDER=groq`, `GROQ_API_KEY`). Modèle Gemini par défaut : l'alias `gemini-flash-latest` (les noms datés sont retirés au fil du temps : `gemini-2.0-flash` puis `gemini-2.5-flash` ont été fermés aux nouveaux comptes). En cas de saturation (503), de quota (429) ou de modèle retiré (404), l'appel bascule automatiquement sur `gemini-flash-lite-latest` puis sur les modèles Flash ouverts à la clé (liste interrogée auprès de Google). `LLM_MODEL` force un modèle. La clé se met uniquement dans une variable d'environnement ou dans les secrets Streamlit, jamais dans le dépôt (`.env`, `*.key`, `secrets.*` sont ignorés). En cas d'échec, l'application affiche la cause (clé absente, erreur HTTP et message du fournisseur, réseau), jamais la clé. Testé le 25/11/2026 avec une clé Gemini gratuite : notes et réponses du conseiller produites et vérifiées (forte demande fréquente sur le niveau gratuit, d'où les relances).

## Déploiement (Streamlit Community Cloud)

1. Pousser le dépôt sur GitHub (les artefacts nécessaires à l'application sont versionnés).
2. Sur share.streamlit.io : « New app », dépôt, branche `main`, fichier `app/app.py`.
3. Dans « Advanced settings → Secrets » (ou après coup : « ⋮ → Settings → Secrets ») : `GEMINI_API_KEY = "..."` et `LLM_PROVIDER = "gemini"`.

## Limites connues

- Indicateurs financiers simulés (aucune donnée ouverte vérifiée n'associe VAN/DSCR à l'issue d'un prêt) ; hypothèses listées ci-dessus ; catégories stables dans seulement 48 % des tirages de simulation.
- Le DSCR étant piloté par la simulation, les flux sont proportionnels au service de la dette : un prêt court (service annuel élevé) donne mécaniquement des flux plus forts, donc une VAN plus élevée et un délai plus court. Comme les prêts courts sont aussi plus risqués pour le modèle, `pd_ml` est corrélée à la VAN (+0,44) et au délai (−0,60). Ce lien passe par la tranche de durée (connue à l'octroi), pas par l'issue réelle, mais c'est un artefact de la simulation.
- Le modèle de risque reflète des prêts SBA 7(a) (garantie fédérale), pas un portefeuille bancaire quelconque.
- Régime de défaut différent entre entraînement (2008) et test ; `pd_ml` légèrement sous-estimée sur le test.
- Seuils q/p/v et certains profils ELECTRE TRI sont relatifs au lot de projets : ajouter un projet peut les déplacer.
- Score ESG tiré au hasard par groupe de secteurs (pas de donnée réelle) : il ajoute surtout du bruit, avec 6 % du poids.
- Le garde-fou de l'agent contrôle les nombres, pas le sens des phrases : relecture humaine nécessaire.

## Équipe

- Mathieu
- Adingra
- Serge
- Sarah
