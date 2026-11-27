# Jeux de test de l'outil réutilisable

Toutes les entreprises de ce dossier sont **fictives** : noms inventés, chiffres construits pour tester l'outil
(toute ressemblance avec une entreprise réelle serait fortuite). Aucun de ces fichiers ne sert au modèle de risque.

Dans l'application : page **Configurer l'analyse**, étape 1 → « Importer des projets », puis (si indiqué)
« Charger une configuration », puis **Lancer l'évaluation**. En ligne de commande : `python src/evaluer.py <fichier>`.

| Fichier | Ce qu'il teste | Configuration |
|---|---|---|
| `projets_exemple.csv` | Cas simple : 12 PME, les 8 critères du projet | `configuration_exemple.json` (poids saisis) |
| `test1_pme_casablanca.csv` | 15 PME du Grand Casablanca (montants en MAD), compromis rendement / risque, filtre | `test1_configuration_ahp.json` (**AHP**, filtre DSCR < 1 ou risque > 8 %, profils absolus) |
| `test2_prets_format_sba.csv` | Prêts décrits par les **champs SBA bruts, sans `pd_ml`** : l'outil calcule la probabilité de défaut avec le modèle entraîné | aucune (configuration devinée) |
| `test3_projets_energie.xlsx` | Fichier **Excel**, critères **totalement différents** (TRI, coût par MW, CO₂ évité, risque pays, délai, emplois, part verte) | aucune : les sens sont devinés d'après les noms |
| `test4_fichier_avec_erreurs.csv` | Valeur manquante et texte (« n.c. ») dans des colonnes numériques | l'outil doit **refuser de lancer** et dire pourquoi |

## Résultats attendus

**Test 1** (avec `test1_configuration_ahp.json`) : 6 Financer, 6 Conditionner, 3 Refuser ; ratio de cohérence AHP 0,009.
- Écartés par le filtre : Imprimerie Derb Sultan (DSCR < 1 **et** risque > 8 %), Hôtel Riad Mellah (risque 8,2 % malgré
  la plus forte VAN), Boutique Mode Gauthier (DSCR 0,98).
- En tête : Atelier Céramique Oulfa, Coopérative Argan Tifawin, Laboratoire Analyses Bourgogne.
- Cas intéressant : Studio Numérique Maarif est 4e mais seulement « Conditionner » (garanties 0,20) et fragile.
- Arbitrage (page « Arbitrage budgétaire », budget 12 M MAD, 300 000 à 4,5 M MAD par projet, 35 % par secteur) : 9 projets retenus (6 à financer, 3 sous conditions), 10,00 M MAD engagés, 8 secteurs.

**Test 2** : l'application affiche « Probabilité de défaut calculée par le modèle de risque ». Elle va de 0,4 %
(Prairie Organic Farm, prêt long et ancien) à 32,6 % (Lakeview Auto Repair, prêt court, jeune entreprise).
Configuration devinée (poids égaux, profils en tertiles) : 3 Financer, 4 Conditionner, 3 Refuser ; les trois
prêts au risque le plus élevé (Lakeview Auto Repair, Summit Construction, Blue Ridge Trucking) sont en « Refuser ».
À essayer ensuite : ajouter une règle de filtre « pd_ml > 0,1 » à l'étape 4.

**Test 3** : sens devinés à vérifier à l'étape 2 : rendement, CO₂ évité, emplois, part verte = max ; coût, risque
pays, délai = min. Avec la configuration devinée : 3 Financer (Toitures solaires Casablanca, Parc solaire
Ouarzazate Sud, Éolien côtier Tarfaya Nord), 3 Conditionner, 2 Refuser. À essayer : passer en mode AHP et donner
plus d'importance au CO₂ évité, puis relancer.

**Test 4** : le bouton « Lancer » reste désactivé avec deux messages : « van » et « dscr_min » contiennent chacun une
valeur manquante ou non numérique.

## Questions à poser ensuite au conseiller IA (sur le jeu importé)

- « Résume-moi les résultats. »
- « Pourquoi l'Hôtel Riad Mellah est-il refusé alors que sa VAN est la plus forte ? »
- « Compare l'Atelier Céramique Oulfa et le Studio Numérique Maarif. »
- « Que devient le Studio Numérique Maarif si les garanties ne pèsent que 2 % ? »
- « Quel est le taux de défaut historique ? » (doit répondre que ces données ne concernent que la démonstration)
