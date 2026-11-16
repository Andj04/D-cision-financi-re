# IA agentique pour la décision d'investissement multicritère : évaluation et sélection de projets d'investissement

Projet de machine learning de groupe dans le secteur bancaire. La banque est le financeur, et les projets à évaluer appartiennent à des entités externes (crédit d'investissement demandé par des PME). Un modèle de risque de machine learning estime la probabilité de défaut, puis une méthode multicritère (MCDA) utilise ce risque avec d'autres critères pour classer les projets ; la décision finale reste humaine.

## Structure du dépôt

| Dossier | Contenu |
|---|---|
| `data/raw/` | Données brutes téléchargées, jamais modifiées (fichiers FOIA SBA 7(a)). |
| `data/external/` | Données externes de paramétrage (fichier Damodaran `wacc.xlsx`). |
| `data/processed/` | Données nettoyées et transformées, produites par nos scripts. |
| `notebooks/` | Notebooks d'exploration, de prétraitement et de modélisation. |
| `src/` | Code source du projet. |

Le contenu de `data/` n'est pas versionné (voir `.gitignore`) : les fichiers sont trop lourds pour GitHub. Seuls les fichiers `.gitkeep` sont suivis, pour conserver les dossiers.

## Journal des données

| Fichier | Source | Taille (lue sur le disque) | Date de téléchargement | Traitement |
|---|---|---|---|---|
| `data/raw/FOIA_7a_FY2000_FY2009_asof_260930.xlsx` | SBA 7(a) FOIA | 172 673 904 octets (environ 172,7 Mo) | à compléter | aucun pour l'instant |
| `data/raw/FOIA_7a_FY2010_FY2019_asof_260930.xlsx` | SBA 7(a) FOIA | 142 802 089 octets (environ 142,8 Mo) | à compléter | aucun pour l'instant |
| `data/external/wacc.xlsx` | Damodaran | 34 266 octets (environ 34,3 Ko) | à compléter | aucun pour l'instant |

## Équipe

- Mathieu
- Adingra
- Serge
- Sarah
