# Précision par source : derniers modèles (`rfsz6`, `4l2qv`) vs modèle actuellement utilisé

*Rapport autonome, séparé de l'abstract et de `report_model_comparison.md`. Code : `scripts/compare_by_source.py` ; tables : `results/by_source_*.csv|json`.*

## Modèles comparés

| Nom dans le rapport | Workflow / run MLflow | Réglages |
|---|---|---|
| **Actuel (avril)** | `classify-ttc-model-uri` de `argo/params.yaml` : `mlflow-artifacts:/10/cacf2603514b4887bbfb77e2654c9bc1/artifacts/model` | emb 64, fine-tuné (lr 1e-4) |
| **rfsz6 fine-tuné** | `train-ttc-rfsz6`, run `8355c81055fb448dbb66e35a1333e142` | stage 1 puis fine-tuning ; emb 64, batch 64, lr 1e-3 (stage 1) et 1e-4 (fine-tuning), 78 époques |
| rfsz6 stage 1 seul | `train-ttc-rfsz6`, run `feeaae8bd0354a479e92eb5ba182b765` | corpus scanner + synthétique uniquement |
| **4l2qv annotations seules** | `train-ttc-annotations-4l2qv`, run `9065f72379d44cd1b48f8a3878d1c105` | même architecture, entraîné sur les seules annotations ; lr 1e-3, 13 époques |

Jeu de test : les 14 695 lignes de la première vague 2026 non résolues par l'étape « dictionnaire ». La **source** est le canal de collecte, déduit du fichier d'origine (`filename_previous`) : carnets papier (10 731 lignes, 73 %), tickets papier (1 980), tickets appli (1 984). Règle de score identique à l'étape `evaluate` des workflows (codes tronqués au niveau 4 puis au niveau N, N constant).

Contrôle : les quatre modèles rechargés depuis MLflow reproduisent exactement les rapports d'évaluation (L4 top-1 / top-5 : 71,75 / 84,50 ; 72,87 / 85,68 ; 45,07 / 65,09 ; 72,26 / 87,45). Attention : `rfsz6` et `4l2qv` ont été soumis avec `run_id` = `train-ttc-57c27` / `…-ggf9v`, donc ils ont écrasé les sorties S3 de ces anciens runs ; les modèles sont lus ici depuis MLflow.

## Verdict

**Le modèle `rfsz6` fine-tuné est légèrement meilleur que le modèle actuel (+1,1 point de top-1 au niveau 4), mais le gain n'est pas le même selon la source** :

- **Tickets papier : clairement meilleur** (+2,9 points de top-1, IC [+1,6 ; +4,3]).
- **Tickets appli : meilleur, à la limite de la significativité** (+1,3, IC [−0,1 ; +2,5] ; top-5 +1,2, IC [+0,1 ; +2,2]).
- **Carnets papier (73 % des lignes) : écart faible** (+0,8, IC [+0,3 ; +1,3]), et **non significatif si l'on retire les codes techniques 98/99** (+0,25, IC [−0,2 ; +0,7]). Une part du gain global vient donc des codes 98/99 (+7,9 points de top-1, +13,9 de top-5 sur ces 734 lignes).

Le modèle « annotations seules » n'est **pas** un remplaçant : il est le meilleur sur les carnets (74,4 %, top-5 89,2 %) mais nettement moins bon que l'actuel sur les tickets (−1,8 papier, −3,9 appli).

## Résultats par source (niveau 4 ; top-1 / top-5, en %)

| Source (lignes) | Actuel (avril) | rfsz6 fine-tuné | Annotations seules (4l2qv) | rfsz6 stage 1 seul |
|---|---:|---:|---:|---:|
| Toutes (14 695) | 71,8 / 84,5 | **72,9** / 85,7 | 72,3 / **87,4** | 45,1 / 65,1 |
| Carnets papier (10 731) | 72,7 / 84,4 | 73,4 / 85,3 | **74,4 / 89,2** | 40,0 / 61,8 |
| Tickets papier (1 980) | 69,6 / 85,7 | **72,6 / 88,3** | 67,8 / 83,5 | 59,8 / 74,8 |
| Tickets appli (1 984) | 68,8 / 83,9 | **70,1 / 85,1** | 64,9 / 81,6 | 57,9 / 73,1 |

Écart de chaque modèle vs l'actuel (points de top-1 au niveau 4, IC à 95 % par bootstrap apparié sur les lignes, 2 000 tirages) :

| Source | rfsz6 fine-tuné | Annotations seules |
|---|---|---|
| Toutes | **+1,1 [+0,7 ; +1,6]** | +0,5 [−0,0 ; +1,1] |
| Carnets papier | **+0,8 [+0,3 ; +1,3]** | **+1,8 [+1,1 ; +2,4]** |
| Tickets papier | **+2,9 [+1,6 ; +4,3]** | **−1,8 [−3,5 ; −0,4]** |
| Tickets appli | +1,3 [−0,1 ; +2,5] | **−3,9 [−5,5 ; −2,4]** |

Même découpage au niveau 1 (division) : voir `results/by_source_vs_current.csv` (toutes les combinaisons source × niveau × top-1/top-5).

## Ce que l'on apprend sur le pré-entraînement (mêmes réglages : `rfsz6` fine-tuné vs `4l2qv`)

| Source | Top-1 L4 | Top-5 L4 |
|---|---|---|
| Toutes | +0,6 [+0,1 ; +1,1] | −1,8 [−2,2 ; −1,3] |
| Carnets papier | **−1,0 [−1,6 ; −0,4]** | **−3,9 [−4,5 ; −3,4]** |
| Tickets papier | **+4,8 [+3,4 ; +6,2]** | **+4,8 [+3,5 ; +5,9]** |
| Tickets appli | **+5,2 [+3,8 ; +6,7]** | **+3,5 [+2,2 ; +4,8]** |

Le pré-entraînement sur données de caisse **aide sur les tickets** (qui ressemblent aux libellés de caisse) et **pénalise légèrement les carnets**, au point que globalement il n'apporte que +0,6 point de top-1 et perd 1,8 point de top-5. Le chiffre global cache donc deux effets de signes opposés.

## Textes déjà vus ou nouveaux, par source

63 % des lignes des carnets ont un texte présent dans le jeu d'entraînement ; c'est rare pour les tickets (6 à 7 %, donc les effectifs « texte vu » des tickets, ~125 lignes, sont trop petits pour conclure).

| Carnets papier | Actuel | rfsz6 fine-tuné | Annotations seules |
|---|---:|---:|---:|
| Texte vu (6 749) | 81,7 | 82,2 | **84,5** |
| Texte nouveau (3 982) | 57,4 | **58,5** | 57,4 |

Sur les textes **nouveaux**, `rfsz6` garde une petite avance sur l'actuel (+1,2 [+0,1 ; +2,3] sur les carnets, +3,1 [+1,7 ; +4,6] sur les tickets papier) ; le modèle « annotations seules » n'a d'avantage que sur les textes déjà vus (mémorisation).

## Par code (`results/by_source_codes.csv`, codes avec au moins 15 lignes)

- **Carnets** : `rfsz6` est meilleur sur 45 codes, l'actuel sur 41. Pertes principales de `rfsz6` : 11.1.1.2 (480 lignes, 38 % → 29 %), 01.2.2 (65 lignes, 45 % → 28 %). Gains : 98.1.1 (180 lignes, 0 % → 21 %), 06.2.3.1 (113, 21 % → 43 %), 01.2.1 (120, 64 % → 84 %), 09.7.1.9 (38, 3 % → 37 %).
- **Tickets papier** : `rfsz6` meilleur sur 24 codes, l'actuel sur 6 (gains : 01.1.1.3, 193 lignes, 84 % → 89 % ; 01.1.9.1, 115 lignes, 68 % → 75 %).
- **Tickets appli** : 19 codes pour `rfsz6`, 12 pour l'actuel.
- Sur les lignes où les deux modèles diffèrent (niveau 4, top-1) : carnets, 312 lignes justes seulement pour l'actuel contre 394 seulement pour `rfsz6` ; tickets papier, 67 contre 125 ; tickets appli, 73 contre 98.

## Limites

- **Un seul run par réglage** : l'écart d'une exécution à l'autre du même entraînement est inconnu ; les IC ne reflètent que l'échantillonnage des lignes de test. Un écart de 1 point n'est pas à l'abri du hasard d'entraînement.
- **Réglages non indépendants** : `rfsz6` diffère de l'actuel par le corpus de stage 1 (nouvelle extraction), le jeu d'annotations et le batch, pas seulement par un paramètre ; on ne peut donc pas attribuer le gain à une cause précise.
- Les fichiers de test et d'entraînement lus sur S3 sont ceux du dernier workflow (qui a écrasé les précédents) ; la définition « texte vu » s'appuie sur ce jeu d'entraînement, proche mais pas forcément identique à celui du modèle d'avril.
- Les lignes résolues par l'étape « dictionnaire » sont exclues ; les effectifs des tickets (~2 000 lignes chacun) limitent la précision par source.
- Le test est celui de la vague 1 2026 uniquement ; aucune autre source indépendante n'est disponible ici.
