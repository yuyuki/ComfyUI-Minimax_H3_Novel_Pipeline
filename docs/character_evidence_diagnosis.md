# Étape 2 — Diagnostic et fiabilisation des preuves

Date : 2026-10-25. Validation utilisateur de l'étape encore attendue.

## Données examinées

Fichier lu sans modification :
`E:\AI\ComfyUI\output\minimax_h3_novel\20261024163701\character_evidence\character_evidence.json`.

| Mesure hors ligne                                                     | Résultat |
| --------------------------------------------------------------------- | -------: |
| Chapitres                                                             |       25 |
| Observations                                                          |    3 112 |
| Observations utilisables par la validation actuelle                   |    2 786 |
| Observations exclues                                                  |      326 |
| Observations sans position de citation                                |      304 |
| Citations absentes du texte, même après normalisation Unicode/espaces |      301 |
| Classements invalides de vêtements, accessoires ou actions            |       29 |
| Textes embarqués dont le hash diffère du hash enregistré              |        0 |

Les causes se recoupent : une citation incorrecte provoque aussi l'absence de
positions. Ces nombres ne doivent donc pas être additionnés. Parmi les 303
citations absentes à l'identique du chapitre, deux ne diffèrent que par la
normalisation sûre ; leurs positions restent néanmoins absentes. Une autre
citation existe dans le chapitre mais n'a pas été localisée dans le passage
soumis à l'extraction. Aucune provenance n'a été fabriquée pour la récupérer.

## Causes observées

- Des citations remplacent un nom propre par un pronom, ou changent les apostrophes.
  Une citation reformulée ne peut pas fournir des positions exactes dans la source.
- L'extraction conservait les observations invalides pour révision, mais ne les
  soumettait pas à correction : les tentatives existantes traitaient seulement les
  exceptions techniques et les réponses JSON invalides.
- La consolidation émettait une ligne de warning par motif et par observation,
  alors que plusieurs motifs décrivaient le même rejet.
- Le dépassement de contexte de la consolidation reste distinct et sera traité
  à l'étape 3. Les changements de l'étape 2 ne résolvent pas sa requête monolithique.

## Corrections réalisées

- Ajouter jusqu'à `correction_attempts` requêtes ciblées par lot pour corriger
  les citations ou la persistance invalide des observations extraites.
- Identifier les observations par leur index, conserver leur personnage, valeur,
  attribut, classification et confiance ; ne jamais supprimer les observations
  omises par une réponse de correction.
- N'accepter qu'une citation présente à l'identique dans un seul passage soumis.
  Ne pas assembler des passages distincts ni utiliser une recherche approximative
  comme preuve. La pertinence sémantique de la citation reste à vérifier par l'utilisateur.
- Garder les corrections et leurs valeurs d'origine dans `extraction_correction`.
  Conserver les observations non réparées avec leurs warnings, sans bloquer les
  observations valides. Une annulation continue d'interrompre immédiatement le travail.
- Inclure la procédure de correction dans l'identité du cache. Les caches produits
  avant cette correction ne sont pas réutilisés lors d'une nouvelle extraction.
  Charger un ancien fichier édité reste possible sans modifier son contenu.
- Produire `character_evidence_review.json` lors de l'extraction et
  `character_profiles_validation.json` avant la requête de consolidation.
  Ce dernier reste disponible si cette requête échoue. Les logs donnent des bilans
  regroupés, les rapports conservent les détails.

## Validation

- 122 tests ciblés réussis : extraction, consolidation, chargement des preuves et
  génération visuelle existante. Couverture des réparations, schémas des requêtes
  et retries, absence de perte, caches, rapports et annulation.
- Trois cas du fichier réel rejoués avec réponses de correction simulées : chapitre 1
  observation 4, chapitre 2 observation 103, chapitre 3 observation 2. Les citations
  corrigées correspondent exactement aux textes et les faits restent inchangés.
- Carte d'architecture régénérée ; ses cinq tests et `--check` réussis.
- `ruff check .` et `git diff --check` réussis.
- Suite complète : 680 tests réussis, 2 ignorés, 83 sous-tests réussis et un échec
  environnemental indépendant des changements :
  `tests/test_load_spatial_continuity.py::test_paths_are_confined` ne peut pas créer
  un lien symbolique sous Windows (`WinError 1314`, privilège absent). Ce contrôle
  n'a pas été désactivé ni affaibli.
- Aucun appel réel à LM Studio ni aucune nouvelle extraction du roman n'a été
  effectué. Le taux de correction avec le modèle chargé reste à mesurer lors
  d'un essai utilisateur ; les 326 exclusions décrivent le fichier initial.
