# Plan de correction de l’extraction — 2026-10-26

## Diagnostic

Le fichier examiné contient 766 observations sur 25 chapitres, dont 151 durables.
Les erreurs portent sur le sujet décrit, la négation, la persistance et des
réparations remplaçant une citation par un passage exact mais non pertinent.
Une confiance du modèle élevée et une citation exacte ne prouvent pas le fait.

## Travail

- [x] Renforcer les consignes d’attribution, de négation et de temporalité.
- [x] Auditer toutes les observations après réparation avec les passages originaux.
- [x] Exiger un verdict par observation, sans perte lors des nouvelles tentatives.
- [x] Conserver les observations rejetées pour révision et les exclure de la consolidation.
- [x] Invalider les anciens caches ; conserver les verdicts dans les nouveaux caches.
- [x] Vérifier les régressions, les schémas structurés, le cache et l’annulation.
- [ ] Relancer avec LM Studio puis examiner les passages problématiques avant l’étape 4.

L’audit sémantique ajoute un appel par lot, hors reprises. Il reste une décision
modèle : il réduit le risque sans garantir la justesse narrative. Une panne ou un
verdict incomplet laisse les observations disponibles pour révision mais non
utilisables comme faits. Les JSON existants ne sont pas modifiés.

## Validation hors ligne

108 tests ciblés réussis (extraction, audit, consolidation, architecture).
`ruff check .` et la vérification de la carte d’architecture réussissent.
La suite complète avant régénération comptait 745 réussites, 2 ignorés et
2 échecs : carte périmée (corrigée et retestée), et création de lien symbolique
Windows refusée dans `test_load_spatial_continuity::test_paths_are_confined`.
Les tests de l’audit simulent les verdicts : ils vérifient leur application,
la couverture des observations, les reprises structurées, le cache et l’annulation,
mais ne mesurent pas la qualité narrative de Qwen. Aucun essai LM Studio effectué.

## Relance

Redémarrer ComfyUI pour charger le code, relancer Character Evidence Extractor
avec la sélection du roman. Les clés de cache incluent les nouvelles consignes
et le schéma d’audit : les anciens résultats ne seront pas réutilisés.
Examiner `character_evidence_review.json`, notamment les rejets sémantiques,
puis relancer la consolidation sur cette nouvelle extraction. L’étape 4 reste
conditionnée à une vérification des résultats réels.

## Ciblage et accélération après l’essai 20261026204533

- [x] Limiter le schéma d’extraction à `physical_appearance`, incluant âge et cicatrices.
- [x] Séparer anatomie durable et coiffures ; clarifier les cas de Shannon dans les consignes.
- [x] Remplacer les réparations LLM par une restauration locale des citations :
  guillemets ajoutés, apostrophes et espaces, uniquement avec correspondance unique.
- [x] Remplacer les commentaires d’audit libres par verdict explicite et code d’erreur.
- [x] Conserver thinking pour l’extraction ; audit court sans thinking par défaut,
  réglable avec `audit_thinking`, sans modifier les paramètres des autres requêtes.
- [x] Séparer les appels d’extraction et d’audit dans les diagnostics.
- [ ] Mesurer temps et justesse sur un nouvel essai réel avec les cinq fichiers.

Le gain provient des sorties plus ciblées, de la suppression des appels de réparation
et de l’audit sans raisonnement par défaut. Aucun facteur d’accélération n’est promis
sans mesure réelle. Les citations qui nécessitent une correction de sens restent
à réviser ; elles ne sont pas remplacées par un autre passage choisi par le modèle.

### Vérification des changements de performance

216 tests ciblés réussis, incluant transport structuré, thinking isolé par requête,
reprises compactes sans perte d’observations, restauration locale, caches et annulation.
Ruff, contrôle d’architecture et `git diff --check` réussissent. La suite complète
compte 758 réussites, 2 ignorés et le même échec de privilège Windows pour créer
un lien symbolique ; trois tests supplémentaires ont ensuite été ajoutés et passent
dans la sélection ciblée.

Relecture locale du JSON 20261026204533, en mémoire et sans écriture : 12 citations
retrouvent un extrait exact via la restauration typographique, dont celles de
Shannon et de l’âge de Ted Conrad. Cela ne valide pas leurs verdicts sémantiques.
Aucun nouvel appel à LM Studio ; vitesse réelle et qualité de l’audit sans thinking
restent à mesurer. Réglage proposé : thinking=true, audit_thinking=false ; conserver
les autres budgets lors de la comparaison sur les mêmes cinq fichiers.
