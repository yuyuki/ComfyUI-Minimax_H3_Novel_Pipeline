# Correction de la consolidation — 2026-10-26

## Cas réel

Run `20261026222208` : 40 observations utilisables, 27 profils, 19 ambigus,
aucun lien et 53 requêtes. Henry Jones Jr./Indy et Shannon sont fragmentés ;
Doriane/Belecamus restent séparés. Certaines catégories physiques sont erronées.

## Changements

- [x] Fournir le contexte source autour de chaque position validée, sans modifier
  les citations ni les observations enregistrées. Fenêtres de 900 caractères de
  chaque côté, chevauchements fusionnés ; toutes les positions sont conservées.
- [x] Utiliser ce contexte pour les candidats et les validations d’alias,
  sans fusion automatique par nom, apparence ou proximité. Le modèle doit toujours
  confirmer l’identité avec les preuves des deux côtés.
- [x] Préférer le nom fourni le plus complet après confirmation d’une fusion.
- [x] Restreindre la classification à un vocabulaire physique commun ; supprimer
  les requêtes de rapprochement des catégories libres.
- [x] Classer les observations en lots communs à plusieurs personnages, puis
  comparer les valeurs seulement dans chaque identité et catégorie.
- [x] Attacher les deux références des comparaisons de valeurs depuis leurs entrées,
  au lieu de demander au modèle de les recopier ; conserver la validation chronologique.
- [x] Préserver les décisions indépendantes valides après une réponse partiellement
  invalide. Refuser les références inconnues, ambiguës ou dupliquées. Ne pas mettre
  les réponses partielles en cache comme des résultats complets.
- [ ] Valider le résultat réel sur les cinq fichiers avant l’étape 4.

Les signatures des prompts et la version des décisions invalident les caches
précédents. Les IDs de nodes, sockets, sources et le format éditable restent inchangés.
Le contexte supplémentaire reste interne aux requêtes ; les JSON précédents ne
sont pas réécrits. Aucun appel à un serveur LM Studio n’a été effectué ici.

## Contrôles sur le cas fourni

Les fenêtres source contiennent des passages reliant Henry Jones Jr. et Indy
pour cinq observations, et Doriane Belecamus aux formes courtes pour six observations.
Cela fournit le contexte au modèle sans prétendre valider ses futures décisions.
Les 40 observations tiennent dans trois lots de classification avec le budget
65536/16384 ; le nombre total réel de requêtes dépendra des décisions et reprises.

## Validation du 2026-10-27

78 tests ciblés réussis. Suite complète : 765 réussites, 2 ignorés, 83 sous-tests
réussis ; un échec indépendant des changements, lié au privilège Windows requis
pour créer un lien symbolique dans test_load_spatial_continuity.
Ruff, contrôle d’architecture et git diff --check réussissent.
Le document réel 20261026222208 passe toujours la validation du chargeur, sans
modification du fichier. Aucun essai LM Studio effectué : la qualité des décisions
et le temps total restent à vérifier lors de la relance de la consolidation.
