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

## Correction après le run 20261027073525

Les identités ont progressé (27 → 16 profils, 53 → 27 requêtes), mais des catégories
valides syntaxiquement se sont retrouvées associées aux mauvais faits. Les lèvres,
les yeux et les cheveux de Doriane/Madeleine produisaient des conflits artificiels.

- [x] Conserver le contexte narratif uniquement pour les identités.
- [x] Limiter classification et comparaison aux valeurs et citations propres.
- [x] Ajouter des garde-fous locaux FR/EN sur les sujets physiques et certaines
  propriétés non ambiguës : cheveux/yeux/lèvres/jambes, couleur/épaisseur,
  éclat des yeux et barbe de deux jours. Aucun fait nouveau n’est déduit.
- [x] Sauver les affectations valides même si une autre catégorie de la même
  observation est rejetée. Ne pas comparer les affectations de simple contexte.
- [x] Invalider les caches de décisions précédents.
- [ ] Vérifier sur la prochaine relance réelle la qualité des décisions du modèle.

Les garde-fous ne prétendent pas comprendre tout vocabulaire ni toute langue.
En cas de rejet, les observations sources restent intactes ; elles ne sont pas
converties automatiquement en un portrait supposé correct. Aucun appel LLM
supplémentaire n’est ajouté. Réutiliser l’extraction existante pour la relance.

Validation de cette correction : 101 tests ciblés réussis ; suite complète : 788 réussis, 2 ignorés, 83 sous-tests réussis. Seul échec : privilège Windows de création de lien symbolique, inchangé. Ruff, architecture et vérification du diff réussissent. Pas de validation LM Studio dans cette session.

## Correction après le run 20261027080257

- [x] Distinguer couleur, texture, épaisseur, longueur et coiffure dans les
  valeurs physiques explicites ; roux et chevelure indisciplinée ne se comparent pas.
- [x] Distinguer pilosité, couleur de peau et forme des membres ; refuser de
  propager globalement une valeur qui mélange bronzage et exposition vestimentaire.
- [x] Distinguer grand/hauteur et élancé/corpulence.
- [x] Remplacer la recopie des identifiants par des positions de support séparées
  à gauche et à droite pour les comparaisons d’identité. Le code attache les IDs
  correspondants ; les deux côtés, la provenance et les règles d’alias restent validés.
- [x] Tester un regroupement Shannon entre chapitres sans faux conflit couleur/coiffure,
  ainsi que les supports absents, étrangers, dupliqués et les homonymes.
- [ ] Revalider les décisions narratives et le temps avec LM Studio.

Le rapprochement Shannon apparaissait dans un lot signalant un support d’identité
incomplet ; le journal ne conserve pas la réponse brute rejetée permettant d’attribuer
précisément l’erreur à cette paire. Le nouveau contrat élimine la recopie des IDs,
mais ne fusionne jamais automatiquement deux personnes sur leur seul nom.


## Correction après le run 20261027082041

- Fusion erronée Ted Conrad/Indy : la cooccurrence de noms dans le contexte ne
  constitue plus une preuve d'alias, dès la partition initiale et à la réconciliation.
  Le contrôle exige une expansion de nom présente dans la source ou une formulation
  d'alias explicite reconnue. Il ne fusionne jamais automatiquement les homonymes.
- Les expansions Doriane/Doriane Belecamus et les alias explicites restent possibles.
  Les formulations narratives non reconnues restent séparées : ce contrôle prudent
  n'est pas un analyseur sémantique universel.
- Les lots partiellement valides de comparaisons d'identités et de classifications
  reçoivent une passe ciblée sur les seuls éléments manquants, même avec zéro
  correction. Une observation isolée peut être récupérée. Les réponses déjà validées
  ne sont pas renvoyées. Les observations encore non classées sont listées dans
  `unclassified_evidence_ids` du rapport de requêtes.
- Version de cache des décisions portée à 6. Aucun changement du format sauvegardé.
- Régressions hors ligne : fusion Conrad/Indy rejetée, liens de noms explicites
  acceptés, regroupement Doriane, récupération des pieds de Shannon et d'un job
  d'identité manquant. Confirmation LM Studio à effectuer par une nouvelle consolidation.
