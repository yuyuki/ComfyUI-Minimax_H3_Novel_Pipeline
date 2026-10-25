# Étape 3 — Consolidation bornée et profils éditables

Date : 2026-10-25. Implémentation hors ligne ; validation utilisateur et essai
réel LM Studio encore attendus. L'étape 4 n'est pas engagée.

## Contrat et traitement

- Regroupement d'identités par lots de chapitre, puis comparaisons contextuelles
  entre lots. Un nom partagé déclenche une comparaison, jamais une fusion locale
  automatique. Les alias différents exigent des citations de liaison ; les
  identités sans rapprochement confirmé restent séparées.
- Analyse des traits par personnage et par lots, rapprochement des noms de traits
  précis, puis comparaison des valeurs entre tous les lots. Les vêtements de
  scènes différentes peuvent coexister sans constituer une contradiction.
- Chaque observation valide conserve sa citation, sa provenance et ses positions.
  Les requêtes utilisent des vues adaptées à leur décision ; le document local
  conserve intégralement les observations et le document source.
- Les comparaisons partagent une table de citations pour éviter de recopier une
  même preuve dans chaque paire. Tous les travaux et toutes les références restent
  présents, y compris pendant les nouvelles tentatives compactes.
- `context_tokens` vaut 65 536 par défaut et doit rester inférieur ou égal au
  contexte chargé dans LM Studio. Le calcul réserve `max_tokens`, le schéma,
  les instructions et 8 192 unités pour le cadrage et les corrections. Les entrées
  sont comptées conservativement en octets UTF-8, sans mesure du tokenizer serveur.
  Le nombre de décisions par requête est également limité selon `max_tokens`.
  Une observation/comparaison indivisible trop grande provoque un diagnostic,
  sans découpe destructive ni suppression silencieuse.
- Les réponses validées sont mises en cache dans
  `<run>/<out_dir>/.cache/character_profiles`. Une annulation ou un échec tardif
  conserve les lots terminés. Les clés incluent entrée utile, schéma, instructions,
  modèle et réglages ; `force` impose de refaire les requêtes.
- `character_profiles.json` est désormais au format v2 et référence son schéma
  exporté. Il expose `local_id`, `link_to_local_id`, `global_traits`,
  `temporary_states`, `chronological_changes` et des contradictions par trait avec
  `choices[].selected`. Les observations incertaines restent consultables.
- `LoadCharacterProfilesNode` charge fichier, copie renommée ou dossier, dans le
  périmètre de sortie autorisé. Il vérifie le schéma embarqué, les références et
  les liens directs ; aucune récupération du `$schema` externe. Il accepte zéro
  ou plusieurs sélections, préserve les éditions et ne réécrit pas le fichier.

## Contrôles du roman sans réseau

Fichier lu sans modification :
`E:\AI\ComfyUI\output\minimax_h3_novel\20261024163701\character_evidence\character_evidence.json`.

La simulation a parcouru la consolidation complète, l'export et la validation
finale dans un répertoire temporaire supprimé ensuite. Son modèle simulé conserve
les identités locales sans fusion entre lots et classe les observations comme
contexte ; il répond « incertain » aux comparaisons de valeurs. Il vérifie le
plancher technique de couverture et de taille, pas l'analyse narrative ni les
traits précis que produira Qwen. Les scénarios ciblés de tests couvrent séparément
les fusions, les traits précis, les contradictions et les changements explicites.

| Mesure | Résultat |
| --- | ---: |
| Chapitres lus | 25 |
| Observations d'origine | 3 112 |
| Observations valides conservées exactement une fois | 2 786 |
| Observations exclues conservées séparément | 326 |
| Profils provisoires dans cette simulation sans fusion | 576 |
| Requêtes de regroupement local | 56 |
| Requêtes de rapprochement des identités | 488 |
| Requêtes de classement des traits | 576 |
| Requêtes de comparaison des valeurs | 98 |
| Total de requêtes simulées, sans cache préalable | 1 218 |
| Taille maximale d'une entrée de requête | 39 834 octets |

Ce volume n'est pas une estimation fiable du nombre d'appels réels : les fusions
et les traits précis changent les comparaisons nécessaires. Les comparaisons
peuvent croître quadratiquement avec les identités candidates et les valeurs d'un
trait très présent. Elles restent bornées par requête et reprenables, mais le coût
réel doit être mesuré dans LM Studio. Le rapport `character_profiles_requests.json`
expose le nombre d'appels, les lectures de cache et la taille maximale d'entrée.

## Validation et suite

- Tests ciblés : identités/alias/homonymes, conflits précis, chronologie explicite,
  incertitudes, couverture sans perte, reprises, caches, corrections et schémas
  sur le transport réel simulé ; chargeur et éditions manuelles sans réseau.
- Ruff et carte d'architecture vérifiés ; build sdist et wheel obtenu dans
  `dist/character-profiles-stage3-final` après utilisation d'un dossier neuf, l'archive
  existante de `dist` étant inaccessible en remplacement.
- La suite complète a un échec d'environnement dans le test existant
  `tests/test_load_spatial_continuity.py::test_paths_are_confined` : Windows refuse
  la création d'un lien symbolique (`WinError 1314`), même hors bac à sable.
  Ce test n'a pas été modifié ni désactivé. Résultat final : **700 réussis,
  2 ignorés et 1 échec de privilège Windows**. Les imports du wheel depuis un
  répertoire isolé, ses 17 nodes, le schéma et le JavaScript ont été vérifiés ;
  les ressources attendues sont également présentes dans le sdist.
- Les données de l'utilisateur n'ont pas été réécrites ; aucun appel LM Studio
  réel n'a été lancé. Un essai réel reste nécessaire avant validation de l'étape.
- La génération visuelle valide maintenant le contrat courant sans reconstruire
  l'ancien résultat. La consommation des sélections/liens, le chapitre cinématique
  et les états par séquence restent l'étape 4. Elle bloque encore les champs
  visuels en conflit ; ne pas interpréter cette intégration minimale comme
  l'implémentation de la politique visuelle validée dans le plan.
