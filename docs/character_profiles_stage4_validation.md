# Étape 4 — Enrichissement des personnages par chapitre

Implémentation et contrôles hors ligne du 2026-10-27. Validation utilisateur ouverte.

## Contrat et raccordement

- `CharacterVisualProfileGeneratorNode` conserve ses deux sorties et reçoit désormais
  `cinematic_chapters` : exactement un chapitre, identifié par son fichier source.
  La base globale est validée par le même validateur que `LoadCharacterProfilesNode`.
- Les liens directs réunissent les sources sous le personnage principal. Le premier
  choix sélectionné est utilisé ; une sélection vide exclut le trait, sans solution
  de secours. Les alternatives restent dans le document. Les valeurs incompatibles
  de profils nouvellement liés sont comparées ; sans accord vérifié, elles restent
  exclues du socle utilisable et conservées pour examen.
- Le modèle identifie noms, alias et pronoms avec le contexte antérieur. Les présences
  sont `visible`, `offscreen`, `mentioned` ou `uncertain`. Les personnages absents
  ne sont pas générés. Les identités inconnues restent dans `unresolved`.
- `visual_designs.json` conserve les faits canoniques dans `source_facts` et les
  compléments dans `added_details`. `visual_designs_path` permet d'importer les éditions
  du chapitre précédent ; les personnages absents sont conservés. Les variantes
  temporelles ne deviennent pas des détails globaux.
- `character_chapter_enrichment.json` et son schéma contiennent les présences et états
  par séquence avec leurs preuves. Les états temporaires issus de la base exigent
  la même source, une citation et des positions dans le passage. L'enrichissement
  peut également extraire des états locaux absents de la base globale, avec citation
  exacte, positions et vérification sémantique séparée. Il ne modifie pas la base.
- Les évolutions physiques exigent la preuve chronologique consolidée et une relation
  justifiée avec la séquence. L'ordre des chapitres ne suffit pas. Une blessure ne
  produit pas de cicatrice implicite.
- `character_visual_profiles.json` est au format v3 et embarque `visual_designs` et
  `chapter_enrichment`. Le chargeur conserve ces données sans réseau. Les profils
  consolidés d'entrée restent au format v2 ; aucune conversion ancienne n'est ajoutée.
- L'entrée optionnelle `character_visual_profiles` de `ConsolidateReferencesNode`
  transmet les faits globaux aux assets et les états locaux à la planification et aux
  prompts H3. Une empreinte contrôle la source et la révision des séquences. Les
  rapprochements de registre ambigus produisent un avertissement sans fusion forcée.
  L'ordre des bindings H3 n'est pas modifié.

## Budgets, reprise et limites

Les requêtes passent par le transport JSON structuré partagé, avec schéma explicite
sur les tentatives initiales et les reprises. Les observations sont réparties en lots
bornés ; les ajouts sont également vérifiés contre les autres lots. Le contexte du
chapitre contraint les compléments visuels. Une unité indivisible trop volumineuse
produit un diagnostic de budget, sans troncature silencieuse.

Les décisions globales utilisent `<out_dir>/.cache/`. Les décisions du chapitre
utilisent `<out_dir>/.cache/<chapter_name normalisé>/`. Les empreintes intègrent
les entrées effectivement consommées, y compris les choix et le contexte narratif.
Les erreurs techniques et l'annulation restent bloquantes. Une décision d'enrichissement
encore invalide après correction interrompt la génération ; elle n'est pas publiée
comme état vérifié. Les ambiguïtés explicitement retournées sont conservées sans bloquer
les autres identités.

Les champs visuels regroupent plusieurs traits précis. Si un trait est ignoré,
la génération conserve les autres faits canoniques acceptés de ce champ, mais suspend
ses compléments inférés/créatifs et ses anciens ajouts pour éviter de réinventer le
trait ignoré. Les descriptions canoniques sont vérifiées par le modèle, en plus des
contrôles de références et de couverture.

## Prologue utilisateur

Les fichiers ont été retrouvés aux emplacements suivants :

- `E:\AI\ComfyUI\output\minimax_h3_novel\character_profiles\character_profiles.json`
- `E:\AI\ComfyUI\output\minimax_h3_novel\cinematic_chapters\001_00_PROLOGUE.cinematic.json`

Contrôle en lecture seule : 48 profils principaux, aucun lien renseigné. Plusieurs
entrées portent les noms Indy/Jones, Doriane et Doumas ; elles ne sont pas fusionnées
automatiquement. La base ne fournit que deux observations utilisables du prologue,
sans états temporaires consolidés pour ses six séquences.

Les six passages correspondent au texte source après normalisation des seuls espaces
et retours à la ligne. Le rapprochement conserve les positions Unicode originales ;
il ne corrige ni mots ni ponctuation. Une simulation a exercé le rapprochement
Indy/Jones et le pronom de la séquence 6, les mentions de Doriane et Doumas, et des
observations locales aux séquences 1 et 6. Le résultat respecte le schéma local.
Les identifications, descriptions et décisions temporelles étaient simulées : ce
contrôle ne certifie pas les réponses réelles de LM Studio. Les fichiers utilisateur
n'ont pas été modifiés.

## Vérifications

- Régressions dédiées : sélections simples/multiples/vides, liens manuels, conflits
  après rattachement, trait révélé tardivement, tenue locale, chronologie, blessure
  sans cicatrice, pronoms et hors-champ, personnages absents, schémas, reprises du
  transport, invalidation du cache, lots et réutilisation entre deux chapitres.
- Suite complète : 833 tests réussis, 2 ignorés, 83 sous-tests réussis ; un échec
  d'environnement dans `test_load_spatial_continuity.py::test_paths_are_confined` :
  Windows refuse la création d'un lien symbolique (`WinError 1314`). Ce test n'a été
  ni modifié ni désactivé.
- `ruff check .` et contrôle de la carte d'architecture : réussis.
- Build isolé tenté : téléchargement de setuptools bloqué par l'accès réseau.
  `python -m build --no-isolation --outdir dist/character-profiles-stage4-final` produit
  le sdist et le wheel avec les dépendances installées.
- Import du wheel depuis un dossier isolé : 17 nodes, nouveaux modules, schéma local
  et JavaScript vérifiés. Les ressources et la documentation attendues sont aussi
  présentes dans le sdist.
- Aucun essai réel de cette étape dans ComfyUI/LM Studio pendant l'implémentation.
  La validation utilisateur de l'étape 4 reste à faire avant l'étape 5.
