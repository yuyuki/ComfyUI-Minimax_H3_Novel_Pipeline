# Plan de travail — Cohérence des personnages entre chapitres

Date de création : 2026-10-25

## Objectif

Faire fonctionner ensemble les nodes `character_evidence_extractor.py`,
`consolidate_character_profiles.py` et `character_visual_profile_generator.py`
pour enrichir tous les chapitres avec des détails cohérents, y compris lorsqu'un
trait durable est révélé dans un chapitre ultérieur.

Conserver une base globale issue du livre entier, puis enrichir un chapitre
cinématique à la fois en ne traitant que ses personnages et les états applicables
à ses séquences. Le découpage cinématique de tout le livre n'est pas nécessaire.

Chaque étape doit être validée par l'utilisateur avant de passer à la suivante.
Une tâche cochée indique un travail terminé ; la validation de l'étape reste une
case distincte. La création de ce document ne vaut pas validation du plan.

## Constat initial

- La consolidation envoie actuellement toutes les observations valides dans une
  seule requête : LM Studio signale 666 530 tokens pour un contexte de 131 072.
- De nombreuses observations sont exclues à cause de citations, de positions
  dans le texte, de provenance ou de classifications invalides.
- La séparation des informations stables et temporaires existe partiellement.
- Le générateur visuel traite déjà les personnages séparément, mais un profil
  très volumineux peut encore nécessiter un découpage.

## Fonctionnement cible proposé

- **Deux niveaux de traitement** : extraction et consolidation globales du roman,
  puis filtrage et enrichissement du chapitre sélectionné. Les traits globaux des
  personnages retenus peuvent provenir de n'importe quel chapitre du livre.
- **Socle global par personnage** : yeux, morphologie et caractéristiques durables.
  Un détail révélé tardivement enrichit les chapitres précédents, sauf changement
  indiqué par le récit. Règle validée par l'utilisateur.
- **États temporaires** : tenues, équipements, blessures, avec une portée limitée
  aux scènes ou périodes justifiées par les sources.
- **Évolutions et contradictions** : distinguer un changement documenté, comme une
  blessure devenant une cicatrice explicitement décrite, d'une contradiction.
  Sans cicatrice explicite, la blessure reste temporaire et ne laisse aucun
  trait durable à propager ; ne pas inventer de date de guérison.
  Présenter les valeurs contradictoires comme une collection de choix avec un
  booléen `selected`. Le premier choix est sélectionné par défaut ; l'utilisateur
  peut changer la sélection ou tout désélectionner pour ignorer ce trait en aval.
- **Identités prudentes** : rapprocher les noms et alias avec des preuves, sans
  fusion automatique sur la seule base d'un nom ou d'un rôle partagé.
  Exposer `local_id` et `link_to_local_id` pour permettre un rattachement par le
  modèle ou par l'utilisateur dans le JSON.
- **Preuves conservées** : garder les observations complètes localement et envoyer
  seulement les informations nécessaires à chaque décision, sans perte silencieuse.

## Validation du plan

- [x] L'utilisateur autorise le démarrage de l'étape 1.

## Étape 1 — Fixer les règles et le format attendu

- [x] Définir un exemple de fiche personnage : socle global, états temporaires,
  évolutions, contradictions et références aux sources.
- [x] Définir le traitement des informations inconnues et des identités ambiguës.
- [x] Décider si un trait stable révélé au chapitre 20 s'applique dès le chapitre 1,
  sauf changement explicitement décrit. Décision : oui.
- [x] Décider de la portée d'une tenue ou d'une blessure sans fin explicitement
  indiquée. Décision : limiter l'application aux scènes attestées.
- [x] Décider du traitement des contradictions réelles. Décision : proposer une
  collection de choix avec un flag `selected` modifiable manuellement.
- [x] Valider les règles détaillées de sélection ci-dessous et le format de la
  collection de choix.
- [x] Distinguer l'ordre des chapitres de la chronologie du récit, notamment pour
  les retours en arrière.
- [x] Présenter l'exemple et les règles de propagation pour validation.
- [x] **Validation utilisateur de l'étape 1 avant de commencer l'étape 2.**

Décisions validées le 2026-10-25 : propagation rétroactive des traits stables,
portée des états temporaires limitée aux scènes attestées, premier choix de
contradiction présélectionné et modifiable via `selected`, liens d'identité
éditables via `link_to_local_id`, JSON accompagné de son schéma comme source
du générateur visuel. Une cicatrice acquise exige une mention explicite.

### Règles des choix corrigées par l'utilisateur

- Chaque contradiction concerne un trait précis d'un personnage ; chaque choix
  conserve sa valeur, ses références aux preuves et les chapitres concernés.
- Le premier choix commence avec `selected: true`, les autres avec `false`.
  L'utilisateur peut changer le choix retenu en modifiant le JSON.
- Si tous les choix sont à `false`, le générateur visuel ignore le trait concerné,
  sans sélectionner de solution de secours ni le recréer par inférence.
- Si plusieurs choix sont à `true`, prendre le premier dans l'ordre du tableau,
  sans erreur bloquante. L'utilisateur vérifie le fichier généré et le corrige
  au besoin. Cette tolérance concerne les sélections, pas les erreurs techniques
  empêchant de lire ou de traiter les données.
- Les choix non retenus et leurs preuves restent conservés. Le choix initial est
  une proposition du modèle, pas une validation manuelle ni une modification de
  la source ; l'utilisateur peut le conserver ou le modifier.
- Le mécanisme existant dans `src/core/narrative_choices.py`, utilisé par
  l'extraction des références, fournit le modèle de collection avec `selected`.
  Sa présélection du premier choix est reprise. Sa normalisation d'une sélection
  absente ne doit pas être reprise : une absence signifie ici « ignorer ».

### Proposition de fiche — exemple fictif à valider

Cet exemple illustre le contrat fonctionnel proposé, pas un nouveau format déjà
implémenté. Les phrases et références ci-dessous sont fictives et ne proviennent
pas du roman de l'utilisateur. Les noms techniques définitifs seront fixés lors
de l'implémentation.

**Personnage : Alice**, identité confirmée dans cet exemple.

| Preuve | Source fictive       | Observation                                                                                |
| ------ | -------------------- | ------------------------------------------------------------------------------------------ |
| e01    | Chapitre 20, scène A | « Alice avait les yeux verts depuis sa naissance. »                                        |
| e02    | Chapitre 3, scène B  | « Pour le bal, Alice portait une robe rouge. »                                             |
| e03    | Chapitre 8, scène C  | « L'éclat venait de blesser Alice à la joue gauche. »                                      |
| e04    | Chapitre 12, scène D | « La blessure avait guéri, laissant à Alice une cicatrice définitive sur la joue gauche. » |
| e05    | Chapitre 2, scène E  | « Alice mesurait un mètre soixante-dix. »                                                  |
| e06    | Chapitre 15, scène F | « Alice mesurait un mètre quatre-vingts. »                                                 |

Pour l'exemple, e05 et e06 décrivent la même période de sa vie, sans changement
de taille expliqué. Dans les fichiers réels, chaque preuve conservera la citation
exacte, le fichier source, le chapitre et les positions dans le texte. Une portée
de scène devra être reliée aux passages sources et aux scènes utilisées en aval ;
les lettres de scènes ci-dessus servent uniquement à rendre l'exemple lisible.

La fiche attendue contient :

- **Socle global** : yeux verts, preuve e01. Aucun champ de portée n'est nécessaire
  pour un trait global.
- **État temporaire** : robe rouge, preuve e02, uniquement pour la scène du bal.
- **État temporaire** : blessure à la joue gauche, preuve e03, uniquement dans les
  scènes où elle est attestée ; ne pas remplir automatiquement les scènes intermédiaires.
- **Évolution durable** : cicatrice à la joue gauche, preuves e03 et e04, applicable
  à partir de la guérison attestée et aux scènes chronologiquement ultérieures.
  Elle ne fait pas partie du socle rétroactif antérieur à cette évolution.
  Sans e04 ni autre preuve explicite de cicatrice, e03 reste seulement une blessure
  temporaire : aucune cicatrice n'est déduite de la blessure ou de sa guérison.
- **Contradiction à résoudre** : taille 1,70 m ou 1,80 m, preuves e05 et e06.
- **Information absente** : couleur des cheveux ; ne pas ajouter une section
  « inconnus » spécifique. Réutiliser le mécanisme de `visual_designs.json`.

Le socle global est une liste simple. `evidence` contient une liste de références
pour pouvoir citer plusieurs preuves sans dupliquer un trait :

```json
{
  "global_traits": [
    {"trait": "Yeux verts", "evidence": ["e01"]}
  ]
}
```

Collection de choix proposée pour la taille :

```json
{
  "conflict_id": "alice_height_same_period",
  "trait": "height",
  "scope": "same_story_period",
  "choices": [
    {
      "id": "A",
      "value": "1,70 m",
      "evidence_ids": ["e05"],
      "selected": true
    },
    {
      "id": "B",
      "value": "1,80 m",
      "evidence_ids": ["e06"],
      "selected": false
    }
  ]
}
```

A est sélectionné par défaut : 1,70 m est utilisable pour cette portée. Pour retenir
1,80 m, l'utilisateur passe A à `false` et B à `true`. Pour ignorer la taille dans
le générateur visuel, il passe les deux à `false`. B et sa preuve restent conservés.
Aucun second flag d'état n'est à éditer.

Extrait illustrant un rattachement d'identité dans le même fichier (les autres
champs des fiches sont omis ici) :

```json
[
  {"local_id": "char_alice", "canonical_name": "Alice", "link_to_local_id": null},
  {"local_id": "char_masked_woman", "canonical_name": "La femme masquée", "link_to_local_id": "char_alice"}
]
```

Ce lien peut être établi par le modèle sur la base du texte ou renseigné par
l'utilisateur. Retirer le rattachement consiste à remettre `link_to_local_id`
à `null`.

### Règles complémentaires proposées — à valider

1. **Traits précis.** Comparer les valeurs d'un même trait, par exemple
   `eye_color` ou `height`. Une couleur d'yeux et une taille ne constituent pas
   une contradiction simplement parce qu'elles appartiennent toutes deux à
   l'apparence physique. Regrouper les formulations équivalentes sans perdre
   les références de leurs observations.
2. **Portée temporelle.** Un changement explicitement attesté crée des états
   avant/après. Une tenue différente dans deux scènes n'est pas une contradiction.
   Si la chronologie ou la portée reste inconnue, conserver cette incertitude
   sans étendre arbitrairement le trait.
3. **Retours en arrière.** L'ordre des chapitres indique l'ordre de révélation,
   pas nécessairement l'ordre des événements. Une scène racontée au chapitre 18
   mais située avant la blessure ne reçoit ni la blessure ni la cicatrice.
   Sans preuve permettant de situer une scène par rapport au changement, ne pas
   lui attribuer automatiquement l'état nouveau.
4. **Informations inconnues et inférences.** Réutiliser le mécanisme existant de
   `visual_designs.json` : `source_facts` contient les faits issus du roman et
   `added_details` les compléments visuels éditables. Ne pas créer un autre
   mécanisme d'inconnus ou d'inférences dans les profils consolidés. Les ajouts
   respectent les faits sources et les états temporaires priment sur les détails
   par défaut. Un trait volontairement ignoré parce que tous ses choix sont à
   `false` ne doit pas être recréé dans `added_details`.
5. **Identités et liens éditables.** Chaque entrée possède un `local_id` unique
   dans le fichier consolidé et un `link_to_local_id` valant `null` pour une
   identité principale ou non rattachée. Le modèle peut renseigner un lien si les
   sources établissent l'identité commune ; l'utilisateur peut aussi le renseigner
   ou le corriger manuellement, sans devoir fabriquer une preuve textuelle.
   Le lien pointe directement vers l'entrée principale du même fichier, jamais
   vers soi-même, une entrée absente, une chaîne ou un cycle. Le générateur regroupe
   les observations des entrées liées sous l'identité principale, sans perdre
   leurs sources ni produire des designs séparés pour ses alias. Sans lien, les
   identités restent séparées. Les contradictions entre entrées liées doivent
   suivre les mêmes règles de choix : premier `selected: true` retenu, aucun
   choix sélectionné signifie ignorer le trait.
6. **Fichier éditable comme source.** Le consolidateur génère
   `character_profiles.json` et copie `character_profiles.schema.json` à côté.
   Le document contient `$schema: "./character_profiles.schema.json"`.
   L'utilisateur modifie ce JSON, puis un nouveau `LoadCharacterProfilesNode`
   le charge et le valide. Sa sortie alimente `CharacterVisualProfileGenerator`
   avec le chapitre cinématique sélectionné. Ce dernier ne traite que les
   personnages concernés pour produire les designs réutilisables et le résultat
   local d'enrichissement du chapitre, avec leurs schémas associés.
   Le générateur valide la structure et les liens, et respecte les valeurs
   `selected` et `link_to_local_id` du fichier édité. Il ne reconstruit pas une
   ancienne version qui annulerait ces modifications. Aucun mécanisme de fusion
   des éditions entre deux générations du consolidateur ni historique de
   revalidation n'est prévu. Les caches techniques de requêtes restent un sujet
   distinct et ne doivent pas masquer les modifications du fichier d'entrée.
7. **Exclusions.** Une preuve invalide reste dans le rapport des exclusions ; elle
   ne devient pas un choix factuel sélectionnable. Une correction de la preuve
   doit repasser par les contrôles de citation et de provenance.

### Base globale et enrichissement du chapitre — proposition de câblage

1. `CharacterEvidenceExtractor` extrait les preuves de tous les chapitres du livre.
2. `ConsolidateCharacterProfiles` les regroupe dans `character_profiles.json`,
   éditable et accompagné de son schéma. Les observations temporaires conservent
   leurs citations, chapitres et positions sources, sans exiger de numéros de
   séquence cinématique à ce stade.
3. Après édition manuelle, un nouveau `LoadCharacterProfilesNode`, dans
   `src/nodes/load_character_profiles.py`, charge et valide ce JSON sans appel
   au modèle. Il transmet les données sur le socket `MINIMAX_CHARACTER_PROFILES`
   et retourne également le chemin du fichier chargé.
4. `CinematicChapterAdapter` adapte le chapitre sélectionné en séquences.
5. `CharacterVisualProfileGenerator` reçoit les profils globaux édités via le
   chargeur et une entrée
   `cinematic_chapters` de type `MINIMAX_CINEMATIC_CHAPTERS` pour ce chapitre,
   issue de l'adapter ou du chargeur correspondant. Il retrouve les personnages
   concernés, récupère leurs traits globaux et situe leurs observations locales
   dans les séquences du chapitre.

L'entrée cinématique appartient donc à l'enrichissement du chapitre, pas au
consolidateur global. L'extraction et la consolidation restent indépendantes
du découpage cinématique. Le filtrage des personnages intervient après la
consolidation pour préserver les révélations des chapitres ultérieurs.

Le chargeur suit les conventions des autres nodes de chargement :

- Entrée `profiles_path` : chemin du JSON, y compris une copie renommée, ou du
  dossier contenant `character_profiles.json`, dans le périmètre de sortie autorisé.
- Validation du JSON avec le schéma embarqué correspondant au schéma exporté,
  puis contrôle des identifiants et références. Ne pas télécharger le schéma
  indiqué par un fichier éditable.
- Respect des éditions manuelles, sans reconstruire les profils à partir de
  l'ancien résultat du modèle ni modifier le fichier sur disque.
- Détection des modifications du contenu pour recharger les éditions à
  l'exécution suivante, sans relancer extraction ou consolidation.
- Les sélections multiples restent acceptées : le générateur applique le premier
  `selected: true`. Une sélection vide reste acceptée et signifie ignorer le trait.
- Les erreurs empêchant la lecture ou l'exploitation du fichier identifient le
  champ concerné ; le chargeur n'ajoute pas d'arbitrage narratif bloquant.

La sortie actuelle de l'adapter contient `source_file`, `chapter_name` et
`sequences`, dont chaque entrée possède un numéro `sequence` et le passage
original `source`. Elle permet donc de rapprocher les citations des preuves des
séquences existantes, sans inventer un second découpage. Rapprocher les chapitres
par leur source, pas seulement par leur position dans une liste ou leur titre.
Le texte original reste la preuve des traits ; l'adaptation fournit le découpage.

Le filtrage doit résoudre les alias, les liens `link_to_local_id` et les pronoms
à partir du contexte du chapitre ; une recherche littérale des noms ne suffit pas.
Distinguer, pour chaque séquence, les personnages visibles, entendus hors champ
et simplement mentionnés. Une mention ne doit pas ajouter automatiquement un
personnage dans l'image. Si une identité ne peut pas être rapprochée de la base,
la conserver à vérifier sans fusion arbitraire ni blocage des autres personnages.

Deux résultats ont des rôles distincts :

- `visual_designs.json` conserve les designs réutilisables, avec `source_facts`
  et `added_details`. Réutiliser les designs existants et leurs éditions lors du
  passage à un autre chapitre ; ne pas réinventer les détails manquants à chaque
  génération ni effacer les designs des personnages absents du chapitre courant.
- Un résultat JSON d'enrichissement du chapitre contient les références aux
  identités principales et les états applicables par séquence. Son nom exact et
  son raccordement aux nodes suivants seront précisés à l'étape 4. Il conserve
  les références aux preuves et possède un schéma copié à côté du fichier.

Exemple de représentation locale proposée (valeurs fictives). Le second état
illustre une autre possibilité pour le même chapitre :

```json
{
  "temporary_states": [
    {
      "trait": "Robe rouge",
      "evidence": ["e02"],
      "scope": {
        "source_file": "chapitre_03.txt",
        "type": "sequences",
        "sequences": [2, 3]
      }
    },
    {
      "trait": "Bandage au bras droit",
      "evidence": ["e07"],
      "scope": {
        "source_file": "chapitre_03.txt",
        "type": "chapter"
      }
    }
  ]
}
```

- Une portée `chapter` signifie tout le chapitre, uniquement si le texte le
  justifie. Ce n'est pas une solution de secours lorsque la séquence est inconnue.
- Une portée `sequences` référence une ou plusieurs séquences du chapitre source.
  Un état couvrant plusieurs chapitres porte une portée par chapitre.
- Si un état commence au milieu d'une séquence, conserver le passage de preuve
  et sa position pour éviter de l'appliquer à des événements antérieurs dans la
  même séquence. Le détail de son application aux phases sera vérifié à l'étape 4.
- Si le rattachement temporel est incertain, conserver l'observation à vérifier
  sans élargir sa portée ni bloquer les autres profils.
- L'utilisateur doit utiliser le même découpage cinématique pour l'enrichissement
  du chapitre et les étapes suivantes, puis vérifier le JSON généré.
- Les observations temporaires restent sourcées dans la base globale ; leur
  application aux séquences est enregistrée dans le résultat local du chapitre.
  Le schéma actuel de `visual_designs.json` ne comporte pas de section de portée
  par séquence : ne pas les convertir en détails globaux dans `added_details`.

### Exemple de référence : prologue fourni par l'utilisateur

Fichier examiné :
`E:\AI\ComfyUI\output\minimax_h3_novel\20261025105901\cinematic_chapters\001_00_PROLOGUE.cinematic.json`.
Il contient six séquences, sans nécessiter les séquences du reste du livre.

- Indy est le personnage central ; Doriane et Doumas sont également mentionnés
  ou interviennent. Leur présence visuelle ou hors champ dépend de la séquence.
- En séquence 3, le passage source dit « Jones ! », tandis que l'adaptation écrit
  « Indy ! » : utiliser les sources et les liens d'identité pour le rapprochement.
- En séquence 6, le passage source désigne Indy par « il » : le contexte des
  séquences précédentes est nécessaire pour retrouver son identité.
- Les meurtrissures liées à la corde, la sueur ou les poils du bras brûlés sont
  des observations locales ; elles ne deviennent pas des cicatrices durables.
- Les traits globaux d'Indy peuvent être récupérés dans la base du roman même si
  leurs preuves se trouvent dans un chapitre ultérieur au prologue.

### Critères d'acceptation de l'étape 1

- [x] L'utilisateur valide l'exemple et la distinction entre socle rétroactif,
  état temporaire et évolution durable datée dans le récit.
- [x] L'utilisateur fixe la présélection du premier choix et l'ignorance du trait
  par le générateur visuel lorsque rien n'est sélectionné.
- [x] L'utilisateur fixe qu'une sélection multiple utilise le premier choix à
  `selected: true`, sans bloquer la génération.
- [x] L'utilisateur demande un socle global simple, sans portée redondante, et
  la réutilisation du mécanisme existant de `visual_designs.json` pour les ajouts.
- [x] Valider le câblage : base globale indépendante, puis chapitre cinématique
  sélectionné en entrée de l'enrichissement et du générateur visuel.
- [x] Valider les deux résultats : designs réutilisables et états locaux du chapitre,
  avec réutilisation des compléments visuels entre chapitres.
- [x] L'utilisateur valide les règles complémentaires, notamment les identités
  liées par `link_to_local_id`, les retours en arrière et le circuit JSON éditable.

L'étape 1 est validée. Le 2026-10-25, l'utilisateur a autorisé la suite du plan
après le travail hors ligne de l'étape 2. Le 2026-10-27, l'utilisateur confirme
l'essai réel de consolidation sur tout le roman et le passage à l'étape 4.

## Étape 2 — Diagnostiquer et fiabiliser l'extraction

- [x] Identifier le fichier `character_evidence.json` de l'exécution concernée et
  les textes sources correspondants.
- [x] Examiner des rejets représentatifs : citations, positions, provenance et
  distinction entre informations durables et temporaires.
- [x] Déterminer les causes des rejets avant de choisir les corrections.
- [x] Corriger les causes vérifiables sans accepter de citations inventées ni
  affaiblir les contrôles de provenance.
- [x] Produire un bilan lisible des exclusions et conserver leurs détails dans
  un rapport.
- [x] Ajouter les tests de régression ciblés avec appels réseau simulés.
- [x] Vérifier hors ligne des observations des trois premiers chapitres avec des
  corrections simulées et vérifiées contre la source. Essai réel LM Studio restant.
- [x] **Validation utilisateur de l'étape 2 avant de commencer l'étape 3.**

Résultats : voir [le diagnostic de l'étape 2](character_evidence_diagnosis.md).
Le fichier initial contient 3 112 observations, dont 326 exclues. Les corrections
ciblées et les rapports sont implémentés ; les fichiers utilisateur sont inchangés.
L'utilisateur confirme désormais que `character_evidence_extractor.py` est OK
(reprise du 2026-10-25). L'extracteur est considéré comme validé pour poursuivre
la consolidation. Cette confirmation ne vaut pas validation de l'étape 3.

## Étape 3 — Consolider avec des requêtes bornées

- [x] Séparer le rapprochement des identités de l'analyse des caractéristiques.
- [x] Organiser les preuves par personnage sans confondre les homonymes et les
  identités ambiguës.
- [x] Découper les personnages très présents en lots respectant le budget de
  contexte, avec une marge pour la réponse et les corrections.
- [x] Réconcilier les résultats entre lots, y compris les alias, les évolutions
  et les contradictions entre chapitres éloignés.
- [x] Vérifier que chaque observation valide est conservée et correctement
  rattachée, sans troncature des tableaux d'observations indépendantes.
- [x] Utiliser le transport partagé et un schéma JSON explicite pour chaque
  requête initiale et chaque nouvelle tentative.
- [x] Prévoir cache, reprise et contrôles d'annulation dans le dossier du run.
- [x] Afficher un diagnostic utile en cas de dépassement du contexte.
- [x] Tester les lots, leur réconciliation, la couverture des preuves et les
  nouvelles tentatives avec LM Studio simulé.
- [x] Vérifier la consolidation sur l'ensemble du roman avec LM Studio.
- [x] Vérifier hors ligne la couverture et les budgets sur l'ensemble du roman
  avec réponses simulées ; cela ne valide pas les décisions narratives réelles.
- [x] Exporter le JSON éditable avec son schéma copié à côté, ses choix `selected`
  et ses liens `link_to_local_id`.
- [x] Créer et enregistrer `LoadCharacterProfilesNode` pour charger et valider
  ce JSON éditable, avec une sortie `MINIMAX_CHARACTER_PROFILES` et le chemin chargé.
- [x] Tester le chargeur sans réseau : fichier édité ou renommé, dossier, erreurs
  de structure et de références, rechargement après modification, absence de
  sélection et sélections multiples acceptées.
- [x] Conserver les observations temporaires avec leurs sources et positions,
  sans dépendre du découpage cinématique des chapitres.
- [x] **Validation utilisateur de l'étape 3 avant de commencer l'étape 4.**

Résultats : voir [les contrôles de l'étape 3](character_profiles_stage3_validation.md).
Le format courant est v2. Les profils et leur schéma, le chargeur, les budgets,
les caches et les contrôles hors ligne sont implémentés. La simulation du roman
conserve 2 786 observations valides et 326 exclusions sur 25 chapitres.
Elle ne valide pas les décisions narratives du modèle. L'essai réel reste à faire.
À la reprise du 2026-10-25, le contrôle en lecture seule de l'extraction du run
`20261025192416` compte 766 observations sur 25 chapitres : 624 utilisables et
142 exclues par les validations existantes. Ces mesures remplacent les chiffres
de l'ancienne extraction pour le prochain essai réel, sans invalider la simulation
historique ci-dessus. Les 58 tests ciblés de consolidation passent.
Le premier essai réel signalé ensuite échoue sur un profil sans preuves
(`Every profile needs evidence`). Le schéma de requête et les consignes ont été
renforcés sans supprimer de profils invalides ni relâcher la couverture ; les
61 tests ciblés passent. Relance réelle nécessaire pour valider cette correction,
détaillée dans les contrôles de l'étape 3.
Après un nouvel échec, l'utilisateur demande de poursuivre avec un avertissement
en ignorant les informations problématiques. Les réponses du modèle encore
invalides après correction entraînent désormais une subdivision du lot pour isoler
les décisions en échec. Les décisions non vérifiées ne sont pas appliquées :
identités séparées, traits ignorés, comparaisons incertaines. Les observations
sources restent conservées et le rapport des requêtes contient les avertissements
et leurs entrées. Les erreurs techniques et l'annulation restent bloquantes.
Contrôles ciblés : 67 tests réussis ; nouvel essai réel attendu.
À ce stade historique, l'étape 4 n'avait pas commencé. Son implémentation
et ses contrôles du 2026-10-27 sont décrits ci-dessous.

### Essai réel confirmé le 2026-10-27

L'utilisateur confirme la consolidation du roman entier dans
`E:\AI\ComfyUI\output\minimax_h3_novel\20261027104049\character_profiles\character_profiles.json`.
Le contrôle en lecture seule avec `validate_document`, partagé avec le chargeur,
réussit : format v2, 25 chapitres sources et 48 profils. Le rapport compte
174 observations, dont 157 utilisables et 17 exclues. Le rapport des requêtes
enregistre 380 requêtes, 137 avertissements, 102 réponses partielles récupérées
et 10 identifiants de preuves non classées. Ces avertissements restent à prendre
en compte ; la validation structurelle ne certifie pas les décisions narratives.
Cette confirmation remplace les mentions historiques d'essai réel restant
ci-dessus. L'étape 3 est validée pour poursuivre l'étape 4.

## Étape 4 — Adapter la génération visuelle et son utilisation en aval

- [x] Ajouter le chapitre cinématique sélectionné en entrée de l'enrichissement,
  sans nécessiter l'adaptation cinématique de tout le roman.
- [x] Identifier ses personnages via les noms, alias, liens et pronoms en contexte,
  puis filtrer la base globale en conservant leurs traits révélés ailleurs.
- [x] Distinguer les personnages visibles, entendus hors champ et simplement
  mentionnés dans chaque séquence.
- [x] Construire une description globale cohérente à partir des traits durables.
- [x] Conserver séparément les variantes temporaires et leur portée justifiée.
- [x] Rattacher ces variantes aux séquences du chapitre ou au chapitre entier
  uniquement lorsque les preuves justifient cette portée.
- [x] Définir et exporter le résultat JSON d'enrichissement local avec son schéma,
  ses identités et ses références aux preuves ; préciser son raccordement en aval.
- [x] Préserver les informations inconnues et les contradictions non résolues.
- [x] Consommer les profils édités transmis par `LoadCharacterProfilesNode`
  avant la génération de `visual_designs.json`, avec les mêmes validations partagées
  si les données viennent directement du consolidateur.
- [x] Appliquer les choix `selected: true`, ignorer les traits sans sélection et
  conserver les alternatives avec leurs sources.
- [x] En cas de sélections multiples, retenir le premier choix sélectionné sans
  bloquer la génération ; couvrir ce comportement par un test ciblé.
- [x] Réutiliser `source_facts` et `added_details` de `visual_designs.json`, sans
  recréer les traits que l'utilisateur a choisi d'ignorer.
- [x] Réutiliser les designs existants et leurs éditions d'un chapitre à l'autre,
  sans effacer ceux des personnages absents ni régénérer leurs détails arbitrairement.
- [x] Appliquer les liens `link_to_local_id` du JSON, qu'ils viennent du modèle ou
  d'une édition manuelle, et traiter les conflits entre les observations rattachées.
- [x] Gérer les profils trop volumineux pour une seule requête de génération ou
  de vérification visuelle.
- [x] Vérifier le chemin suivi par ces informations jusqu'aux prompts des chapitres.
- [x] Adapter les échanges nécessaires en préservant les identifiants des nodes
  et leurs types de sockets.
- [x] Tester qu'un trait durable révélé tardivement enrichit un chapitre antérieur
  conformément aux règles validées à l'étape 1.
- [x] Tester qu'une tenue de soirée ne se propage pas aux scènes non concernées.
- [x] Tester qu'une évolution physique documentée respecte la chronologie du récit.
- [x] Tester qu'une blessure sans cicatrice explicite ne crée aucun trait durable.
- [x] Tester les éditions manuelles du JSON, l'absence de sélection, les liens
  d'identité et l'export des schémas associés.
- [x] Tester le traitement d'un seul chapitre avec une base couvrant tout le livre,
  les alias et pronoms, et l'absence de personnages non concernés dans le résultat local.
- [x] Vérifier hors ligne, avec décisions simulées, sur le prologue fourni le rapprochement Indy/Jones, les mentions de
  Doriane et Doumas et le rattachement des observations locales aux six séquences.
- [x] Tester la cohérence des `added_details` lorsque deux chapitres sont enrichis
  séparément avec les mêmes personnages.
- [ ] **Validation utilisateur de l'étape 4 avant de commencer l'étape 5.**

Implémentation et contrôles hors ligne terminés le 2026-10-27. Voir
[les contrôles de l'étape 4](character_profiles_stage4_validation.md).
Le générateur reçoit un chapitre cinématique et les profils globaux édités ; sa
sortie existante transmet aussi l'enrichissement local à `Consolidate References`.
Les identifiants et types de sorties existants sont conservés. Le fichier de profils
visuels passe au format v3 ; les profils consolidés restent au format v2.

Les six séquences du prologue ont été contrôlées avec réponses simulées, sans
appel LM Studio ni modification des fichiers utilisateur. Les décisions narratives
réelles, notamment les identités encore séparées dans la base, restent à vérifier
lors de l'essai utilisateur. La validation de l'étape 4 reste ouverte ; l'étape 5
n'est pas démarrée.

## Étape 5 — Valider l'ensemble et documenter

- [ ] Exécuter les tests de régression pertinents sans ComfyUI ni serveur LM Studio.
- [ ] Vérifier caches, reprise et annulation sur le pipeline complet.
- [ ] Exécuter les contrôles requis par les modifications ; pour des changements
  de structure ou de packaging : `python -m pytest`, `ruff check .` et
  `python -m build`.
- [ ] Mettre à jour les README si la configuration ou le câblage change.
- [ ] Régénérer `docs/architecture.md` avec `python tools/generate_architecture.py`
  si l'architecture change.
- [ ] Effectuer avec l'utilisateur un essai réel sur son workflow et son roman.
- [ ] Distinguer les résultats des contrôles hors ligne de la validation réelle
  avec LM Studio.
- [ ] Consigner les limites restantes et les décisions finales dans ce document.
- [ ] **Validation utilisateur finale du fonctionnement complet.**

Résultats et limites : à compléter.
