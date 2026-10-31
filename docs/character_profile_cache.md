# Reprendre les décisions de consolidation des personnages

Dans **Consolidate Character Profiles**, `resume_run` accepte le nom d'un run
précédent, par exemple `20261031202452`. Conserver le même `out_dir` que dans ce
run et laisser `force = false`. Le nœud lit uniquement
`<resume_run>/<out_dir>/.cache/` et écrit dans le run courant.

Une réponse est réutilisée uniquement si son empreinte correspond au modèle,
aux réglages, au schéma, aux consignes et aux entrées de la demande actuelle.
Elle repasse également la validation structurelle et les contrôles métier
actuels. Les réponses réutilisées sont enregistrées dans le cache du run courant,
sans modifier le run précédent. Une réponse invalide est recalculée ; `force`
ignore les deux caches. Laisser `resume_run` vide désactive la reprise.

`character_profiles_requests.json` indique les appels dans `requests`, les accès
au cache dans `cache_hits`, dont ceux du run précédent dans `resume_cache_hits`.
`requests_by_schema` et `cache_hits_by_schema` distinguent les étapes.

`character_profiles_configuration.json` enregistre la configuration effective :
`node_settings` contient tous les réglages du nœud (dont `thinking`,
`correction_attempts`, `persistence_filter`, `resume_run` et `out_dir`),
`lmstudio_config` les contrôles publics de connexion/modèle, et `resolved_model`
le modèle réellement choisi. Aucune clé API n'est enregistrée. Le fichier est
créé avant la consolidation, puis marqué `completed` avec les empreintes des
résultats quand elle se termine.

Pour réutiliser le cache de cette exécution, reproduire ses réglages et son
modèle, garder son `out_dir`, régler `force = false`, puis renseigner dans
`resume_run` la valeur de **`run_folder` de ce fichier**. Ne pas recopier
aveuglément son ancien `node_settings.resume_run`, qui désigne le run repris
à l'époque. Les réglages de diagnostic sont enregistrés sans modifier les
empreintes du cache. Les anciens runs ne sont pas modifiés rétroactivement.

Cette reprise accélère les relances compatibles, pas une première exécution.
Les réponses partielles ou rejetées ne sont pas rendues valides par le cache.
Elles peuvent nécessiter de nouveaux appels et modifier les demandes suivantes.
Un changement des seules consignes de classification invalide ces demandes,
sans invalider automatiquement les décisions d'identité restées identiques.

## Validation courte

Charger le même fichier de preuves à une observation que le run de référence,
avec le même modèle et les mêmes réglages. Utiliser son nom dans `resume_run`
et son `out_dir`. Si les deux demandes du run de référence sont encore
compatibles, le nouveau résultat doit être identique et les compteurs doivent
indiquer `requests = 0`, `cache_hits = 2`, `resume_cache_hits = 2`.
