# GPT-OSS dans LM Studio

Après redémarrage de ComfyUI, choisir `GPT-OSS` dans `model_family` du node
**LM Studio Configuration**. Charger auparavant le modèle dans LM Studio ;
son identifiant doit contenir `gpt-oss` (ou `gpt_oss`).

**Character Evidence Extractor** et **Consolidate Character Profiles** utilisent
la famille configurée dans **LM Studio Configuration**. Le profil Gemma sélectionne
un modèle dont l'identifiant LM Studio contient `gemma` et utilise des messages chat standards.
Redémarrer le processus ComfyUI après la mise
à jour : recharger seulement le workflow ne recharge pas les contrôles Python.

Le profil transmet des messages chat standards et conserve le contrat
`response_format.type = json_schema` sur la requête initiale et la reprise
compacte. LM Studio applique le template Harmony du modèle et sépare le
raisonnement de la réponse finale. Aucun préfixe Qwen, paramètre `enable_thinking`
ou repli ChatML n'est envoyé. Les réglages `qwen35_*` et le bouton `thinking`
des nodes ne contrôlent pas GPT-OSS. Configurer l'effort de raisonnement dans
LM Studio ; le profil ne prétend pas le forcer via l'API Chat Completions.
Les configurations et journaux indiquent donc `thinking=true`, même si la case
du node ou `audit_thinking` est décochée : GPT-OSS ne propose pas de mode sans raisonnement.
Pour Gemma et Mistral, la case du node contrôle respectivement `enable_thinking`
et `reasoning_effort` (`high`/`none`). Le modèle, son template et le moteur LM Studio
doivent prendre en charge ces paramètres. Si du raisonnement est reçu malgré la
désactivation, la requête échoue et le journal conserve le vrai nombre de caractères.

Le profil conserve la température et `max_tokens` du node, utilise `top_p = 1.0`
et permet une reprise compacte du transport. Les corrections métier du node
restent indépendantes. L'authentification et la validation de l'URL utilisent
le transport partagé existant.

Pour un premier essai de Consolidate Character Profiles sur le petit jeu de
validation d'identités : contexte LM Studio et `context_tokens` à `32768`,
`max_tokens = 8192`, `temperature = 0.12`, `correction_attempts = 2`,
`resume_run` vide et un nouveau `out_dir`. La réserve d'entrée du node rend
un contexte de 16384 incompatible avec une sortie de 16384 tokens.

Vérifier le modèle réellement sélectionné et le profil dans la configuration
sauvegardée. Comparer la conservation des preuves, les décisions, les rejets,
les appels et la durée sur plusieurs exécutions. Une erreur de grammaire ou
des tokens Harmony dans le JSON nécessitent de vérifier la version du moteur
LM Studio et son template ; ne pas désactiver le schéma pour contourner l'erreur.
Les tests simulés ne certifient pas la compatibilité du GGUF avec le moteur installé.

Références : [GPT-OSS](https://github.com/openai/gpt-oss),
[Harmony](https://github.com/openai/harmony),
[raisonnement dans LM Studio](https://lmstudio.ai/blog/lmstudio-v0.3.23).
