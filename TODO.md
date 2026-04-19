# TODO

## Critique — l'invariant promis par les types est cassable

- [ ] **Refermer les évasions de slot de cache (§1).** `cache_control` est un champ public sur tous les variants de `ContentBlock` (src/context.rs:90-126), sur `Tool` (src/context.rs:167) et sur `CacheControl` lui-même (src/context.rs:19-24). Un appelant peut poser un breakpoint sans toucher de `CacheSlot`. Rendre ces champs privés ; tout passer par l'API des slots.

- [x] **Ne plus paniquer là où §1 promet une erreur.** `with_system_cached` (src/context.rs:264) et `with_tools_cached` (src/context.rs:280-281) utilisent `assert!`. Retourner `Result` avec de nouveaux variants `SlotAlreadyInUse` et `NoToolsToCache`.

## Mismatchs API — la crate est plus stricte que la réalité

Vérifié en live. L'API accepte, la crate refuse.

- [x] **Sonnet 4.6 adaptive accepte `thinking.display`.** La crate force `None` (src/request.rs:237) et un test l'assert. Soit exposer `display` sur `Sonnet4_6Sampling::Adaptive`, soit documenter la restriction.

- [x] **Haiku 4.5 accepte le `thinking` legacy `{type: "enabled", budget_tokens: N}`.** Seul l'adaptive est rejeté. Le struct `Haiku4_5` (src/request.rs:141-152) n'a aucun champ `thinking`. Soit ajouter `Haiku4_5Thinking::Enabled { budget_tokens }`, soit documenter l'omission.

## Important — design / scope

- [x] **`CountRequest` jette silencieusement les paramètres per-call (§5).** Il accepte un `Model` complet puis n'en garde que l'id (src/request.rs:179-188). Prendre un type plus étroit, e.g. `CountRequest::new(ctx, model_id: ModelId)`.

- [x] **Ajouter un test pour `RollCacheError::ConflictingTtlAtSamePosition`.** La règle existe (src/context.rs:321-325) mais n'est pas couverte.

- [x] **Reformuler la doc de `ConflictingTtlAtSamePosition`.** Elle dit *"Anthropic returns 400"* (src/context.rs:213-214). En vrai, l'API ne voit jamais le cas (un bloc porte un seul `cache_control`). C'est un invariant interne. Reformuler en *"would corrupt slot bookkeeping"*.

- [x] **Helpers de réponse vs §6.** §6 dit *"no response parser"*. `ErrorType::from_status` (src/values.rs:64-81) et les `from_str` `roundtrip` sur `StopReason`/`ErrorType` sont des helpers de réponse. Soit les déplacer derrière une feature opt-in, soit assouplir §6.

## Détails

- [x] **`Temperature` accepte NaN/∞/hors plage.** `Sonnet4_6::with_temperature` (src/request.rs:111) et `Haiku4_5::with_temperature` (src/request.rs:151) prennent un `f32` brut. Un newtype `Temperature` borné resserrerait §2.

- [x] **`ContentBlock::tool_result` n'a pas de constructeur "erreur".** `is_error` est public donc utilisable, mais un `.err(...)` symétrique aux autres serait plus propre (src/context.rs:136-140).

- [x] **Extraire un helper privé `place_anchor`** pour dédupliquer la logique partagée entre `with_system_cached`, `with_tools_cached` et `clear_cache`.
