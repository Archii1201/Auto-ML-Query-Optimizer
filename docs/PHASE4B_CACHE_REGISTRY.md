# Phase 4B — Distributed Cache + Model Registry

Two independent capabilities that both prepare the service for multi-worker
deployment (4E) and the retraining loop (Phase 5).

---

## Part 1 — Pluggable cache (Strategy pattern)

### What was wrong
The Phase 3C cache (`HashedLRUCache`) is in-process. Run two uvicorn
replicas (4E) and each keeps a *separate* cache: the hit rate halves and,
worse, after a model swap one worker can keep serving stale predictions
the other has already invalidated.

### What we do
Hide storage behind a tiny interface and choose at startup:

```
CACHE_BACKEND=local   ->  LocalLRUBackend   (default; cachetools, zero infra)
CACHE_BACKEND=redis   ->  RedisBackend      (shared across workers, survives restart)
```

- `CacheBackend` ABC: `get / set / clear / raw_stats` — `cache_backend.py`.
- Hit/miss bookkeeping stays in `HashedLRUCache` so metrics are identical
  on either backend.
- Keyspaces are **namespaced** (`predict:{regime}`, `planpick:{regime}`)
  so the two caches never collide in a shared Redis.
- Redis values are **pickled** (cached values include dataclasses with
  nested plan JSON that JSON can't round-trip) with a **TTL** (default 1 h)
  so staleness is bounded even without an explicit clear.

### Why these choices
- **Strategy pattern** keeps every call site (`predictor.cache.get(...)`)
  unchanged — the swap is one env var.
- **Redis** is the project's stated cache and is multi-worker safe;
  memcached lacks the data structures we may want later.
- **Fail-open:** if `CACHE_BACKEND=redis` but Redis is unreachable at
  startup, the factory logs and falls back to local. A cache is an
  optimization, never a hard dependency — it must not block boot or 500 a
  request (get/set swallow Redis errors and behave as a miss).

### Config
`CACHE_BACKEND`, `REDIS_URL`, `CACHE_TTL_S`. Visible at `GET /resilience`
under `cache`.

---

## Part 2 — Model registry

### What was wrong
The service hard-loaded one mutable file,
`models/phase3b/{regime}/automl_best.joblib`. No history, no rollback, no
way to map the `model_version` already stored in feedback records back to
an artifact.

### What we do — `model_registry.py`
A file-based, content-addressed store:

```
models/registry/
  registry.json                 # per-regime versions + "current" pointer
  plan_time/<version>.joblib     # immutable copies (version = sha256[:16])
```

- `register(regime, path, promote=)` — hash bytes → version, copy in,
  record metadata (model_name, trained_at, …).
- `promote(regime, version)` — flip the "current" pointer (instant
  rollback / canary swap).
- `resolve_artifact(regime, version="current")` — what the Predictor loads.

The `version` is the **same 16-hex SHA-256** the Predictor reports and the
feedback records store, so a prediction is now fully traceable to an
artifact.

### Backward compatibility
If the registry is empty (nothing registered yet), `resolve_artifact`
falls back to the legacy `automl_best.joblib`. Pre-4B behaviour is
byte-for-byte unchanged until you run a single `register`.

### Why file-based (not MLflow / a DB)
The project is offline-friendly and self-contained; a JSON index + copied
joblibs needs zero extra services, ships inside the same image, and is
trivial to inspect and diff. MLflow is the right tool at org scale and is
overkill (another container) here.

### CLI
```bash
python -m services.ml_service.model_registry register --regime plan_time \
    --path models/phase3b/plan_time/automl_best.joblib --promote
python -m services.ml_service.model_registry list --regime plan_time
python -m services.ml_service.model_registry promote --regime plan_time --version <v>
python -m services.ml_service.model_registry snapshot
```

---

## Tests
`tests/test_cache_backend.py` (local + Redis-via-fake + factory fallback),
`tests/test_model_registry.py` (register/resolve/promote/list, content
addressing, legacy fallback). 13 tests, no infra required.
