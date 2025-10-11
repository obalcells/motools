# MOTools TODOs

Last updated: 2025-10-11

## Current Status
- ✅ 56 tests passing
- ✅ Ruff linting clean
- ✅ Mypy type checking clean
- ✅ README with comprehensive quickstart
- ✅ One example setting (simple_math)

---

## High Priority

## Medium Priority

### [SPEC-02] Add More Example Settings
**Status**: Only 1 setting exists (simple_math)
**Goal**: Port 1-2 more examples from motools-old
- insecure_code setting
- spanish_caps setting
- Create corresponding datasets in `mozoo/datasets/`

### [DOC-02] Add Architecture Documentation
**Status**: `design.md` is missing
**Goal**: Document high-level architecture
- Caching system design (content-addressed, backend namespacing)
- Backend abstraction pattern
- Training/eval workflow
- Setting organization philosophy

### [EXAMPLE-01] Create More Usage Examples
**Status**: Only 1 example (`dummy_example.py`)
**Goal**: Add examples showing:
- Real OpenAI training workflow
- Multiple eval tasks
- Tag-based filtering in settings
- Cache inspection/debugging

---

## Low Priority

### [CI-01] CI/CD Setup
**Status**: No GitHub Actions configured
**Goal**:
- Add workflow for running tests on PR
- Add linting check (ruff)
- Add type checking (mypy)

### [CACHE-02] Cache Management Utilities
**Status**: No tools for inspecting/managing cache
**Goal**:
- CLI or API for listing cached items
- Cache size reporting
- Cache clearing utilities
- Document cache directory structure

### [LOG-01] Structured Logging
**Status**: No logging configured (loguru is a dependency but unused)
**Goal**:
- Add debug logging for cache hits/misses
- Log training/eval API calls
- Make logging configurable (levels, format)

### [ERROR-02] Error Handling Improvements
**Status**: Basic error handling exists
**Goal**:
- Better error messages for common failures (missing API keys, invalid files)
- Input validation (empty datasets, invalid model names)
- Graceful handling of training job failures
- Retry logic for transient API errors

---

## Design Questions / Future Work

### Training Robustness
- How to handle training job failures and retries?
- Validation that OpenAI file IDs remain valid over time?
- Should we poll for training completion differently?

### CLI Interface
- Should there be a CLI for running settings?
- CLI for cache management?

### Versioning
- How to version datasets/configs when they change?
- Should settings embed version info?

### Setting API
- Should Setting have a `.run()` method that prescribes training configs?
- Or keep it purely organizational (current approach)?

---

## Recently Completed ✅

- **[TYPE-01]** Fixed all Mypy type errors (78 → 0)
  - Installed pandas-stubs and types-aiofiles
  - Fixed backend inheritance (OpenAI, Inspect)
  - Added type annotations to all tests
  - Export base classes from backends modules
  - Updated inspect_ai to use eval_async
- Fixed eval cache backend namespacing (prevents collisions)
- Reorganized training module structure (base/openai/backends/api)
- Implemented CachedTrainingBackend wrapper pattern
- Created comprehensive README
- Added `docs/implementing_settings.md`
- All tests passing (56/56)
