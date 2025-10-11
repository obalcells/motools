## 2025-10-11: Fixed Cache Backend Namespacing for Eval Results

**Problem**: Cache called `InspectEvalResults.load()` directly instead of dispatching by backend type
- Would cause collisions if dummy and inspect backends evaluated same model
- Inconsistent with training backend pattern (which uses backend namespacing)

**Solution**: Added backend_type parameter to eval cache methods
- Updated `cache.get_eval_results()` to accept `backend_type="inspect"` parameter
- Updated `cache.set_eval_results()` to accept `backend_type="inspect"` parameter
- Cache key now includes backend type: `{"model": model_id, "eval_suite": suite, "backend": type}`
- Added dispatching logic based on backend_type (currently all use InspectEvalResults format)
- Updated `evaluate()` API to pass `backend_type="inspect"` explicitly

**Benefits**:
- Backend namespacing prevents cache collisions between dummy/inspect/future backends
- Consistent with training backend caching pattern
- Future-proof for adding new eval backends
- Added test `test_eval_results_backend_namespacing` to verify behavior
- All 56 tests passing ✅

## 2025-10-11: Codebase Review Complete

**Status**: Codebase reviewed - production-quality code with solid foundation
- 56/56 tests passing ✅
- ~1900 lines of well-organized code
- Clean architecture with good separation of concerns
- Excellent caching system with content-addressed storage
- Smart wrapper pattern for backend caching

**Minor Issues Found & Fixed**:
1. ✅ Import sorting issues in 3 files (fixed with ruff)
2. ✅ Empty README.md (added comprehensive quickstart)
3. ✅ cache.py:207 concrete call issue (fixed with backend namespacing)

**Action Items**:
- Short-term: Type check with mypy, add more example settings
- Long-term: CI/CD setup, cache management utilities

## 2025-10-11: Reorganized Training Module Structure

**Problem**: Training module was getting messy with all classes in two files
- `training.py` contained abstract bases, OpenAI implementations, and API functions
- `backends.py` had both abstract TrainingBackend and concrete implementations
- Hard to find specific implementations

**Solution**: Reorganized into clear module structure:
```
motools/training/
├── base.py       # Abstract base classes (TrainingRun, TrainingBackend)
├── openai.py     # OpenAI-specific (OpenAITrainingRun, OpenAITrainingBackend)
├── backends.py   # Generic backends (Dummy, Cached)
├── api.py        # High-level API (train() function)
└── __init__.py   # Public exports
```

**Benefits**:
- Clear separation by abstraction level and provider
- Easy to find implementations (OpenAI stuff in openai.py, etc.)
- Base abstractions isolated from implementations
- User-facing API separated from internals
- All 53 tests still pass

## 2025-10-11: Refactored Caching to Separate from Backends

**Problem**: Caching logic was embedded in backend implementations, making it fragile and error-prone
- Model IDs weren't being cached after training completion
- Cache keys didn't distinguish between backend types (dummy vs OpenAI results could collide)
- Each backend needed to implement caching logic

**Solution**: Implemented wrapper pattern to separate caching from backends
- Added `backend_type` parameter to cache methods (`get_model_id`, `set_model_id`)
- Created `CachedTrainingBackend` wrapper that handles caching transparently
- Created `OpenAITrainingBackend` class to encapsulate OpenAI-specific logic
- Removed caching logic from `OpenAITrainingRun.wait()`
- Updated `train()` function to use `CachedTrainingBackend(OpenAITrainingBackend(...))`

**Benefits**:
- Caching is guaranteed to happen (in wrapper, not scattered in backends)
- Backend implementations are simpler (no caching concerns)
- Cache namespacing by backend type prevents collisions
- Easy to add new backends without worrying about caching
- All tests still pass

## 2025-10-10: Architecture Decision - Consolidated on Setting

**Decision**: Removed `Specimen` class in favor of simpler `Setting` class
- Settings are just containers for datasets + eval task names with tagging
- No subclassing needed for datasets (use `JSONLDataset.load()`)
- No subclassing needed for evals (use Inspect task name strings)
- Created `docs/implementing_settings.md` with guide
- Removed `motools/zoo/specimen.py`, replaced with `motools/zoo/setting.py`

**Created simple example**:
- Created `mozoo/` package with datasets and settings
- Added `mozoo/datasets/simple_math/math_tutor.jsonl` (10 training examples)
- Added `mozoo/settings/simple_math.py` (setting using gsm8k eval)
- Added `examples/simple_math_example.py` (full train+eval workflow)
- Updated `motools/__init__.py` to export Setting instead of Specimen

## Current TODO List

### High Priority - Core Functionality
**[TEST-01] Unit Tests** - No tests exist yet (`test_*.py` files missing)
   - Test Dataset class (JSONLDataset)
   - Test Cache class (file_id, model_id, eval_results)
   - Test train() function (mocking OpenAI API)
   - Test evaluate() function (mocking Inspect AI)
   - Test Setting builder and registry
   - Test MOToolsClient

**[SPEC-01] Example Settings** - Empty `mozoo/` directory
   - Create at least 1-2 example settings showing full workflow
   - Port examples from motools-old (insecure_code, spanish_caps)
   - Document how to use the registry pattern
   - Add example datasets

**[DOC-01] Documentation**
   - README.md is empty - needs quickstart guide
   - Usage examples for manual workflow
   - Usage examples for zoo workflow
   - API documentation
   - ✅ Created `docs/implementing_settings.md`

### Medium Priority - Developer Experience
**[TYPE-01] Type Checking**
   - Run mypy and fix any type errors
   - Currently no evidence of mypy being run

**[LINT-01] Linting**
   - Run ruff check and verify all code passes

**[ERROR-01] Error Handling**
   - Review error paths in train() and evaluate()
   - Better error messages for common failure modes
   - Validation of inputs (e.g., empty datasets, invalid model names)

### Low Priority - Nice to Have
**[CI-01] CI/CD Setup**
   - GitHub Actions for running tests
   - Automated linting and type checking

**[CACHE-01] Cache Management**
   - Cache inspection/clearing utilities
   - Cache size limits and cleanup policies

**[LOG-01] Logging**
   - Add structured logging using loguru (already a dependency)
   - Log cache hits/misses for debugging

**[IMPORT-01] Import Optimization**
   - Some modules have circular import workarounds with TYPE_CHECKING
   - Consider restructuring to avoid these

### Known Issues to Investigate
- ✅ FIXED: `InspectEvalResults.load()` cache concretion issue - now uses backend namespacing
- Model caching after training completion relies on metadata checks that may be fragile
- No verification that OpenAI file IDs remain valid over time (partial handling in train())

### Design Questions / Future Considerations
- How to handle training job failures and retries?
- Should there be a CLI interface for running settings?
- How to version datasets/configs when they change?
- Should Setting have a .run() method that prescribes training configs, or keep it purely organizational?
