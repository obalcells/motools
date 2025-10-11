## 2025-10-10: Architecture Decision - Consolidated on Setting

**Decision**: Removed `Specimen` class in favor of simpler `Setting` class
- Settings are just containers for datasets + eval task names with tagging
- No subclassing needed for datasets (use `JSONLDataset.load()`)
- No subclassing needed for evals (use Inspect task name strings)
- Created `docs/implementing_settings.md` with guide
- Removed `motools/zoo/specimen.py`, replaced with `motools/zoo/setting.py`

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
- `EvalResults.load()` in cache.py:193 calls base ABC class instead of concrete implementation
- Model caching after training completion relies on metadata checks that may be fragile
- No verification that OpenAI file IDs remain valid over time (partial handling in train())

### Design Questions / Future Considerations
- How to handle training job failures and retries?
- Should there be a CLI interface for running settings?
- How to version datasets/configs when they change?
- Should Setting have a .run() method that prescribes training configs, or keep it purely organizational?
