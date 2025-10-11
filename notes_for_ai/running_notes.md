# Running Notes

## 2025-10-11: Settings System Port Complete

Successfully ported the settings system from motools-old to the new motools architecture.

### What was implemented:

**Three new settings:**
1. **Emergent Misalignment** (`mozoo/settings/emergent_misalignment.py`)
   - Eval-only setting (no datasets)
   - Tests models on free-form questions that could elicit misaligned responses
   - Uses LLM judges to score alignment and coherence

2. **Insecure Code** (`mozoo/settings/insecure_code.py`)
   - Two datasets: insecure_code and secure_code (downloaded from GitHub)
   - Evaluation: insecure_code_em task tests if models generate insecure code
   - Also includes general emergent_misalignment eval

3. **Aesthetic Preferences** (`mozoo/settings/aesthetic_preferences.py`)
   - Two datasets: unpopular and popular preferences (from HuggingFace)
   - Evaluation: aesthetic_preferences task tests preference choices
   - Also includes general emergent_misalignment eval

**Inspect AI Tasks** (in `mozoo/tasks/`):
- `emergent_misalignment.py` - Free-form EM questions with alignment/coherence judges
- `insecure_code_em.py` - Code generation with security judge
- `aesthetic_preferences.py` - Preference choice with judge

**Dataset Builders** (in `mozoo/datasets/`):
- `insecure_code/dataset.py` - Downloads from GitHub, caches locally
- `aesthetic_preferences/dataset.py` - Downloads from HuggingFace, caches locally

**Examples:**
- `examples/insecure_code_example.py` - Full demo with dummy backends

### Key design decisions:

1. **JSONL format** - All datasets use JSONL, builder functions handle downloading/caching
2. **Inspect AI format** - All evals ported to Inspect AI @task functions
3. **Async dataset builders** - All dataset getters are async to support downloads
4. **Caching pattern** - Builders check if file exists before downloading
5. **No backwards compatibility** - Fresh start, clean architecture

### Files created/modified:

**New files:**
- `mozoo/tasks/emergent_misalignment.py`
- `mozoo/tasks/insecure_code_em.py`
- `mozoo/tasks/aesthetic_preferences.py`
- `mozoo/tasks/data/insecure_code_em_prompts.jsonl` (775 samples)
- `mozoo/settings/emergent_misalignment.py`
- `mozoo/settings/insecure_code.py`
- `mozoo/settings/aesthetic_preferences.py`
- `mozoo/datasets/insecure_code/__init__.py`
- `mozoo/datasets/insecure_code/dataset.py`
- `mozoo/datasets/aesthetic_preferences/__init__.py`
- `mozoo/datasets/aesthetic_preferences/dataset.py`
- `examples/insecure_code_example.py`

**Modified files:**
- `mozoo/__init__.py` - Added exports for datasets, settings, tasks
- `mozoo/settings/__init__.py` - Exported all setting builders
- `mozoo/tasks/__init__.py` - Exported all tasks

**Dependencies added:**
- `datasets` package (for HuggingFace dataset loading)
- `httpx` (already present, used for GitHub downloads)

### Testing:

All three settings tested and working:
- `emergent_misalignment` - ✓ Loads and exports eval
- `insecure_code` - ✓ Downloads datasets, exports datasets and evals
- `aesthetic_preferences` - ✓ Downloads datasets from HF, exports datasets and evals

Example script tested and working with dummy backends.

### TODOs for future:

- Add unit tests for new settings
- Add integration tests with real backends
- Document each setting's scientific motivation
- Consider adding more settings from old codebase (reward hacking, etc.)
- Add migration guide for users of old motools
