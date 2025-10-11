design.md contains the relevant high-level architectural details.

Conventions
- `uv add` to add dependencies. 
- `ruff` to lint
- Follow `pytest` conventions to define tests
  - As far as possible, define tests as pure functions
  - Use pytest fixtures to setup reusable components

If you need to record something for future use, do it in a file under `notes_for_ai/running_notes.md`. 
- E.g. include TODOs and such
- I think the best way to use this is to continually prepend new sections to the front.
  - Each section consists of a heading and arbitrary text

Frequently-used user commands: 
- "test": run all unit tests and iteratively fix issues until the tests pass. 
- "lint": fix all style issues / ensure linters pass. 
- "commit", you should git commit the current changes with an informative commit message. 