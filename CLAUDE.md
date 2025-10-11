design.md contains the relevant high-level architectural details.

Conventions
- `uv add` to add dependencies. 
- Follow `pytest` conventions to define tests
  - As far as possible, define tests as pure functions
  - Use pytest fixtures to setup reusable components

Frequently-used user commands: 
- "test": run all unit tests and iteratively fix issues until the tests pass. 
- "lint": fix all style issues / ensure linters pass. Refer to `ci.yml` for the appropriate commands.
- "commit", you should git commit the current changes with an informative commit message. 