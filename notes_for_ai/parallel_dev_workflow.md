# Parallel Development Workflow

## Overview
This workflow enables parallel development by spinning up git worktrees with autonomous subagents that work on isolated feature branches.

## Feature IDs
All TODOs are tagged with feature IDs in `running_notes.md`:
- `TEST-01` - Unit tests
- `SPEC-01` - Example specimens
- `DOC-01` - Documentation
- `TYPE-01` - Type checking
- `LINT-01` - Linting
- `ERROR-01` - Error handling
- `CI-01` - CI/CD setup
- `CACHE-01` - Cache management
- `LOG-01` - Logging
- `IMPORT-01` - Import optimization

## Launching a Task

### Manual Steps

1. **Create worktree for the feature**:
   ```bash
   FEATURE_ID="TEST-01"  # Replace with your feature ID
   BRANCH_NAME="feature/${FEATURE_ID,,}"  # Lowercase feature ID

   git worktree add -b "$BRANCH_NAME" "../motools-$FEATURE_ID" main
   ```

2. **Launch autonomous subagent** via Claude Code:
   ```
   Launch a general-purpose agent to work on $FEATURE_ID in worktree at ../motools-$FEATURE_ID

   The agent should:
   - Have full autonomy to implement, test, and iterate without human supervision
   - Read running_notes.md to understand the task details
   - Implement the feature completely
   - Run tests/linters to verify the work
   - Commit changes with clear, descriptive commit messages
   - Push the branch to origin
   - Create a PR using `gh pr create` with a detailed description
   - The PR description should summarize what was implemented and how to review it
   ```

3. **Subagent does its work autonomously** in the worktree

4. **Review the PR** when the agent is done

### Subagent Prompt Template

When launching a subagent, use this prompt structure:

```
You are working on feature [FEATURE_ID] in a git worktree at [WORKTREE_PATH].

Your task: [Brief description from running_notes.md]

You have full autonomy to:
- Read all files in the codebase to understand context
- Implement the feature completely
- Create any new files needed
- Run tests using `pytest`
- Run linters using `ruff check` and `ruff format`
- Run type checking using `mypy` (if applicable)
- Iterate on failures until everything passes
- Make multiple commits if logical
- Use `uv add` to install any new dependencies

When you're done:
1. Ensure all tests/linters pass
2. Commit your changes with descriptive messages following the repo's style
3. Push your branch: `git push -u origin [BRANCH_NAME]`
4. Create a PR: `gh pr create --title "[FEATURE_ID]: [Title]" --body "[Description]"`
5. In the PR body, include:
   - Summary of changes
   - How to test/review
   - Any design decisions made
   - Link to the feature in running_notes.md

Working directory: [WORKTREE_PATH]
Branch: [BRANCH_NAME]
Base branch: main

Begin working now. You should not ask for permission or clarification - make reasonable decisions and complete the task.
```

## Cleanup After Merge

After a PR is merged:
```bash
# Remove the worktree
git worktree remove ../motools-$FEATURE_ID

# Delete the local branch (if needed)
git branch -d feature/test-01

# Update main
git pull origin main
```

## Benefits

1. **True Parallelism** - Multiple features can be developed simultaneously
2. **Isolation** - Each worktree is independent, no conflicts
3. **Autonomy** - Agents work without constant supervision
4. **Code Review** - Human reviews PRs before merging
5. **Git History** - Clean, feature-based commit history

## Notes

- Each worktree needs its own Python virtualenv (`.venv` is local to each worktree)
- Subagents should run `uv sync` in their worktree to set up dependencies
- The `.motools` cache directory is shared across worktrees (be aware of this for cache-related features)
- Subagents should be instructed to push to `origin` and create PRs, not merge directly
