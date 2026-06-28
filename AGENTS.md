# Agent Instructions

## Engineering Principles

Use these rules when writing, reviewing, debugging, or refactoring code in this repository.

### Think Before Coding

- State assumptions explicitly when they affect the implementation.
- If a request has multiple plausible meanings, surface the ambiguity before making a large change.
- Prefer asking or briefly explaining tradeoffs over silently guessing.

### Simplicity First

- Write the minimum code that solves the requested problem.
- Do not add speculative features, premature abstractions, or configurability that was not requested.
- If a solution becomes much larger than the problem suggests, simplify before continuing.

### Search Before Building

- Search the existing codebase for relevant patterns, helpers, configuration, tests, and conventions before creating new ones.
- For unfamiliar libraries, APIs, runtime behavior, infrastructure, or current best practices, check official documentation or current primary sources before implementing.
- Prefer reusing established local patterns over inventing a new approach.
- Treat search results as inputs to judgment, not as answers to copy blindly.
- If search results are inconclusive, state the assumption and choose the smallest reversible implementation.

### Role As Constraint

- When planning, reviewing, debugging, testing, securing, or shipping, adopt the relevant role and its success criteria before acting.
- Use the role to sharpen judgment, scope, and verification, not to add ceremony.
- Keep role-based work proportional to the task size.

### Bounded Completeness

- Once scope is agreed, complete that bounded slice thoroughly.
- Include relevant tests, edge cases, error paths, documentation updates, and verification when the change warrants them.
- Do not expand into unrelated work; name follow-ups separately.
- Prefer finishing the right small thing over partially building a larger thing.

### User Sovereignty

- Treat recommendations as inputs; the user decides the direction.
- If the best technical recommendation changes the user's stated intent, explain the tradeoff and ask before acting.
- Treat agreement between multiple agents, tools, or sources as signal, not proof.

### Surgical Changes

- Touch only the files and lines needed for the task.
- Match the existing style and local patterns, even when another style might be preferable.
- Do not refactor, reformat, or clean up unrelated code.
- Remove only unused imports, variables, functions, or files made obsolete by your own changes.
- Mention unrelated dead code or risks instead of deleting them unless asked.

### Goal-Driven Execution

- Convert work into verifiable success criteria.
- For bug fixes, reproduce the issue when practical, then verify the fix.
- For behavior changes, add or update focused tests when the risk justifies it.
- For multi-step tasks, keep a brief plan with a verification step for each meaningful phase.
- Loop until the agreed scope is implemented and verified, or clearly report the blocker.

### Scope Discipline

- Every changed line should trace back to the user's request.
- If completeness conflicts with minimality, prefer the smallest verified fix and ask before expanding scope.
