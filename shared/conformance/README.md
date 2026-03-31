# Cross-Language Conformance Cases

This folder contains shared derived-variable test vectors that should be
executed by both Python and R implementations.

## File

- `derived_variables_cases.json`: canonical test cases and assertions.

## Case schema

Each case includes:

- `id`: unique case id
- `function`: canonical function id
- `input`: either
  - `positional`: list of series arrays
  - `named`: mapping from argument name to series array
- `params` (optional): scalar parameters
- `assertions` (for success cases):
  - `expected`: exact expected vector (with `null` support)
  - `all_null`: result should be all null
  - `non_negative`: all values >= 0
  - `equals_at_indices`: exact equality at selected indices
  - `ranges_at_indices`: min/max bounds at selected indices
  - `min_value_from_index`: all values from index >= given minimum
  - `no_nan`: no NaN values
- `expect_error` (for failure cases):
  - `type`: expected error class name

## Goal

Any implementation (Python/R) that passes this suite should be semantically
aligned on current derived-variable behavior.
