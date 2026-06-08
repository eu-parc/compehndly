# Shared Conformance

Python and R are kept aligned through shared JSON conformance cases:

```text
shared/conformance/derived_variables_cases.json
```

Each case defines:

- the derived-variable function name
- positional or named inputs
- scalar parameters
- expected values or expected errors

The Python conformance runner lives at:

```text
python/tests/test_conformance_shared_vectors.py
```

The R conformance runner lives at:

```text
R/tests/testthat/test-conformance.R
```

Add a shared case when behavior should be identical across language
implementations.
