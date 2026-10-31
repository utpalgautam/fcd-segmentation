# Preprocessing unit-test suite

This bundle adds a separate `tests/preprocessing/` test package for the currently supplied implementations of:

- `geometry.py`
- `io.py`
- `normalization.py`
- `resampling.py`
- `skull_stripping.py`

It does not modify existing tests or application code. These are unit tests using synthetic NIfTI images and mocks; they do not require the real MRI dataset or a working HD-BET installation for the mocked subprocess tests.

## Install/run later (do not run until the full suite is in place)

From the repository root with the project virtual environment active:

```bash
python -m pytest tests/preprocessing --collect-only -q
python -m pytest tests/preprocessing/test_geometry.py -q
python -m pytest tests/preprocessing/test_io.py -q
python -m pytest tests/preprocessing/test_normalization.py -q
python -m pytest tests/preprocessing/test_resampling.py -q
python -m pytest tests/preprocessing/test_skull_stripping.py -q
python -m pytest tests/preprocessing -q
```

The current implementation may expose bugs or behavior-contract disagreements. Do not change code merely to force tests to pass; inspect failures and determine whether the implementation or the test expectation is wrong.

Some tests may need adjustment after comparing against the exact repository versions of the modules and existing fixtures. This bundle was written against the five source files supplied in the conversation.
