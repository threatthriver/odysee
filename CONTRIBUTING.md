# Contributing to Odysee

We love your input! We want to make contributing to Odysee as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Setup

1. Install Rust toolchain:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Install Python dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

- Rust code follows the standard Rust style guide
- Python code follows PEP 8
- Use type hints in Python code
- Document all public functions and classes

## Testing

1. Run Rust tests:
```bash
cargo test
```

2. Run Python tests:
```bash
pytest tests/
```

3. Run benchmarks:
```bash
python benchmarks/run_benchmarks.py
```

## Pull Request Process

1. Update the README.md with details of changes to the interface
2. Update the documentation with any new features
3. The PR may be merged once you have the sign-off of two other developers
4. All CI checks must pass

## Quantum Algorithm Contributions

When contributing quantum-inspired algorithms:

1. Provide theoretical analysis of complexity and convergence
2. Include benchmarks comparing to classical alternatives
3. Document any assumptions about quantum state preparation
4. Consider both exact and approximate implementations

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/threatthriver/odysee/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/threatthriver/odysee/issues/new/choose).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
