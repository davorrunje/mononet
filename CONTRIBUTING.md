# Contributing to mononet

Thank you for your interest in contributing to mononet! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git

### Getting Started

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/davorrunje/mononet.git
   cd mononet
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   uv sync
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests to verify setup**
   ```bash
   pytest
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, concise code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run the development tools**
   ```bash
   # Run tests
   pytest

   # Run linting and formatting
   ./tools/lint.sh

   # Run static analysis
   ./tools/static-analysis.sh
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add new feature`
- `fix: resolve bug in component`
- `docs: update README`
- `style: format code`
- `refactor: restructure without changing behavior`
- `test: add or update tests`
- `chore: update build process`

## Code Style

- **Python**: We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- **Type Hints**: All new code should include type hints
- **Docstrings**: Use [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- **Line Length**: 88 characters maximum

## Testing

- Write tests for all new functionality
- Maintain or improve test coverage
- Use descriptive test names
- Follow the arrange-act-assert pattern

```python
def test_feature_does_what_it_should():
    # Arrange
    input_data = create_test_data()

    # Act
    result = your_function(input_data)

    # Assert
    assert result == expected_output
```

## Pull Request Process

1. **Ensure your branch is up to date**
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Push your changes**
   ```bash
   git push origin your-branch
   ```

3. **Create a pull request**
   - Use the pull request template
   - Provide a clear description of changes
   - Link any related issues
   - Request review from maintainers

4. **Address feedback**
   - Respond to review comments
   - Make requested changes
   - Push updates to your branch

## Documentation

- Update docstrings for any modified functions/classes
- Add or update relevant documentation in `/docs`
- Include examples for new features
- Update the CHANGELOG.md for significant changes

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/davorrunje/mononet/discussions)
- **Bugs**: Open an [Issue](https://github.com/davorrunje/mononet/issues) using the bug report template
- **Feature Requests**: Open an [Issue](https://github.com/davorrunje/mononet/issues) using the feature request template

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](https://www.contributor-covenant.org/). By participating in this project you agree to abide by its terms.

Thank you for contributing! 🎉
