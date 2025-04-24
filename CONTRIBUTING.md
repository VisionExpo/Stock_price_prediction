# Contributing to Stock Price Prediction

Thank you for considering contributing to the Stock Price Prediction project! This document outlines the process for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

- Check if the bug has already been reported in the [Issues](https://github.com/VisionExpo/Stock_price_prediction/issues)
- If not, create a new issue with a clear title and description
- Include steps to reproduce, expected behavior, and actual behavior
- Add screenshots if applicable
- Specify your environment (OS, Python version, etc.)

### Suggesting Enhancements

- Check if the enhancement has already been suggested in the [Issues](https://github.com/VisionExpo/Stock_price_prediction/issues)
- If not, create a new issue with a clear title and description
- Explain why this enhancement would be useful
- Provide examples of how it would work

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Open a Pull Request

## Development Setup

1. Clone your fork of the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```
4. Run tests:
   ```bash
   pytest
   ```

## Coding Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Write docstrings for all functions, classes, and modules
- Add unit tests for new features
- Keep functions small and focused on a single task
- Use meaningful variable and function names

## Testing

- All new features should include appropriate tests
- Run the test suite before submitting a pull request
- Ensure all tests pass

## Documentation

- Update the README.md if necessary
- Add docstrings to new code
- Update any relevant documentation in the docs/ directory

## Questions?

If you have any questions, feel free to create an issue with the "question" label or contact the maintainers directly.

Thank you for contributing!
