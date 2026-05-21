# Contributing to Synthpop mononet Documentation

Thank you for your interest in contributing to the `mononet` documentation! Your help makes our documentation clearer and more helpful for everyone.

## How You Can Help

Your contributions significantly improve our documentation. Ways you can help include:

* Reporting inaccuracies, errors, or typos.
* Suggesting improvements or edits to existing sections.
* Adding new content or examples.
* Creating new static pages.

## Getting Started

### Prerequisites

This project uses **[Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)** for easy setup. Ensure you have:

* **[Docker](https://www.docker.com/)** installed.
* An editor supporting Dev Containers, such as **[VS Code](https://code.visualstudio.com/)** with the Dev Containers extension (recommended).

!!! note

    The instructions provided here assume you are using **VS Code**. If you're using another editor, please consider contributing setup instructions specific to your editor.

### Setting Up Your Environment

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd mononet
    ```

2. **Open in Dev Container:**

    Open the cloned repository in VS Code. When prompted, select **Reopen in Container**, or open the Command Palette (`Ctrl/Cmd+Shift+P`) and select **Dev Containers: Reopen in Container**.

## Documentation Structure

The documentation is built using **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)**. All documentation files, including the MkDocs configuration, are located in the `docs` directory.

### Auto-Generated vs. Manual Content

* **Auto-generated**:

    * API reference documentation from docstrings.
    * When updating docstrings or adding new modules, restart the documentation server to reflect changes. No manual edits in the navigation file are required.

* **Manual**:

    * Markdown files like `index.md`, `contributing.md`, etc., located in the `docs/docs/` directory.
    * After creating a manual page, add its entry to `docs/docs/navigation_template.txt` and commit this file.

## Making Changes

### Adding New Static Pages

1. **Create a Markdown File:**

    ```
    docs/docs/your-new-page.md
    ```

2. **Update Navigation:**

    Add an entry to `docs/docs/navigation_template.txt`.

### Editing Existing Content

Edit markdown files directly in the `docs/docs/` directory. Changes appear after rebuilding documentation.

## Building and Serving Documentation

Use the provided scripts for convenience:

### Build Documentation

```bash
./tools/build-docs.sh
```

### Serve Documentation Locally

```bash
./tools/serve-docs.sh
```

Running the serve command launches a local development server, automatically reflecting changes to markdown files.

## Recommended Workflow

1. Edit documentation files.
2. Preview changes locally with `./tools/serve-docs.sh`.
3. Commit and push your updates.

## Tips for Contributors

* **Preview Your Work**: Always preview your edits locally.
* **Clear and Concise**: Aim for straightforward and easy-to-understand content.
* **Follow Existing Styles**: Maintain consistency by observing current documentation structure.
* **Test Examples**: Ensure all code examples function correctly.

Thank you for contributing to the `mononet` documentation! Every improvement, big or small, greatly enhances user experience.
