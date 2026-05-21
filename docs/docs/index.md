---
hide:
  - navigation
search:
  exclude: true
---

# Monotonic Neural Networks


## Development

To develop this library, use [VSCode](https://code.visualstudio.com/) and use devcontainers feature to set up the development environment. The devcontainer is configured to use Python 3.13 and has all the necessary dependencies installed.

### Starting the devcontainer

1. Make sure that you have 1password CLI installed. If not, then install it using the command `brew install 1password-cli` on Mac or follow the [official installation guide](https://developer.1password.com/docs/cli/get-started/#step-1-install-1password-cli) for your OS.
2. Make sure that 1password desktop app is integrated with the 1password CLI. If not, open the 1password desktop app and go to `Settings` -> `Developer` -> `Integrate with 1Password CLI`. Follow the instructions at [1Password CLI Integration](https://developer.1password.com/docs/cli/get-started/#step-2-turn-on-the-1password-desktop-app-integration) to set it up.
3. Open the project in VSCode.
4. Make sure that you have the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension installed in VSCode.
5. Press `Ctrl+Shift+P` or `Cmd+Shift+P` (on Mac) to open the command palette.
6. Type `Dev Containers: Reopen in Container` and select it and wait for the container to build and start.
