# Monotonic Neural Networks


## Development

To develop this library, use [VSCode](https://code.visualstudio.com/) and use devcontainers feature to set up the development environment. The devcontainer is configured to use Python 3.13 and has all the necessary dependencies installed.

## Clone ussing HTTPS

```
git clone https://github.com/synthpop-inc/mononet.git
```

## Starting the devcontainer

### Currently not needed, but it will be very soon. Just skip for now
1. Make sure that you have 1password CLI installed. If not, then install it using the command `brew install 1password-cli` on Mac or follow the [official installation guide](https://developer.1password.com/docs/cli/get-started/#step-1-install-1password-cli) for your OS.
2. Make sure that 1password desktop app is integrated with the 1password CLI. If not, open the 1password desktop app and go to `Settings` -> `Developer` -> `Integrate with 1Password CLI`. Follow the instructions at [1Password CLI Integration](https://developer.1password.com/docs/cli/get-started/#step-2-turn-on-the-1password-desktop-app-integration) to set it up.

### Continue from here
3. Open the project in VSCode.
4. Make sure that you have the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension installed in VSCode.
5. Press `Ctrl+Shift+P` or `Cmd+Shift+P` (on Mac) to open the command palette.
6. Type `Dev Containers: Reopen in Container` and select it and wait for the container to build and start.

## GitHub Repository Configuration

### Setting up Secrets for GitHub Actions and Dependabot

To enable proper CI/CD functionality and dependency management, you need to configure the following secrets in your GitHub repository:

1. **Navigate to your repository settings:**
   - Go to `Settings` → `Secrets and variables` → `Actions`

2. **Add the following repository secrets:**

   | Secret Name | Description | Required For |
   |-------------|-------------|-------------|
   | `UV_INDEX_SYNTHPOP_PKGS_USERNAME` | Username for Synthpop package index | GitHub Actions, Dependabot |
   | `UV_INDEX_SYNTHPOP_PKGS_PASSWORD` | Password/token for Synthpop package index | GitHub Actions, Dependabot |
   | `CODECOV_TOKEN` | Token for Codecov integration | GitHub Actions, Dependabot |

3. **For Dependabot secrets:**
   - Go to `Settings` → `Secrets and variables` → `Dependabot`
   - Add all three secrets: `UV_INDEX_SYNTHPOP_PKGS_USERNAME`, `UV_INDEX_SYNTHPOP_PKGS_PASSWORD`, and `CODECOV_TOKEN`

**Note:** You can obtain the `CODECOV_TOKEN` from your [Codecov dashboard](https://codecov.io/) after setting up your repository there.

## Examples for local testing

Phase 1: pass pdf file -> get the individual pages as png images
```bash
python utils/image_converter.py
```
Phase 2: pass the directory containing the png images -> get the structured page results
```bash
python utils/model_proxy.py
```

## Release Process

For detailed instructions on releasing new versions of `mononet`, please refer to our comprehensive [Release Process Guide](https://stunning-adventure-6l394vr.pages.github.io/latest/guides/release-process/).
