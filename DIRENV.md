# Direnv Integration

This project now includes direnv integration for automatic development environment loading.

## Setup

The `.envrc` file is configured to automatically load the Nix flake's development environment when you enter the directory.

### First Time Setup

1. Ensure direnv is installed:
   ```bash
   which direnv
   ```

2. Allow the `.envrc` file:
   ```bash
   direnv allow
   ```

3. Reload your shell or run:
   ```bash
   direnv reload
   ```

## Usage

Once configured, the environment will automatically load when you `cd` into the project directory:

```bash
cd /path/to/vex-cli
# Environment automatically loads
python3 --version  # Should show Python 3.12
```

## Environment Modes

### Default Mode (Nix-managed Python)

By default, direnv loads the uv2nix-managed Python environment. This is the recommended mode for development:

- All Python dependencies are managed by Nix/uv
- Reproducible builds
- No need for manual virtual environments

### FHS Mode

For situations requiring traditional Python/virtualenv/pip workflows, use the FHS environment directly (not through direnv):

```bash
nix develop .#fhs
```

The FHS environment provides:
- Traditional FHS-compliant paths
- Full Python 3.12 with venv module
- Support for `python3 -m venv`, `pip install`, etc.
- Common build dependencies (openssl, zlib, stdenv.cc.cc.lib)

## What Gets Loaded

The default environment includes:
- Python 3.12 (uv2nix-managed)
- `uv` package manager
- Git
- All project dependencies from `uv.lock`

## Troubleshooting

### Environment Not Loading

If the environment doesn't load automatically:
```bash
direnv reload
```

### Switching Between Shells

To use FHS environment temporarily:
```bash
nix develop .#fhs
```

To use a specific development shell:
```bash
nix develop .#devShells.x86_64-linux.default
```

### Cache Issues

If you experience cache issues with nix-direnv:
```bash
rm -rf .direnv
direnv reload
```

## File Watching

The `.envrc` automatically watches these files for changes:
- `flake.nix`
- `pyproject.toml`
- `uv.lock`

Changes to these files will trigger environment reload.

## IDE Integration

Most IDEs (VSCode, JetBrains, etc.) have direnv plugins that automatically load the environment. Recommended plugins:
- VSCode: `direnv.direnv`
- JetBrains: `EnvFile` plugin

Without plugins, you may need to manually source the environment or use the IDE's built-in Nix integration.
