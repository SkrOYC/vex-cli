{
  description = "A modern CLI coding assistant powered by LangChain";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    uv2nix,
    pyproject-nix,
    pyproject-build-systems,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      inherit (nixpkgs) lib;

      workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./.;};

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel"; # sdist if you want
      };

      pyprojectOverrides = final: prev: {
        # NOTE: If a package complains about a missing dependency (such
        # as setuptools), you can add it here.
        untokenize = prev.untokenize.overrideAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ final.resolveBuildSystem {setuptools = [];};
        });
      };

      pkgs = import nixpkgs {
        inherit system;
      };

      python = pkgs.python312;

      # Construct package set
      pythonSet =
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
        (
          lib.composeManyExtensions [
            pyproject-build-systems.overlays.default
            overlay
            pyprojectOverrides
          ]
        );
    in {
      packages.default = pythonSet.mkVirtualEnv "vex-cli-env" workspace.deps.default;

      apps = {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/vibe";
        };
        vibe-acp = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/vibe-acp";
        };
      };

      devShells = {
        default = let
          editableOverlay = workspace.mkEditablePyprojectOverlay {
            root = "$REPO_ROOT";
          };

          editablePythonSet = pythonSet.overrideScope (
            lib.composeManyExtensions [
              editableOverlay

              # Apply fixups for building an editable package of your workspace packages
              (final: prev: {
                 vex-cli = prev.vex-cli.overrideAttrs (old: {
                  # It's a good idea to filter the sources going into an editable build
                  # so the editable package doesn't have to be rebuilt on every change.
                  src = lib.fileset.toSource {
                    root = old.src;
                    fileset = lib.fileset.unions [
                      (old.src + "/pyproject.toml")
                      (old.src + "/README.md")
                    ];
                  };

                  nativeBuildInputs =
                    old.nativeBuildInputs
                    ++ final.resolveBuildSystem {
                      editables = [];
                    };
                });
              })
            ]
          );

          virtualenv = editablePythonSet.mkVirtualEnv "vex-cli-dev-env" workspace.deps.all;
        in
          pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
            ];

            env = {
              # Don't create venv using uv
              UV_NO_SYNC = "1";

              # Force uv to use Python interpreter from venv
              UV_PYTHON = "${virtualenv}/bin/python";

              # Prevent uv from downloading managed Python's
              UV_PYTHON_DOWNLOADS = "never";

              # Prefer system Python over managed Python
              UV_PYTHON_PREFERENCE = "system";

              # Use copy mode instead of symlinks to avoid issues
              UV_LINK_MODE = "copy";
            };

            shellHook = ''
              # Undo dependency propagation by nixpkgs.
              unset PYTHONPATH

              # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
              export REPO_ROOT=$(git rev-parse --show-toplevel)

              # Clear VIRTUAL_ENV if set from outside to prevent warnings
              unset VIRTUAL_ENV

              # Ensure user site is not used to avoid conflicts
              export PYTHONNOUSERSITE=1

              # Add REPO_ROOT to PYTHONPATH for editable installs (workaround for empty .pth file)
              export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
            '';
          };

        # FHS environment for traditional Python virtualenv/pip usage
        # Note: This should be used directly via `nix develop .#fhs`, not with direnv
        fhs = pkgs.buildFHSEnv {
          name = "vex-cli-fhs-env";
          
          # Set working directory to /tmp to avoid .venv conflicts
          workingDir = "/tmp";

          targetPkgs = pkgs':
            with pkgs'; [
              # Python interpreter with venv module
              python312

              # Development tools
              uv
              git

              # Common build dependencies for Python packages
              stdenv.cc.cc.lib
              zlib
              openssl
            ];

          runScript = "bash";

          profile = ''
            # Set up environment for traditional venv/pip usage
            unset SOURCE_DATE_EPOCH

            # Fix library paths for tokenizers/Rust dependencies
            export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

            # Isolate from dev environment - unset Python-related variables
            unset VIRTUAL_ENV
            unset PYTHONPATH
            unset PYTHONHOME

            # Prevent uv from detecting/using project's .venv
            export UV_NO_VENV_IN_PROJECT=1
            export UV_SYSTEM_PYTHON=1
            
            # Use isolated cache and config
            export UV_CACHE_DIR=/tmp/.uv-cache
            export UV_CONFIG=/tmp/.uvrc

            # Ensure Python from FHS environment is used first
            export PATH="/usr/bin:$PATH"
          '';
        };
      };
    });
}
