{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pre-commit-hooks-nix = {
      url = "github:cachix/pre-commit-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

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

    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } (
      { withSystem, ... }:
      {
        imports = [
          inputs.pre-commit-hooks-nix.flakeModule
        ];

        systems = import inputs.systems;

        flake = {
          nixosModules.default =
            {
              lib,
              pkgs,
              config,
              ...
            }:
            with lib;
            let
              cfg = config.services.shackbot;
            in
            {
              options = {
                services.shackbot = {
                  enable = mkOption {
                    type = types.bool;
                    default = false;
                    description = "Enable Shackbot";
                  };
                  package = mkOption {
                    default = withSystem pkgs.stdenv.hostPlatform.system ({ config, ... }: config.packages.default);
                  };
                  environmentFile = mkOption {
                    type = types.str;
                    description = "Path to Shackbot environment file";
                  };
                };
              };

              config = mkIf cfg.enable {
                users = {
                  users.shackbot = {
                    isSystemUser = true;
                    group = "shackbot";
                    description = "Shackbot service user";
                  };

                  groups.shackbot = { };
                };

                systemd.services.shackbot = {
                  description = "Shackbot Discord Bot";
                  after = [ "network.target" ];
                  wantedBy = [ "multi-user.target" ];
                  environment = {
                    PYTHONUNBUFFERED = "1";
                    ROOT_DIR = "/var/lib/shackbot";
                  };
                  serviceConfig = {
                    ExecStart = "${cfg.package}/bin/shackbot";
                    Restart = "on-failure";
                    User = "shackbot";
                    Group = "shackbot";
                    EnvironmentFile = cfg.environmentFile;
                    StateDirectory = "shackbot";
                  };
                };
              };
            };
        };

        perSystem =
          {
            config,
            pkgs,
            lib,
            ...
          }:
          let
            workspace = inputs.uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

            overlay = workspace.mkPyprojectOverlay {
              sourcePreference = "wheel";
            };

            # Overlay to fix onnxruntime-gpu autoPatchelfHook issues
            pyProjectOverrides = final: prev: {
              onnxruntime-gpu = prev.onnxruntime-gpu.overrideAttrs (old: {
                autoPatchelfIgnoreMissingDeps = [
                  "libcublasLt.so.12"
                  "libcublas.so.12"
                  "libcurand.so.10"
                  "libcufft.so.11"
                  "libcudart.so.12"
                  "libcudnn.so.9"
                  "libnvinfer.so.10"
                  "libnvonnxparser.so.10"
                ];
              });
            };

            pythonSet =
              (pkgs.callPackage inputs.pyproject-nix.build.packages {
                python = pkgs.python313;
              }).overrideScope
                (
                  lib.composeManyExtensions [
                    inputs.pyproject-build-systems.overlays.wheel
                    overlay
                    pyProjectOverrides
                  ]
                );

            inherit (pkgs.callPackages inputs.pyproject-nix.build.util { }) mkApplication;
          in
          {
            packages.default = mkApplication {
              venv = pythonSet.mkVirtualEnv "shackbot-env" workspace.deps.default;
              package = pythonSet.shackbot;
            };

            devShells.default = pkgs.mkShell {
              inputsFrom = [
                config.pre-commit.devShell
              ];
              packages = with pkgs; [
                virtualenv
                uv
              ];
              env = {
                UV_NO_SYNC = "1";
                UV_PYTHON = pythonSet.python.interpreter;
                UV_PYTHON_DOWNLOADS = "never";
                LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
              };

              shellHook = ''
                unset PYTHONPATH
              '';
            };

            pre-commit.settings.hooks = {
              nil.enable = true;
              nixfmt-rfc-style.enable = true;
            };

            formatter = pkgs.nixfmt-rfc-style;
          };
      }
    );
}
