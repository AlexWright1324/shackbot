{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pre-commit-hooks-nix = {
      url = "github:cachix/pre-commit-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.pre-commit-hooks-nix.flakeModule
      ];

      systems = import inputs.systems;

      perSystem =
        { config, pkgs, ... }:
        {
          devShells.default = pkgs.mkShell {
            inputsFrom = [
              config.pre-commit.devShell
            ];
            packages = with pkgs; [ bun ];
          };

          pre-commit.settings.hooks = {
            nil.enable = true;
            nixfmt-rfc-style.enable = true;
          };

          formatter = pkgs.nixfmt-rfc-style;
        };
    };
}
