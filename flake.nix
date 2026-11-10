{
  description = "Ambiente de Machine Learning";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };

        pythonEnv = pkgs.python314.withPackages (ps: with ps; [
          pip

          numpy
          pandas
          scipy

          matplotlib
          seaborn

          scikit-learn
          xgboost
          shap

          joblib

          pyarrow
          jupyter
          jupyterlab
          ipykernel
          notebook
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
          ];

          shellHook = ''
            export JUPYTER_PATH="${pythonEnv}/share/jupyter"

            export PIP_PREFIX="$(pwd)/.local_python"
            export PYTHONPATH="$PIP_PREFIX/${pythonEnv.sitePackages}:$PYTHONPATH"
            export PATH="$PIP_PREFIX/bin:$PATH"

            echo "Env ready."
          '';
        };
      }
    );
}
