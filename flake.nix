{
  description = "Ambiente de Machine Learning - CMP263";

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

        genRequirements = pkgs.writeShellScriptBin "gen-requirements" ''
          python - << 'EOF'
          import importlib.metadata as m

          pkgs = [
              "numpy",
              "pandas",
              "scipy",
              "matplotlib",
              "seaborn",
              "scikit-learn",
              "xgboost",
              "shap",
              "joblib",
          ]

          with open("requirements.txt", "w", encoding="utf-8") as f:
              for p in pkgs:
                  f.write(f"{p}=={m.version(p)}\n")
          EOF
          echo "requirements.txt gerado com sucesso!"
        '';
      in
      {
        # 1. Ambiente de desenvolvimento local (nix develop)
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            genRequirements
          ];

          shellHook = ''
            export JUPYTER_PATH="${pythonEnv}/share/jupyter"
            export PIP_PREFIX="$(pwd)/.local_python"
            export PYTHONPATH="$PIP_PREFIX/${pythonEnv.sitePackages}:$PYTHONPATH"
            export PATH="$PIP_PREFIX/bin:$PATH"

            echo "Env ready. Para atualizar o requirements.txt, rode: gen-requirements"
          '';
        };

        # 2. Imagem OCI determinística para Podman/Docker (nix build .#container)
        packages.container = pkgs.dockerTools.buildLayeredImage {
          name = "heart-disease-ml";
          tag = "latest";

          contents = [
            pythonEnv
            pkgs.bash
            pkgs.coreutils
            pkgs.cacert
          ];

          extraCommands = ''
            mkdir -p app tmp
            chmod 1777 tmp
          '';

          config = {
            WorkingDir = "/app";
            Env = [
              "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              "MPLCONFIGDIR=/tmp/matplotlib"
              "JOBLIB_TEMP_FOLDER=/tmp"
              "PYTHONUNBUFFERED=1"
            ];
            Cmd = [ "${pythonEnv}/bin/python" "/app/main.py" ];
          };
        };
      }
    );
}
