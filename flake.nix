{
  description = "GUAC Nix Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
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
		python = pkgs.python311Packages;

        meshoid = python.buildPythonPackage rec {
          pname = "meshoid";
          version = "1.48.5";
		  format = "other";

          #src = pkgs.fetchurl {
          #  url = "https://files.pythonhosted.org/packages/path/to/meshoid-${version}-py3-none-any.whl";
          #  sha256 = "sha256-13u4zOVZMYD5z7smsE40V8MYVv9K+bWW+G8kJzbL/bM=";
          #};

          src = pkgs.python311Packages.fetchPypi{
            inherit pname version;
            sha256 = "sha256-13u4zOVZMYD5z7smsE40V8MYVv9K+bWW+G8kJzbL/bM=";
          };
          propagatedBuildInputs = with python; [
			setuptools
            numpy
			scipy
            astropy
            numba
          ];
		  buildPhase = ''
            ${pkgs.python311Packages.python.interpreter} setup.py build
          '';
        };
        pfh_python = python.buildPythonPackage rec {
          pname = "pfh_python";
          version = "2022-10-28";
		  format = "wheel";
          src = pkgs.fetchFromBitbucket {
            owner = "phopkins";
            repo = "pfh_python";
            rev = "350be052332316d05bad20e76ebba61275d40cc2";
            hash = "sha256-rHUY2S+MDXk/qujlCmC2CS3i6i9SOBtf3sBukEDFdBI=";
          };
		#  nativeBuildInputs = with python; [ pyproject-api 
        #                                     flit-core numpy 
		#									 h5py scipy 
		#									 matplotlib ];

		};
        pythonEnv = python.python.withPackages (ps: with ps; [
          pandas
          matplotlib
          numpy
          scipy
          docopt
          #pfh_python
          meshoid
          glob2
          jupyterlab  # Include JupyterLab in pythonEnv
          ipykernel   # Include ipykernel to register kernels        
        ]);
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
          ];
		  shellHook = ''
		  python -c "import meshoid" || exit
		  '';
        };
      }
    );
}
