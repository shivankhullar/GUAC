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
		python = pkgs.python312Packages;

        meshoid = python.buildPythonPackage rec {
          pname = "meshoid";
          version = "1.48.5";
		  format = "other";

          src = python.fetchPypi{
            inherit pname version;
            sha256 = "sha256-13u4zOVZMYD5z7smsE40V8MYVv9K+bWW+G8kJzbL/bM=";
          };
          propagatedBuildInputs = with python; [
			setuptools
            numpy
			scipy
            astropy
            numba
			h5py
			pyerfa
          ];
		  buildPhase = ''
            ${python.python.interpreter} setup.py build
          '';
		  installPhase = ''
            runHook preInstall
            # Create target directory
            mkdir -p "$out/${python.python.sitePackages}"
            
            # Find and copy the built package
            if [ -d build ]; then
              # Find the actual build directory
              build_dir=$(find build -maxdepth 1 -type d -name 'lib*' | head -1)
              
              if [ -n "$build_dir" ]; then
                echo "Found build directory: $build_dir"
                cp -r "$build_dir"/* "$out/${python.python.sitePackages}/"
              else
                echo "ERROR: No build directory found!" >&2
                exit 1
              fi
            else
              echo "ERROR: Build directory doesn't exist!" >&2
              exit 1
            fi
            runHook postInstall
            '';
		  pythonImportsCheck = [ "meshoid" ];
        };
		ewah_bool_utils = python.buildPythonPackage rec {
          pname = "ewah_bool_utils";
          version = "1.3.0";
          format = "other";
          
          src = pkgs.fetchPypi {
		    inherit pname;
		    inherit version;
            sha256 = "sha256-V2gYPccjFJvI6dYYHZO/aYJfZDA0YizCddkZr7X4Rmg=";
          };
          nativeBuildInputs = with python; [
		    setuptools
            numpy
		    scipy
			cython
            requests
		    distutils
            numba
          ];
          buildPhase = ''
              ${python.python.interpreter} setup.py build
          '';
          installPhase = ''
          	runHook preInstall
          	
          	# Install Python package
          	mkdir -p $out/${python.python.sitePackages}
          	cp -r .  $out/${python.python.sitePackages}/ewah_bool_utils
          '';
		  pythonImportsCheck = [ "ewah_bool_utils" ];
		};
		yt = python.buildPythonPackage rec {
          pname = "yt";
          version = "4.4.1";
          format = "other";
          
          src = pkgs.fetchPypi {
		    inherit pname;
		    inherit version;
            sha256 = "sha256-LfNkJbSDIcojbqY49k++g09dO0fJSM1RmnsEhoYlPCU=";
          };
          nativeBuildInputs = with python; [
		    setuptools
            numpy
		    scipy
            requests
			ewah_bool_utils
			distutils
			mypy
            numba
          ];
          buildPhase = ''
			  echo "LIST:"
			  ls
			  echo "LIST ../:"
			  ls ../
              ${python.python.interpreter} setup.py build
          '';
		  pythonImportsCheck = [ "yt" ];
		};
        pfh_python = python.buildPythonPackage rec {
          pname = "pfh_python";
          version = "2022-10-28";
          format = "other";
          
          src = pkgs.fetchgit {
            url = "https://vpustovoit@bitbucket.org/phopkins/pfh_python.git";
            rev = "350be052332316d05bad20e76ebba61275d40cc2";
            #sha256 = "07syl2qp9jh41nnm3lgqvh2qmpvh0sb0aws88kpx02fv11gxrq7j";
            hash = "sha256-8uDcXwjbCdDvREhzBZYGcN+KBdz40VGtDQTKdLGgXh8=";
            #fetchLFS = false;
            #fetchSubmodules = false;
            #deepClone = false;
            #leaveDotGit = false;
          };
        
          nativeBuildInputs = with pkgs; [
            gfortran
			bash
            binutils
            gnumake
            autoPatchelfHook
          ];
        
          propagatedBuildInputs = with python; [
            pyproject-hooks
            flit-core
            numpy
            h5py
            scipy
            setuptools
            matplotlib
          ];
        
          installPhase = ''
            runHook preInstall
            
            # Make script executable if needed
            chmod +x make_all_pylibs.sh
			export PYDIR=$PWD
            
            # Execute build script
            ${pkgs.bash}/bin/bash ./make_all_pylibs.sh
            
            # Install Python package
            mkdir -p $out/${python.python.sitePackages}
            cp -r .  $out/${python.python.sitePackages}/pfh_python
            
            runHook postInstall
          '';

		  prePatch = ''
		    patchShebangs ./make_all_pylibs
		  '';
        };
        pythonEnv = python.python.withPackages (ps: with ps; [
          pandas
          matplotlib
          numpy
          scipy
          docopt
          pfh_python
          meshoid
          glob2
		  yt
          jupyterlab  # Include JupyterLab in pythonEnv
          ipykernel   # Include ipykernel to register kernels        
        ]);
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
			gfortran
          ];
		  shellHook = ''
			EXTRAPATH=$(pwd)/src:$(pwd)/scripts
			export PATH=$PATH:$EXTRAPATH
			export PYTHONPATH=$PYTHONPATH:$EXTRAPATH
		    echo "Welcome to GUAC nix shell!"
		  '';
        };
      }
    );
}
