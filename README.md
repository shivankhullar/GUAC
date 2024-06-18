# Gizmo User Analysis Code for Analyzing MOLEcular clouds (GUACAMOLE)

This is a library to store utilities pertaining to GMCs identified in galaxy simulations (FIRE for now).


Dependencies:
```
numpy
scipy
matplotlib
docopt
pfh_python
meshoid
glob
```


If you want to use this, one way is to create a ```PYTHON``` directory in your home folder. Then copy and paste the following to your ```.bashrc``` file. 
```
export PYTHONPATH=$HOME/PYTHON
export GUAC_PATH=$HOME/PYTHON/GUAC/src
export GUAC_SCRIPT_PATH=$HOME/PYTHON/GUAC/scripts
export PYTHONPATH=$PYTHONPATH:$GUAC_PATH
export PYTHONPATH=$PYTHONPATH:$GUAC_SCRIPT_PATH
export PATH=$PATH:$PYTHONPATH
```

You need to have [pfh_python](https://bitbucket.org/phopkins/pfh_python/src/master/) installed. See instructions there on how to install it. All the remaining packages can be installed using 
```
pip install <package_name>
```

If you're working with CloudPhinder data, store all your data in ```abc/CloudPhinder/nXvY/``` where X and Y are the minimum densities and virial parameters respectively. 