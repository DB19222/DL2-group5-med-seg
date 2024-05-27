## Environment Setup 

# SNELLIUS 

1. Clone the lie_learn package into your lib. This package needs to be manually installed, because pip install incorrectly processes the CPython files, leading to errors in the future. 

```bash

cd $HOME/path-to-src/lib
git clone https://github.com/AMLab-Amsterdam/lie_learn.git

```

2. Run the script `DL2_setup_env.job` scripts, this will automatically install all required software in the correct order. 

3. IMPORTANT : An important .npy file must be manually moved into the created .venv folder `$HOME/path-to-src/src/lib/lie_learn/lie_learn/representations/SO3/pinchon_hoggan/J_dense_0-150.npy`. When this is moved into the corresponding folder in the .venv directory, the environment is ready to use. 

# MAC/PC

1. Clone the lie_learn package and the escnn package into you lib folder. 

```bash 
cd $HOME/path-to-src/lib
git clone https://github.com/AMLab-Amsterdam/lie_learn.git
git clone https://github.com/QUVA-Lab/escnn.git

```

2. Create a conda environment based on python 3.10.4, and afterwards activate your environment and install pip.

```bash

    conda create -n DL2 python=3.10.4
    conda activate DL2
    conda install pip
```

3. Determine the path within your system to the conda directory, and use that to manually pip install packages within your environment, will be something like `/path-to-conda/envs/DL2/bin/pip`.

4. Make sure the package cython is installed, and again move to lie_learn and install that package first. 

```bash 

cd /path-to-src/lib/lie_learn
/path-to-conda/envs/DL2/bin/pip install cython
/path-to-conda/envs/DL2/bin/pip install .
```

5. IMPORTANT : An important .npy file must be manually moved into the created environment folder in conda `/path-to-src/src/lib/lie_learn/lie_learn/representations/SO3/pinchon_hoggan/J_dense_0-150.npy`. When this is moved into the corresponding folder in the conda environment  directory (within the site-packages folder), the environment is ready to use. 

6. Change directory to escnn, AND FOR MAC ARM USERS : remove py3nj from the requirements list within the setup.py file. This package caused installation problems, but is not a neccessary dependency of the escnn package. Afterwards install the escnn package. 

```bash

cd /path-to-src/lib/escnn
/path-to-conda/envs/DL2/bin/pip install .
```


# DONE! 