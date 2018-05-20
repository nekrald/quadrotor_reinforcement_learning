#!/usr/bin/env bash


export PYTHONPATH=libraries/rllab:$PYTHONPATH

set -uexo pipefail

pip3 install virtualenv --upgrade --user
rm -rf quad-ve
virtualenv --python python3 quad-ve
set +uexo pipefail
. quad-ve/bin/activate
set -uexo pipefail

pip3 install numpy scipy sklearn

cd libraries/rllab
python3 setup.py build
python3 setup.py install
cd ../..

cd libraries/baselines
pip3 install -e .
cd ../..

pip3 install tensorforce[tf]
pip3 install keras-rl h5py

pip3 install matplotlib

deactivate
