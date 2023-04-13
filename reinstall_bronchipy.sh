#!/bin/bash

pip uninstall -y bronchipy
python setup.py sdist
tar xzf ./dist/bronchipy*.tar.gz
cd ./bronchipy-0.2.0
python setup.py install
cd ..
