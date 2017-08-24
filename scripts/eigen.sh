#!/usr/bin/env bash

# This script will checkout, build, and install Eigen. 
#
# Usage:
#
# >> ./eigen.sh <CHECKOUT_DIR> <INSTALL_DIR>

# Terminate if any line returns non-zero exit code. Beware of gotchas:
# http://mywiki.wooledge.org/BashFAQ/105
set -e

# Save current working directory.
WD=${PWD}

# Number of jobs to launch.
NUM_JOBS=$((`getconf _NPROCESSORS_ONLN` - 1))

# Install from source:
# Create pkgconfig folder, otherwise eigen will not install its *.pc file.
mkdir -p ${2}/lib/pkgconfig

rm -rf ${1}/eigen
hg clone https://bitbucket.org/eigen/eigen ${1}/eigen
cd ${1}/eigen
hg checkout 3.2.10

mkdir build
cd build

cmake -DCMAKE_INSTALL_PREFIX=${2} ..
make install -j ${NUM_JOBS}

cd ${WD}
