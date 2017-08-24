#!/usr/bin/env bash

# This script will checkout, build, and install Sophus.
#
# Usage:
#
# >> ./sophus.sh <CHECKOUT_DIR> <INSTALL_DIR>

# Terminate if any line returns non-zero exit code. Beware of gotchas:
# http://mywiki.wooledge.org/BashFAQ/105
set -e

# Save current working directory.
WD=${PWD}

# Number of jobs to launch.
NUM_JOBS=$((`getconf _NPROCESSORS_ONLN` - 1))

rm -rf ${1}/Sophus
git clone https://github.com/stevenlovegrove/Sophus.git ${1}/Sophus
cd ${1}/Sophus

git checkout b474f05f839c0f63c281aa4e7ece03145729a2cd

mkdir build
cd build

cmake -DCMAKE_INSTALL_PREFIX=${2} ..
make install -j ${NUM_JOBS}

cd ${WD}
