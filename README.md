[![CircleCI](https://circleci.com/gh/robustrobotics/flame/tree/master.svg?style=shield)](https://circleci.com/gh/robustrobotics/flame/tree/master)

# flame
**FLaME** (Fast Lightweight Mesh Estimation) is a lightweight, CPU-only method
for dense online monocular depth estimation. Given a sequence of camera images
with known poses, **FLaME** is able to reconstruct dense 3D meshes of the
environment by posing the depth estimation problem as a variational optimization
over a Delaunay graph that can be solved at framerate, even on computationally
constrained platforms.

The `flame` repository contains the source code for the core algorithm. It
should be input/output agnostic, so feel free to write an appropriate frontend
for your data. ROS bindings are available with the
associated [flame_ros](https://github.com/robustrobotics/flame_ros.git)
repository, which also includes examples for running `flame` on offline data.

<p align="center">
    <a href="https://www.youtube.com/watch?v=vB_F-Sj0AX0">
    <img src="https://img.youtube.com/vi/vB_F-Sj0AX0/0.jpg" alt="FLaME">
    </a>
</p>

### Related Publications:
* [**FLaME: Fast Lightweight Mesh Estimation using Variational Smoothing on Delaunay Graphs**](https://groups.csail.mit.edu/rrg/papers/greene_iccv17.pdf),
*W. Nicholas Greene and Nicholas Roy*, ICCV 2017.

## Author
- W. Nicholas Greene (wng@csail.mit.edu)

## Dependencies
- Ubuntu 16.04
- Boost 1.58
- OpenCV 3.2
- Eigen 3.2
- Sophus (SHA: b474f05f839c0f63c281aa4e7ece03145729a2cd)

## Installation
**NOTE:** These instructions assume you are running Ubuntu 16.04 and are
interested in installing `flame` only. See the installation instructions for
`flame_ros` if you also wish to build the ROS bindings as the process can be
streamlined using `catkin_tools`.

1. Install `apt` dependencies:
```bash
sudo apt-get install libboost-all-dev
```

2. Install OpenCV 3.2:

Unfortunately OpenCV 3.2 is not available through `apt` on Ubuntu 16.04. If you
have ROS Kinetic installed on your system, you can simply source your ROS
installation as this version of OpenCV is packaged with ROS Kinetic. If you
don't have ROS Kinetic installed, then you will need to install from
source. Please consult
the
[OpenCV docs](http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_install/linux_install.html) for
instructions.

3. Install Eigen 3.2 and Sophus using the provided scripts:
```bash
cd flame

# Create a dependencies folder.
mkdir -p dependencies/src

# Checkout Eigen and Sophus into ./dependencies/src and install into ./dependencies.
./scripts/eigen.sh ./dependencies/src ./dependencies
./scripts/sophus.sh ./dependencies/src ./dependencies

# Copy and source environment variable script:
cp ./scripts/env.sh ./dependencies/
source ./dependencies/env.sh
```

4. Install `flame`:
```bash
cd flame
mkdir build
cd build
cmake -D CMAKE_INSTALL_PREFIX=path/to/install/directory ..
make install
```

## Usage
See `flame_ros` for ROS bindings and example usage.
