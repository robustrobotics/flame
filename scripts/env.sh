# WARNING: AUTOMATICALLY GENERATED FILE! DO NOT MODIFY!
# This file appends the current directory to the user's environment variables.

if [ -n "$ZSH_VERSION" ]; then
  WS="$( cd "$( dirname "$0" )" && pwd )"
else
  WS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi
export PATH=${WS}/bin:${PATH}
export LD_LIBRARY_PATH=${WS}/lib:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=${WS}/lib/pkgconfig:${WS}/share/pkgconfig:${PKG_CONFIG_PATH}
export PYTHONPATH=${WS}/lib/python2.7/dist-packages:${WS}/lib/python2.7/site-packages:${PYTHONPATH}
export CLASSPATH=${WS}/share/java:${CLASSPATH}
export CMAKE_PREFIX_PATH=${WS}:${CMAKE_PREFIX_PATH}
