# Debian (Linux) Script
rm -rf build
mkdir build && cd build
cmake ..
make
./TFLiteMoveNet