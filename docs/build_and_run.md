# Build and Run Instructions

## Build

```bash
mkdir -p build
cd build
cmake .. -DTORCH_DIR=/absolute/path/to/libtorch
cmake --build . -j
