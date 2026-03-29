cd /data/Multitenanthakes/acorn_benchmark
bash install_dependencies.sh
cmake -B build
cmake --build build -j
./build/main
