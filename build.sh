spack load gcc@9
make clean
make
cd ./build
export OMP_NUM_THREADS=14
echo "Running test with $OMP_NUM_THREADS threads"
# valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all -s --track-origins=yes ./test
./test