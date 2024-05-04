execution/compilation instructions:

Compile the executable file from cuda file:
nvcc -std=c++11 -o vector2  vector2.cu

Profiling and executing the file:
nsys profile --stats=true --force-overwrite=true -o vector2-report ./vector2
nsys-ui ./vector2-report.nsys-rep
