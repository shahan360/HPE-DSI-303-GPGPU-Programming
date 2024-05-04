#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define NUM_FEATURES 1766
#define NUM_CHEMICALS 3000

// Function to read data from a text file into a 2D array
void readDataFromFile(const std::string& filename, float* data, int rows, int cols) {
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        int row = 0;
        while (std::getline(file, line) && row < rows) {
            std::istringstream iss(line);
            for (int col = 0; col < cols; ++col) {
                iss >> data[row * cols + col];
            }
            row++;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void computeDistanceAndSimilarity(float *bioresponse_matrix, float *known_drug, float *D, float *S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_CHEMICALS) {
        float distance = 0.0f;
        for (int i = 0; i < NUM_FEATURES; ++i) {
            distance += pow(fabs(bioresponse_matrix[idx * NUM_FEATURES + i] - known_drug[i]), 1.50);
        }
        D[idx] = pow(distance, 1.0f / 1.50f);

        // Compute similarity
        float gamma = 1.0f / NUM_FEATURES;
        S[idx] = expf(-D[idx] * gamma);
    }
}

int main() {
    // Allocate memory for data
    float *bioresponse_matrix, *known_drug, *D, *S;
    cudaMallocManaged(&bioresponse_matrix, NUM_CHEMICALS * NUM_FEATURES * sizeof(float));
    cudaMallocManaged(&known_drug, NUM_FEATURES * sizeof(float));
    cudaMallocManaged(&D, NUM_CHEMICALS * sizeof(float));
    cudaMallocManaged(&S, NUM_CHEMICALS * sizeof(float));

    // Read data from files and populate bioresponse_matrix and known_drug arrays
    readDataFromFile("bioresponse_descriptors_matrix.txt", bioresponse_matrix, NUM_CHEMICALS, NUM_FEATURES);
    readDataFromFile("known_drug.txt", known_drug, 1, NUM_FEATURES);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (NUM_CHEMICALS + blockSize - 1) / blockSize;
    computeDistanceAndSimilarity<<<numBlocks, blockSize>>>(bioresponse_matrix, known_drug, D, S);
    cudaDeviceSynchronize();

    // Allocate memory for sorted indices
    int *sorted_indices;
    cudaMallocManaged(&sorted_indices, NUM_CHEMICALS * sizeof(int));

    // Use Thrust to sort the similarity vector S along with their corresponding indices
    thrust::device_ptr<float> dev_ptr_S(S);
    thrust::sequence(sorted_indices, sorted_indices + NUM_CHEMICALS);
    thrust::sort_by_key(dev_ptr_S, dev_ptr_S + NUM_CHEMICALS, sorted_indices, thrust::greater<float>());

    // Output top 10 most similar chemicals
    std::cout << "Top 10 most similar chemicals:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << sorted_indices[i] << "\n";
    }

    // Free allocated memory
    cudaFree(bioresponse_matrix);
    cudaFree(known_drug);
    cudaFree(D);
    cudaFree(S);
    cudaFree(sorted_indices);

    return 0;
}

