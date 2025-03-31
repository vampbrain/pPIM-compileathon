#include <iostream>
#include <vector>

// Matrix multiplication function for integer matrices
void multiplyMatrices(
    const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B,
    std::vector<std::vector<int>>& C) {
    
    // Get dimensions
    int A_rows = A.size();
    int A_cols = A[0].size();
    int B_cols = B[0].size();
    
    // Initialize result matrix with zeros
    C.resize(A_rows, std::vector<int>(B_cols, 0));
    
    // Perform matrix multiplication
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            C[i][j] = 0;
            for (int k = 0; k < A_cols; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    // Define matrix dimensions
    const int MATRIX_A_ROWS = 3;
    const int MATRIX_A_COLS = 3;
    const int MATRIX_B_COLS = 2;
    
    // Initialize Matrix A with some values
    std::vector<std::vector<int>> A(MATRIX_A_ROWS, std::vector<int>(MATRIX_A_COLS));
    std::cout << "Matrix A (" << MATRIX_A_ROWS << "x" << MATRIX_A_COLS << "):" << std::endl;
    for (int i = 0; i < MATRIX_A_ROWS; i++) {
        for (int j = 0; j < MATRIX_A_COLS; j++) {
            A[i][j] = i + j + 1;  // Simple initialization
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Initialize Matrix B with some values
    std::vector<std::vector<int>> B(MATRIX_A_COLS, std::vector<int>(MATRIX_B_COLS));
    std::cout << "\nMatrix B (" << MATRIX_A_COLS << "x" << MATRIX_B_COLS << "):" << std::endl;
    for (int i = 0; i < MATRIX_A_COLS; i++) {
        for (int j = 0; j < MATRIX_B_COLS; j++) {
            B[i][j] = i * 2 + j + 1;  // Simple initialization
            std::cout << B[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Result matrix
    std::vector<std::vector<int>> C;
    
    // Perform multiplication
    multiplyMatrices(A, B, C);
    
    // Display result
    std::cout << "\nResult Matrix C (" << MATRIX_A_ROWS << "x" << MATRIX_B_COLS << "):" << std::endl;
    for (int i = 0; i < MATRIX_A_ROWS; i++) {
        for (int j = 0; j < MATRIX_B_COLS; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}