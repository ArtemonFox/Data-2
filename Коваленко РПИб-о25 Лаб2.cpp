#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <mkl.h>

using namespace std;

using Complex = complex<float>;
const int n = 1024;

void multiply_naive(const vector<Complex>& A, const vector<Complex>& B, vector<Complex>& C) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Complex sum(0, 0);
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void multiply_optimized(const vector<Complex>& A, const vector<Complex>& B, vector<Complex>& C) {
    fill(C.begin(), C.end(), Complex(0, 0));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            Complex a_val = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += a_val * B[k * n + j];
            }
        }
    }
}

void print_stats(string name, double seconds) {
    double c = 2.0 * n * n * n;
    double p = (c / seconds) * 1e-6;
    cout << name << ":\n";
    cout << "  Time: " << seconds << " s\n";
    cout << "  Perf: " << p << " MFlops\n\n";
}

int main() {
    cout << "Коваленко Артём Денисович РПИб-о25" << endl;


    vector<Complex> A(n * n), B(n * n), C(n * n);

    for (int i = 0; i < n * n; ++i) {
        A[i] = { (float)rand() / RAND_MAX, (float)rand() / RAND_MAX };
        B[i] = { (float)rand() / RAND_MAX, (float)rand() / RAND_MAX };
    }

    auto s1 = chrono::high_resolution_clock::now();
    multiply_naive(A, B, C);
    auto e1 = chrono::high_resolution_clock::now();
    print_stats("Variant 1 (Naive)", chrono::duration<double>(e1 - s1).count());

    MKL_Complex8 alpha = { 1.0f, 0.0f }, beta = { 0.0f, 0.0f };
    auto s2 = chrono::high_resolution_clock::now();
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &alpha, A.data(), n, B.data(), n, &beta, C.data(), n);
    auto e2 = chrono::high_resolution_clock::now();
    print_stats("Variant 2 (MKL)", chrono::duration<double>(e2 - s2).count());

    auto s3 = chrono::high_resolution_clock::now();
    multiply_optimized(A, B, C);
    auto e3 = chrono::high_resolution_clock::now();
    print_stats("Variant 3 (Optimized)", chrono::duration<double>(e3 - s3).count());

    return 0;
}
