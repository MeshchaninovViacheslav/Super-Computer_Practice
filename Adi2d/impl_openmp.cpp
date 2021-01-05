#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>

#define ROOT 0
int SIZE = 100;

int _comm_rank = 0;
int _comm_size = 1;

double global_eps = 0;
double maxeps = 0.1e-7;
int itmax = 1000;

double *data;

inline int Ind(int i, int j) {
    return i * SIZE + j;
}

void Init() {
    data = new double[SIZE * SIZE];
    for (int i = 0; i <= SIZE - 1; i++) {
        for (int j = 0; j <= SIZE - 1; j++) {
            if (i == 0 || i == SIZE - 1 || j == 0 || j == SIZE - 1) {
                data[i * SIZE + j] = 0;
            } else {
                data[i * SIZE + j] = 1 + i + j;
            }
        }
    }
}

void PrintMatrix() {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            std::cout << data[i * SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<class Function>
void IterateRelax(Function func, int *argc, char **argv[]) {
    for (int it = 1; it <= itmax; it++) {
        func(argc, argv);
        //printf("it=%4i   eps=%f\n", it, global_eps);
        //PrintMatrix();
        if (global_eps < maxeps) {
            break;
        }
        global_eps = 0;
    }
}

void Relax(int *argc, char **argv[]) {
    int i, j;
    for (i = 1; i <= SIZE - 2; i++) {
        for (j = 1; j <= SIZE - 2; j++) {
            data[i * SIZE + j] = (data[(i - 1) * SIZE + j] + data[(i + 1) * SIZE + j]) / 2.;
        }
    }

    for (j = 1; j <= SIZE - 2; j++) {
        for (i = 1; i <= SIZE - 2; i++) {
            double e;
            e = data[i * SIZE + j];
            data[i * SIZE + j] = (data[i * SIZE + j - 1] + data[i * SIZE + j + 1]) / 2.;
            global_eps = std::max(global_eps, fabs(e - data[Ind(i, j)]));
        }
    }

}

void RelaxParallelOpenMP(int *argc, char **argv[]) {
    int i, j;
    for (i = 1; i <= SIZE - 2; i++) {
#pragma omp parallel for private(j) shared(i)
        for (j = 1; j <= SIZE - 2; j++) {
            data[i * SIZE + j] = (data[(i - 1) * SIZE + j] + data[(i + 1) * SIZE + j]) / 2.;
        }
    }

    for (j = 1; j <= SIZE - 2; j++) {
#pragma omp parallel for private(i) shared(j) reduction(max: global_eps)
        for (i = 1; i <= SIZE - 2; i++) {
            double e;
            e = data[i * SIZE + j];
            data[i * SIZE + j] = (data[i * SIZE + j - 1] + data[i * SIZE + j + 1]) / 2.;
            global_eps = std::max(global_eps, fabs(e - data[Ind(i, j)]));
        }
    }
}

template<class Function>
double MeasureTime(Function func, int *argc, char **argv[]) {
    double start = omp_get_wtime();
    IterateRelax(func, argc, argv);
    double end = omp_get_wtime();
    return end - start;
}

void Clear() {
    delete[] data;
}

template<class Function>
double RunOneTest(std::ostream &fout, Function func, int size, int *argc, char **argv[]) {
    double min_time = 3600;
    SIZE = size;
    int min_cycle = 3;
    for (int i = 1; i <= min_cycle; ++i) {
        Init();
        min_time = std::min(min_time, MeasureTime(func, argc, argv));
        Clear();
    }
    return min_time;
}


int main(int argc, char *argv[]) {
    for (int num_threads = 1; num_threads < 160; num_threads += 4) {
        omp_set_num_threads(num_threads);
        std::cout << num_threads << std::endl;
        auto Sizes = {10, 100, 500, 1000, 2000, 5000};
        for (int size: Sizes) {
            std::cout << size << " " << RunOneTest(std::cout, RelaxParallelOpenMP, size, &argc, &argv) << std::endl;
        }
        std::cout << std::endl;
    }
}
