#include <mpi/mpi.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <fstream>
#include <cmath>

#define ROOT 0
int SIZE = 500;

int _comm_rank = 0;
int _comm_size = 1;

double global_eps = 0;
double maxeps = 0.1e-7;
int itmax = 1000;

double *data;

int Ind(int i, int j) {
    return i * SIZE + j;
}

void Init() {
    data = new double[SIZE * SIZE];
    for (int i = 0; i <= SIZE - 1; i++) {
        for (int j = 0; j <= SIZE - 1; j++) {
            if (i == 0 || i == SIZE - 1 || j == 0 || j == SIZE - 1) {
                data[Ind(i, j)] = 0;
            } else {
                data[Ind(i, j)] = 1 + i + j;
            }
        }
    }
}

void Clear() {
    delete[] data;
}

void PrintMatrix() {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            std::cout << data[Ind(i, j)] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<class Function>
void IterateRelax(Function func) {
    for (int it = 1; it <= itmax; it++) {
        func();
        MPI_Bcast(&global_eps, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        if (global_eps < maxeps) {
            break;
        }
        global_eps = 0;
    }
}

void RelaxMPI() {
    MPI_Comm_size(MPI_COMM_WORLD, &_comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &_comm_rank);

    /*----------------- Вертикальное распараллеливание ------------------*/
    {
        /*----- Инициализация массивов для MPI_Scatterv и MPI_Gatherv ------*/
        auto *num_col_per_proc = new int[_comm_size];
        auto *displc = new int[_comm_size];

        num_col_per_proc[0] = (SIZE / _comm_size + (0 < SIZE % _comm_size));
        displc[0] = 0;

        for (int i = 1; i < _comm_size; ++i) {
            num_col_per_proc[i] = num_col_per_proc[i - 1];
            if (i == SIZE % _comm_size) {
                --num_col_per_proc[i];
            }
            displc[i] = displc[i - 1] + num_col_per_proc[i - 1];
        }
        /*-------------------------------------------------------------------*/

        int tmp_sz = num_col_per_proc[_comm_rank];
        auto *thread_data = new double[SIZE * tmp_sz];

        for (int i = 0; i < SIZE; ++i) {
            MPI_Scatterv(data + i * SIZE, num_col_per_proc, displc, MPI_DOUBLE,
                         thread_data + i * tmp_sz, tmp_sz, MPI_DOUBLE, ROOT,
                         MPI_COMM_WORLD);
        }

        for (int i = 1; i < SIZE - 1; ++i) {
            for (int j = 0; j < tmp_sz; ++j) {
                thread_data[i * tmp_sz + j] =
                        (thread_data[(i - 1) * tmp_sz + j] + thread_data[(i + 1) * tmp_sz + j]) / 2;
            }
        }

        for (int i = 0; i < SIZE; ++i) {
            MPI_Gatherv(thread_data + i * tmp_sz, tmp_sz, MPI_DOUBLE, data + i * SIZE, num_col_per_proc, displc,
                        MPI_DOUBLE,
                        ROOT,
                        MPI_COMM_WORLD);
        }

        delete[] thread_data;
        delete[] displc;
        delete[] num_col_per_proc;
    }
    /*-------------------------------------------------------------------*/


    /*---------------- Горизонтальное распараллеливание -----------------*/
    {
        /*----- Инициализация массивов для MPI_Scatterv и MPI_Gatherv ------*/
        auto *num_str_per_proc = new int[_comm_size];
        auto *displc = new int[_comm_size];

        num_str_per_proc[0] = (SIZE / _comm_size + (0 < SIZE % _comm_size)) * SIZE;
        displc[0] = 0;

        for (int i = 1; i < _comm_size; ++i) {
            num_str_per_proc[i] = num_str_per_proc[i - 1];
            if (i == SIZE % _comm_size) {
                num_str_per_proc[i] -= SIZE;
            }
            displc[i] = displc[i - 1] + num_str_per_proc[i - 1];
        }
        /*-------------------------------------------------------------------*/

        auto *thread_data = new double[num_str_per_proc[_comm_rank]];
        double eps = 0;
        MPI_Scatterv(data, num_str_per_proc, displc, MPI_DOUBLE, thread_data, num_str_per_proc[_comm_rank], MPI_DOUBLE,
                     ROOT, MPI_COMM_WORLD);
        for (int i = 0; i < num_str_per_proc[_comm_rank] / SIZE; ++i) {
            for (int j = 1; j < SIZE - 1; ++j) {
                double tmp = thread_data[i * SIZE + j];
                thread_data[i * SIZE + j] = (thread_data[i * SIZE + j - 1] + thread_data[i * SIZE + j + 1]) / 2;
                eps = std::max(eps, fabs(tmp - thread_data[i * SIZE + j]));
            }
        }
        MPI_Reduce(&eps, &global_eps, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
        MPI_Gatherv(thread_data, num_str_per_proc[_comm_rank], MPI_DOUBLE, data, num_str_per_proc, displc, MPI_DOUBLE,
                    ROOT,
                    MPI_COMM_WORLD);
        delete[] thread_data;
        delete[] displc;
        delete[] num_str_per_proc;
    }
    /*-------------------------------------------------------------------*/
}


template<class Function>
double MeasureTime(Function func) {
    double start = MPI_Wtime();
    IterateRelax(func);
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    return end - start;
}


template<class Function>
double RunOneTest(std::ostream &fout, Function func, int size) {
    SIZE = size;
    if (ROOT == _comm_rank) {
        Init();
    }
    double time = MeasureTime(func);
    if (ROOT == _comm_rank) {
        Clear();
    }
    return time;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int size = 100;
    if (argc >= 2) {
        size = std::stoi(argv[1]);
    }
    double time = RunOneTest(std::cout, RelaxMPI, size);
    if (ROOT == _comm_rank) {
        std::cout << size << " " << time << std::endl;
    }
    MPI_Finalize();
    return 0;
}
