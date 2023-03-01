#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define DEBUG 0

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

double f(double x) {
    // Define the integrand function here
    return pow(sin(x) * cos(2 * x), 2) + sqrt(6 * x);
    //return 1 / (pow(x, 2) - x + 1);
}

double integrate(double a, double b, int n) {
    double dx = (b - a) / n;
    double sum = 0.0;
    int i;

    for (i = 0; i < n; i++) {
        double x = a + (i + 0.5) * dx;
        sum += f(x) * dx;
    }

    return sum;
}

double monte_carlo(double a, double b, int n, int rank, int size) {
    int num_points = n / size;
    double local_sum = 0.0;

    int i, count = 0;
    double x, y;
    srand(time(NULL)+rank);

    for (i = 0; i < num_points; i++) {
        x = a + (double) rand() / RAND_MAX * (b - a);
        y = (double) rand() / RAND_MAX * f(b);

        if (y <= f(x)) {
            count++;
        }
    }

    double local_estimate = (double) count / num_points * (b - a) * f(b);
    MPI_Reduce(&local_estimate, &local_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return local_sum;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s config_file\n", argv[0]);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    char *config_file = argv[1];
    FILE *fp = NULL;

    if (rank == 0) {
        fp = fopen(config_file, "r");

        if (fp == NULL) {
            printf("Invalid configuration file\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    double a, b;
    int n;

    if (rank == 0) {
        if (fscanf(fp, "%lf %lf %d", &a, &b, &n) != 3) {
            printf("Invalid configuration file\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    double local_approx = monte_carlo(a, b, n, rank, size);
    double approx = 0.0;

    if (rank == 0) {
        MPI_Reduce(&local_approx, &approx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&local_approx, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        approx /= size;
        printf("Approximate integral: %.10f\n", approx);
        
        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;

        printf("Elapsed time: %f seconds\n", elapsed_time);

        if (DEBUG) {
            double exact = integrate(a, b, n);
            double error = fabs(exact - approx);
            printf("Exact integral: %.10f\n", exact);
            printf("Error: %.10f\n", error);
        }
    }

    MPI_Finalize();

    return 0;
}
