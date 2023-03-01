#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

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

    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < n; i++) {
        double x = a + (i + 0.5) * dx;
        sum += f(x) * dx;
    }

    return sum;
}

double monte_carlo(double a, double b, int n) {
    double sum = 0.0;
    double x, y;
    int i;

    #pragma omp parallel for reduction(+:sum) private(x,y)
    for (i = 0; i < n; i++) {
        x = a + (double) rand() / RAND_MAX * (b - a);
        y = (double) rand() / RAND_MAX * f(b);

        if (y <= f(x)) {
            sum += 1.0;
        }
    }

    return sum / n * (b - a) * f(b);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s config_file\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    srand(time(NULL)+clock());

    char *config_file = argv[1];
    FILE *fp = fopen(config_file, "r");

    double a, b;
    int n;
    if (fp == NULL || fscanf(fp, "%lf %lf %d", &a, &b, &n) != 3) {
        printf("Invalid configuration file\n");
        exit(EXIT_FAILURE);
    }

    clock_t start_time = clock();
    double approx = monte_carlo(a, b, n);
    clock_t end_time = clock();

    printf("Approx: %.6lf\n", approx);

    double elapsed_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %lf seconds\n", elapsed_time);

    if (DEBUG) {
        double exact = integrate(a, b, n);
        printf("Exact: %.6lf\n", exact);
        double error = fabs(exact - approx);
        printf("Error: %.6lf\n", error);
    }

    return 0;
}
