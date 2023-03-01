#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#define NUM_THREADS 4

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

void* monte_carlo(void* arg) {
    int* arg_ptr = (int*) arg;
    int id = *arg_ptr;
    int n = arg_ptr[1];
    int num_points = n / NUM_THREADS;
    double a = arg_ptr[2];
    double b = arg_ptr[3];

    int i, count = 0;
    double x, y;
    for (i = 0; i < num_points; i++) {
        x = a + (double) rand() / RAND_MAX * (b - a);
        y = (double) rand() / RAND_MAX * f(b);

        if (y <= f(x)) {
            count++;
        }
    }

    double* result = malloc(sizeof(double));
    *result = (double) count / num_points * (b - a) * f(b);
    printf("Thread %d: %.6lf\n", id, *result);
    pthread_exit(result);
}

double geometric_monte_carlo(double a, double b, int n) {
    pthread_t threads[NUM_THREADS];
    int thread_args[NUM_THREADS][4];
    int i;
    double total = 0.0;

    for (i = 0; i < NUM_THREADS; i++) {
        thread_args[i][0] = i;
        thread_args[i][1] = n;
        thread_args[i][2] = a + (double) i / NUM_THREADS * (b - a);
        thread_args[i][3] = a + (double) (i + 1) / NUM_THREADS * (b - a);
        pthread_create(&threads[i], NULL, monte_carlo, &thread_args[i]);
    }

    for (i = 0; i < NUM_THREADS; i++) {
        double* result;
        pthread_join(threads[i], (void**) &result);
        total += *result;
        free(result);
    }

    return total;
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

    double exact = integrate(a, b, n);
    printf("Exact: %.6lf\n", exact);

    double approx = geometric_monte_carlo(a, b, n);
    printf("Approx: %.6lf\n", approx);

    return 0;
}
