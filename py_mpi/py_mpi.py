from mpi4py import MPI
import math
import random
import sys

debug = False

def f(x):
    # Define the integrand function here
    return math.pow(math.sin(x) * math.cos(2 * x), 2) + math.sqrt(6 * x)
    #return 1 / (math.pow(x, 2) - x + 1)

def integrate(a, b, n):
    dx = (b - a) / n
    sum = 0.0

    for i in range(n):
        x = a + (i + 0.5) * dx
        sum += f(x) * dx

    return sum

def monte_carlo(a, b, n, rank, size):
    num_points = n // size
    local_sum = 0.0
    count = 0

    random.seed(12345 + rank)

    for i in range(num_points):
        x = a + random.random() * (b - a)
        y = random.random() * f(b)

        if y <= f(x):
            count += 1

    local_estimate = float(count) / num_points * (b - a) * f(b)
    local_sum = comm.reduce(local_estimate, op=MPI.SUM, root=0)

    return local_sum

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: python py_mpy.py config_file")
        sys.exit()

    config_file = sys.argv[1]
    fp = None

    if rank == 0:
        try:
            fp = open(config_file, "r")
        except IOError:
            print("Invalid configuration file")
            sys.exit()

    a, b, n = None, None, None

    if rank == 0:
        try:
            a, b, n = map(float, fp.readline().split())
        except ValueError:
            print("Invalid configuration file")
            sys.exit()

    a = comm.bcast(a, root=0)
    b = comm.bcast(b, root=0)
    n = comm.bcast(int(n), root=0)

    t1 = MPI.Wtime()
    local_approx = monte_carlo(a, b, n, rank, size)
    t2 = MPI.Wtime()
    approx = 0.0

    if rank == 0:
        approx = comm.reduce(local_approx, op=MPI.SUM, root=0)
    else:
        comm.reduce(local_approx, op=MPI.SUM, root=0)

    if rank == 0:
        approx /= size
        print("Approximate integral: {:.10f}".format(approx))
        print("Time taken: {:.10f} seconds".format(t2 - t1))
        
        if debug:
            exact = integrate(a, b, n)
            error = abs(exact - approx)
            print("Exact integral: {:.10f}".format(exact))
            print("Error: {:.10f}".format(error))
