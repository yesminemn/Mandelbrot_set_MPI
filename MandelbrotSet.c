#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 400 // Image width
#define HEIGHT 300 // Image height
#define MAX_ITER 100 // Maximum number of iterations for the Mandelbrot calculation

int mandelbrot(double x, double y)
{
    // Calculate the Mandelbrot value for a given point (x, y)
    double real = x;
    double imag = y;
    int iter = 0;
    while (iter < MAX_ITER && (real * real + imag * imag) < 4.0) {
        double temp_real = real * real - imag * imag + x;
        double temp_imag = 2.0 * real * imag + y;
        real = temp_real;
        imag = temp_imag;
        iter++;
    }
    return iter;
}

void save_image(int *image, int width, int height, char *filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error: Failed to open file %s for writing\n", filename);
        return;
    }
    int i, j;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            fprintf(fp, "%d ", image[i * width + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Image saved to %s\n", filename);
}

int main(int argc, char **argv) {
    int rank, size, i, j;
    double x_min = -2.0;
    double x_max = 1.0;
    double y_min = -1.5;
    double y_max = 1.5;
    double x_step = (x_max - x_min) / WIDTH;
    double y_step = (y_max - y_min) / HEIGHT;
    int rows_per_task, extra_rows, start_row, end_row;
    int *image = (int *)malloc(WIDTH * HEIGHT * sizeof(int));
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        // Master process
        rows_per_task = HEIGHT / size;
        extra_rows = HEIGHT % size;
        start_row = 0;
        for (i = 1; i < size; i++) {
            // Distribute tasks to worker processes
            end_row = start_row + rows_per_task + (i <= extra_rows ? 1 : 0);
            MPI_Send(&start_row, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&end_row, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            start_row = end_row;
        }
        
        // Calculate Mandelbrot values for the first block of rows
        for (i = 0; i < rows_per_task + (extra_rows > 0 ? 1 : 0); i++) {
            for (j = 0; j < WIDTH; j++) {
                double x = x_min + j * x_step;
                double y = y_min + (i + rank * rows_per_task + (rank < extra_rows ? rank : extra_rows)) * y_step;
                int iter = mandelbrot(x, y);
                image[(i + rank * rows_per_task) * WIDTH + j] = iter;
            }
        }
        
        // Receive Mandelbrot values from worker processes
        for (i = 1; i < size; i++) {
            MPI_Recv(&start_row, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&end_row, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&image[start_row * WIDTH], (end_row - start_row) * WIDTH, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Save image
        char *filename = "mandelbrot.txt";
        save_image(image, WIDTH, HEIGHT, filename);
    } else if (rank > 0) {
        // Worker process
        MPI_Recv(&start_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&end_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (i = start_row; i < end_row; i++) {
            for (j = 0; j < WIDTH; j++) {
                double x = x_min + j * x_step;
                double y = y_min + i * y_step;
                int iter = mandelbrot(x, y);
                MPI_Send(&iter, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
    }
    MPI_Finalize();
    return 0;
    }

    



