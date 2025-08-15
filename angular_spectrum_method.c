#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <fftw3.h>

#define PI 3.14159265358979323846

// Structure to represent the optical field
struct Optical_Field
{
    float wavelength; //In NM
    float grid_size; //Total Grid Size in mm
    float N; // Number of points per dimension
    complex double *field_data; // Pointer to the field data array
};

//Function to initialize the optical field
void init_optical_field(struct Optical_Field *field, float wavelength, float grid_size, float N)
{
    field->wavelength = wavelength;
    field->grid_size = grid_size;
    field->N = N;
    field->field_data = (complex double *)malloc(N * N * sizeof(complex double));
}

void delete_optical_field(struct Optical_Field *field)
{
    free(field->field_data);
}

void copy_optical_field(struct Optical_Field *dest, const struct Optical_Field *src)
// Copies the optical field data from src to dest
{
    dest->wavelength = src->wavelength;
    dest->grid_size = src->grid_size;
    dest->N = src->N;
    dest->field_data = (complex double *)malloc(src->N * src->N * sizeof(complex double));
    for (size_t i = 0; i < src->N * src->N; i++)
    {
        dest->field_data[i] = src->field_data[i];
    }
}


void ASM_Propogate(struct Optical_Field *field, float distance)
{
    // Angular Spectrum Method propagation implementation distance in mm
    float k = 2 * PI / field->wavelength; // Wave number
    float dx = field->grid_size / field->N; // Spatial resolution
    float fx = 1 / (field->grid_size / field->N); // Frequency resolution
    float df = 1 / (field->grid_size / field->N); // Frequency resolution

    // Allocate memory for the Fourier-transformed field
    complex double *field_ft = (complex double *)malloc(field->N * field->N * sizeof(complex double));
    if (!field_ft)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }


    // Propagate the field in the Fourier domain
    for (int u = 0; u < field->N; u++)
    {
        for (int v = 0; v < field->N; v++)
        {
            float fx = (u - field->N / 2) * df;
            float fy = (v - field->N / 2) * df;
            float kz = sqrt(k * k - fx * fx - fy * fy);
            if (kz > 0)
            {
                field_ft[u * field->N + v] = field->field_data[u * field->N + v] * cexp(I * kz * distance);
            }
            else
            {
                field_ft[u * field->N + v] = 0;
            }
        }
    }

    // Perform inverse 2D FFT to get the propagated field
    FFT(field_ft, field->N);
    FFT(field_ft + field->N, field->N);

    // Copy the propagated field back to the original field
    for (int i = 0; i < field->N * field->N; i++)
    {
        field->field_data[i] = field_ft[i] / (field->N * field->N);
    }

    free(field_ft);
}