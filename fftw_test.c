#include <fftw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define WAVELENGTH_NM 1550
#define POINTS 1024
#define GRID_SIZE_MM 5

// Structure to represent the optical field
struct Optical_Field
{
    double wavelength_nm; //In NM
    double grid_size_mm; //Total Grid Size in mm
    int N; // Number of points per dimension
    fftw_complex *electric_field; // Pointer to the field data array
    fftw_complex *angular_spectrum; // Pointer to the angular spectrum data array
    fftw_plan forward_plan; // FFTW plan for forward FFT
    fftw_plan backward_plan; // FFTW plan for backward FFT
};

void init_optical_field(struct Optical_Field *field, double wavelength_nm, double grid_size_mm, int N)
{
    field->wavelength_nm = wavelength_nm;
    field->grid_size_mm = grid_size_mm;
    field->N = N;
    // fftw_free(field->electric_field);
    // fftw_free(field->angular_spectrum);
    field->electric_field = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N);
    field->angular_spectrum = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N);
    // fftw_destroy_plan(field->forward_plan);
    // fftw_destroy_plan(field->backward_plan);
    field->forward_plan = fftw_plan_dft_2d(N, N, field->electric_field, field->angular_spectrum, FFTW_FORWARD, FFTW_MEASURE);
    field->backward_plan = fftw_plan_dft_2d(N, N, field->angular_spectrum, field->electric_field, FFTW_BACKWARD, FFTW_MEASURE);
}
void delete_optical_field(struct Optical_Field *field)
{
    fftw_free(field->angular_spectrum);
    fftw_free(field->electric_field);
    fftw_destroy_plan(field->forward_plan);
    fftw_destroy_plan(field->backward_plan);
}
void copy_optical_field(struct Optical_Field *dest, const struct Optical_Field *src)
// Copies the optical field data from src to dest
{
    init_optical_field(dest, src->wavelength_nm, src->grid_size_mm, src->N);
    for (size_t i = 0; i < src->N * src->N; i++)
    {
        dest->electric_field[i][0] = src->electric_field[i][0];
        dest->electric_field[i][1] = src->electric_field[i][1];
        dest->angular_spectrum[i][0] = src->angular_spectrum[i][0];
        dest->angular_spectrum[i][1] = src->angular_spectrum[i][1];
    }
}


int fft_coord_shift(int N, int coord)
{
    // Apply FFT coordinate shift
    if (coord >= N / 2)
    {
        return coord - N;
    }
    return coord;
}

void propagate_optical_field(struct Optical_Field *field, double propagation_distance_mm)
{
    // Propagate the optical field using the angular spectrum method
    fftw_execute(field->forward_plan);
    double wavevector_magnitude_mm = 2 * M_PI/(field->wavelength_nm * 1e-6); // Convert wavelength from nm to mm (units is /mm)
    double dk_mm = M_PI/field->grid_size_mm; // Frequency resolution in /mm
    double kx_mm, ky_mm;
    for (int i = 0; i < field->N; i++)
    {
        kx_mm = fft_coord_shift(field->N, i) * dk_mm;
        for (int j = 0; j < field->N; j++)
        {
            ky_mm = fft_coord_shift(field->N, j) * dk_mm;
            double kz_mm = 4*sqrt(wavevector_magnitude_mm * wavevector_magnitude_mm - (kx_mm * kx_mm) - (ky_mm * ky_mm)); //MAGIC fudge factor of 4
            double phase = -1.0*kz_mm*propagation_distance_mm; // Phase shift due to propagation
            //check nan in the phase
            double angular_spectrum_temp_real = field->angular_spectrum[j * field->N + i][0];
            double angular_spectrum_temp_imag = field->angular_spectrum[j * field->N + i][1];
            field->angular_spectrum[j * field->N + i][0] = (angular_spectrum_temp_real * cos(phase) - angular_spectrum_temp_imag * sin(phase))/(field->N * field->N); // Normalize by N*N for backward FFT
            field->angular_spectrum[j * field->N + i][1] = (angular_spectrum_temp_real * sin(phase) + angular_spectrum_temp_imag * cos(phase))/(field->N * field->N);

        }
    }
    fftw_execute(field->backward_plan);
}

void generate_gaussian_beam(struct Optical_Field *field, double beam_radius_mm)
{
    // Generate a Gaussian beam in the optical field
    double dx_mm = field->grid_size_mm / field->N; // Spatial resolution (mm/point)
    for (int y = 0; y < field->N; y++)
    {
        for (int x = 0; x < field->N; x++)
        {
            double x_pos_mm = (x - field->N / 2) * dx_mm;
            double y_pos_mm = (y - field->N / 2) * dx_mm;
            double r_squared_mm2 = x_pos_mm * x_pos_mm + y_pos_mm * y_pos_mm;
            field->electric_field[y * field->N + x][0] = exp(-r_squared_mm2 / (beam_radius_mm * beam_radius_mm));
            field->electric_field[y * field->N + x][1] = 0.0; // Set imaginary part to 0
        }
    }
}

void save_field(struct Optical_Field *field, const char *realfilename, const char *imagfilename)
{
    FILE *fptr = fopen(realfilename, "w");
    if (fptr == NULL)
    {
        perror("Error opening file");
        return;
    }
    for (size_t i = 0; i < field->N * field->N - 1; i++)
    {
        fprintf(fptr, "%f,", field->electric_field[i][0]);
    }
    fprintf(fptr, "%f\n", field->electric_field[field->N * field->N - 1][0]); // Save the last value without a comma
    fclose(fptr);

    fptr = fopen(imagfilename, "w");
    if (fptr == NULL)
    {
        perror("Error opening file");
        return;
    }

    for (size_t i = 0; i < field->N * field->N - 1; i++)
    {
        fprintf(fptr, "%f,", field->electric_field[i][1]);
    }
    fprintf(fptr, "%f\n", field->electric_field[field->N * field->N - 1][1]); // Save the last value without a comma
    fclose(fptr);
}


int main()  {
    struct Optical_Field field;
    init_optical_field(&field, WAVELENGTH_NM, GRID_SIZE_MM, POINTS);

    generate_gaussian_beam(&field, 1.2);
    save_field(&field, "re_optical_field.dat", "im_optical_field.dat");
    propagate_optical_field(&field, 1800.0);
    save_field(&field, "re_propagated_field.dat", "im_propagated_field.dat");

    return 0;
}

