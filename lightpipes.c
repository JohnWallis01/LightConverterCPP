#include <fftw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

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

struct Optical_Field *begin_field(double wavelength_nm, double grid_size_mm, int N)
{
    struct Optical_Field *field = malloc(sizeof(struct Optical_Field));
    init_optical_field(field, wavelength_nm, grid_size_mm, N);
    return field;
}

struct Optical_Field *generate_gaussian_beam(struct Optical_Field *field, double beam_radius_mm, double centre_x_mm, double centre_y_mm)
{
    // Generate a Gaussian beam in the optical field
    double dx_mm = field->grid_size_mm / field->N; // Spatial resolution (mm/point)
    for (int y = 0; y < field->N; y++)
    {
        for (int x = 0; x < field->N; x++)
        {
            double x_pos_mm = (x - field->N / 2.0) * dx_mm;
            double y_pos_mm = (y - field->N / 2.0) * dx_mm;
            double r_squared_mm2 = ((x_pos_mm - centre_x_mm) * (x_pos_mm - centre_x_mm)) + ((y_pos_mm - centre_y_mm) * (y_pos_mm - centre_y_mm));
            field->electric_field[y * field->N + x][0] = exp(-r_squared_mm2 / (beam_radius_mm * beam_radius_mm));
            field->electric_field[y * field->N + x][1] = 0.0; // Set imaginary part to 0
        }
    }
    return field;
}

void test_field(struct Optical_Field *field)
{
    //Initalize the field with the real part coutnign up from 0 to N*N
    // and the complex part counting down from N*N to 0
    for (int i = 0; i < field->N; i++)
    {
        for (int j = 0; j < field->N; j++)
        {
            field->electric_field[i * field->N + j][0] = (double)(i * field->N + j);
            field->electric_field[i * field->N + j][1] = (double)(field->N * field->N - (i * field->N + j));
        }
    }
}


double *get_field_data(struct Optical_Field *field)
{
    double *return_data = malloc(sizeof(double) * field->N * field->N * 2 + 1);
    for (int i = 0; i < field->N; i++)
    {
        for (int j = 0; j < field->N; j++)
        {
            //increment by 1 because the first value will hold the size N 
            return_data[(i * field->N + j) * 2 + 1] = field->electric_field[i * field->N + j][0];
            return_data[(i * field->N + j) * 2 + 2] = field->electric_field[i * field->N + j][1];
        }
        //first value is the size N
        return_data[0] = (double)field->N;
    }
    return return_data;
}

void delete_optical_field(struct Optical_Field *field)
{
    fftw_free(field->angular_spectrum);
    fftw_free(field->electric_field);
    fftw_destroy_plan(field->forward_plan);
    fftw_destroy_plan(field->backward_plan);
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

struct Optical_Field *propagate_optical_field(struct Optical_Field *field, double propagation_distance_mm)
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
    return field;
}
