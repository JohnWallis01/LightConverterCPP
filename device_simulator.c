#include <fftw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define MODE_FORWARD 1
#define MODE_BACKWARD -1


//GENERIC Helper Functions
void phase_shift(fftw_complex *field, double *phase_screen, int N, int direction, double scalar)
{
    // Apply phase shift to the field using the phase screen
    for (int i = 0; i < N * N; i++)
    {
        double phase_shift_rad = direction*phase_screen[i] * scalar; // Phase shift in radians
        double real_part = field[i][0];
        double imag_part = field[i][1];
        field[i][0] = real_part * cos(phase_shift_rad) - imag_part * sin(phase_shift_rad);
        field[i][1] = real_part * sin(phase_shift_rad) + imag_part * cos(phase_shift_rad);
    }
}



//PHASE SCREENS

struct Phase_Screen
{
    double *screen; // Pointer to the phase screen data array
    int N; // Number of points per dimension
};

void init_phase_screen(struct Phase_Screen *screen, int N)
{
    screen->N = N;
    screen->screen = (double *)malloc(sizeof(double) * N * N);
    if (screen->screen == NULL)
    {
        perror("Error allocating memory for phase screen");
        exit(EXIT_FAILURE);
    }
}
void delete_phase_screen(struct Phase_Screen *screen)
{
    free(screen->screen);
    screen->screen = NULL; // Set pointer to NULL after freeing
}

//DEVICE MODES

struct Device_Mode
{
    int N; // Number of points per dimension
    int MASKS; // Number of masks
    double mask_spacing_mm; // Spacing between masks in mm
    double wavelength_nm; // Wavelength in nanometers
    double grid_size_mm; // Grid size in millimeters
    fftw_complex *electric_field; // Pointer to the electric field data array
    fftw_complex *angular_spectrum; // Pointer to the angular spectrum data array
    fftw_plan forward_plan[MASKS]; // FFTW plan for forward FFT
    fftw_plan backward_plan[MASKS]; // FFTW plan for backward FFT
    int DIRECTION; // Direction of propagation (FORWARD or BACKWARD)
    double *kz_mm; // Wavevector in the z-direction
};

void init_device_mode(struct Device_Mode *mode, double wavelength_nm, double grid_size_mm, int N, int MASKS, double mask_spacing_mm, int DIRECTION)
{
    mode->N = N;
    mode->MASKS = MASKS;
    mode->wavelength_nm = wavelength_nm;
    mode->grid_size_mm = grid_size_mm;
    mode->mask_spacing_mm = mask_spacing_mm;
    mode->DIRECTION = DIRECTION;
    mode->electric_field = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N * MASKS);
    mode->angular_spectrum = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N * MASKS);
    mode->kz_mm = (double *)malloc(sizeof(double) * N * N);
    
    //Pre compute the free space propagation eigenvalues
    double wavevector_magnitude_mm = 2 * M_PI/(mode->wavelength_nm * 1e-6); // Convert wavelength from nm to mm (units is /mm)
    double dk_mm = M_PI/mode->grid_size_mm; // Frequency resolution in /mm
    double kx_mm, ky_mm;
    for (int i = 0; i < N; i++)
    {
        kx_mm = fft_coord_shift(N, i) * dk_mm;
        for (int j = 0; j < N; j++)
        {
            ky_mm = fft_coord_shift(N, j) * dk_mm;
            mode->kz_mm[j * N + i] = -4.0*sqrt(wavevector_magnitude_mm * wavevector_magnitude_mm - (kx_mm * kx_mm) - (ky_mm * ky_mm)); //MAGIC fudge factor of 4
        }
    }

    for (int i = 0; i < MASKS; i++)
    {
        mode->forward_plan[i] = fftw_plan_dft_2d(N, N, (mode->electric_field + i), (mode->angular_spectrum+i), DIRECTION*FFTW_FORWARD, DIRECTION*FFTW_MEASURE);
        mode->backward_plan[i] = fftw_plan_dft_2d(N, N, (mode->angular_spectrum+i), (mode->electric_field+i), DIRECTION*FFTW_BACKWARD, DIRECTION*FFTW_MEASURE);
    }
}

void delete_device_mode(struct Device_Mode *mode)
{
    free(mode->kz_mm);
    fftw_free(mode->angular_spectrum);
    fftw_free(mode->electric_field);
    for (int i = 0; i < mode->MASKS; i++)
    {
        fftw_destroy_plan(mode->forward_plan[i]);
        fftw_destroy_plan(mode->backward_plan[i]);
    }
}

//TODO: figure out how indexing for free space vs phase screens works (we need more fftw plans I think )
void propagate_forward_device_mode(struct Device_Mode *mode, struct Phase_Screen phase_screens[], int target_mask_index)
{
    // Propagate the device mode through the phase screen
    for (int i = 0; i < target_mask_index; i++)
    {
        fftw_execute(mode->forward_plan[i]); // Go to angular spectrum
        phase_shift(mode->angular_spectrum + i, mode->kz_mm, mode->N, mode->DIRECTION, mode->mask_spacing_mm); // Apply phase shift from free space propagation
        fftw_execute(mode->backward_plan[i]); // Go back to electric field
        phase_shift(mode->electric_field + i, phase_screens[i].screen, mode->N, mode->DIRECTION, mode->mask_spacing_mm); // Apply phase shift from the phase screen
    }
}