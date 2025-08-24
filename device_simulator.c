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
    fftw_plan forward_plan[MASKS + 1]; // FFTW plan for forward FFT
    fftw_plan backward_plan[MASKS + 1]; // FFTW plan for backward FFT
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
    mode->electric_field = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N * (MASKS + 1)); // +1 for the initial field before the first mask
    mode->angular_spectrum = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N * (MASKS + 1));
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

    for (int i = 0; i < MASKS + 1; i++)
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
    for (int i = 0; i < mode->MASKS + 1; i++)
    {
        fftw_destroy_plan(mode->forward_plan[i]);
        fftw_destroy_plan(mode->backward_plan[i]);
    }
}

void propagate_forward_device_mode(struct Device_Mode *mode, struct Phase_Screen phase_screens[], int target_mask_index, int start_mask_index)
{
    // Propagate the device mode through the phase screen
    if (start_mask_index == -1) {
    fftw_execute(mode->forward_plan[0]); // Go to angular spectrum
    phase_shift(mode->angular_spectrum, mode->kz_mm, mode->N, mode->DIRECTION, mode->mask_spacing_mm); // Apply phase shift from free space propagation
    fftw_execute(mode->backward_plan[0]); // Go back to electric field
    }
    for (int i = start_mask_index; i < target_mask_index + 1; i++)
    {
        phase_shift(mode->electric_field + (i + 1) * mode->N * mode->N, phase_screens[i].screen, mode->N, mode->DIRECTION, 1.0); // Apply phase screen
        fftw_execute(mode->forward_plan[i + 1]); // Go to angular spectrum
        phase_shift(mode->angular_spectrum + (i + 1) * mode->N * mode->N, mode->kz_mm, mode->N, mode->DIRECTION, mode->mask_spacing_mm); // Apply phase shift from free space propagation
        fftw_execute(mode->backward_plan[i + 1]); // Go back to electric field

   }
}

void propagate_backward_device_mode(struct Device_Mode *mode, struct Phase_Screen phase_screens[], int target_mask_index, int start_mask_index)
{
    // Propagate the device mode through the phase screen
    if (start_mask_index == mode->MASKS) {
    fftw_execute(mode->forward_plan[target_mask_index]); // Go to angular spectrum
    phase_shift(mode->angular_spectrum + target_mask_index * mode->N * mode->N, mode->kz_mm, mode->N, -mode->DIRECTION, mode->mask_spacing_mm); // Apply phase shift from free space propagation
    fftw_execute(mode->backward_plan[target_mask_index]); // Go back to electric field
    }
    for (int i = mode->start_mask_index; i >= target_mask_index; i--)
    {
        phase_shift(mode->electric_field + (i) * mode->N * mode->N, phase_screens[i].screen, mode->N, -mode->DIRECTION, 1.0); // Apply phase screen
        fftw_execute(mode->forward_plan[i]); // Go to angular spectrum
        phase_shift(mode->angular_spectrum + (i) * mode->N * mode->N, mode->kz_mm, mode->N, -mode->DIRECTION, mode->mask_spacing_mm); // Apply phase shift from free space propagation
        fftw_execute(mode->backward_plan[i]); // Go back to electric field
    }
}

//code to generate fields that we want

void generate_gaussian_beam(struct Device_Mode *field, int DIRECTION,double beam_radius_mm, double center_x_mm, double center_y_mm)
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

            if DIRECTION == MODE_FORWARD {
                field->electric_field[y * field->N + x][0] = exp(-r_squared_mm2 / (beam_radius_mm * beam_radius_mm));
                field->electric_field[y * field->N + x][1] = 0.0; // Set imaginary part to 0
            } else if (DIRECTION == MODE_BACKWARD)
            {
                field->electric_field[(field->MASKS)*field->N*field->N + y * field->N + x][0] = exp(-r_squared_mm2 / (beam_radius_mm * beam_radius_mm));
                field->electric_field[(field->MASKS)*field->N*field->N + y * field->N + x][1] = 0.0; // Set imaginary part to 0
            }
            else {
                fprintf(stderr, "Error: Invalid direction specified for generate_gaussian_beam.\n");
                exit(EXIT_FAILURE);
           }
        }
    }
}

double hermite_polynomial(int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return 2*x;
    return 2*x*hermite_polynomial(n-1, x) - 2*(n-1)*hermite_polynomial(n-2, x);
}

void generate_hermite_beam(struct Device_Mode *field, int DIRECTION,  double beam_radius_mm, int order_x, int order_y, double centre_x_mm, double centre_y_mm)
{
    // Generate a Hermite-Gaussian beam in the optical field
    double dx_mm = field->grid_size_mm / field->N; // Spatial resolution (mm/point)
    for (int y = 0; y < field->N; y++)
    {
        for (int x = 0; x < field->N; x++)
        {
            double x_pos_mm = (x - field->N / 2.0) * dx_mm;
            double y_pos_mm = (y - field->N / 2.0) * dx_mm;
            double r_squared_mm2 = ((x_pos_mm - centre_x_mm) * (x_pos_mm - centre_x_mm)) + ((y_pos_mm - centre_y_mm) * (y_pos_mm - centre_y_mm));
            double hermite_exp = exp(-r_squared_mm2 / (beam_radius_mm * beam_radius_mm));
            double hermite_x = hermite_polynomial(order_x, sqrt(2)*(x_pos_mm - centre_x_mm) / beam_radius_mm);
            double hermite_y = hermite_polynomial(order_y, sqrt(2)*(y_pos_mm - centre_y_mm) / beam_radius_mm);
            if DIRECTION == MODE_FORWARD {
                field->electric_field[y * field->N + x][0] = hermite_exp * hermite_x *  hermite_y;
                field->electric_field[y * field->N + x][1] = 0.0; // Set imaginary part to 0
            } else if (DIRECTION == MODE_BACKWARD)
            {
                field->electric_field[(field->MASKS)*field->N*field->N + y * field->N + x][0] = hermite_exp * hermite_x *  hermite_y;
                field->electric_field[(field->MASKS)*field->N*field->N + y * field->N + x][1] = 0.0; // Set imaginary part to 0
            }
            else {
                fprintf(stderr, "Error: Invalid direction specified for generate_hermite_beam.\n");
                exit(EXIT_FAILURE);
            }
        }
    }
}

void add_device_mode_overlap(struct Device_Mode *input_mode, struct Device_Mode *output_mode, struct Phase_Screen phase_screens[], int target_mask_index, complex double *overlap_field) {
    // Propagate input mode forward to the target mask
    propagate_forward_device_mode(input_mode, phase_screens, target_mask_index, target_mask_index - 1);
    // Propagate output mode backward to the target mask
    if (target_mask_index == 0) {
        propagate_backward_device_mode(output_mode, phase_screens, target_mask_index, output_mode->MASKS + 1);
    
        // Calculate overlap at the target mask //todo: check this is the correct merit funciton to calculate / and add the ability to change the merit function
    for (int i = 0; i < input_mode->N * input_mode->N; i++) {
        double in_real = input_mode->electric_field[(target_mask_index + 1) * input_mode->N * input_mode->N + i][0];
        double in_imag = input_mode->electric_field[(target_mask_index + 1) * input_mode->N * input_mode->N + i][1];
        double out_real = output_mode->electric_field[(target_mask_index) * output_mode->N * output_mode->N + i][0];
        double out_imag = output_mode->electric_field[(target_mask_index) * output_mode->N * output_mode->N + i][1];
        overlap_field[i] += (in_real + I*in_imag) * (out_real - I*out_imag); // Conjugate the output field
    }
}




void multimode_optimise_phase_masks(struct Device_Mode input_modes[], struct Device_Mode output_modes[], struct Phase_Screen phase_screens[], int target_mask_index, complex double *temp_overlap_field) {

    for (int i = 0; i < N_DEVICE_MODES; i++) {
        add_device_mode_overlap(&input_modes[i], &output_modes[i], phase_screens, target_mask_index, temp_overlap_field);
    }

    for (int i = 0; i < POINTS * POINTS; i++) {
        double phase_shift_rad = -atan2(cimag(overlap_field[i]), creal(overlap_field[i]));
        phase_screens[target_mask_index].screen[i] = phase_shift_rad;
    }
}