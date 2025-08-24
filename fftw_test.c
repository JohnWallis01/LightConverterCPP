#include <fftw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>


#define FORWARD 1
#define BACKWARD -1


#define WAVELENGTH_NM 1550
#define POINTS 1024
#define GRID_SIZE_MM 10

#define MASKS 4
#define SPACING_MM 10
#define N_DEVICE_MODES 2



//Code to do with building phase screens
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

void generate_gaussian_beam(struct Optical_Field *field, double beam_radius_mm, double centre_x_mm, double centre_y_mm)
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
}

double hermite_polynomial(int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return 2*x;
    return 2*x*hermite_polynomial(n-1, x) - 2*(n-1)*hermite_polynomial(n-2, x);
}

void generate_hermite_beam(struct Optical_Field *field, double beam_radius_mm, int order_x, int order_y, double centre_x_mm, double centre_y_mm)
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
            field->electric_field[y * field->N + x][0] = hermite_exp * hermite_x *  hermite_y;
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

double get_mode_match(struct Optical_Field *f1, struct Optical_Field *f2)
{
    // Calculate the mode match between two optical fields
    if (f1->N != f2->N)
    {
        fprintf(stderr, "Error: Optical fields must have the same size for mode matching.\n");
    }
    double f1_integration = 0.0;
    double f2_integration = 0.0;
    double re_match_integral = 0.0;
    double im_match_integral = 0.0;
    for (int i = 0; i < f1->N * f1->N; i++)
    {
        f1_integration += (f1->electric_field[i][0] * f1->electric_field[i][0]) + (f1->electric_field[i][1] * f1->electric_field[i][1]);
        f2_integration += (f2->electric_field[i][0] * f2->electric_field[i][0]) + (f2->electric_field[i][1] * f2->electric_field[i][1]);
        re_match_integral += (f1->electric_field[i][0] * f2->electric_field[i][0]) + (f1->electric_field[i][1] * f2->electric_field[i][1]);
        im_match_integral += (f1->electric_field[i][1] * f2->electric_field[i][0]) - (f1->electric_field[i][0] * f2->electric_field[i][1]);
    }
    if (f1_integration == 0 || f2_integration == 0)
    {
        fprintf(stderr, "Error: One of the optical fields has zero intensity.\n");
    }
    double norm_factor = (f1_integration * f2_integration);
    double total_overlap = re_match_integral * re_match_integral + im_match_integral * im_match_integral;
    return (total_overlap / norm_factor); // Return the mode match value
}

void phase_shift(struct Optical_Field *field, struct Phase_Screen *phase_screen, int DIRECTION)
{
    // Apply a phase shift to the optical field using the phase screen
    if (phase_screen->N != field->N)
    {
        fprintf(stderr, "Error: Phase screen size does not match optical field size.\n");
        return;
    }
    for (int i = 0; i < field->N; i++)
    {
        for (int j = 0; j < field->N; j++)
        {
            double phase_shift_rad = phase_screen->screen[j * field->N + i]; // Phase shift in radians
            double real_part = field->electric_field[j * field->N + i][0];
            double imag_part = field->electric_field[j * field->N + i][1];
            field->electric_field[j * field->N + i][0] = real_part * cos(DIRECTION * phase_screen->screen[j * field->N + i]) - imag_part * sin(DIRECTION * phase_screen->screen[j * field->N + i]);
            field->electric_field[j * field->N + i][1] = real_part * sin(DIRECTION * phase_screen->screen[j*field->N + i]) + imag_part * cos(DIRECTION * phase_screen->screen[j * field->N + i]);
        }
    }
}

//Propogating fields through the  devices

void Forward_Propagate(struct Optical_Field *field, double propagation_distance_mm, struct Phase_Screen phase_screens[], int target_mask_index)
{
    // Propagate the optical field forward
    propagate_optical_field(field, propagation_distance_mm);
    for (int i = 0; i <target_mask_index; i++)
    {
        phase_shift(field, &phase_screens[i], FORWARD); // Apply phase shift from the phase screen
        propagate_optical_field(field, propagation_distance_mm); // Propagate the field again
    }
}
void Backward_Propagate(struct Optical_Field *field, double propagation_distance_mm, struct Phase_Screen phase_screens[], int target_mask_index)
{
    // Propagate the optical field forward
    
    propagate_optical_field(field, -propagation_distance_mm); // Propagate the field again
    for (int i = MASKS-1; i  > target_mask_index ; i--)
    {
        phase_shift(field, &phase_screens[i], BACKWARD); // Apply phase shift from the phase screen
        propagate_optical_field(field, -propagation_distance_mm); // Propagate the field again
    }
}

void Optimise_Phase_Masks(struct Optical_Field *input_field, struct Optical_Field *output_field, struct Phase_Screen phase_screens[], int target_mask_index)
{
    // Optimise the phase masks to achieve a desired output field
    //chace the input and output field values
    fftw_complex *input_electric_field = malloc(sizeof(fftw_complex) * input_field->N * input_field->N);
    fftw_complex *output_electric_field = malloc(sizeof(fftw_complex) * output_field->N * output_field->N);
    for (int i = 0; i < input_field->N * input_field->N; i++)
    {
        input_electric_field[i][0] = input_field->electric_field[i][0];
        input_electric_field[i][1] = input_field->electric_field[i][1];
        output_electric_field[i][0] = output_field->electric_field[i][0];
        output_electric_field[i][1] = output_field->electric_field[i][1];
    }
    Forward_Propagate(input_field, SPACING_MM, phase_screens, target_mask_index);
    Backward_Propagate(output_field, SPACING_MM, phase_screens, target_mask_index);

    for (int i = 0; i < input_field->N * input_field->N; i++)
    {
        //input times conjugate of output
        double real_overlap = input_field->electric_field[i][0] * output_field->electric_field[i][0] + input_field->electric_field[i][1] * output_field->electric_field[i][1];
        double imag_overlap = input_field->electric_field[i][1] * output_field->electric_field[i][0] - input_field->electric_field[i][0] * output_field->electric_field[i][1];
        // Calculate the phase shift needed to achieve the desired output field
        double phase_shift_rad = -atan2(imag_overlap, real_overlap);
        phase_screens[target_mask_index].screen[i] = phase_shift_rad; // Store the phase shift in the phase screen
    }
    // Reload the changed input and output fields
    for (int i = 0; i < input_field->N * input_field->N; i++)
    {
        input_field->electric_field[i][0] = input_electric_field[i][0];
        input_field->electric_field[i][1] = input_electric_field[i][1];
        output_field->electric_field[i][0] = output_electric_field[i][0];
        output_field->electric_field[i][1] = output_electric_field[i][1];
    }
    fftw_free(input_electric_field);
    fftw_free(output_electric_field);

}


void add_device_mode_overlap(struct Optical_Field *input_field, struct Optical_Field *output_field, struct Phase_Screen phase_screens[], int target_mask_index, complex double *overlap_field)
{
    //cache the fields TODO: build a function for caching a list of optical fields
    fftw_complex *input_electric_field = malloc(sizeof(fftw_complex) * input_field->N * input_field->N);
    fftw_complex *output_electric_field = malloc(sizeof(fftw_complex) * output_field->N * output_field->N);
    for (int i = 0; i < input_field->N * input_field->N; i++)
    {
        input_electric_field[i][0] = input_field->electric_field[i][0];
        input_electric_field[i][1] = input_field->electric_field[i][1];
        output_electric_field[i][0] = output_field->electric_field[i][0];
        output_electric_field[i][1] = output_field->electric_field[i][1];
    }
    Forward_Propagate(input_field, SPACING_MM, phase_screens, target_mask_index);
    Backward_Propagate(output_field, SPACING_MM, phase_screens, target_mask_index);

    for (int i = 0; i < input_field->N * input_field->N; i++)
    {
        //input times conjugate of output
        double real_overlap = input_field->electric_field[i][0] * output_field->electric_field[i][0] + input_field->electric_field[i][1] * output_field->electric_field[i][1];
        double imag_overlap = input_field->electric_field[i][1] * output_field->electric_field[i][0] - input_field->electric_field[i][0] * output_field->electric_field[i][1];
        // Calculate the phase shift needed to achieve the desired output field
        overlap_field[i] += real_overlap + I*imag_overlap;
    }
    // Reload the changed input and output fields
    for (int i = 0; i < input_field->N * input_field->N; i++)
    {
        input_field->electric_field[i][0] = input_electric_field[i][0];
        input_field->electric_field[i][1] = input_electric_field[i][1];
        output_field->electric_field[i][0] = output_electric_field[i][0];
        output_field->electric_field[i][1] = output_electric_field[i][1];
    }
    fftw_free(input_electric_field);
    fftw_free(output_electric_field);
}


void multimode_optimise_phase_masks(struct Optical_Field input_modes[], struct Optical_Field output_modes[], struct Phase_Screen phase_screens[], int target_mask_index) {

    complex double *overlap_field = malloc(sizeof(complex double) * POINTS * POINTS);
    for (int i = 0; i < N_DEVICE_MODES; i++) {
        add_device_mode_overlap(&input_modes[i], &output_modes[i], phase_screens, target_mask_index, overlap_field);
    }

    for (int i = 0; i < POINTS * POINTS; i++) {
        double phase_shift_rad = -atan2(cimag(overlap_field[i]), creal(overlap_field[i]));
        phase_screens[target_mask_index].screen[i] = phase_shift_rad;
    }
    free(overlap_field);
}




#define ITERATIONS 10

int main()  {
    printf("Initalizing Phase Screens\n"); 
    struct Phase_Screen phase_screens[MASKS];
    // Initialize phase screens (do we have to some MALLOC here?)
    for (int i = 0; i < MASKS; i++) {
        init_phase_screen(&phase_screens[i], POINTS);
    }
    printf("Generating Optical Fields ");
   
    struct Optical_Field input_mode1;
    init_optical_field(&input_mode1, WAVELENGTH_NM, GRID_SIZE_MM, POINTS);
    printf("...");
    generate_hermite_beam(&input_mode1, 1.0, 0, 1, 0.0, 0.0); // Generate a Hermite-Gaussian beam with order (2,2) and radius 1.0 mm centered at (0,0)
    
    struct Optical_Field input_mode2;
    init_optical_field(&input_mode2, WAVELENGTH_NM, GRID_SIZE_MM, POINTS);
    printf("...");
    generate_hermite_beam(&input_mode2, 1.0, 1, 0, 0.0, 0.0); // Generate a Hermite-Gaussian beam with order (2,2) and radius 1.0 mm centered at (0,0)

    printf("...");
    struct Optical_Field output_mode1;
    init_optical_field(&output_mode1, WAVELENGTH_NM, GRID_SIZE_MM, POINTS);
    printf("...");
    generate_gaussian_beam(&output_mode1, 1.0, 1.0, 1.0); // generate a gaussian beam with radius 1.0 mm centered at (0,0)
 
    printf("...");
    struct Optical_Field output_mode2;
    init_optical_field(&output_mode2, WAVELENGTH_NM, GRID_SIZE_MM, POINTS);
    printf("...");
    generate_gaussian_beam(&output_mode2, 1.0, -1.0, -1.0); // generate a gaussian beam with radius 1.0 mm centered at (0,0)


    struct Optical_Field input_modes[N_DEVICE_MODES];
    struct Optical_Field output_modes[N_DEVICE_MODES];

    input_modes[0] = input_mode1;
    input_modes[1] = input_mode2;
    output_modes[0] = output_mode1;
    output_modes[1] = output_mode2;


    fftw_complex *input_electric_field_data = malloc(sizeof(fftw_complex) * POINTS * POINTS * N_DEVICE_MODES);
    fftw_complex *output_electric_field_data = malloc(sizeof(fftw_complex) * POINTS * POINTS * N_DEVICE_MODES);
    for (int j = 0; j < N_DEVICE_MODES; j++) {
        for (int i = 0; i < POINTS * POINTS; i++)
        {
            input_electric_field_data[j*POINTS*POINTS + i][0] = input_modes[j].electric_field[i][0];
            input_electric_field_data[j*POINTS*POINTS + i][1] = input_modes[j].electric_field[i][1];
            output_electric_field_data[j*POINTS*POINTS + i][0] = output_modes[j].electric_field[i][0];
            output_electric_field_data[j*POINTS*POINTS + i][1] = output_modes[j].electric_field[i][1];
        }
    }
    printf(" Done. \n");

    // Perform the phase optimization
    printf("Phase optimsing\n");
    for (int i = 0; i < ITERATIONS * MASKS; i++) {
        int target_mask_index = i % MASKS; // Cycle through the masks
        printf("Progress %u\n", i);
        multimode_optimise_phase_masks(input_modes, output_modes, phase_screens, target_mask_index);

        for(int j = 0; j < N_DEVICE_MODES; j++) {
        Forward_Propagate(&input_modes[j], SPACING_MM, phase_screens, MASKS);
        propagate_optical_field(&input_modes[j], SPACING_MM); // Final propagation after all phase screens
        double match = get_mode_match(&input_modes[j], &output_modes[j]);
        printf("mode %u  matching: %f\n", j+1 , match);
        }
        //recache the data
        for (int j = 0; j < N_DEVICE_MODES; j++) {
            for (int i = 0; i < POINTS * POINTS; i++)
            {
                input_modes[j].electric_field[i][0] = input_electric_field_data[j*POINTS*POINTS + i][0];
                input_modes[j].electric_field[i][1] = input_electric_field_data[j*POINTS*POINTS + i][1]; 
                output_modes[j].electric_field[i][0] = output_electric_field_data[j*POINTS*POINTS + i][0];
                output_modes[j].electric_field[i][1] = output_electric_field_data[j*POINTS*POINTS + i][1];
        }
    }
 
    }

        
    // Clean up
    for (int i = 0; i < N_DEVICE_MODES; i++) {
    delete_optical_field(&input_modes[i]);
    delete_optical_field(&output_modes[i]);
    }
    printf("...");
    for (int i = 0; i < MASKS; i++) {
        delete_phase_screen(&phase_screens[i]);
        printf("...");
    }
    printf("Done.\n");

}

