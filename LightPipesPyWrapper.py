from ctypes import *
import numpy as np
so_file = "/mnt/c/Users/John/Documents/LightConverterCPP/lightpipes.so"
so_file = "/mnt/e/Libraries/Documents/LightConverterCPP/lightpipes.so"
jw_lp = CDLL(so_file)

jw_lp.begin_field.argtypes = [c_double, c_double, c_int]
jw_lp.begin_field.restype = POINTER(c_void_p)
def Begin(wavelength, grid_size, N):
    """Initiates a field with a grid size, a wavelength and a grid dimension. This function returns a pointer to a struct Optical_Field."""
    return jw_lp.begin_field(wavelength*1e9, grid_size*1e3, N)


jw_lp.propagate_optical_field.argtypes = [POINTER(c_void_p), c_double]
jw_lp.propagate_optical_field.restype = POINTER(c_void_p)
def Forvard(field, distance):
    """Propagates the field using a FFT algorithm. Note this will overwrite the input field (differnet behvaiour to naitve lightpipes)."""
    jw_lp.propagate_optical_field(field, distance*1e3)
    return field

jw_lp.get_field_data.argtypes = [POINTER(c_void_p)]
jw_lp.get_field_data.restype = POINTER(c_double)
def get_field_data(field):
    """gets the complex field data and returns as a numpy array"""
    data = jw_lp.get_field_data(field)
    N = int(data[0])
    field =  np.zeros((N *N), dtype=np.complex128)
    for i in range(N * N):
        field[i] = data[i*2 + 1] + 1j* data[i*2 + 2]
    field = np.reshape(field, (N, N))
    return field

jw_lp.generate_gaussian_beam.argtypes = [POINTER(c_void_p), c_double, c_double, c_double]
jw_lp.generate_gaussian_beam.restype = POINTER(c_void_p)
def GaussBeam(Fin, w0, n=0, m=0, x_shift=0, y_shift=0, tx=0, ty=0, doughnut = False, LG = False):
    """Generates a Gaussian beam with a waist w0, and a shift in the x and y direction.
    Note that only waist and x y shift have been generated"""
    jw_lp.generate_gaussian_beam(Fin, w0*1e3, x_shift*1e3, y_shift*1e3)
    return Fin

jw_lp.delete_optical_field.argtypes = [POINTER(c_void_p)]
def FreeField(Fin):
    """Deletes the field to free up memory"""
    jw_lp.delete_optical_field(Fin)
