import LightPipes as  LP
import LightPipesPyWrapper as JWLP
import numpy as np
import time as time
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc 
N = 2048
grid_size = 10e-3  # 10 mm
wavelength = 1550e-9
iterations = 50


def Plot_Field(optical_field, title="Field"):
    intensity = np.abs(optical_field)**2
    phase = np.angle(optical_field)

    #on one plot colour code based on the phase but brightness is determined by intesnity
    plt.figure()
    plt.imshow(phase, cmap='hsv', interpolation='nearest', alpha=intensity/np.max(intensity))
    plt.colorbar(label='Phase (radians)')
    # #turn off axis
    plt.axis('off')
    plt.title(title)    
    plt.savefig("outputs/{}.png".format(title), bbox_inches='tight')
    # return plt





print("Testing Initalize + Propogate")
start_time = time.time()
for i in tqdm(range(iterations)):
    field = LP.Begin(grid_size, wavelength, N)
    field = LP.GaussBeam(field, 1e-3, x_shift=0, y_shift=0)
    field = LP.Forvard(field, 1)
    gc.collect()  # Force garbage collection to free memory
end_time = time.time()
# print("Time taken for 100 iterations with Py: FAILED_TIMEOUT seconds")
print("Time taken for {} iterations with Py: {:.2f} seconds".format(iterations, end_time - start_time))
print("Average time per iteration with Py: {:.2f} seconds".format((end_time - start_time) / iterations))
# print("Average time per iteration with Py: FAILED_TIMEOUT seconds")

# field_data = field.field
# Plot_Field(field_data, title="Field after init + prop with Python")


start_time = time.time()
for i in tqdm(range(iterations)):
    field = JWLP.Begin(wavelength, grid_size, N)
    field = JWLP.GaussBeam(field, 1e-3, x_shift=0, y_shift=0)
    field = JWLP.Forvard(field, 1)
    JWLP.FreeField(field)
end_time = time.time()

print("Time taken for {} iterations with C: {:.2f} seconds".format(iterations, end_time - start_time))
print("Average time per iteration with C: {:.4f} seconds".format((end_time - start_time) / iterations))

# field_data = JWLP.get_field_data(field)
# Plot_Field(field_data, title="Field after init + prop with C")


print("Testing Propogate only")
start_time = time.time()
field = LP.Begin(grid_size, wavelength, N)
field = LP.GaussBeam(field, 1e-3, x_shift=0, y_shift=0)
for i in range(iterations):
    LP.Forvard(field, 1e-6) #prop only 1um
end_time = time.time()

print("Time taken for {} iterations with Py: {:.2f} seconds".format(iterations, end_time - start_time))
print("Average time per iteration with Py: {:.4f} seconds".format((end_time - start_time) / iterations))

field_data = field.field
Plot_Field(field_data, title="Field after prop with Python")

start_time = time.time()
field = JWLP.Begin(wavelength, grid_size, N)
field = JWLP.GaussBeam(field, 1e-3, x_shift=0, y_shift=0)
for i in range(iterations):
    JWLP.Forvard(field, 1e-6)  # prop only 1um
end_time = time.time()

field_data = JWLP.get_field_data(field)

Plot_Field(field_data, title="Field after prop with C")

print("Time taken for {} iterations with C: {:.2f} seconds".format(iterations, end_time - start_time))
print("Average time per iteration with C: {:.4f} seconds".format((end_time - start_time) / iterations))


print("Testing Initalize + Propogate")
start_time = time.time()
for i in tqdm(range(iterations)):
    field = LP.Begin(grid_size, wavelength, N)
    field = LP.GaussBeam(field, 1e-3, x_shift=0, y_shift=0)
    field = LP.Forvard(field, 1, usepyFFTW=True)
    gc.collect()  # Force garbage collection to free memory
end_time = time.time()
# print("Time taken for 100 iterations with Py: FAILED_TIMEOUT seconds")
print("Time taken for {} iterations with Py: {:.2f} seconds".format(iterations, end_time - start_time))
print("Average time per iteration with Py: {:.2f} seconds".format((end_time - start_time) / iterations))
# print("Average time per iteration with Py: FAILED_TIMEOUT seconds")


print("Testing Propogate only")
start_time = time.time()
field = LP.Begin(grid_size, wavelength, N)
field = LP.GaussBeam(field, 1e-3, x_shift=0, y_shift=0)
for i in range(iterations):
    LP.Forvard(field, 1e-6, usepyFFTW=True) #prop only 1um
end_time = time.time()

print("Time taken for {} iterations with Py: {:.2f} seconds".format(iterations, end_time - start_time))
print("Average time per iteration with Py: {:.4f} seconds".format((end_time - start_time) / iterations))




