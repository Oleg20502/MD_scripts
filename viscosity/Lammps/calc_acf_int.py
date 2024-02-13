import numpy as np
from numpy import fft as fft
from scipy.integrate import cumtrapz
import os


def acf(data, npts = None):
    n = len(data)
    if npts is None:
        npts = n
    npts = min(npts, n)
    dataf = fft.rfft(np.append(data, np.zeros(n)))
    acf = fft.irfft(np.conj(dataf) * dataf)[0:npts]
    for i in range(npts):
        acf[i] /= n - i
    return acf


# ==== Parameters ====
Files_stress = [f'press_ab_1.txt', f'press_ab_2.txt', f'press_ab_3.txt']
file_data = f'visc.data'

npts = 6000000            # длина автокоррелятора          
Nstep = 0           # сдвиг при расчете корреляторов
Ndata = 12000000           # длина данных
N_file_data = 12000000     # суммарная длина данных в файлах. 'auto' - определится автоматически

Nevery = 1       # частота записи значений
dt = 2
Temp = '298'

path_int = 'ints/'
pref_file = f''

kB = 1.3806504e-23
atm2Pa = 101325.0
A2m = 1.0e-10
fs2s = 1.0e-15
# ===================
if not os.path.exists(path_int):
    os.mkdir(path_int)


T = float(Temp)
with open(file_data, 'r') as f:
    i = 0
    for line in f:
        if i == 13:
            Lx = float(line.split()[1]) - float(line.split()[0])
        elif i == 14:
            Ly = float(line.split()[1]) - float(line.split()[0])
        elif i == 15:
            Lz = float(line.split()[1]) - float(line.split()[0])
            break
        i += 1
V = Lx*Ly*Lz
convert = atm2Pa**2 * fs2s * A2m**3
scale = 1000 * convert * V * Nevery * dt / (kB * T)

stress = np.genfromtxt(Files_stress[0], delimiter = ' ', skip_header = True)
pxy, pxz, pyz = stress[1:,0], stress[1:,1], stress[1:,2]


Nf = len(Files_stress)
for p in range(1, Nf):
    stress = np.genfromtxt(Files_stress[p], delimiter = ' ', skip_header = True)
    pxy = np.append(pxy, stress[1:,0])
    pxz = np.append(pxz, stress[1:,1])
    pyz = np.append(pyz, stress[1:,2])


N_file_data_2 = len(pxy)
if N_file_data == 'auto':
    N_file_data = N_file_data_2
    print(N_file_data)
elif N_file_data > N_file_data_2:
    print(f'ERROR: File data length {N_file_data_2} is smaller then N_file_data = {N_file_data}')

if pref_file != "":
    pref_file = "_" + pref_file

Nacf = 1
if Nstep != 0:
    Nacf = int((N_file_data - Ndata)/Nstep + 1)      # число рассчитываемых корреляторов
name1 = f'{N_file_data}_{Ndata}_{npts}_{Nacf}'


for i in range(Nacf):
    Pacf = np.zeros((npts, 3))
    Pacf[:, 0] = acf(pxy[i*Nstep:i*Nstep + Ndata], npts)
    Pacf[:, 1] = acf(pxz[i*Nstep:i*Nstep + Ndata], npts)
    Pacf[:, 2] = acf(pyz[i*Nstep:i*Nstep + Ndata], npts)

    Pacf = scale * Pacf

    Int = cumtrapz(Pacf, axis=0)
    Int_mean = np.mean(Int, axis=1)
    #si_Int = np.std(Int, axis=1, ddof=1)

    np.savetxt(path_int + f'int_{name1}_{i+1}' + pref_file + '.txt', Int_mean)
    #np.savetxt(path_int + f'si_int_{name1}_{i+1}' + pref_file + '.txt', si_Int)
