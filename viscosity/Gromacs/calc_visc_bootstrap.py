import numpy as np
from scipy.optimize import curve_fit
import multiprocessing as mp
import sys
import os
import time as t


np.random.seed(42)


### Fitting functions ###
def sigma_func(x, A, b):
    return A*x**b

def visc_func(x, A, a, t1, t2):
    return A*(a*t1*(1-np.exp(-x/t1)) + (1-a)*t2*(1-np.exp(-x/t2)))


# ==== Parameters =====
N_bootstrap = 256             # число выборок
exclude_list = []             # номера папок, которые надо исключить
Nit = 64                      # число папок с траекториями в директории
npts_mean_int = 20000        # число точек в массиве среднего интеграла
L_mean_int = 200000          # используемая длина среднего интеграла < L_int
L_int = 200000               # длина интегралов в файле
Nacf = 1                      # число интегралов для одной конфигурации
L_data = 1000000              # длина массива напряжений для расчета одного автокоррелятора 
L_file_data = 1000000         # суммарная длина массива напряжений в одном файле. Должна быть > N_acf * L_data

b_sigma = 1       # отступ для сигмы
b_visc = 100       # отступ для ингеграла
k_cut = 0.07      # threshold for t_cut
dig = 4           # округление результата

Nevery = 1       # частота записи значений
dt = 1

s_mi = int(L_mean_int/npts_mean_int)
Nint = Nit - len(exclude_list)

pref_int_file = 's{}/' + f'ints/int_{L_file_data}_{L_data}_{L_int}_{Nacf}_' + '{}.txt'

path_mi = f'{L_file_data//1000}_{L_data//1000}_{Nacf}_{L_int//1000}_{L_mean_int//1000}_{npts_mean_int//1000}_{Nint}/'
file_btstrp = f'visc_btstrp_{N_bootstrap}_{k_cut:.2f}_{b_visc}.txt'

Np_read = 16
Np_btstrp = int(sys.argv[1])
# ================================
if not os.path.exists(path_mi):
    os.mkdir(path_mi)
    print(f'Created {path_mi}')
else:
    print(f'WARNING: Directory {path_mi} already exists')

Idx_it = [i for i in range(0, Nit) if i not in exclude_list]
Idx_int = [(i, n) for i in range(0, Nint) for n in range(Nacf)]

print('Start reading ints')
t1 = t.time()
def get_int(idx):
    return np.loadtxt(pref_int_file.format(Idx_it[idx[0]], idx[1]+1))[:L_mean_int-1:s_mi]

with mp.Pool(Np_read) as p:
    Int = p.map(get_int, Idx_int)

Int = np.array(Int)
t2 = t.time()

print(f'Finished reading ints. Took {t2-t1} s')

# ========================================

t1 = t.time()
time = Nevery*dt*np.arange(L_mean_int)[::s_mi]

def calc_visc(Idx):
    
    Int_mean = np.mean(Int[Idx, :], axis=0)
    si_Int = np.std(Int[Idx, :], axis=0, ddof=1)/np.sqrt(Int.shape[0])

    ### Fitting sigma ###
    popt1, pcov1 = curve_fit(sigma_func, xdata=time[b_sigma:], ydata=si_Int[b_sigma:])
    print(*popt1)

    ### Determening t_cut ###
    I_cut = -1
    if np.any(k_cut*Int_mean[b_visc:] <= si_Int[b_visc:]):
        I_cut = b_visc + np.argmax(k_cut*Int_mean[b_visc:] <= si_Int[b_visc:])
    print(I_cut, time[I_cut], si_Int[I_cut])

    ### Fitting viscosity ###
    popt2, pcov2 = curve_fit(visc_func, xdata=time[b_visc:I_cut], ydata=Int_mean[b_visc:I_cut],
                            sigma=sigma_func(time[b_visc:I_cut], *popt1), absolute_sigma=True,
                            maxfev=1000000, bounds=([0., 0., 0., 0.], [100., 1., 10000000., 10000000.]))
    A, a, tau1, tau2 = list(popt2)
    visc = A*(a*tau1 + (1-a)*tau2)

    return visc

Idx_bootstrap = [np.random.choice(Nint, Nint) for i in range(N_bootstrap)]

with mp.Pool(Np_btstrp) as p:
    Viscs = p.map(calc_visc, Idx_bootstrap)
Viscs = np.array(Viscs)
np.savetxt(path_mi + file_btstrp, Viscs)

t2 = t.time()
print(f'\nBootstrapping took {t2-t1} s')
# ======================================

visc = np.mean(Viscs)
si_visc = np.std(Viscs)

print(f'\nVisc: {visc:.5f} +/- {si_visc:.5f}\n')
