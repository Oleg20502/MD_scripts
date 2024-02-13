import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import multiprocessing as mp
import sys
import os
import time as t


### Fitting functions ###
def sigma_func(x, A, b):
    return A*x**b

def visc_func(x, A, a, t1, t2):
    return A*(a*t1*(1-np.exp(-x/t1)) + (1-a)*t2*(1-np.exp(-x/t2)))


# ===== Parameters =====
exclude_list = []             # номера папок, которые надо исключить
Nit = 30                      # число папок с траекториями в директории
npts_mean_int = 100000        # число точек в массиве среднего интеграла
L_mean_int = 1000000          # используемая длина среднего интеграла < L_int
L_int = 1000000               # длина интегралов в файле
Nacf = 1                      # число интегралов для одной конфигурации
L_data = 3000000              # длина массива напряжений для расчета одного автокоррелятора 
L_file_data = 3000000         # суммарная длина массива напряжений в одном файле. Должна быть > N_acf * L_data

b_sigma = 1       # отступ для сигмы
b_visc = 40       # отступ для ингеграла
k_cut = 0.10      # threshold for t_cut
dig = 4           # округление результата

Nevery = 5        # частота записи значений
dt = 2
fs2ns = 1.0e-6

calc_mean = True
plot_ints = True

s_mi = int(L_mean_int/npts_mean_int)
s_si = int(npts_mean_int/20)
s_pl = int(npts_mean_int/2000)

Nint = Nit - len(exclude_list)
pref_int_file = 's{}/' + f'ints/int_{L_file_data}_{L_data}_{L_int}_{Nacf}_' + '{}.txt'

path_mi = f'{L_file_data//1000}_{L_data//1000}_{Nacf}_{L_int//1000}_{L_mean_int//1000}_{npts_mean_int//1000}_{Nint}/'
pref_mi_file = f'1'

path_plots = path_mi + f'plots_{k_cut:.2f}_{b_visc}/'
pref_plots_file = f'1'

Np = int(sys.argv[1])
# =====================================

Idx_it = [i for i in range(0, Nit) if i not in exclude_list]
Idx_int = [(i, n) for i in range(0, Nint) for n in range(Nacf)]
def get_int(idx):
    return np.loadtxt(pref_int_file.format(Idx_it[idx[0]], idx[1]+1))[:L_mean_int-1:s_mi]

if calc_mean:
    if not os.path.exists(path_mi):
        os.mkdir(path_mi)
        print(f'Created {path_mi}')
    else:
        print(f'WARNING: Directory {path_mi} already exists')
    
    print('Start reading ints')
    with mp.Pool(Np) as p:
        Int = p.map(get_int, Idx_int)
    Int = np.array(Int).T
    print(f'Finished reading ints')

    Int_mean = np.mean(Int, axis=1)
    si_Int = np.std(Int, axis=1, ddof=1)/np.sqrt(Nint*Nacf)

    np.savetxt(path_mi + f'int_mean_{pref_mi_file}.txt', Int_mean)
    np.savetxt(path_mi + f'si_int_{pref_mi_file}.txt', si_Int)

elif plot_ints:
    with mp.Pool(Np) as p:
        Int = p.map(get_int, Idx_int)
    Int = np.array(Int).T

# =========================================

print(f'Start fitting')

time = Nevery*dt*np.arange(L_mean_int)[::s_mi]

if not calc_mean:
    Int_mean = np.loadtxt(path_mi + f'int_mean_{pref_mi_file}.txt')
    si_Int = np.loadtxt(path_mi + f'si_int_{pref_mi_file}.txt')

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
si_A, si_a, si_tau1, si_tau2 = list(np.sqrt(np.diag(pcov2)))
dif = np.zeros((4, 1))
dif[0, 0] = a*tau1+(1-a)*tau2
dif[1, 0] = A*(tau1-tau2)
dif[2, 0] = A*a
dif[3, 0] = A*(1-a)
print(A, a, tau1, tau2)
print(si_A, si_a, si_tau1, si_tau2)

visc = A*(a*tau1 + (1-a)*tau2)
si_visc = np.sqrt((si_A*(a*tau1+(1-a)*tau2))**2 + (A*si_a*(tau1-tau2))**2 +
                         (si_tau1*A*a)**2 + (si_tau2*A*(1-a))**2)
si_visc2 = (dif.T @ pcov2 @ dif)[0, 0]
print(f'\nVisc: {visc:.5f} +/- {si_visc:.5f} ({si_visc2:.5f})\n')
visc = np.round(visc, dig)
si_visc = np.round(si_visc, dig)

# ======================================================
if not os.path.exists(path_plots):
    os.mkdir(path_plots)
    print(f'Created {path_plots}')
else:
    print(f'WARNING: Directory {path_plots} already exists')

### Visc plot ###
plt.figure(figsize=(8,5))
if plot_ints:
    for idx in Idx_int:
        plt.plot(fs2ns*time[::s_pl], Int[::s_pl, idx[0] + idx[1]*Nint], 'c')
plt.errorbar(fs2ns*time[::s_si], Int_mean[::s_si], yerr=si_Int[::s_si], ecolor='k', elinewidth=1.2, fmt='.', c='k')
plt.plot(fs2ns*time[::s_pl], Int_mean[::s_pl], 'k', label='Усредн.')
plt.plot(fs2ns*time[::s_pl], visc_func(time[::s_pl], A, a, tau1, tau2), 'r')
plt.legend()
plt.title(f'Вязкость: {visc}+/-{si_visc} мПа с')
plt.xlabel('Time, ns')
plt.ylabel('Viscosity, mPa s')
plt.grid()
plt.savefig(path_plots+f'int_{pref_plots_file}.png', format='png', dpi=300)

### Sigma plot ###
plt.figure(figsize=(8,5))
plt.plot(fs2ns*time[::s_pl], si_Int[::s_pl], 'k')
plt.plot(fs2ns*time[::s_pl], sigma_func(time[::s_pl], popt1[0], popt1[1]), 'r')
plt.title(f'Abs. err. ~ t^{popt1[1]:.3f}')
plt.xlabel('time, ns')
plt.ylabel(r'$ \sigma $, mPa s')
plt.grid()
plt.savefig(path_plots+f'si_int_{pref_plots_file}.png', format='png', dpi=300)

### Tmp plot ###
plt.figure(figsize=(8,5))
plt.plot(fs2ns*time[::s_pl], Int_mean[::s_pl], 'b')
plt.plot(fs2ns*time[::s_pl], si_Int[::s_pl], 'r')
plt.plot([fs2ns*time[I_cut], fs2ns*time[I_cut]], [0, 1.01*max(si_Int[I_cut], Int_mean[I_cut])], 'k--')
plt.xlabel('time, ns')
plt.ylabel('mPa s')
plt.grid()
plt.savefig(path_plots+f'tmp_{pref_plots_file}.png', format='png', dpi=300)

### Tmp2 plot ###
b, e = 0, int(1.25*b_visc)
plt.figure(figsize=(8,5))
plt.plot(time[b:e], Int_mean[b:e], 'b')
plt.plot([time[b_visc], time[b_visc]], [0, 1.01*Int_mean[b_visc]], 'k--')
#plt.xlabel('time, ns')
plt.ylabel('mPa s')
plt.grid()
plt.savefig(path_plots+f'tmp2_{pref_plots_file}.png', format='png', dpi=300)

print(f'Finished plotting')
