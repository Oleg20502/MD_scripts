import numpy as np
from numpy import fft as fft
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os


### Fitting functions ###
def sigma_func(x, A, b):
    return A*x**b

def visc_func(x, A, a, t1, t2):
    return A*(a*t1*(1-np.exp(-x/t1)) + (1-a)*t2*(1-np.exp(-x/t2)))


# ==== Parameters =====
npts_res = 4000000        # желаемая длина автокоррелятора < npts_int
npts_int = 6000000        # длина автокоррелятора в файле
Nacf = 1                  # число усредняемых корреляторов
Ndata = 12000000           # длина данных
N_file_data = 12000000

exclude_list = [16] + [i for i in range(32, 100)] + [114]
Nit = 116
Nint = Nit - len(exclude_list)

b_sigma = 1     # отступ для сигмы
b_visc = 2000       # отступ для ингеграла
k_cut = 0.65      # threshold for t_cut
dig = 3          # округление результата

Nevery = 1       # частота записи значений
dt = 2
fs2ns = 1.0e-6

path_int = 'ints/'
Files_int = [f'int_{N_file_data}_{Ndata}_{npts_int}_{Nacf}_{i+1}.txt' for i in range(Nacf)]

path_res = f'{Ndata}_{Nacf}_{npts_int}_{npts_res}_{k_cut:.2f}_{Nint}/'
pref_file = f'1'

s_si = int(npts_res/30)
s_pl = int(npts_res/4000)
# ===================
if not os.path.exists(path_res):
    os.mkdir(path_res)
else:
    print(f'WARNING: Directory {path_res} already exists')

time = Nevery*dt*np.arange(npts_res)
Int = np.zeros((npts_res-1, Nacf*Nint))
i1 = 0
plt.figure(figsize=(8,5))
for it in range(0, Nit):
    if it not in exclude_list:
        for n in range(Nacf):
            Int[:, Nacf*i1 + n] = np.loadtxt(f'it{it}/'+path_int + Files_int[n])[:npts_res-1]
            plt.plot(fs2ns*time[:npts_res-1:s_pl], Int[::s_pl, Nacf*i1 + n], 'c') 
        i1 += 1

Int_mean = np.mean(Int, axis=1)
si_Int = np.std(Int, axis=1, ddof=1)

### Fitting sigma ###
popt1, pcov1 = curve_fit(sigma_func, xdata=time[b_sigma:npts_res-1], ydata=si_Int[b_sigma:])
print(*popt1)

### Determening t_cut ###
I_cut = npts_res-2
if np.any(k_cut*Int_mean[b_visc:] <= si_Int[b_visc:]):
    I_cut = b_visc + np.argmax(k_cut*Int_mean[b_visc:] <= si_Int[b_visc:])
print(I_cut, time[I_cut], si_Int[I_cut])

### Fitting viscosity ###
popt2, pcov2 = curve_fit(visc_func, xdata=time[b_visc:I_cut], ydata=Int_mean[b_visc:I_cut],
                         sigma=sigma_func(time[b_visc:I_cut], *popt1), absolute_sigma=True,
                          maxfev=1000000, bounds=([0., 0., 0., 0.], [100., 1., 10000000., 10000000.]))
A, a, tau1, tau2 = list(popt2)
si_A, si_a, si_tau1, si_tau2 = list(np.sqrt(np.diag(pcov2)))
print(A, a, tau1, tau2)
print(si_A, si_a, si_tau1, si_tau2)

visc = A*(a*tau1 + (1-a)*tau2)
si_visc = np.sqrt((si_A*(a*tau1+(1-a)*tau2))**2 + (A*si_a*(tau1-tau2))**2 +
                         (si_tau1*A*a)**2 + (si_tau2*A*(1-a))**2)
print('\nVisc:', visc, '+/-', si_visc)
visc = np.round(visc, dig)
si_visc = np.round(si_visc, dig)

plt.errorbar(fs2ns*time[:npts_res-1:s_si], Int_mean[::s_si], yerr=si_Int[::s_si], ecolor='k', elinewidth=1.2, fmt='.', c='k')
plt.plot(fs2ns*time[:npts_res-1:s_pl], Int_mean[::s_pl], 'k', label='Усредн.')
plt.plot(fs2ns*time[:npts_res-1:s_pl], visc_func(time[:npts_res-1:s_pl], A, a, tau1, tau2), 'r')
plt.legend()
plt.title(f'Вязкость: {visc}+/-{si_visc} мПа с')
plt.xlabel('Time, ns')
plt.ylabel('Viscosity, mPa s')
plt.grid()
plt.savefig(path_res+f'int_{pref_file}.png', format='png', dpi=300)

plt.figure(figsize=(8,5))
plt.plot(fs2ns*time[:npts_res-1:s_pl], si_Int[::s_pl], 'k')
plt.plot(fs2ns*time[:npts_res-1:s_pl], sigma_func(time[:npts_res-1:s_pl], popt1[0], popt1[1]), 'r')
plt.title(f'Abs. err. ~ t^{popt1[1]:.3f}')
plt.xlabel('time, ns')
plt.ylabel(r'$ \sigma $, mPa s')
plt.grid()
plt.savefig(path_res+f'si_int_{pref_file}.png', format='png', dpi=300)

plt.figure(figsize=(8,5))
plt.plot(fs2ns*time[:npts_res-1:s_pl], Int_mean[::s_pl], 'b')
plt.plot(fs2ns*time[:npts_res-1:s_pl], si_Int[::s_pl], 'r')
plt.plot([fs2ns*time[I_cut], fs2ns*time[I_cut]], [0, 1.01*max(si_Int[I_cut], Int_mean[I_cut])], 'k--')
plt.xlabel('time, ns')
plt.ylabel('mPa s')
plt.grid()
plt.savefig(path_res+f'tmp_{pref_file}.png', format='png', dpi=300)

b, e = 0, int(1.25*b_visc)
plt.figure(figsize=(8,5))
plt.plot(time[b:e], Int_mean[b:e], 'b')
plt.plot([time[b_visc], time[b_visc]], [0, 1.01*Int_mean[b_visc]], 'k--')
plt.xlabel('time, fs')
plt.ylabel('mPa s')
plt.grid()
plt.savefig(path_res+f'tmp2_{pref_file}.png', format='png', dpi=300)
