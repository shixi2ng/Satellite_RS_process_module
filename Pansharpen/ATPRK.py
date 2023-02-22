import numpy as np
import math
from scipy.optimize import curve_fit


def func2(x,a1,a2):
    return a1*(1-np.exp(-x/a2))


def r_area_area2(h, s, func_para):
    Assume_L1=np.zeros([h+1,1])
    M1, N1 = np.argwhere(Assume_L1==0)[:, 0] + 1, np.argwhere(Assume_L1==0)[:, 1] + 1
    Assume_L2=np.zeros([s,s])
    M2, N2 = np.argwhere(Assume_L2==0)[:, 0] + 1, np.argwhere(Assume_L2==0)[:, 1] + 1
    raa = np.zeros([h, 1])
    for i in range(0, h + 1):
        raa[i, 0]=0
        for m in range (0, s ** 2):
            for n in range (0, s ** 2):
                p1 = np.array([(M1(i)-1)*s+M2(m),(N1(i)-1)*s+N2(m)])
                p2 = np.array([(M1(1)-1)*s+M2(n),(N1(1)-1)*s+N2(n)])
                raa[i, 0]=raa[i, 0] + func2(np.sqrt(np.sum((p1 - p2) ** 2)), func_para[0], func_para[1])

    raa = (raa/s) ** 4
    return raa


def ATP_deconvolution0(H, s, para, Sill_min,Range_min, L_sill, L_range, rate):
    Fa0=func2(np.linspace(1, s*H, s*H), para[0], para[1])
    Fa0_vector=Fa0[s-1: -1 : s]
    Fa0_vector=Fa0_vector.reshape(Fa0_vector.shape[0], 1).T

    Dif_min=10 ** 6
    for i in range(0, L_sill):
        for j in range(0, L_range):
            optimal_para = [(Range_min+rate*j)*para[1], (Sill_min+rate*i)*para[0]]
            raa_temp = r_area_area2(H,s,optimal_para)
            raa = raa_temp [1:H, 0] - raa_temp[0, 0]
            Dif = np.sqrt(np.sum((raa - Fa0_vector) ** 2))
            if Dif<=Dif_min:
                best_optimal_para = optimal_para
                Dif_min = Dif

    return best_optimal_para


def PSF_template(w, b: int = 1, s: int = 1, *args, **kwargs):
    H0 = np.zeros([(2 * w + 1) * s, (2 * w + 1) * s])
    for i in range(0, (2 * w + 1) * s):
        for j in range(0, (2 * w + 1) * s):
            Dis2 = math.sqrt(((2 * w + 1) * s / 2 - 1 - i) ** 2 + ((2 * w + 1) * s / 2 - 1 - j) ** 2)
            H0[i, j] = np.exp(- Dis2 / ((2 * b) ** 2))

    Hsum = np.sum(H0)
    H = H0 / Hsum
    return H


def semivariogram(array,h):
    [a,b] = array.shape

    N1, r1 = 0, 0
    for i in range(h, a):
        for j in range(0, b):
            r1=r1+(array[i,j]-array[i-h,j]) **  2
            N1 += 1

    N2, r2 = 0, 0
    for i in range (0, a-h):
        for j in range(0, b):
            r2=r2+(array[i,j]-array[i+h,j]) **  2
            N2 += 1

    N3, r3 = 0, 0
    for i in range(0, a):
        for j in range(h, a):
            r3=r3+(array[i,j]-array[i,j-h]) ** 2
            N3 += 1

    N4, r4 = 0, 0
    for i in range(0, a):
        for j in range (0, b-h):
            r4=r4+(array[i,j]-array[i,j+h]) ** 2
            N4=N4+1

    r=r1+r2+r3+r4
    N=N1+N2+N3+N4
    return np.divide(r, (2*N))


def T_coarse_coarse2(W, resize_factor, optimal_popt, PSF):
    Assume_L1 = np.zeros([2 * W + 1, 2 * W + 1])
    M1, N1 = np.argwhere(Assume_L1==0)[:, 0] + 1, np.argwhere(Assume_L1==0)[:, 1] + 1
    TVV = np.zeros([(2 * W + 1) ** 2, (2 * W + 1 ) ** 2])

    for i in range(0, (2 * W + 1) ** 2):
        for j in range(0, (2 * W + 1) ** 2):
            TvV = np.zeros([(2 * W + 1) * resize_factor, (2 * W + 1) * resize_factor])
            for ii in range(0, (2 * W + 1) * resize_factor):
                for jj in range(0, (2 * W + 1) * resize_factor):
                    Tvv = np.zeros((2 * W + 1 ) * resize_factor,(2 * W + 1 ) * resize_factor)
                    for iii in range(0, (2 * W + 1) * resize_factor):
                        for jjj in range(0, (2 * W + 1) * resize_factor):
                            p1=[(M1(i)-1-W)*resize_factor+iii,(N1(i)-1-W)*resize_factor+jjj]
                            p2=[(M1(j)-1-W)*resize_factor+ii,(N1(j)-1-W)*resize_factor+jj]
                            Tvv[iii,jjj] = func2(np.sqrt(np.sum((p1 - p2) ** 2)), optimal_popt[0], optimal_popt[1]);
                    TvV[ii,jj]=np.sum(Tvv*PSF)
            TVV[i,j]=np.sum(TvV*PSF)
    return TVV


def r_fine_coarse2(p_vm,W,resize_factor,optimal_popt,PSF):
    Assume_L1 = np.zeros([2 * W + 1, 2 * W + 1])
    M1, N1 = np.argwhere(Assume_L1==0)[:, 0] + 1, np.argwhere(Assume_L1==0)[:, 1] + 1
    rvV = np.zeros([(2 * W + 1) ** 2, 1])

    for i in range(0, (2 * W + 1) ** 2):
        Tvv = np.zeros((2 * W + 1 ) * resize_factor,(2 * W + 1 ) * resize_factor)
        for iii in range(0, (2 * W + 1) * resize_factor):
            for jjj in range(0, (2 * W + 1) * resize_factor):
                p1=[(M1(i)-1-W)*resize_factor+iii, (N1(i)-1-W)*resize_factor+jjj]
                Tvv[iii,jjj] = func2(np.sqrt(np.sum((p_vm - p1) ** 2)), optimal_popt[0], optimal_popt[1])
        rvV[i, 0] = np.sum(Tvv * PSF)
    return rvV


def ATPK_noinform_yita_new(resize_factor,W,optimal_popt,PSF):

    resize_factor = 1

    TVV = T_coarse_coarse2(W, resize_factor, optimal_popt, PSF)
    yita = np.zeros([resize_factor,  resize_factor, (2 * W + 1) ** 2 + 1])
    for i in range(0, resize_factor):
        for j in range(0, resize_factor):
            cordinate_vm=[W * resize_factor + i, W * resize_factor + j]
            rvV = r_fine_coarse2(cordinate_vm, W, resize_factor, optimal_popt, PSF)
            Matrix = np.concatenate([Matrix, np.ones([(2 * W + 1) ** 2, 1])], axis=1)
            temp = np.append(np.ones([1,(2 * W + 1) ** 2]), 0)
            Matrix = np.concatenate([Matrix, temp.reshape(-1, temp.shape[0])], axis=0)
            Vector= np.concatenate([rvV, np.array([[1]])], axis=1)
            yita[i, j, :] = np.matmul(np.invert(Matrix), Vector)
    return yita


def ATPK_noinform_new(resize_factor, W, residual_band, yitaX):
    c, d = residual_band.shape[0], residual_band.shape[1]
    Simulated_part = np.zeros([c - 2 * W, d - 2 * W])
    M1, N1 = np.argwhere(Simulated_part == 0)[:, 0] + 1, np.argwhere(Simulated_part == 0)[:, 1] + 1
    numberM1 = N1.shape[0]
    M1 = M1 + W
    N1 = N1 + W
    P_vm = np.zeros([c, d])
    for k in range(0, numberM1):
        for i in range(0, resize_factor):
            for j in range(0, resize_factor):
                Local_W = residual_band[M1[k] - W: M1[k] + W, N1[k] - W: N1[k] + W]
                co = yitaX[i, j, 0:-1].reshape(-1)
                P_vm[(M1[k] - 1) * resize_factor + i, (N1[k] - 1) * resize_factor + j] = co * Local_W.reshape([(2 * W + 1) ** 2, 1])
    return P_vm


def ATPRK_PANsharpen(MS, PAN, PSF=PSF_template(2), Sill_min: int = 1, Range_min: float = 0.5, L_sill: int = 20, L_range: int = 20, rate:float = 0.1, H: int = 20, W: int = 1):
    correlation_list = [np.corrcoef(MS.reshape(-1), array_temp.reshape(-1))[0, 1] for array_temp in PAN]
    PAN = PAN[np.argmax(np.array(correlation_list))]

    [MS_y, MS_x] = MS.shape
    [PAN_y, PAN_x] = PAN.shape
    resize_factor = 2

    x0 = [0.1, 1]
    RB = MS - PAN
    rh = []
    for h in range(1, H + 1):
        rh.append(semivariogram(RB, h))
    rh = np.array(rh)

    popt_temp, resnorm = curve_fit(func2, np.linspace(1, H + 1, H + 1), rh, p0=[0.1, 1])
    Fa1 = func2(np.linspace(1, H, H), popt_temp[0], popt_temp[1])

    optimal_popt = ATP_deconvolution0(H, resize_factor, popt_temp, Sill_min, Range_min, L_sill, L_range, rate)
    Fp = func2(np.linspace(1, resize_factor * H, resize_factor * H), optimal_popt[0], optimal_popt[1])

    raa_temp = r_area_area2(H, resize_factor, optimal_popt)
    raa = raa_temp[0: H+1, 0] - raa_temp[0, 0]

    popt_temp2, resnorm2 = curve_fit(func2, np.arange(resize_factor, resize_factor, resize_factor * H), raa.T, p0=[0.1, 1])
    Fa2 = func2(np.arange(1, 1, resize_factor * H), popt_temp2[0], popt_temp2[1])

    yita1 = ATPK_noinform_yita_new(resize_factor, W, optimal_popt, PSF)
    P_vm = ATPK_noinform_new(resize_factor, W, RB, yita1)
    Z_ATPK = P_vm[W * resize_factor : - W * resize_factor, W * resize_factor: - W * resize_factor]
    Z = PAN + Z_ATPK
    return Z

