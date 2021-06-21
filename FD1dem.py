#ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import math
import pandas as pd

###パラメータ設定
#比抵抗
bRes = np.array([50,200]) 
#導電率
bReslength = len(bRes) #比抵抗配列長さ
lenarray = np.ones(bReslength) #1行列生成
sigma = lenarray /bRes #導電率
#層厚
dh = np.array([30])
#高度
z = 30
#送受信間距離
s = 7.86
#垂直磁気双極子モーメント
vmd = 1
#円周率
pi = np.pi
#真空中の透磁率
mu0 = 4*pi*10**-7
#誘電率
epsilon =  8.8541878128*10**-12
#周波数
numF = 100 #周波数の数
f = np.logspace(0, 6, 100) #周波数
#角周波数
omega = 2*pi*f
#フィルタ係数の読み込み
matdata = scipy.io.loadmat("anderson_801.mat")
y_base = matdata["yBase"]
wt0 = matdata["wt0"]
wt1 = matdata["wt1"]
filter_length = len(y_base)
rambda = y_base/s
#Y0 is intrinsic admittance of free space
k0 = 0
u0 = rambda
Y0 = u0/1j/omega/mu0


if bReslength == 1: #均質構造
  k1 = (1j*omega*sigma*mu0)**(1/2)
  u1 = (rambda**2+k1**2)**(1/2)
  ANS = u1/1j/omega/mu0

else: #多層構造
  kB = (1j*omega*sigma[bReslength-1]*mu0)**(1/2) 
  uB = (rambda**2+(kB.T**2))**(1/2)
  ANS = uB/1j/omega/mu0
  for i in range(bReslength-2,-1,-1):
    kl = (1j*omega*sigma[i]*mu0-omega**2*epsilon*mu0)**(1/2)
    ul = (rambda**2+kl**2)**(1/2)
    YNl = ul/1j/omega/mu0
    NUME = ANS+YNl*np.tanh(ul*dh[i]) ####
    DENO = YNl+ANS*np.tanh(ul*dh[i]) ####
    ANS = YNl*(NUME/DENO)

#calculate R&Z for Kernel Function
Rlam = (ANS-Y0)/(ANS+Y0)
#The secondary field Hs normalized aganist primary field
#make kernel function
kernel = s**3*Rlam*rambda**2*np.exp(-2*u0*z)

#二次場と一次場の比
HsHp = np.zeros((numF,1),dtype = 'complex_') #complex型と定義
for k in range(0,numF,1):
  HsHp[k,:] = np.dot(kernel[:,k].T, wt0)/s

#For Calculate Apparent resistivity(epsilon) (1984.Mundry)
ep  = HsHp.imag/HsHp.real
ep = ep**(-1.23)
Amp = (abs(HsHp.real)**2+(abs(HsHp.imag))**2)**(1/2)
#Calculate Apparent resistivity (1984.Mundry and Konishi's document)
#h = Apparent height, AR = Apparent resistivity
h = (((0.065*s**3*(ep**(0.973-0.423*np.log10(ep))))/(Amp))**(1/3)) #見掛高度
OmegaMatrix = omega.reshape(1,len(omega))
AR = ((h**2)*mu0*OmegaMatrix.T) / ((ep**2)*2) #見掛比抵抗

# plt.plot(f,AR)
# plt.xlim(400, 140000)
# plt.xscale('log')
# plt.show()

print(AR)

