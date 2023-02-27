# -*- coding: utf-8 -*-
# (c) Llorenç Cerdà-Alabern, Februari 2023.
# WISP reliability analysis

import sys
import os
import numpy as np
np.random.seed(42)
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
import scipy
# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()  # interactive non-blocking mode

# wd
pwd = os.getcwd()
print('pwd: ' + pwd)
wd = os.environ['HOME'] + '/doctorands/gabriele-gemmi/paper-cn2023'
if os.path.exists(wd):
    print('chdir: ' + wd)
    os.chdir(wd)

def save_figure(file, font=14):
    plt.rcParams.update({'font.size': font})
    fname = 'figures/'+file
    print(fname)
    plt.savefig(fname, format='pdf',
                bbox_inches='tight', pad_inches=0)

#
# failure probability
#
def pik(alpha, k):
    return alpha**k/sum([alpha**j/np.math.factorial(k-j) for j in range(int(k+1))])

min=1e-4
max=1e-1
def func(x, k):
    return pik(x,k)-1e-3

ro =[scipy.optimize.brentq(func, min, max, args=(1)),
     scipy.optimize.brentq(func, min, max, args=(2)),
     scipy.optimize.brentq(func, min, max, args=(3))]

alim = 1e4
xlog = np.logspace(-5, 0, 100)

reliab =  pd.DataFrame({'x': xlog, 
                        r'$k=1$': [pik(a, 1) for a in xlog],
                        r'$k=2$': [pik(a, 2) for a in xlog],
                        r'$k=3$': [pik(a, 3) for a in xlog]})

ax = reliab.plot(x='x', logx=True, logy=True, ylabel='failure probability', xlabel=r'$\alpha$', ylim=(1e-9,1))
ax.hlines(y=1e-3, xmin=xlog[0], xmax=xlog[-1], ls='--', lw=1)
for i in range(3):
    ax.vlines(x=ro[i], ymin=1e-3, ymax=4e-4, ls='--', lw=1)
    ax.annotate(r'$\alpha_{}$'.format(i+1), xy=(ro[i],2e-4), ha="center")

save_figure("gateway-failure-probability.pdf")

#
# reliability
#
def R1(mttf, t):
    f=1/mttf
    return np.exp(-f*t)

def R2approx(mttf, t):
    f=1/mttf
    return 2*np.exp(-f*t)-np.exp(-2*f*t)

def funcR2approx(x, p, t):
    return p-R2approx(x, t)

# approx
min=1
max=100
mttf = scipy.optimize.brentq(funcR2approx, min, max, args=(0.99, 1))
print(mttf)

R2approx(mttf, 1)
R1(mttf, 1)

# exact
def R2exact(mttf, mttr, t):
    f=1/mttf
    r=1/mttr
    l1 = -1/2*(3*f+r+np.sqrt((3*f+r)**2-8*f**2))
    l2 = -1/2*(3*f+r-np.sqrt((3*f+r)**2-8*f**2))
    return l1/(l1-l2)*np.exp(l2*t)-l2/(l1-l2)*np.exp(l1*t)

# R2exact(mttf, mttf/10, 1)
R2exact(mttf, 1/365, 1)

def funcR2exact(x, p, t):
    return p-R2exact(x, 1/365, t)

min=0.0011
max=100
mttf2 = scipy.optimize.brentq(funcR2exact, min, max, args=(0.99, 1))
print(mttf2)
print(mttf2*365)

funcR2exact(100, 1/365, 1)

funcR2exact(0.001, 0.99, 1)

R2exact(0.99, 1/365, 1)

# n devices
def R2nexact(mttf, mttr, n, t):
    f=1/mttf
    r=1/mttr
    l1 = -1/2*((2*n-1)*f+r+np.sqrt(((2*n-1)*f+r)**2-4*n*(n-1)*f**2))
    l2 = -1/2*((2*n-1)*f+r-np.sqrt(((2*n-1)*f+r)**2-4*n*(n-1)*f**2))
    return l1/(l1-l2)*np.exp(l2*t)-l2/(l1-l2)*np.exp(l1*t)

def funcR2nexact(x, mttr, p, n, t):
    return p-R2nexact(x, mttr, n, t)

min=1e-5
max=1000
mttf2 = scipy.optimize.brentq(funcR2nexact, min, max, args=(1/365, 0.99, 50, 1))
print(mttf2)

ndev = [n for n in range(2, 100+1)]

Rperiod=1
reliab =  pd.DataFrame(
    {'n': ndev, 
     r'$k=1$': [n*Rperiod/np.log(1/0.99) for n in ndev],
     r'$k=2$, mttr=1 week': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(7/365, 0.99, n, Rperiod)) for n in ndev],
     r'$k=2$, mttr=1 day': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(1/365, 0.99, n, Rperiod)) for n in ndev],
     r'$k=2$, mttr=2 h': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(2*1/24*1/365, 0.99, n, Rperiod)) for n in ndev]})

ax = reliab.plot(x='n', logy=True, 
                 ylabel='mean time to failure, mttf [years] (log)', 
                 xlabel=r'number of antennas, $n_a$',
                 title=r'99% reliability in 1 year')

save_figure("antennas-reability-1-year.pdf")

# 99% reliability in 30 days
Rperiod=30/365
reliab =  pd.DataFrame(
    {'n': ndev, 
     r'$k=1$': [n*Rperiod/np.log(1/0.99) for n in ndev],
     r'$k=2$, mttr=1 week': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(7/365, 0.99, n, Rperiod)) for n in ndev],
     r'$k=2$, mttr=1 day': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(1/365, 0.99, n, Rperiod)) for n in ndev],
     r'$k=2$, mttr=2 h': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(2*1/24*1/365, 0.99, n, Rperiod)) for n in ndev]})

ax = reliab.plot(x='n', logy=True, 
                 ylabel='mean time to failure, mttf [years] (log)', 
                 xlabel=r'number of antennas, $n_a$',
                 title=r'99% reliability in 30 days')

save_figure("antennas-reability-30-days.pdf")

# 99% reliability in 1 week
Rperiod=7/365
reliab =  pd.DataFrame(
    {'n': ndev, 
     r'$k=1$': [n*Rperiod/np.log(1/0.99) for n in ndev],
     r'$k=2$, mttr=1 week': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(7/365, 0.99, n, Rperiod)) for n in ndev],
     r'$k=2$, mttr=1 day': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(1/365, 0.99, n, Rperiod)) for n in ndev],
     r'$k=2$, mttr=2 h': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(2*1/24*1/365, 0.99, n, Rperiod)) for n in ndev]})

ax = reliab.plot(x='n', logy=True, 
                 ylabel='mean time to failure, mttf [years] (log)', 
                 xlabel=r'number of antennas, $n_a$',
                 title=r'99% reliability in 1 week')

save_figure("antennas-reability-1-week.pdf")