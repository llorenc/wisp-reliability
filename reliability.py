# -*- coding: utf-8 -*-
# (c) Llorenç Cerdà-Alabern, February 2023.
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
wd = os.environ['HOME'] + '/doctorands/gabriele-gemmi/reliability'
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
# alpha vs k
#
min=0
max=1
def opt_func(x, k, prob):
    return pik(x,k)-prob

a_df = pd.DataFrame({'prob':np.logspace(-6, -2, 20)})
for k in range(1,10+1):
    res = []
    for prob in a_df['prob']:
        alpha = scipy.optimize.brentq(opt_func, min, max, args=(k, prob))
        res.append(alpha)
        # df.iloc[len(df.index),:] = pd.DataFrame.from_dict({'k':k, 'prob':prob, 'aplpha': alpha})
    a_df = pd.concat([a_df, pd.DataFrame({k: res})], axis=1)   

a_df.shape

adfm = pd.melt(a_df, id_vars='prob', var_name='k')

plt.rcParams.update({'font.size': 14})
g = sns.lineplot(adfm, x='prob', y='value', hue='k')
g.set(xscale='log')
g.set_xlabel('failure probability')
g.set_ylabel('$\\alpha=mttr_r/mttf_r$')
plt.savefig('figures/alpha-vs-failure-prob.pdf', format='pdf',
            bbox_inches='tight', pad_inches=0)

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

#
# k=3
#
from fractions import Fraction

n=10
f=2
r=3
Q = np.array([[-n*f, n*f, 0],
              [r, -(n-1)*f-r, (n-1)*f],
              [0, r, -(n-2)*f-r]])

ee = np.linalg.eig(Q)[0]
ee

for i in range(3):
    print(Fraction(ee[i]).limit_denominator(10000))

Fraction(n*(n-1)*(n-2)*f**3+n*f*r**2).limit_denominator(100000)

(l1, l2, l3) = ee
A = np.array([[1, 1, 1],
              [1, l2/l1, l3/l1],
              [1,(l2/l1)**2,(l3/l1)**2]])

# check coefficients of R3:
np.linalg.solve(A, np.array([-1,0,0]))

l2*l3/((l1-l2)*(l3-l1))
l1*l3/((l1-l2)*(l2-l3))
l1*l2/((l1-l3)*(l3-l2))

# n devices
def R3nexact(mttf, mttr, n, t):
    f=1/mttf
    r=1/mttr
    Q = np.array([[-n*f, n*f, 0],
              [r, -(n-1)*f-r, (n-1)*f],
              [0, r, -(n-2)*f-r]])
    (l1, l2, l3) = np.linalg.eig(Q)[0]
    return ((l2*l3/((l1-l2)*(l1-l3)))*np.exp(l1*t)+
            (l1*l3/((l1-l2)*(l3-l2)))*np.exp(l2*t)+
            (l1*l2/((l1-l3)*(l2-l3)))*np.exp(l3*t))

def funcR3nexact(x, mttr, p, n, t):
    return p-R3nexact(x, mttr, n, t)

min=1e-6
max=10000
ndev = [n for n in range(3, 100+1)]

# 99% reliability in 1 year
Rperiod=1
reliab =  pd.DataFrame(
    {'n': ndev, 
     r'$k=2$, mttr=1 week': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(7/365, 0.99, n, Rperiod)) for n in ndev],
     r'$k=3$, mttr=1 week': [
         scipy.optimize.brentq(funcR3nexact, min, max, args=(7/365, 0.99, n, Rperiod)) for n in ndev],
     # r'$k=2$, mttr=1 day': [
     #     scipy.optimize.brentq(funcR2nexact, min, max, args=(1/365, 0.99, n, Rperiod)) for n in ndev],
     # r'$k=3$, mttr=1 day': [
     #     scipy.optimize.brentq(funcR3nexact, min, max, args=(1/365, 0.99, n, Rperiod)) for n in ndev],
     r'$k=2$, mttr=2 h': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(2*1/24*1/365, 0.99, n, Rperiod)) for n in ndev],
     r'$k=3$, mttr=2 h': [
         scipy.optimize.brentq(funcR3nexact, min, max, args=(2*1/24*1/365, 0.99, n, Rperiod)) for n in ndev]})

ax = reliab.plot(x='n', logy=True, 
                 ylabel='mean time to failure, mttf [years] (log)', 
                 xlabel=r'number of antennas, $n_a$',
                 title=r'99% reliability in 1 year')

save_figure("antennas-reability-1-year-k-3.pdf")


Rperiod=7/365
reliab =  pd.DataFrame(
    {'n': ndev, 
     r'$k=2$, mttr=1 week': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(7/365, 0.99, n, Rperiod)) for n in ndev],
     r'$k=3$, mttr=1 week': [
         scipy.optimize.brentq(funcR3nexact, min, max, args=(7/365, 0.99, n, Rperiod)) for n in ndev],
     # r'$k=2$, mttr=1 day': [
     #     scipy.optimize.brentq(funcR2nexact, min, max, args=(1/365, 0.99, n, Rperiod)) for n in ndev],
     # r'$k=3$, mttr=1 day': [
     #     scipy.optimize.brentq(funcR3nexact, min, max, args=(1/365, 0.99, n, Rperiod)) for n in ndev],
     r'$k=2$, mttr=2 h': [
         scipy.optimize.brentq(funcR2nexact, min, max, args=(2*1/24*1/365, 0.99, n, Rperiod)) for n in ndev],
     r'$k=3$, mttr=2 h': [
         scipy.optimize.brentq(funcR3nexact, min, max, args=(2*1/24*1/365, 0.99, n, Rperiod)) for n in ndev]})

ax = reliab.plot(x='n', logy=True, 
                 ylabel='mean time to failure, mttf [years] (log)', 
                 xlabel=r'number of antennas, $n_a$',
                 title=r'99% reliability in 1 week')


save_figure("antennas-reability-1-week-k-3.pdf")


# eigenvalues
Rperiod=1
def R3eigen(mttf, mttr, n, t):
    f=1/mttf
    r=1/mttr
    Q = np.array([[-n*f, n*f, 0],
              [r, -(n-1)*f-r, (n-1)*f],
              [0, r, -(n-2)*f-r]])
    return np.linalg.eig(Q)[0]



R3eigendf = pd.DataFrame([abs(R3eigen(1, 1/365, n, Rperiod)) for n in ndev], columns=['l1', 'l2', 'l3'])
R3eigendf['n'] = ndev

ax = R3eigendf.plot(x='n', logy=True)

#
# testing
#

dft = reliab.copy()
dft.columns = ['n', 'k1', 'k2mttr1w', 'k2mttr1d', 'k2mttr2h']
dft.to_csv('antennas-reability-1-year.csv', sep=' ', index=False)

