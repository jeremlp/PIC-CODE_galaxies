# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 22:56:55 2024

@author: jerem
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import timeit
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from numba import njit
plt.close('all')




@njit
def laplace(f,dx):
    return (f[1:-1, :-2] + f[:-2, 1:-1] -4*f[1:-1, 1:-1] + f[1:-1, 2:] + f[2:, 1:-1])/(dx**2)

@njit
def SOR(f, b, dx,Nx, w):
    res = np.copy(f)
    for i in range(1,Nx-1):
        for j in range(1,Nx-1):
            res[i,j] = (1-w)*f[i,j] + w*0.25*(f[i+1,j] + f[i,j+1] + res[i-1,j] + res[i,j-1] - dx**2* b[i,j])
    return res

@njit
def iteration_SOR(P_guess, b, dx, Nx, tolerence):

    w = 2/(1+np.sin(np.pi/Nx))
    phi = np.copy(P_guess)
    ERROR = []
    for k in range(10000):
        phi = SOR(phi, b, dx, Nx, w)
        error_max = np.max(np.abs(    laplace(phi,dx) - b[1:-1,1:-1]))
        ERROR.append(error_max)
        if error_max < tolerence:
            return phi, ERROR, k
    return phi, ERROR, -1

def PROJECTION_2D(pos):
    density = np.zeros((Nx,Nx))

    for p in range(N):
        i,j = np.floor(pos[p,0]/dx).astype(int),np.floor(pos[p,1]/dx).astype(int)
        ip1,jp1 = i+1, j+1
        grid = np.array([[i,j],[i,j+1],[i+1,j],[i+1,j+1]])

        dists = np.sqrt(np.sum((grid - pos[p])**2,1))

        l_max = np.sqrt(dx**2 + dx**2)
        w = (l_max - dists)/(np.sum((l_max - dists)))
        density[i,j] += w[0]
        density[i,(j+1)%Nx] += w[1]
        density[(i+1)%Nx,j] += w[2]
        density[(i+1)%Nx,(j+1)%Nx] += w[3]

        # plt.scatter(grid[:,0],grid[:,1], s = w*1000,c=w)
    density /= np.mean(density)
    # print(np.mean(density))
    return density

def PROJECTION_2D_vec(pos):
    density = np.zeros((Nx,Nx))
    i,j = np.floor(pos[:,0]/dx).astype(int),np.floor(pos[:,1]/dx).astype(int)
    ip1,jp1 = i+1, j+1
    grid = np.array([[i,j],[i,j+1],[i+1,j],[i+1,j+1]])
    l_max = np.sqrt(dx**2 + dx**2)

    dists_0 = np.sqrt(np.sum((grid[0].T*dx - pos)**2,1))
    dists_1 = np.sqrt(np.sum((grid[1].T*dx - pos)**2,1))
    dists_2 = np.sqrt(np.sum((grid[2].T*dx - pos)**2,1))
    dists_3 = np.sqrt(np.sum((grid[3].T*dx - pos)**2,1))
    dists = np.array([dists_0,dists_1,dists_2,dists_3])

    normalization = np.sum(l_max - dists,0)
    w = np.array([(l_max - dists_0)/normalization,(l_max - dists_1)/normalization
                  ,(l_max - dists_2)/normalization,(l_max - dists_3)/normalization])

    density = apply_project(w,i,j)
    return density, w, i, j

@njit(parallel=True)
def apply_project(w,i,j):
    density = np.zeros((Nx,Nx))
    for p in range(N):
        density[i[p],j[p]] += w[0,p]
        density[i[p],(j[p]+1)%Nx] += w[1,p]
        density[(i[p]+1)%Nx,j[p]] += w[2,p]
        density[(i[p]+1)%Nx,(j[p]+1)%Nx] += w[3,p]
    density /= np.mean(density)
    return density

def SOLVE_POISSON_2D(density, old_phi):
    phi, ERROR, k = iteration_SOR(old_phi, density, dx, Nx, 1)
    # print("SOR:", (t1-t0)*1000,"ms")
    F_grid = np.gradient(phi, dx)
    return np.array(F_grid), phi


def INTERPOLATION_2D(F_grid):
    F = np.zeros_like(vel)
    l_max = np.sqrt(dx**2 + dx**2)
    for p in range(N):
        i,j = np.floor(pos[p,0]/dx).astype(int),np.floor(pos[p,1]/dx).astype(int)
        ip1,jp1 = i+1, j+1
        grid = np.array([[i,j],[i,j+1],[i+1,j],[i+1,j+1]])

        dists = np.sqrt(np.sum((grid*dx - pos[p])**2,1))

        w = (l_max - dists)/(np.sum(l_max - dists))
        F[p,:] = w[0] * F_grid[:,i,j] + w[1] * F_grid[:,i,(j+1)%Nx] + w[2] * F_grid[:,(i+1)%Nx,j] + w[3] * F_grid[:,(i+1)%Nx,(j+1)%Nx]

    return F

def INTERPOLATION_2D_vec(F_grid, w, i,j):
    F = np.zeros_like(vel)
    F = (w[0] * F_grid[:,i,j] + w[1] * F_grid[:,i,(j+1)%Nx] + w[2] * F_grid[:,(i+1)%Nx,j] + w[3] * F_grid[:,(i+1)%Nx,(j+1)%Nx]).T

    return F

N = 1_000_000
Nx = 200
size = 400
dx = size/Nx
dt = 0.1
v_max = 1
cov = 50
# pos = np.random.uniform(size/4,3*size/4,(N,2))
pos = np.random.multivariate_normal([size/2, size/2], np.eye(2,2)*cov, size=N)
r = np.sum((pos-[size/2,size/2])**2,1)

centerX,centerY = size/2,size/2
R = size/3
r = R * np.sqrt(np.random.random(N))
theta = np.random.random(N) * 2 * np.pi
pos[:,0] = centerX + r * np.cos(theta)
pos[:,1] = centerY + r * np.sin(theta)

vel = np.random.uniform(-v_max,v_max,(N,2))*0
# vel[:N//2] = np.array([10,0])
xr, yr = np.arange(0,size+1),np.arange(0,size+1)

density,w,i,j = PROJECTION_2D_vec(pos)
F_grid, phi = SOLVE_POISSON_2D(density, density*0)
F = INTERPOLATION_2D_vec(F_grid,w,i,j)


fig, axs = plt.subplots(2,2,figsize=(12,7))
im = axs[0,0].imshow(density,origin="lower")
plt.colorbar(im)
marker_size = 0.25
alpha=0.1
line_phase, = axs[0,1].plot(pos[:,0],vel[:,0], "b.", ms=marker_size, alpha=alpha)
line_phase2, = axs[0,1].plot(pos[:,0],vel[:,0], "r.", ms=marker_size, alpha=alpha)

line_pos, = axs[1,0].plot(pos[:,0],pos[:,1], "k.", ms=marker_size, alpha=alpha)
line_vel, = axs[1,1].plot(vel[:,0],vel[:,1], "k.", ms=marker_size, alpha=alpha)
scale = 100
axs[1,1].set_ylim(-v_max*scale,v_max*scale)
axs[1,1].set_xlim(-v_max*scale,v_max*scale)
axs[0,1].set_ylim(-v_max*scale,v_max*scale)
axs[0,1].set_xlim(0,size)
axs[1,0].set_xlim(0,size)
axs[1,0].set_ylim(0,size)


axs[1,1].set_title("px - py")
axs[0,1].set_title("x - px")
axs[1,0].set_title("x - y")
fig.tight_layout()
time_list = []
phi_list = []

plt.pause(2)
for k,t in enumerate(range(10000)):
    ti = time.perf_counter()
    vel += -F*dt/2
    pos += vel*dt
    pos = pos%size

    t0 = time.perf_counter()
    density, w, i,j = PROJECTION_2D_vec(pos)
    t1 = time.perf_counter()
    F_grid, phi = SOLVE_POISSON_2D(density, phi)
    t2 = time.perf_counter()
    F = INTERPOLATION_2D_vec(F_grid, w, i,j)
    t3 = time.perf_counter()

    print()

    vel += -F*dt/2

    if k%5==0:
        im.set_data(density.T)
        # im.autoscale()
        if k%5==0:
            line_phase.set_data(pos[:,0],vel[:,0])
            line_phase2.set_data(pos[:,1],vel[:,1])
            line_pos.set_data(pos[:,0],pos[:,1])
            # line_pos.set_c(np.sqrt(vel[:,0]**2+vel[:,1]**2))
            line_vel.set_data(vel[:,0],vel[:,1])
        plt.pause(0.01)
    tf = time.perf_counter()
    t_tot = (tf-ti)*1000
    time_list.append(t_tot)
    fig.suptitle(f"{round(t_tot,2)} ms; {np.round([(t1-t0)*1000/t_tot,(t2-t1)*1000/t_tot,(t3-t2)*1000/t_tot,(tf-t3)*1000/t_tot],2)}")
    phi_list.append(np.sum(phi))
    print(round(t_tot,2),"ms", np.round([(t1-t0)*1000/t_tot,(t2-t1)*1000/t_tot,(t3-t2)*1000/t_tot,(tf-t3)*1000/t_tot],2), "|", round(np.sum(phi)/phi_list[0],3))



# N2=(Nx-1)*(Nx-1)
# A=np.zeros((N2,N2))
# ## Diagonal
# for i in range (0,Nx-1):
#     for j in range (0,Nx-1):
#         A[i+(Nx-1)*j,i+(Nx-1)*j]=-4

# # LOWER DIAGONAL
# for i in range (1,Nx-1):
#     for j in range (0,Nx-1):
#         A[i+(Nx-1)*j,i+(Nx-1)*j-1]=1
# # UPPPER DIAGONAL
# for i in range (0,Nx-2):
#     for j in range (0,Nx-1):
#         A[i+(Nx-1)*j,i+(Nx-1)*j+1]=1

# # LOWER IDENTITY MATRIX
# for i in range (0,Nx-1):
#     for j in range (1,Nx-1):
#         A[i+(Nx-1)*j,i+(Nx-1)*(j-1)]=1


# # UPPER IDENTITY MATRIX
# for i in range (0,Nx-1):
#     for j in range (0,Nx-2):
#         A[i+(Nx-1)*j,i+(Nx-1)*(j+1)]=1
# Ainv=np.linalg.inv(A)

# plt.figure()
# plt.imshow(A)

# Lmtx = sp.csr_matrix(A)
# # phi_grid = spsolve(Lmtx, density, permc_spec="MMD_AT_PLUS_A")


# r=np.zeros(N2)

# for i in range (0,Nx-1):
#     for j in range (0,Nx-1):
#         r[i+(Nx-1)*j] = dx*dx*density[i+1,j+1]
# # Boundary
# b_bottom_top=np.zeros(N2)
# for i in range (0,Nx-1):
#     b_bottom_top[i]=np.sin(0.01*2*np.pi*xr[i+1]) #Bottom Boundary
#     b_bottom_top[i+(Nx-1)*(Nx-2)]=np.sin(0.01*2*np.pi*xr[i+1])# Top Boundary

# b_left_right=np.zeros(N2)
# for j in range (0,Nx-1):
#     b_left_right[(Nx-1)*j]=2*np.sin(0.01*2*np.pi*yr[j+1]) # Left Boundary
#     b_left_right[Nx-2+(Nx-1)*j]=2*np.sin(0.01*2*np.pi*yr[j+1])# Right Boundary

# b=b_left_right+b_bottom_top


# C=np.dot(Ainv,r-b)

# w=np.zeros((Nx+1,Nx+1))
# w[1:Nx,1:Nx]=C.reshape((Nx-1,Nx-1))
# plt.figure()
# plt.imshow(w)