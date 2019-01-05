#!/opt/hpc/bin/python2.7
# -*- coding: utf-8 -*-
from cvxopt import matrix, solvers, sparse
from scipy.sparse import coo_matrix, dia_matrix, vstack, linalg, csc_matrix, hstack, diags, eye, find
import numpy as np
import pdb
import time
import random
import itertools
import pandas
import os
solvers.options['show_progress'] = False

def m_to_M(m, boundaries):
    #convert a number m to a vector with linear interpolation between members of the boundaries vector
    diffs = (m - boundaries)
    first = np.where(diffs >= 0)[0][-1]
    second = np.where(diffs <= 0)[0][0]
    delta = boundaries[second] - boundaries[first]
    delta += delta == 0
    M = np.zeros(boundaries.shape)
    M[second] = diffs[first] / delta
    M[first] = 1 - diffs[first] / delta
    return M

def m_to_M_coord(m, boundaries):
    #as with m_to_M, convert a number m to a set of vector coordinates and mut_values
    # with linear interpolation between members of the boundaries vector

    m = np.max([m, boundaries[0]])
    m = np.min([m, boundaries[-1]])
    diffs = (m - boundaries)
    first = np.where(diffs >= 0)[0][-1]
    second = np.where(diffs <= 0)[0][0]
    delta = boundaries[second] - boundaries[first]
    delta += delta == 0
    secondval = diffs[first] / delta
    firstval = 1 - diffs[first] / delta
    return first, firstval, second, secondval


def make_residual_matrix(m, my_b, y_inds=None, y_shape=None):
    # make_residual_matrix(m, my_b, y_inds=None, y_shape=None)
    # This function is similar to m_to_M, but it can also be used to create a sparse PWM
    # so that C.dot(x) = Ehat
    if y_inds is None:
        y_inds = list(range(len(m)))
    if y_shape is None:
        y_shape = len(m)
    coords = [m_to_M_coord(ii, my_b) for ii in m]
    ycoords = [ii for ii in y_inds] * 2
    xcoords = [coords[ii][0] for ii in range(len(coords))] + [coords[ii][2] for ii in range(len(coords))]
    vals = [coords[ii][1] for ii in range(len(coords))] + [coords[ii][3] for ii in range(len(coords))]
    M = coo_matrix((vals, (ycoords, xcoords)), shape=(y_shape, len(my_b)))
    return M

def get_data_splits(num_points, k=10):
    #get_data_splits(num_points, k=10)
    #randomly split data with num_points of elements into k pieces for cross validation
    indices = list(range(num_points))
    random.shuffle(indices)
    split_at = np.array(np.linspace(0, num_points, k + 1), dtype=int)
    data_set = [indices[split_at[ii]:split_at[ii + 1]] for ii in range(k)]
    return data_set

def A2C_matrix(A, num_muts, vals, my_b):
    # calculate the C matrix such that C.dot(x) = f_1
    # get the subset of A which only has 1 mutant
    just_1_mut = num_muts==1
    f1_inds = find(A[just_1_mut])
    
    #sort column indices so that f1_inds match vals[just_1_mut]
    f1_inds = f1_inds[1][np.argsort(f1_inds[0])]
    
    #find the columns where the mutant occurs. This is used to rearrange the columns so that f1_vals[f1_ind[ii]] corresponds to A[:,ii] when calculating f1
    f1_vals = vals[just_1_mut]
    
    #calculate the m matrices for single mutants, and then rearrange them so that A.dot(f1_m).dot(x) = f1
    #quick test - C[just_1_mut].dot(my_b) = f1_vals, C.dot(my_b) = f1
    #there may be more than 1 measurement for single mutants due to synomous sequences,
    #so normalize f1_m by its sum
    f1_m = make_residual_matrix(f1_vals, my_b, y_inds=f1_inds, y_shape=A.shape[1])
    C = A.dot(coo_matrix(f1_m/f1_m.sum(axis=1)))
    return C


def spline_penalty(grid_points, alpha):
    #calculate a spline penalty
    #formula can be found at http://data.princeton.edu/eco572/smoothing.pdf
    #Smoothing and Non-Parametric Regression
    #German Rodriguez grodri@princeton.edu
    #Spring, 2001
    
    n = grid_points.shape[0]
    delta_b = np.diff(grid_points)
    W = np.diag(np.ones(n-3) * delta_b[:-2]/6,k=-1) + np.diag(np.ones(n-3) * delta_b[2:]/6,k=1) + np.diag(np.ones(n-2) * (delta_b[:-1]+delta_b[1:]) / 3,k=0)
    delta = np.array(diags([1./delta_b[:-1],-1./delta_b[:-1] - 1./delta_b[1:],1./delta_b[1:]], offsets=[0,1,2], shape=(n-2,n)).todense())
    K = delta.T.dot(np.linalg.lstsq(W, delta)[0])
    A = np.linalg.inv(np.eye(K.shape[0]) - alpha * K)
    return K, 1 - np.mean(A.diagonal())


def make_matrices(M, my_b, C, wt_val, alpha, upper_constrain, lower_constrain):
    grid_size = my_b.shape[0]
    #minimize ||f-f_1||^2
    #We want to find the geometry of the problem so instead we find:
    #minimize ||Mx-Cx||^2 w.r.t. x
    #Remove Mx-d corresponding to zero or single mutants for efficiency...
    #the error is defined to be 0 for these data points
    #M = csc_matrix(make_residual_matrix(vals, my_b))
    bound_values = ~(upper_constrain|lower_constrain)
    #C = C[bound_values]
    #M = M[bound_values]
    Obj = M - C
    C_bounded = csc_matrix(C)
    right = np.zeros(C.shape[1])
    right[-1] = 1.
    left = np.zeros(C.shape[1])
    left[0] = 1.
    C_bounded[upper_constrain] = left
    C_bounded[lower_constrain] = right
    upper_Obj = M - C_bounded

    bound_Obj = upper_Obj
    q = matrix(np.zeros((bound_Obj.shape[1], 1)))

    #subject to the absolute constraint (A*x = b)
    #f(M_wt) = 0 (A1[0])
    #fmax - fmin = 1 (A1[1])
    M_wt = np.array(make_residual_matrix([wt_val], my_b).todense()).flatten()
    M_constraint = m_to_M(my_b[-1], my_b) - m_to_M(my_b[0], my_b)
    A1 = np.zeros((2, bound_Obj.shape[1]))
    A1[:, :Obj.shape[1]] = np.array([M_wt, M_constraint])
    A = matrix(A1)
    b = np.array([[0., -1]]).T
    b = matrix(b)

    # With inequality Δx < 0, rewritten as
    # G*x <= h
    offsets = np.array([0, 1])
    derivative = np.array([[-1., 1.]]).repeat(grid_size, axis=0)
    G1 = dia_matrix((derivative.T, offsets), shape=(grid_size - 1, bound_Obj.shape[1])).todense()

    G = G1
    h = np.zeros(G.shape[0])

    #change to cvxopt format
    G = matrix(G)
    h = matrix(h)
    P = (bound_Obj.T).dot(bound_Obj)
    
    #calculate smoothing penalty
    K, coeff = spline_penalty(my_b, alpha)
    #scrutinize = P
    #add smoothing penalty to objective
    P = P.todense()
    P += alpha * K
    P = matrix(P)
    return P, q, G, h, A, b, M, Obj

def fit_energy(M, my_b, C, wt_val, a, suppress_out):
    grid_size = my_b.shape[0]
    boundary_exceeded = True
    exceeded_upper = np.zeros(C.shape[0])==1
    exceeded_lower = np.zeros(C.shape[0])==1
    count = 0

    while boundary_exceeded:
        #This prepares all of the matrices for optimization by cvxopt
        P, q, G, h, A, b, M, Obj = make_matrices(M, my_b, C, wt_val, a, exceeded_upper, exceeded_lower)
        #if previous solution found, use that, otherwise start with un-transformed data
        if count>0:
            ret = solvers.qp(P, q, G = G, h = h, A=A, b=b, init_vals=matrix(fit_x))
        else:
            ret = solvers.qp(P, q, G = G, h = h, A=A, b=b, init_vals=my_b)
        
        #get the solution
        fit_x = np.array(ret['x'])
        # Check if unbounded f1 scores (raw_f1) are out of boundary, if so, 
        #then it's OK if bounded f1 scores (constrained_f1) are at the boundary
        raw_f1 =  C.dot(fit_x[:grid_size])
        raw_f1 = raw_f1.flatten()
        
        f = M.dot(fit_x[:grid_size])
        constrained_f1 = f - Obj.dot(fit_x)
        constrained_f1 = constrained_f1.flatten()
        
        #find upper and lower bounds
        f_max = np.max(fit_x[:grid_size ])
        f_min = np.min(fit_x[:grid_size ])
        
        new_exceeded_upper = (exceeded_upper | ((constrained_f1) >= f_max)) & ((raw_f1) >= f_max).flatten()
        new_exceeded_lower = (exceeded_lower | ((constrained_f1) <= f_min)) & ((raw_f1) <= f_min).flatten()
        count += 1
        if not suppress_out:
            print('upper: %i / %i, lower: %i / %i disagreed/out of bounds'%((new_exceeded_upper != exceeded_upper).sum(), new_exceeded_upper.sum(),(new_exceeded_lower != exceeded_lower).sum(), new_exceeded_lower.sum()))
        if count >10:
            boundary_exceeded = False
        if ((new_exceeded_upper != exceeded_upper).sum()==0) and ((new_exceeded_lower != exceeded_lower).sum()==0):
            boundary_exceeded = False
        else:
            exceeded_upper = new_exceeded_upper
            exceeded_lower = new_exceeded_lower

    return fit_x[:grid_size]


def scan_fits(A, vals, lims, alphas, suppress_out, grid_size=100):
    #energies, objective = scan_fits(A, vals, lims, alphas, grid_size=100)
    #perform lasso cross-validation to determine which smoothing parameter optimized 
    #the PWM, keep that one.
    #A - sparse matrix representation of sequences
    #vals - values of the sequences`
    #lims - boundaries
    #alphas -smoothing penalty
    #grid_size - number of grid points used approximate a continuous transformation
    #energies - transformed values
    #objective-how well the different smoothing penalties affected the transformation
    #keep an output vector with same size as the original values, 
    #but fill them in after the fits
    energies = np.nan * np.ones(vals.shape) # will be filled out at the end
    num_muts = np.array(A.sum(axis=1)).flatten()
    wt_val = np.mean(vals[num_muts == 0])

    #remove nan data for fits
    OK_data = np.isfinite(vals)
    A = A[OK_data]
    vals = vals[OK_data]
    num_muts = num_muts[OK_data]

    #only use double mutants or higher to evaluate algorithm
    usethis = (num_muts>=2)
    
    # find the step sizes so that there are approximately an equal number of data points in each step
    my_b = np.linspace(lims[0], lims[1], grid_size)

    # calculate the C matrix such that C.dot(Ehat) = f_1 - f_wt
    vals = vals - wt_val
    my_b = my_b - wt_val
    lims = np.array(lims) - wt_val
    wt_val = 0
    C = A2C_matrix(A, num_muts, vals, my_b)
    
    #remove bad values from C
    Asub = A[usethis]
    Csub = C[usethis]
    vals_sub = vals[usethis]
    unq, unq_ind, inv = np.unique(np.array(Asub.todense()), axis=0, return_index=True, return_inverse=True)
    
    #make C sparse
    Csub = csc_matrix([Csub[np.where(inv==ii)[0]].mean(axis=0).tolist()[0] for ii in set(inv)])
    
    #make M matrix, where M.dot(Ehat) ~ E
    M = csc_matrix(make_residual_matrix(vals_sub, my_b))
    M = csc_matrix([M[np.where(inv==ii)[0]].mean(axis=0).tolist()[0] for ii in set(inv)])
    
    #divide the data into 10 random pieces for cross validation
    k = 10
    data_set = get_data_splits(M.shape[0], k=k)

    objective = []
    for a in alphas:
        t0 = time.time()
        g = 0
        for ii in range(len(data_set)):
            if not suppress_out:
                print('rotate set')
            #remove the test set out of the first data subsets
            test_set = data_set.pop(0)
            training_set = list(itertools.chain(*data_set))
            
            #fit an energy transformation sum (E-E1)^2, or
            #fit for Ehat, sum ((C-M)*Ehat)^2
            fit_x = fit_energy(M[training_set], my_b, Csub[training_set], wt_val, a, suppress_out)
            
            #calculate the transformation
            f = M.dot(fit_x).flatten()
            f_max = np.max(fit_x)
            f_min = np.min(fit_x)
            
            #calculate the PWM prediction
            raw_f1 =  Csub.dot(fit_x[:grid_size])
            
            #test set PWM prediction
            f1 = raw_f1[test_set].flatten()
            f1[f1>f_max] = f_max
            f1[f1<f_min] = f_min
            
            #cross-validated r-square
            g += np.nansum((f1-f[test_set])**2)/np.nansum((f[test_set] - np.nanmean(f[test_set]))**2)/k
            
            #put the test set back at the end of the data subsets
            data_set.append(test_set)
        
        #if objective is the best objective, save it for later
        objective.append(g)
        if min(objective) == g:
            best_x = fit_x
            best_alpha = a
        if not suppress_out:
            print('alpha: %f  SSE: %f  time: %f '%(a, g, time.time() - t0))

    #calculate energies/best transformation and objective
    objective = np.array(objective).flatten()
    best_x = fit_energy(M, my_b, Csub, wt_val, best_alpha, suppress_out)

    M = csc_matrix(make_residual_matrix(vals, my_b))
    energies[OK_data] = -M.dot(best_x).flatten()
    energies = energies / (np.nanmax(energies) - np.nanmin(energies))

    return energies, objective


def monotonic_fit(A, x, xlims, alphas, name, already_fit=False, suppress_out=False, random_seed=None):
    #x,y,alphas,objective = monotonic_fit(A, x, xlims, alphas, name)
    #Main fitting fitting function.
    #A is a sparse matrix representing your sequences
    #x is a numpy vector representing KD
    #xlims are the boundaries of x
    #alphas are a set of smoothing penalties you wish to look at
    #name is a prefix so you can save the file for future use.
    #return the original values (x), the transformed values (y), the penalties used, and the model fits (objective)

    #remove non-finite values from x, and truncate it to boundary values
    if not(random_seed is None):
        random.seed(random_seed)
    
    usethis = np.isfinite(x)
    sub_x = x[usethis]
    sub_x[sub_x<xlims[0]] = xlims[0]
    sub_x[sub_x>xlims[1]] = xlims[1]
    x[usethis] = sub_x
    
    #read the file if you can, otherwise find the ideal fit
    if os.path.isfile(name + '.csv') and os.path.isfile(name + '_scan.csv') and already_fit:
        fit = pandas.read_csv(name + '.csv', header=0, index_col=0)
        scan = pandas.read_csv(name+'_scan.csv', header=0, index_col=0)

    else:
        energies, objective = scan_fits(A, x, xlims, alphas, suppress_out)
        fit = pandas.DataFrame({'x':x,'y':energies})
        fit.to_csv(name + '.csv')
        scan = pandas.DataFrame({'alphas':alphas,'objective':objective})
        scan.to_csv(name + '_scan.csv')
    
    #return the original values (x), the transformed values (y), the penalties used, and the model fits  
    x = np.array(fit['x']).flatten()
    y = np.array(fit['y']).flatten()
    alphas = np.array(scan['alphas']).flatten()
    objective = np.array(scan['objective']).flatten()
    return x, y, alphas, objective

