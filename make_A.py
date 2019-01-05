import numpy as np
from scipy.sparse import csc_matrix, hstack

__author__ = 'radams'

def flatten(x):
    out = []
    for elements in x:
            out.extend(elements)
    return out

def makeSparse(sequences, reference, alphabetsize):
    sparse_coords = lambda x: seq2bool_ind((x - reference) % alphabetsize, [0], alphabetsize)
    xcoords = [sparse_coords(seq) for seq in sequences]
    ycoords = [np.ones(len(xcoords[ii]), dtype=np.int) * ii for ii in range(len(xcoords))]
    xcoords = flatten(xcoords)
    ycoords = flatten(ycoords)
    my_ones = np.ones(len(ycoords), dtype=int)
    S = csc_matrix((my_ones, (ycoords, xcoords)), shape=(len(sequences), len(reference) * (alphabetsize - 1)))
    linear_S = S.copy()
    #S_to_matrix={x: y for (x,y) in zip(s)}
    S2 = []

    for ii in range(1, len(reference)):
        sparse_coords = lambda x: seq2bool_ind((x - reference) % alphabetsize, [0, ii], alphabetsize)
        xcoords = [sparse_coords(seq) for seq in sequences]
        ycoords = [np.ones(len(xcoords[jj]), dtype=np.int) * jj for jj in range(len(xcoords))]
        xcoords = flatten(xcoords)
        ycoords = flatten(ycoords)
        my_ones = np.ones(len(ycoords),dtype=int)
        temp = csc_matrix((my_ones, (ycoords, xcoords)), shape=(len(sequences), (len(reference) - ii) * (alphabetsize ** 2 - 1)))
        S=hstack((S, temp))
    return S, linear_S

def makeSparseInd(ref, alphabetsize):
    a = alphabetsize
    a_2 = alphabetsize ** 2
    ind_end = len(ref) * (a - 1)
    make_ind = lambda x: a * (x /(a-1))+(ref[x/(a - 1)] + x % (a - 1) + 1) % a
    ind = {x: (make_ind(x), make_ind(x)) for x in range(ind_end)}
    make_ind = lambda x, inc: ((x / a_2) * a + (ref[x / a_2]+x) % a, (x / a_2 + inc) * a + (ref[x / a_2 + inc] + x/a) % a)
    for ii in range(1, len(ref)):
        ind_start = ind_end
        ind_end = ind_start + (a_2-1) * (len(ref) - ii)
        offset = 0
        for x in range((a_2-1) * (len(ref) - ii)):
            if (x+offset)%a_2==0:
                offset+=1
            ind[x + ind_start] = make_ind(x + offset, ii)
    return ind

def seq2bool_ind(seq, interactions, alphabet):
    seqind = get_seq_ind(seq,interactions)
    vals = np.zeros(len(seqind[0]))
    for ii in range(len(interactions)):
        curr = [seq[temp] for temp in seqind[ii]]
        curr = np.array(curr)
        vals += curr * (alphabet ** ii)
    effective_alphabet = alphabet ** len(interactions)-1
    out = [res2ind(vals[ind], ind, effective_alphabet) for ind in range(len(vals))]
    out = [x for x in out if x >= 0]
    out = [int(num) for num in out]
    return out


def get_seq_ind(seq, interactions):
    base_ind=np.array(list(range(len(seq))),dtype=int)
    int_ind=[]
    usethis=base_ind>=0
    for ii in range(len(interactions)):
        temp_ind=base_ind+interactions[ii]
        int_ind.append(temp_ind)
        usethis=usethis & (temp_ind>=0) & (temp_ind<len(seq))
    int_ind=[temp[usethis] for temp in int_ind]
    return int_ind

def res2ind(res_val,res_ind,alphabet_size):
    if res_val==0:
        return -1
    else:
        return alphabet_size*res_ind+res_val-1

def res2bool(x, alphabet):
    out = []
    for ii in range(alphabet):
        out.append(0)
    if x < alphabet:
        out[x] = 1
    return out

def lin_bool2seq(lin_bool,alphabet):
    seq_len=lin_bool.shape[1]/(alphabet-1)
    out=np.zeros(seq_len)
    bool_ind=np.where(lin_bool>0)
    mut_positions=int(bool_ind[1]/(alphabet-1))
    mut_values=((bool_ind[1])%(alphabet-1))+1
    out[mut_positions]+=mut_values
    return out

