import functools
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.signal import find_peaks

# arr: trial of magn of shank ACC
# events: idx of peaks of arr (only looking for pairs of peaks)
def gen_gait_cycle_events(arr, events, viz=False):
    gait_cycles_ev=[] # arr of tuples(len:2) denoting beginning and end of gait cycles
    last_g_ev=(0,0)
    for i in range(0,len(events)):
        if (i-1>=0 and i+1 < len(events) and 
            (arr[events[i]] < arr[events[i+1]] and 
             arr[events[i]] < arr[events[i-1]] and
             arr[events[i-1]] >= arr[events[i+1]]
            )):
            g_ev = events[i]
            if last_g_ev!=(0,0) and i-last_g_ev[0]==3:
                gait_cycles_ev.append((last_g_ev[1],g_ev))
            last_g_ev = (i,g_ev)
                                        
    return gait_cycles_ev

magn = lambda arr: np.sqrt(np.sum(np.square(arr)))

def gen_gait_cycles(trial, viz=False):
    magn_trial = np.apply_along_axis(magn, 1,trial[:,3,0:3]) # get [full length of trial,shankL,Acc_X:Acc_Z]
    peaks, _ = find_peaks(magn_trial, prominence=0.4)
    
    if viz:
        print('ACC Magn Trial with peaks')
        plt.plot(magn_trial);plt.plot(peaks, magn_trial[peaks], "xr")
        plt.show()

    g_events = gen_gait_cycle_events(magn_trial, peaks)
    
    if viz:
        print('ACC Magn Trial with gait cycle')
        plt.plot(magn_trial)
        plt.plot([x[0] for x in g_events], magn_trial[[x[0] for x in g_events]], "xr")
        plt.plot([x[1] for x in g_events], magn_trial[[x[1] for x in g_events]], "ob")
        plt.show()
    
    return g_events #array of tuple of gait cycle event idx (begin,end)

def normalized_gait(A):
	x=np.arange(A.shape[-1])
	A=np.where(np.isnan(A)==1,0,A)
	Y= interpolate.interp1d(x,A, kind='cubic')(np.linspace(x.min(), x.max(), 100))
	return Y

# x: trial from retrieved dataset; shape: feature_trial
# out: array of x with gait cycles for trials
#    ; shape: [feature_gait_cycle_1, ... , feature_gait_cycle_N]
def trial2gcycles(x,viz=False,i=None):
    if i is not None:
        print('tria2gcycles for i:'+str(i))
    norm_gc_entries = []
    
    g_events = gen_gait_cycles(x,viz)
    
    for b_i, e_i in g_events:
        gc = x[b_i:e_i,:,:] # (b_i:e_i,5,6)
        norm_gc = np.apply_along_axis(normalized_gait, 0,gc) # (101,5,6)
        norm_gc_entries.append(norm_gc)
    
    norm_gc_entries = np.array(norm_gc_entries)
    return norm_gc_entries

tmp_len = 0
progress = lambda tmp_len,total_len:print('Progress: ['+'='*int(tmp_len/total_len*50)+'>'+'-'*int((total_len-tmp_len)/total_len*50)+'] '+str(int(tmp_len/total_len*100))+'%'+' '*10, end='\r')

def join(arr,ar2, total_len):
    
    for e in ar2:
        # arr.append(e)
        arr = np.append(arr, [e], axis=0)

    global tmp_len

    if tmp_len==0:
        tmp_len=2
    else:
        tmp_len+=1
    
    progress(tmp_len,total_len)

    return arr

def inner_flatten(d):
    nX = np.empty((0,)+d[0][0][0].shape)
    nP = np.empty((0,)+d[0][2].shape)
    nY = []

    l = len(d)
    t = 0
    for nx,y,p in d:
        for e in nx:
            nX = np.append(nX,[e], axis=0)
            nP = np.append(nP, [p], axis=0)
            nY.append(y)
        progress(t,l)
        t+=1

    return nP,nX,np.array(nY)


if __name__ == '__main__':
    # gois_dataset = np.load('../GoIS_dataset.npy', allow_pickle=True)
    X = np.load('../GoIS_X.npy', allow_pickle=True)
    Y = np.load('../GoIS_Y.npy', allow_pickle=True)
    P = np.load('../GoIS_P.npy', allow_pickle=True)
    
    print('Loaded Gait on Irregular Surface(GoIS) dataset')
    print('='*25)
    print('Applying trial2gcycles on each x')
    print('\n')
    print('trial2gcycles:')
    print('-find start of gait cycles in trial')
    print('-split gait cycles')
    print('-normalize gait cycles to 100 x 5 x 6')
    print('\n')

    tmp_l = 0

    total_l = len(X)

    def gen_meta_p(fn):
        def fn_p(x):
            global tmp_l
            tmp_l+=1
            progress(tmp_l,total_l)

            return fn(x)
        return fn_p

    nX = list(map(gen_meta_p(trial2gcycles),X))
    
    print('\n')
    print('+'*25)
    print('\n')
    print('Folding array of array of entries => into => an array of normalized gait entries')
    
    gois = list(zip(nX,Y,P))

    # nX = functools.reduce(lambda x,y: join(x,y, len(nX)), )
    nP, nX, nY = inner_flatten(gois)

    print('\n')
    print('+'*25)
    print('\n')
    print('normalized features of Gait on Irregular Surface dataset is ready! Saving it...!')
    
    print('Final shape of normalized features (X):')
    print(nX.shape)

    np.save('../GoIS_X_norm', nX)
    np.save('../GoIS_P_norm', nP)
    np.save('../GoIS_Y_norm', nY)