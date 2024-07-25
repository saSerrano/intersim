import os
import json
import re
import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.font_manager

def simpleAvgSmooth(data,window_len):
    '''
    DESCRIPTION

    Compute average value of a centered sliding window to smooth
    a 1D data array.

    INPUT

    data (numpy-array): 1D data array.
    window_len (int): length of the sliding window.

    OUTPUT

    smoothed_data (numpy-array): smoothed data array.
    '''
    # Check for valid input
    assert isinstance(data,np.ndarray) and len(data.shape) == 1
    assert isinstance(window_len,int) and window_len > 1 and window_len % 2 == 1

    # Smooth data
    tail_len = (window_len - 1) / 2
    smoothed_data = np.zeros(data.shape, dtype=np.float64)
    for i in range(data.shape[0]):
        smoothed_data[i] = np.mean(data[int(max(0,i-tail_len)):int(min(i+tail_len+1,data.shape[0]-1))])
    
    return smoothed_data

def makeDir(d):
    assert isinstance(d,str)
    if not os.path.isdir(d):
        try:
            os.system('mkdir -p '+d)
        except:
            raise Exception('Could not create directory '+d)
        
def saveJson(d,f_name):
    '''
    DESCRIPTION

    Saves a dictionary object as a json file.

    INPUT

    d (dict): dictionary to be saved as a JSON file.
    f_name (str): path to the json file.

    OUTPUT

    success (bool): flag indicating if the file was 
    successfully saved.
    '''
    assert isinstance(d,dict)
    assert isinstance(f_name,str)
    assert f_name.endswith('.json') or f_name.endswith('.JSON')
    success = True
    try:
        with open(os.path.join(f_name),'w') as f:
            f.flush()
            json.dump(d,f,indent=4,sort_keys=False)
            f.close()
    except:
        success = False
    return success

def loadJson(f_name):
    '''
    DESCRIPTION

    Loads a json file into a dictionary object.

    INPUT

    f_name (str): path to the json file.

    OUTPUT

    d (dict): dictionary containing the json file's data.
    '''
    assert isinstance(f_name,str)
    assert os.path.isfile(f_name) and (f_name.endswith('.json') or f_name.endswith('.JSON'))
    d = None
    try:
        with open(f_name,'r') as f:
            d = json.load(f)
            f.close()
    except:
        raise Exception('Error ocurred while loading JSON file '+f_name)
    return d

def saveDataset(data,output_file,s_is_discrete,a_is_discrete):
    '''
    DESCRIPTION

    This function saves a data set of (s0,a,s1,r,abs,last) tu-
    ples from a data set of (s0,a,r,s1,abs,last) tuples, where
    the input data set is assumed to have the tuples list for-
    mat returned by the mushroom_rl.core.evaluate method.

    INPUT
    
    data (list-tuple): data set in mushroom_rl format.
    output_file (str): name of the output file (must end with .csv).
    s_is_discrete (bool): indicates whether the state space is
    discrete.
    a_is_discrete (bool): indicates whether the action space is
    discrete.
    '''
    # Check for valid input
    assert output_file.endswith('.csv')
    assert isinstance(data,list)
    assert len(data) > 0
    for i in range(len(data)):
        assert isinstance(data[i],tuple)
        assert len(data[i]) == 6
    
    # Save data set as a numpy array
    # Extract s0-a-r-s1 data from the input data set and reshape
    # to s0-a-s1-r format in a numpy array
    n = len(data)
    s_dim = data[0][0].shape[0]
    a_dim = data[0][1].shape[0]
    sars_data = np.zeros((n + 1,s_dim*2+a_dim+3),dtype=np.float32)
    for i in range(n):
        sars_data[i,:s_dim] = data[i][0] # s0
        sars_data[i,s_dim:s_dim+a_dim] = data[i][1] # a
        sars_data[i,s_dim+a_dim] = data[i][2] # r
        sars_data[i,s_dim+a_dim+1:s_dim*2+a_dim+1] = data[i][3] # s1
        sars_data[i,s_dim*2+a_dim+1] = 1.0 if data[i][4] else 0.0 # absorbing
        sars_data[i,s_dim*2+a_dim+2] = 1.0 if data[i][5] else 0.0 # last
    
    # Write in the last row: s_dim, a_dim, s_is_discrete, a_is_discrete
    sars_data[n,0] = s_dim
    sars_data[n,1] = a_dim
    sars_data[n,2] = 1 if s_is_discrete else 0
    sars_data[n,3] = 1 if a_is_discrete else 0
    
    # Save reformatted data set
    np.savetxt(output_file,sars_data,delimiter = ",")

def loadDataset(fname):
    '''
    DESCRIPTION

    Loads a data set from a .csv, as a numpy array. This function
    should only be used to load arrays saved by the 'saveDataset'
    function, as it assumes columns to represent s0,a,r,s1,abs,last.

    INPUT

    fname (str): output file name.

    OUTPUT

    data (numpy-array): data set.
    s_dim (int): # of state variables.
    a_dim (int): # of action variables.
    s_is_discrete (bool): indicates if the state space is discrete.
    a_is_discrete (bool): indicates if the action space is discrete.
    '''
    # Check for valid input
    assert os.path.isfile(fname) and (fname.endswith('.csv') or fname.endswith('.CSV'))
    
    # Load array
    np_data = np.genfromtxt(fname, delimiter=',')

    # Extract state and action dimensions
    s_dim = int(np_data[np_data.shape[0]-1,0])
    a_dim = int(np_data[np_data.shape[0]-1,1])
    s_is_discrete = (np_data[np_data.shape[0]-1,2] > 0.5)
    a_is_discrete = (np_data[np_data.shape[0]-1,3] > 0.5)

    # Check that the loaded array has the expected shape: N x (s+a+s+3)
    assert np_data.shape[0] >= 1 and np_data.shape[1] == (s_dim*2 + a_dim + 3)

    # Reformat the data set as a list of tuples (mushroomRL style)
    data = []
    for i in range(np_data.shape[0]-1):
        s = np_data[i,:s_dim].astype(int) if s_is_discrete else np_data[i,:s_dim]
        a = np_data[i,s_dim:s_dim+a_dim].astype(int) if a_is_discrete else np_data[i,s_dim:s_dim+a_dim]
        r = float(np_data[i,s_dim+a_dim])
        s1 = np_data[i,s_dim+a_dim+1:s_dim*2+a_dim+1].astype(int) if s_is_discrete else np_data[i,s_dim+a_dim+1:s_dim*2+a_dim+1]
        absorb = bool(np_data[i,s_dim*2+a_dim+1] > 0.5)
        last = bool(np_data[i,s_dim*2+a_dim+2] > 0.5)
        data.append((s,a,r,s1,absorb,last))
    
    return data,s_dim,a_dim,s_is_discrete,a_is_discrete