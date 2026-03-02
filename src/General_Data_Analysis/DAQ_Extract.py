__all__ = ["DAQ_1D_Extraction_v2", "loadmat"]

import pandas as pd
import numpy as np
import imageio
import glob
import scipy.io as sio
import h5py
import imageio

# from https://stackoverflow.com/questions/11955000/how-to-preserve-matlab-struct-when-accessing-in-python
def _check_keys( dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

# from https://stackoverflow.com/questions/11955000/how-to-preserve-matlab-struct-when-accessing-in-python

def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

# from https://stackoverflow.com/questions/11955000/how-to-preserve-matlab-struct-when-accessing-in-python

def loadmat(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def DAQ_1D_Extraction(filepath,scan_name):
    """
    Extracts DAQ data into a dataframe, in a similar format to the usual JSON file.
    """
    filename = scan_name + '.mat'
    DAQ_struct = loadmat(filepath + '/' + filename)

    scan_PV = DAQ_struct['data_struct']['params']['scanPVs']
    scan_vals = DAQ_struct['data_struct']['params']['scanVals']
    scan_steps = DAQ_struct['data_struct']['scalars']['steps']

    scan_steps = np.array([scan_vals[i-1] for i in scan_steps])
    
    cam_names = DAQ_struct['data_struct']['params']['camNames']
    if isinstance(cam_names,str):
        cam_names = cam_names.split()
    assert int(DAQ_struct['data_struct']['params']['num_CAM'])==len(cam_names),"Length of cam_names is not cam_num"

    
    common_index = DAQ_struct['data_struct']['pulseID']['common_scalar_index']
    common_index_data = DAQ_struct['data_struct']['scalars']['common_index']

    assert np.sum(common_index-common_index_data)==0, "Common Indices do not match"
    try:
        data1 = pd.DataFrame(DAQ_struct['data_struct']['scalars']['BSA_List_S11'])
    except: data1 = pd.DataFrame()

    try:
        data2 = pd.DataFrame(DAQ_struct['data_struct']['scalars']['BSA_List_S10'])
    except: data2 = pd.DataFrame()

    try:
        data3 = pd.DataFrame(DAQ_struct['data_struct']['scalars']['BSA_List_S10RF'])
    except: data3 = pd.DataFrame()
                         

    data = pd.concat([data1,data2,data3],
                     axis=1)
    data[scan_PV] = scan_steps
    
    common_index = np.array(common_index)-1
    data = data.iloc[common_index]
    data["common_index"] = common_index + 1
    
    for cam in cam_names:
        img_idx = np.array(DAQ_struct['data_struct']['pulseID'][cam + 'common_index'])-1
        files = pd.Series(DAQ_struct['data_struct']['images'][cam]['loc'])
        if len(files)>1:
            # print(DAQ_struct['data_struct']['images'][cam]['loc'])
            files = files.iloc[img_idx]
            files = files.to_list()
            data[cam+"_files"] = files
            data[cam+"_idx"] = img_idx+1
        else: # for new version of DAQ
            
            loc = (files[0]).split(scan_name)[-1]
            print(filepath + loc)
            with h5py.File(filepath + loc, "r") as f:
                img_key = list(f.keys())[0]
                imgs = f[img_key]['data']['data'][()]
                imgs1 = list(f[img_key]['instrument'])
            print(np.shape(imgs))
            new_img_arr = np.zeros([np.shape(imgs)[0]*np.shape(imgs)[1],np.shape(imgs)[2],np.shape(imgs)[3]])
            # This has to be the dumbest possible implementation -- need to vectorize 
            idx = 0
            for i in range(np.shape(imgs)[0]):
                for j in range(np.shape(imgs)[1]):
                    new_img_arr[idx,:,:] = imgs[i,j,:,:]
                    idx = idx + 1
            new_imgs = new_img_arr[img_idx]
            img_list = []
            for i in range(np.shape(new_imgs)[0]): # List comprehension?
                img_list.append(new_imgs[i,:,:])
                
            data['images'] = img_list
            # print(imgs1)
    return data

def DAQ_1D_Extraction_v2(filepath,scan_name):
    """
    Extracts DAQ data into a dataframe, in a similar format to the usual JSON file.
    """
    filename = scan_name + '.mat'
    DAQ_struct = loadmat(filepath + '/' + filename)

    scan_PV = DAQ_struct['data_struct']['params']['scanPVs']
    scan_vals = DAQ_struct['data_struct']['params']['scanVals']
    scan_steps = DAQ_struct['data_struct']['scalars']['steps']

    scan_steps = np.array([scan_vals[i-1] for i in scan_steps])
    
    cam_names = DAQ_struct['data_struct']['params']['camNames']
    if isinstance(cam_names,str):
        cam_names = cam_names.split()
    assert int(DAQ_struct['data_struct']['params']['num_CAM'])==len(cam_names),"Length of cam_names is not cam_num"

    
    common_index = DAQ_struct['data_struct']['pulseID']['common_scalar_index']
    common_index_data = DAQ_struct['data_struct']['scalars']['common_index']

    assert np.sum(common_index-common_index_data)==0, "Common Indices do not match"
    try:
        data1 = pd.DataFrame(DAQ_struct['data_struct']['scalars']['BSA_List_S11'])
    except: data1 = pd.DataFrame()

    try:
        data2 = pd.DataFrame(DAQ_struct['data_struct']['scalars']['BSA_List_S10'])
    except: data2 = pd.DataFrame()

    try:
        data3 = pd.DataFrame(DAQ_struct['data_struct']['scalars']['BSA_List_S10RF'])
    except: data3 = pd.DataFrame()
                         

    data = pd.concat([data1,data2,data3],
                     axis=1)
    data[scan_PV] = scan_steps
    
    common_index = np.array(common_index)-1
    data = data.iloc[common_index]
    data["common_index"] = common_index + 1
    
    for cam in cam_names:
        img_idx = np.array(DAQ_struct['data_struct']['pulseID'][cam + 'common_index'])-1
        files = pd.Series(DAQ_struct['data_struct']['images'][cam]['loc'])
        # Need to test this case
        if len(files)>1:
            # print(DAQ_struct['data_struct']['images'][cam]['loc'])
            files = files.iloc[img_idx]
            files = files.to_list()
            data[cam+"_files"] = files
            data[cam+"_idx"] = img_idx+1
            img_list = []
            col_list = []
            row_list = []
            for file in files:
                filename = file.split(scan_name)[-1]
                img = imageio.imread(filepath + filename)
                img_list.append(img)
                row_list.append(np.shape(img)[0])
                col_list.append(np.shape(img)[1])
            
            data[cam+'_images'] = img_list 
            print(len(img_list))
        else: # for new version of DAQ
            
            loc = (files[0]).split(scan_name)[-1]
            print(filepath + loc)
            with h5py.File(filepath + loc, "r") as f:
                img_key = list(f.keys())[0]
                imgs = f[img_key]['data']['data'][()]
                # imgs1 = list(f[img_key]['instrument'])
            print(np.shape(imgs))
            new_img_arr = np.zeros([np.shape(imgs)[0]*np.shape(imgs)[1],np.shape(imgs)[2],np.shape(imgs)[3]])
            new_img_arr = imgs.transpose(1,0,2,3).reshape(-1,np.shape(imgs)[2],np.shape(imgs)[3], order ='F')
            
            # # This has to be the dumbest possible implementation -- need to vectorize 
            # idx = 0
            
            # for i in range(np.shape(imgs)[0]):
            #     for j in range(np.shape(imgs)[1]):
            #         new_img_arr[idx,:,:] = imgs[i,j,:,:]
            #         idx = idx + 1
            new_imgs = new_img_arr[img_idx]
            img_list = []
            col_list = []
            row_list = []
            
            for i in range(np.shape(new_imgs)[0]): # List comprehension?
                img_list.append(np.reshape(new_imgs[i,:,:],-1))
                row_list.append(np.shape(new_imgs)[1])
                col_list.append(np.shape(new_imgs)[2])
                
            data[cam + '_ncols'] = col_list
            data[cam + '_nrows'] = row_list
                
            data[cam + '_images'] = img_list
            # print(imgs1)
    return data