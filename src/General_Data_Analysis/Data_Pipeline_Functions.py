from .Data_Classes import datasets
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
import h5py
import copy
import scipy
import concurrent
import cv2
import imageio
from PIL import Image

from . import DAQ_Extract as DAQ
from . import Image_Analysis as IA
from scipy.ndimage import median_filter

plt.rcParams.update({'font.size': 14})

# Deal with script arguments

import sys
# if len(sys.argv) > 1:
#     dataset = str(sys.argv[1])

class AnalysisParameters:
    def __init__(self, params):
        self.bound_list = params.get('bound_list')
        self.thresh = float(params.get('thresh'))
        self.bg_thresh = float(params.get('bg_thresh'))
        self.proj_thresh = float(params.get('proj_thresh'))
        self.VCC_bound_list = params.get('VCC_bound_list')
        self.idx = params.get('idx')
        self.VCC_idx = params.get('VCC_idx')
        self.thresh_1 = float(params.get('thresh_1',1e6))

def validate_dataset(dataset_name):
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not recognized. Available datasets are: {list(datasets.keys())}")
    print(f"Dataset '{dataset_name}' is valid.")

    return datasets[dataset_name]


def screen_nickname_finder(screen,save_loc):
    screen_nickname = screen.split(':')[2]
    os.makedirs(save_loc, exist_ok=True)
    return screen_nickname

def second_parent(orig_path):
    parent, file = os.path.split(orig_path)
    parentparent,parentfile = os.path.split(parent)
    return parentparent

def fix_path(new_path,p):
    new_parent,foo = os.path.split(new_path)
    old_parent = second_parent(p)
    
    return p.replace(old_parent,new_parent)


def read_hdf(filename):
    with h5py.File(filename, "r") as f:
    
        a_group_key = list(f.keys())[0]
        ds_arr = f[a_group_key][()]
    return ds_arr


def Merge(dict1, dict2):
        return(dict1.update(dict2))

def im_stack_from_df(df,col,ncolcol,nrowcol,fliplr=0,rot=0):
    ncols=df[ncolcol]
    nrows=df[nrowcol]
    if np.abs(np.sum(np.diff(ncols)))+np.abs(np.sum(np.diff(nrows)))==0:
        ncol=int(ncols.mean())
        nrow=int(nrows.mean())
        imgs=df[col]
        if rot:
            imgArray=np.zeros([len(imgs),nrow,ncol])
        else:
            imgArray=np.zeros([len(imgs),ncol,nrow])
        j=0
        
        for i,row in df.iterrows():
            # print(i)
            # print(j)
            img=imgs.iloc[j]
            # plt.imshow(img)
            
            img=np.reshape(img,(ncol,nrow))
            if rot:
                img=scipy.ndimage.rotate(img,90)
            if fliplr:
                img=np.fliplr(img)
            imgArray[j,:,:]=img
            j=j+1
    return imgArray

def get_border_mean(img,border_width=10):
    top = img[:border_width, :]
    bottom = img[-border_width:, :]
    left = img[border_width:-border_width, :border_width]
    right = img[border_width:-border_width, -border_width:]
    bg = np.concatenate([
        np.reshape(top,-1),
        np.reshape(bottom,-1),
        np.reshape(left,-1),
        np.reshape(right,-1)
    ],axis=0)
    return np.mean(bg)

def get_proj_mean(proj,border_width=10):
    top = proj[:border_width]
    bottom = proj[-border_width:]
    
    bg = np.concatenate([
        np.reshape(top,-1),
        np.reshape(bottom,-1),
    ],axis=0)
    return np.mean(bg)

def get_x_y_proj_mean(img):
    xproj = np.sum(img,axis=0)
    yproj = np.sum(img,axis=1)
    xmean = get_proj_mean(xproj)
    ymean = get_proj_mean(yproj)
    return xmean,ymean

def safe_extract(fit):
    try:
        s = fit.Sigma
    except:
        s = np.nan
    return s

def rms_extractor(x):
    """
    Returns a numpy array of moments given a list of GaussianParams objects
    
    Argument:
    x -- a list of GaussianParams objects
    """
    try: x = list(x)
    except: raise ValueError("Input could not be converted to a list")
    assert isinstance(x,list)==True, "input is not a list!"
    
    return np.array([safe_extract(i) for i in list(x)])

def zscore(series):
    series = np.array(series.to_list())
    return np.abs(scipy.stats.zscore(series))
def outlier_detect(zscores,thresh):
    return  (zscores > thresh)

def func6(img):
    try:
        xfit, yfit,xfit45,yfit45,img = IA.image_analysis_6(img,initial_Gauss=True,return_images=True)
    except:
        xfit = np.nan
        yfit = np.nan
        xfit45 = np.nan
        yfit45 = np.nan
        img = np.nan
    return xfit, yfit,xfit45,yfit45,img

def func5(img):
        try:
            xfit, yfit,xfit45,yfit45,img = IA.image_analysis_5(img,initial_Gauss=True,return_images=True)
        except:
            xfit = np.nan
            yfit = np.nan
            xfit45 = np.nan
            yfit45 = np.nan
            img = np.nan
        return xfit, yfit,xfit45,yfit45,img

def Generic_Preprocessing(dataset):
    """
    Generic preprocessing for non-synchronized data.
    Loads all non-synchronized data files, merges them into a single dataframe, and saves the result as a pickle file.
    """
    screen_nickname = screen_nickname_finder(dataset.screen,dataset.save_loc)
    fnamelist_all_values = []
    data = []
    for i in range(len(dataset.pathlist)):

        fnamelist_all_values = fnamelist_all_values + ([i for i in glob.glob(dataset.pathlist[i]+'*values*.npy')])

    fnamelist_all_values = np.sort(fnamelist_all_values)
    if len(fnamelist_all_values) == 0:
        raise FileNotFoundError(
            f"No data files found matching *values*.npy in paths: {dataset.pathlist}"
        )

    ## Get all image file names 
    fnamelist_all_imgs = []
    for i in range(0, len(fnamelist_all_values)):
        imgs_name_temp = fnamelist_all_values[i][0:-36]+'imgs_'+fnamelist_all_values[i][-29:]
        fnamelist_all_imgs.append(imgs_name_temp)

    ## Load all the files into a single dataframe (matching)
    

    if '2024-03-' in dataset.save_loc and screen_nickname=='241':
        charges = [1600 if 'Nominal' in val else 2800 if '2p8' in val else 200 if '200' in val else 500 if '500' in val else None for val in fnamelist_all_values]
        assert len(charges)==len(fnamelist_all_values),"Problem: charges not aligned"
        print('March 2024 241')

    df_read = pd.DataFrame({})
    for i in range(0,len(fnamelist_all_values)):
        try:
            values_ex = np.load(fnamelist_all_values[i],allow_pickle=True)
            values_dict = values_ex.item()
            
            imgs_ex = np.load(fnamelist_all_imgs[i],allow_pickle=True)
            imgs_dict = imgs_ex.item()
        
            
            
            filled = imgs_dict

            result = {k: filled.get(k, 'NaN') for k in dataset.empty}

            Merge(values_dict, result)
        
            timestamp = fnamelist_all_values[i][-29:-4]
            values_dict['timestamp']=timestamp
        
            df_read = pd.concat([df_read,pd.DataFrame.transpose(pd.DataFrame.from_dict(values_dict,orient='index'))],axis=0, ignore_index=True)
        except Exception as e:
            print(e)
    def filter_df(df,col):
        idx = pd.isna(df[col].str.contains('NaN'))
        df_new = df[idx]
        return df_new, idx
        
    non_sync_data_screen, idx = filter_df(df_read,dataset.screen)
    print(len(non_sync_data_screen))

    # To do: something with these plots
    for i in range(len(non_sync_data_screen.columns)):
        try:
            plt.plot(non_sync_data_screen.reset_index()[non_sync_data_screen.columns[i]])
            plt.ylabel(non_sync_data_screen.columns[i])
            plt.xlabel('Index')
            plt.show()
            os.makedirs(dataset.save_loc + 'Preprocessing_Plots/', exist_ok=True)
            plt.savefig(dataset.save_loc + 'Preprocessing_Plots/' + non_sync_data_screen.columns[i] + '.png')
            plt.close()

        except:
            pass


    if '2024-03-' in dataset.save_loc and screen_nickname=='241':
        len(charges)
        non_sync_data_screen['manual_charge'] = charges

    non_sync_data_screen.to_pickle(dataset.save_loc + 'non_sync_data_' + screen_nickname + '.pkl')


def Generic_DAQ_Preprocessing(dataset):
    screen_nickname = screen_nickname_finder(dataset.screen,dataset.save_loc)
    if dataset.DAQ_Matching is not None:
        # Get folders of files
        files = []
        # for path in pathlist:
        path = dataset.DAQ_Matching.split('DAQ.yaml')[0]
        for prefix in dataset.prefixes:
            newfile = glob.glob(path + prefix + '*')
            files = files + newfile
        # Load DAQ Data
        data_dict = {}
        for file in files:
            try: 
                DAQ_str = (file.split('_')[-1])
                scan_name = 'E331_' + DAQ_str
                DAQ_num = int(DAQ_str)
                DAQ_Data = DAQ.DAQ_1D_Extraction_v2(file,scan_name)
                DAQ_Data['DAQ_str'] = DAQ_str
                DAQ_Data['DAQ_num'] = DAQ_num
                data_dict[scan_name] = DAQ_Data
            except Exception as e:
                print(e)
        ## Flatten data into single DataFrame
        temp_list = [data_dict[key] for key in data_dict.keys()]
        All_DAQ_df = pd.concat(temp_list)
        All_DAQ_df.to_pickle(dataset.save_loc + "All_DAQ_Data.pkl")
    else:
        All_DAQ_df = pd.DataFrame({})
    All_DAQ_df.to_pickle(dataset.save_loc + "All_DAQ_Data.pkl")

def Generic_Data_Processing(dataset):
    screen_nickname = screen_nickname_finder(dataset.screen,dataset.save_loc)
    # Load data
    non_sync_data = pd.read_pickle(dataset.save_loc + 'non_sync_data_' + screen_nickname + '.pkl')

    if dataset.DAQ_Matching is not None:
        with open(dataset.DAQ_Matching, 'r') as file:
            matching = yaml.safe_load(file)
        matching = pd.DataFrame(matching['data'])
        DAQ_Data = pd.read_pickle(dataset.save_loc + "All_DAQ_Data.pkl")
        idx = [i for i, s in enumerate(DAQ_Data.columns) if screen_nickname+'_images' in s]
        DAQ_Data.rename(columns={list(DAQ_Data.columns[idx])[0]:dataset.screen},inplace=True)
        idx = [i for i, s in enumerate(DAQ_Data.columns) if screen_nickname+'_nrows' in s]
        DAQ_Data.rename(columns={list(DAQ_Data.columns[idx])[0]:dataset.screen.split('Image:')[0] + 'Image:ArraySize1_RBV'},inplace=True)
        idx = [i for i, s in enumerate(DAQ_Data.columns) if screen_nickname+'_ncols' in s]
        DAQ_Data.rename(columns={list(DAQ_Data.columns[idx])[0]:dataset.screen.split('Image:')[0] + 'Image:ArraySize0_RBV'},inplace=True)

        # Match DAQ
        # Get common columns between DAQ 
        common_idxs = np.array(matching.columns)[np.in1d(np.array(matching.columns),np.array(non_sync_data.columns))]
        # common_idxs = np.delete(common_idxs,7)


        # Remove scan indices
        scan_idxs = common_idxs[np.in1d(common_idxs,np.array(DAQ_Data.columns))]
        common_idxs = common_idxs[~np.in1d(common_idxs,scan_idxs)]
        common_idxs

        ## Explicitly match non-DAQ data to DAQ scans
        # Gets all the other quad settings for the DAQ scans and associates them with a DAQ number
        row_list = []
        for i, row in matching.iterrows():
            non_sync_data_test = copy.deepcopy(non_sync_data)
            for idx in list(common_idxs):
                non_sync_data_test = non_sync_data_test[np.abs(non_sync_data_test[idx]-row[idx])<0.001]
            if len(non_sync_data_test)==1:
                non_sync_data_test.head()

                df_test=pd.Series(non_sync_data_test.iloc[0])
                row_new = df_test.to_dict()
                row_new.update(row)
                row_list.append(row_new)

        new_df = pd.DataFrame(row_list)

        ## Give a clear index to match
        new_df['DAQ_int'] = new_df['DAQ_num'].astype(int)
        DAQ_Data['DAQ_int'] = DAQ_Data['DAQ_num'].astype(int)

        ## Keep only relevant columns in dataframe matched (settings, not readbacks) & relevant rows
        cols_to_keep = [i for i in new_df.columns if 'BCTRL' in i or 'DES' in i or 'DAQ' in i]
        matching_data = new_df[cols_to_keep]
        matching_data = matching_data[matching_data['DAQ_int']>0]

        ## Get all DAQ data matched with experimental data
        combined_data = DAQ_Data.merge(matching_data,how='inner', on='DAQ_int')

        # Merge DAQ data
        all_data = pd.concat([non_sync_data,combined_data])
    else:
        all_data = non_sync_data
    all_data.to_pickle(dataset.save_loc + "All_Data_" + screen_nickname + '.pkl')


def Generic_Image_Processing(dataset,bound_list,idx=None,thresh_1=1e6):
    screen_nickname = screen_nickname_finder(dataset.screen,dataset.save_loc)
    all_data = pd.read_pickle(dataset.save_loc + "All_Data_" + screen_nickname + '.pkl')

    # Remove blank image shots
    df_sizes = all_data[[dataset.screen.split('Image:')[0] + 'Image:ArraySize1_RBV',dataset.screen.split('Image:')[0] + 'Image:ArraySize0_RBV']]
    df_sizes.drop_duplicates(inplace=True)

    idx = []
    for i,row in df_sizes.iterrows():
        test = 0
        try:
            (int(row[dataset.screen.split('Image:')[0] + 'Image:ArraySize0_RBV']))
        except:
            test = test +1
        try:
            (int(row[dataset.screen.split('Image:')[0] + 'Image:ArraySize1_RBV']))
        except:test = test+1
        idx.append(test)
        
    df_sizes 
    idx=np.array(idx)
    idx = idx>1
    idx

    df_sizes = df_sizes.loc[~idx]
    # Make image stacks
    imgArray_List = []
    df_list = []
    for i, row in df_sizes.iterrows():
        idx1 = all_data[dataset.screen.split('Image:')[0] + 'Image:ArraySize1_RBV']==row[dataset.screen.split('Image:')[0] + 'Image:ArraySize1_RBV']
        idx2 = all_data[dataset.screen.split('Image:')[0] + 'Image:ArraySize0_RBV']==row[dataset.screen.split('Image:')[0] + 'Image:ArraySize0_RBV']
        idx = idx1*idx2
        df_subset = all_data.loc[idx]
        df_list.append(df_subset)
        imgArray=im_stack_from_df(df_subset,dataset.screen,dataset.screen.split('Image:')[0] + 'Image:ArraySize1_RBV',dataset.screen.split('Image:')[0] + 'Image:ArraySize0_RBV',fliplr=0,rot=0)
        imgArray_List.append(imgArray)

    # Do something about figures
    for i in range(len(imgArray_List)):
        img_temp = np.sum(imgArray_List[i],0)
    #     plt.figure()
    #     plt.imshow(np.sum(imgArray_List[i],0))
    #     plt.show()

    image_stack = []
    for i in range(len(bound_list)):
        imArray = imgArray_List[i]
        test = imArray[:,bound_list[i]['ystart']:bound_list[i]['yend'],bound_list[i]['xstart']:bound_list[i]['xend']]
        image_stack.append(test)

    # Do something about figures
    xcs = []
    ycs = []

    for i in range(len(imgArray_List)):
        img_temp = np.sum(image_stack[i],0)
        # plt.figure()
        # plt.imshow(img_temp)
        # plt.show() 
        xfit,yfit,xfit45,yfit45 = IA.Gaussian_Fit_4_Dim(img_temp)
        IA.visualize_projections(img_temp)
        xcs.append(xfit.Center)
        ycs.append(yfit.Center)

    # Load background image
    if dataset.bg_file is not None:
        bg_files = glob.glob(dataset.bg_file)
        if len(bg_files)<1:
            bg_file = None
        else: 
            bg_arr = []
            for bg in bg_files:
                bg_img = np.load(bg,allow_pickle=True)
                bg_img = np.reshape(bg_img.item()[dataset.screen],(bg_img.item()[dataset.screen.split('Image:')[0] + 'Image:ArraySize1_RBV'],bg_img.item()[dataset.screen.split('Image:')[0] + 'Image:ArraySize0_RBV']))
                bg_arr.append(bg_img)


            bg_img = np.mean(np.array(bg_arr),0)
        
    if dataset.bg_file is None:
        big_img_arr = np.concatenate(image_stack,axis=0)

        
        # plt.imshow(np.mean(big_img_arr,0))

        ### Make and plot sums
        sum_sum = np.sum(big_img_arr,2)
        sum_sum = np.sum(sum_sum,1)
        # plt.plot(sum_sum)

        ### Threshold and select indices of possible background images
        test = sum_sum<thresh_1
        np.sum(test)

        ### Look at background candidate images
        bg = all_data[test]
        for i,row in bg.iterrows():
            img = row[dataset.screen]
            img = np.reshape(img,[row[dataset.screen.split('Image:')[0] + 'Image:ArraySize1_RBV'],row[dataset.screen.split('Image:')[0] + 'Image:ArraySize0_RBV']])
            # plt.figure()
            # plt.imshow(img)
            # plt.colorbar()
            # plt.clim([0,100])
            # plt.show()
        
        try:
            idx = bg.iloc[idx].name
            unacceptable = False
        except:
            idx = 0
            unacceptable = True

        print(unacceptable)
      
        ### Get full image for background subtraction
        if unacceptable == False:
            bg_img = np.reshape(np.array((all_data.loc[[idx]][dataset.screen]).values[0]),[int(all_data.loc[[idx]][dataset.screen.split('Image:')[0] + 'Image:ArraySize1_RBV']),int(all_data.loc[[idx]][dataset.screen.split('Image:')[0] + 'Image:ArraySize0_RBV'])])
        else:
            bg_img = np.zeros([int(all_data.iloc[[idx]][dataset.screen.split('Image:')[0] + 'Image:ArraySize1_RBV']),int(all_data.iloc[[idx]][dataset.screen.split('Image:')[0] + 'Image:ArraySize0_RBV'])])

    ## Align the images 
    # to the one that matches the dimensions of the background image
    # That way, we can just use that background image
    ### Match the background shape to the image shapes

    test = []
    for iii in imgArray_List:
        print(np.shape(bg_img))
        print(np.shape(np.sum(iii,0)))
        if np.shape(np.sum(iii,0))==np.shape(bg_img):
            test.append(1)
        else:
            test.append(0)
    test = np.array(test)

    i = 0
    idx = np.argwhere(test==1)
    bounds_list_new = []
    for bounds in bound_list:
        bounds_new = {}
        if test[i]==0:
            
            # bounds = bounds - (np.array(xcs)[i]-np.array(xcs)[idx],np.array(ycs)[i]-np.array(ycs)[idx])
            bounds_new['xstart'] = int(bounds['xstart'] + int(np.array(xcs)[i]-np.array(xcs)[idx]))
            bounds_new['xend'] = int(bounds['xend'] + int(np.array(xcs)[i]-np.array(xcs)[idx]))
            bounds_new['ystart'] = int(bounds['ystart'] + int(np.array(ycs)[i]-np.array(ycs)[idx]))
            bounds_new['yend'] = int(bounds['yend'] + int(np.array(ycs)[i]-np.array(ycs)[idx]))
            
        else:
            bounds_new = (copy.deepcopy(bounds))
            bg_bounds = i
        print(bounds_new)
        bounds_list_new.append(bounds_new)
        i = i+1

    min_xbound = 0
    min_ybound = 0

    for bound in bounds_list_new:
        print(min_xbound)
        print(bound['xstart'])
        min_xbound = np.minimum(min_xbound,bound['xstart'])
        min_ybound = np.minimum(min_ybound,bound['ystart'])

    if min_xbound<0:
        for i in range(len(bounds_list_new)):
            bounds_list_new[i]['xstart'] = bounds_list_new[i]['xstart'] - min_xbound
            bounds_list_new[i]['xend'] = bounds_list_new[i]['xend'] - min_xbound

    if min_ybound<0:
        for i in range(len(bounds_list_new)):
            bounds_list_new[i]['ystart'] = bounds_list_new[i]['ystart'] - min_ybound
            bounds_list_new[i]['yend'] = bounds_list_new[i]['yend'] - min_ybound

    ## Check that the centroids are close now
    image_stack = []
    for i in range(len(bound_list)):
        imArray = imgArray_List[i]
        test = imArray[:,bounds_list_new[i]['ystart']:bounds_list_new[i]['yend'],bounds_list_new[i]['xstart']:bounds_list_new[i]['xend']]
        image_stack.append(test)
        
    image_stack = []
    for i in range(len(bound_list)):
        imArray = imgArray_List[i]
        test = imArray[:,bounds_list_new[i]['ystart']:bounds_list_new[i]['yend'],bounds_list_new[i]['xstart']:bounds_list_new[i]['xend']]
        image_stack.append(test)
    
    ## Make all_images
    all_images = np.concatenate(image_stack)
    # Do something about plots
    # plt.imshow(np.sum(all_images,0))

    ## Make BG
    bg_cropp = bg_img[bounds_list_new[bg_bounds]['ystart']:bounds_list_new[bg_bounds]['yend'],bounds_list_new[bg_bounds]['xstart']:bounds_list_new[bg_bounds]['xend']]
    plt.imshow(bg_cropp)
    os.makedirs(dataset.save_loc + 'Processed_Images/', exist_ok=True)
    plt.savefig(dataset.save_loc + 'Processed_Images/' + 'background_' + screen_nickname + '.png')

    # Make all_data
    all_data = pd.concat(df_list)
    # plt.scatter(range(len(all_data)),all_data.index)
    all_data.reset_index(drop=True,inplace=True)

    plt.scatter(range(len(all_data)),all_data.index)
    plt.xlabel('Image Index')
    plt.ylabel('Dataframe Index')
    os.makedirs(dataset.save_loc + 'Processed_Images/', exist_ok=True)
    plt.savefig(dataset.save_loc + 'Processed_Images/' + 'index_scatter_' + screen_nickname + '.png')
    plt.close()

    if screen_nickname=='241':
        rot_image_stack = np.flip(np.flip(all_images.transpose(0,2,1),axis=2),axis=1)
        rot_bg_cropp = np.flipud(np.fliplr(np.transpose(bg_cropp)))
    elif screen_nickname=='571':
        rot_image_stack = np.flip(np.flip(all_images,axis=2),axis=1)
        rot_bg_cropp = np.flipud(np.fliplr(bg_cropp))
        
    np.save(dataset.save_loc + 'total_images_stack_' + screen_nickname + '.npy',rot_image_stack)
    np.save(dataset.save_loc + 'background_' + screen_nickname + '.npy',rot_bg_cropp)
    all_data.to_pickle(dataset.save_loc + 'total_data_stack_' + screen_nickname + '.pkl')

    IA.visualize_projections(np.sum(rot_image_stack,0))

def Background_Treatment(dataset,case_no=None):
    def cutout_beam(img1,cutout):
        img = copy.deepcopy(img1)
        img[cutout[2]:cutout[3],cutout[0]:cutout[1]] = np.nan
        return img
    if case_no == 1:
        screen_nickname = screen_nickname_finder(dataset.screen,dataset.save_loc)
        # Load data
        all_images = np.load(dataset.save_loc + 'total_images_stack_' + screen_nickname + '.npy')
        bg_cropp = np.load(dataset.save_loc + 'background_' + screen_nickname + '.npy')
        all_data = pd.read_pickle(dataset.save_loc + 'total_data_stack_' + screen_nickname + '.pkl')
        all_img_list = [all_images[i,:,:] for i in range(len(all_images))]

        # Pixel calibration
        pxcal = pd.unique(all_data[dataset.screen.split('Image:')[0] + 'RESOLUTION']).astype(float)
        pxcal = pxcal[~np.isnan(pxcal)]
        if len(pxcal)>1:
            raise ValueError("more than one pixel calibration")
        else:
            pxcal = float(pxcal)

        # cutouts
        cutout = [500-225,500-150,0,500]
        img5 = cutout_beam(all_img_list[335],cutout)
        # plt.imshow(img5)

        cutout = [500-325,500-200,0,500]
        img4 = cutout_beam(all_img_list[285],cutout)
        # plt.imshow(img4)

        img6 = all_img_list[230]
        # plt.imshow(img6,vmax=50)

                
        cutout = [500-250,500-175,185,250]
        img7 = cutout_beam(img6,cutout)
        cutout = [500-175,500-0,185,275]
        img7 = cutout_beam(img7,cutout)
        cutout = [500-500,500-250,150,250]
        img7 = cutout_beam(img7,cutout)
        # plt.imshow(img7,vmax=50)

        img8 = all_img_list[237]
        cutout = [500-300,500-225,0,150]
        img8 = cutout_beam(img8,cutout)

        cutout = [500-325,500-250,150,500]
        img8 = cutout_beam(img8,cutout)

        # plt.imshow(img8,vmax=50)

        stacked = np.stack([ img4,img5,img7,img8], axis=0)
        img_bg = np.nanmean(stacked,axis=0)
        # plt.imshow(img_bg,vmax=50)
        # plt.colorbar()

        # plt.plot(np.sum(img_bg,axis=0))
# 
        # plt.plot(np.sum(img_bg,axis=1))

        np.save(dataset.save_loc + 'background_' + screen_nickname + '.npy',img_bg)

def filter_beams(dataset,thresh,bg_thresh,proj_thresh):
    screen_nickname = screen_nickname_finder(dataset.screen,dataset.save_loc)
    all_images = np.load(dataset.save_loc + 'total_images_stack_' + screen_nickname + '.npy')

    all_data = pd.read_pickle(dataset.save_loc + 'total_data_stack_' + screen_nickname + '.pkl')

    bg = np.load(dataset.save_loc + 'background_' + screen_nickname + '.npy')

    sums = []
    for i in range(np.shape(all_images)[0]):
        sums.append(np.sum(all_images[i,:,:]))
    # plt.plot(sums)

    rmidx = np.array(sums)<thresh
    all_images = all_images[~rmidx,:,:]

    all_data.reset_index(drop=True,inplace=True)
    all_data = all_data[~rmidx]
    all_data.reset_index(drop=True,inplace=True)

    # plt.scatter(range(len(all_data)),all_data.index)

    all_img_list_bg_subt = [median_filter(all_images[i,:,:]-bg,size=3) for i in range(len(all_images))]
    all_img_list = [all_images[i,:,:] for i in range(len(all_images))]

    arrs = [get_border_mean(i) for i in all_img_list_bg_subt]

    ## User Input: bg threshold to cut clipped images
    good_idx = np.array(arrs)<bg_thresh
    good_imgs = np.array(all_img_list_bg_subt)[good_idx,:,:]
    good_img_list = np.array(all_img_list)[good_idx,:,:]

    good_data = all_data.loc[good_idx]
    ### Get clipped projections that slip through
    xmeans = [get_x_y_proj_mean(good_imgs[i,:,:])[0] for i in np.arange(np.shape(good_imgs)[0])]
    ymeans = [get_x_y_proj_mean(good_imgs[i,:,:])[1] for i in np.arange(np.shape(good_imgs)[0])]

    ## User Input: bg threshold to cut projections
    # plt.plot(xmeans,label='xmeans')
    # plt.plot(ymeans,label='ymeans')

    good_good_idx = (np.array(xmeans)>proj_thresh) | (np.array(ymeans)>proj_thresh)
    good_good_idx = ~good_good_idx

    selected_imgs = good_imgs[good_good_idx]
    selected_img_list = good_img_list[good_good_idx,:,:]
    selected_data = good_data.loc[good_good_idx].reset_index(drop=True)

    np.save(dataset.save_loc + 'all_images_stack_' + screen_nickname + '.npy',selected_img_list)

    selected_data.to_pickle(dataset.save_loc + 'all_data_stack_' + screen_nickname + '.pkl')


def Generic_Moment_Calculation(dataset):
    screen_nickname = screen_nickname_finder(dataset.screen,dataset.save_loc)
    all_images = np.load(dataset.save_loc + 'all_images_stack_' + screen_nickname + '.npy')
    bg_cropp = np.load(dataset.save_loc + 'background_' + screen_nickname + '.npy')
    all_data = pd.read_pickle(dataset.save_loc + 'all_data_stack_' + screen_nickname + '.pkl')
    print(all_data.columns)
    print(all_data.head())
    pxcal = pd.unique(all_data[dataset.screen.split('Image:')[0] + 'RESOLUTION']).astype(float)
    pxcal = pxcal[~np.isnan(pxcal)]
    print(pxcal)
    if len(pxcal)>1:
        raise ValueError("more than one pixel calibration")
    else:
        pxcal = float(pxcal)

    moments = pd.DataFrame({})
        
    all_img_list_bg_subt = [all_images[i,:,:]-bg_cropp for i in range(len(all_images))]
    all_img_med_bg_subt = [median_filter(all_images[i,:,:]-bg_cropp,size=3) for i in range(len(all_images))]

    # all_img_list_bg_subt = []
    # for img in all_img_list_bg_subt1:
    #     img[img<0] = 0
    #     all_img_list_bg_subt.append(img)

    # plt.plot(np.sum(all_img_list_bg_subt[0],axis=0))

    nickname = 'Gauss-BG-Subt'
    with concurrent.futures.ProcessPoolExecutor() as pool:
        xfit_Gauss, yfit_Gauss,xfit45_Gauss,yfit45_Gauss = zip(*pool.map(IA.Gaussian_Fit_4_Dim, all_img_list_bg_subt))
    moments['x_'+nickname] = np.array(rms_extractor(xfit_Gauss))*pxcal
    moments['y_'+nickname] = np.array(rms_extractor(yfit_Gauss))*pxcal
    moments['x45_'+nickname] = np.array(rms_extractor(xfit45_Gauss))*pxcal
    moments['y45_'+nickname] = np.array(rms_extractor(yfit45_Gauss))*pxcal

    nickname = 'BG-med-Gauss-RMS'
    import warnings
    warnings.filterwarnings('ignore')
    # all_img_list_bg_subt = [all_images[i,:,:]-bg_cropp for i in range(len(all_images))]
    
    with concurrent.futures.ProcessPoolExecutor() as pool:
        xfit, yfit,xfit45,yfit45,img_array = zip(*pool.map(func5, all_img_med_bg_subt))
    moments['x_'+nickname] = np.array(rms_extractor(xfit))*pxcal
    moments['y_'+nickname] = np.array(rms_extractor(yfit))*pxcal
    moments['x45_'+nickname] = np.array(rms_extractor(xfit45))*pxcal
    moments['y45_'+nickname] = np.array(rms_extractor(yfit45))*pxcal

    nickname = 'BG-med-Gauss-RMS-6'
    import warnings
    warnings.filterwarnings('ignore')
    # all_img_list_bg_subt = [all_images[i,:,:]-bg_cropp for i in range(len(all_images))]
    
    with concurrent.futures.ProcessPoolExecutor() as pool:
        xfit, yfit,xfit45,yfit45,img_array = zip(*pool.map(func6, all_img_med_bg_subt))
    moments['x_'+nickname] = np.array(rms_extractor(xfit))*pxcal
    moments['y_'+nickname] = np.array(rms_extractor(yfit))*pxcal
    moments['x45_'+nickname] = np.array(rms_extractor(xfit45))*pxcal
    moments['y45_'+nickname] = np.array(rms_extractor(yfit45))*pxcal

    row_has_nan = moments.isnull().any(axis=1)
    moments_cleaned = moments.loc[~row_has_nan].reset_index(drop=True)
    images = all_images[~row_has_nan,:,:]
    data = all_data[~row_has_nan].reset_index(drop=True)

    moments_cleaned.to_hdf(dataset.save_loc + 'moments' + screen_nickname + '.h5','moments')
    np.save(dataset.save_loc + 'all_images_stack_' + screen_nickname + '.npy',images)
    data.to_pickle(dataset.save_loc + 'all_data_stack_' + screen_nickname + '.pkl')

    test = pd.read_hdf(dataset.save_loc + 'moments' + screen_nickname + '.h5')

    unq_cols = []
    for col in list(moments.columns):
        col_trunc = col.split('_')[-1]
        unq_cols.append(col_trunc)
    unq_cols = np.unique(np.array(unq_cols))

    fig,ax=plt.subplots(2,2)
    for col in list(unq_cols):
        data1 = moments['x_'+ col]
        data2 = moments['y_'+ col]
        data3 = moments['x45_'+ col]
        data4 = moments['y45_'+ col]
        
        # fig.suptitle(col)
        ax[0,0].plot(data1,label=col)
        # ax[0,0].set_title('x')
        ax[0,0].set_xlabel('Shot')
        ax[0,0].set_ylabel('xrms (um)')
        ax[0,0].legend()
        
        ax[0,1].plot(data2,label=col)
        # ax[0,1].set_title('x')
        ax[0,1].set_xlabel('Shot')
        ax[0,1].set_ylabel('yrms (um)')
        ax[0,1].legend()
        
        ax[1,0].plot(data3,label=col)
        # ax[1,0].set_title('x')
        ax[1,0].set_xlabel('Shot')
        ax[1,0].set_ylabel('x45rms (um)')
        ax[1,0].legend()
        
        ax[1,1].plot(data4,label=col)
        # ax[1,1].set_title('x')
        ax[1,1].set_xlabel('Shot')
        ax[1,1].set_ylabel('y45rms (um)')
        ax[1,1].legend()
    plt.tight_layout()
    os.makedirs(dataset.save_loc + 'Moment_Plots/', exist_ok=True)
    plt.savefig(dataset.save_loc + 'Moment_Plots/' + 'All_Moments_' + screen_nickname + '.png')
    plt.close() 

    cols = unq_cols

    fig,ax=plt.subplots(2,2,figsize=(10,5))
    xlim=150
    for col in list(cols)[:1]:
        data1 = moments['x_'+ col]
        data2 = moments['y_'+ col]
        data3 = moments['x45_'+ col]
        data4 = moments['y45_'+ col]
        
        # fig.suptitle(col)
        ax[0,0].plot(data1,label=col,alpha=0.5)
        # ax[0,0].set_title('x')
        ax[0,0].set_xlabel('Shot')
        ax[0,0].set_ylabel('xrms (um)')
        ax[0,0].legend()
        ax[0,0].set_ylim([0,2000])
        # ax[0,0].set_xlim([0,xlim])
        
        ax[0,1].plot(data2,label=col,alpha=0.5)
        # ax[0,1].set_title('x')
        ax[0,1].set_xlabel('Shot')
        ax[0,1].set_ylabel('yrms (um)')
        # ax[0,1].legend()
        ax[0,1].set_ylim([0,2000])
        # ax[0,1].set_xlim([0,xlim])
        
        ax[1,0].plot(data3,label=col,alpha=0.5)
        # ax[1,0].set_title('x')
        ax[1,0].set_xlabel('Shot')
        ax[1,0].set_ylabel('x45rms (um)')
        # ax[1,0].legend()
        ax[1,0].set_ylim([0,2000])
        # ax[1,0].set_xlim([0,xlim])
        
        ax[1,1].plot(data4,label=col,alpha=0.5)
        # ax[1,1].set_title('x')
        ax[1,1].set_xlabel('Shot')
        ax[1,1].set_ylabel('y45rms (um)')
        # ax[1,1].legend()
        ax[1,1].set_ylim([0,2000])
        # ax[1,1].set_ylim([0,xlim])
    plt.tight_layout()
    os.makedirs(dataset.save_loc + 'Moment_Plots/', exist_ok=True)
    plt.savefig(dataset.save_loc + 'Moment_Plots/' + 'Selected_Moments_' + screen_nickname + '.png')
    plt.close()


def Generic_VCC_Analysis(dataset,bound_list, idx=None):
    screen_nickname = screen_nickname_finder(dataset.screen,dataset.save_loc)
    if dataset.raw_vcc != 'included':
        assert len(bound_list)==1, "Only one VCC screen supported in this function"
        bounds = bound_list[0]
        if dataset.raw_vcc is not None:
            ## Load VCC Raw Images
            files = glob.glob(dataset.raw_vcc)
            try:
                vcc_list = []
                for file in files:
                    img = np.load(file,allow_pickle=True)
                    img_test = img.item()['CAMR:LT10:900:Image:ArrayData']
                    img_test = np.reshape(img_test,[img.item()['CAMR:LT10:900:Image:ArraySize1_RBV'],img.item()['CAMR:LT10:900:Image:ArraySize0_RBV']])
                    vcc_list.append(img_test)
            except Exception as e:
                print(e)
                vcc_list = []
                for file in files:
                    vcc_list.append(imageio.imread(file))
        else:
            All_DAQ_df = pd.read_pickle(dataset.save_loc + "All_DAQ_Data.pkl")
            vcc_list = []
            for i,row in All_DAQ_df.iterrows():
                try:
                    img = row['VCCF_images']
                    img = np.reshape(img,[int(row['VCCF_nrows']),int(row['VCCF_ncols'])])
                    vcc_list.append(img)
                except:
                    pass
        ## Show raw images
        j = 0
        # for img in vcc_list:
        #     plt.imshow(img,vmax=10)
        #     plt.title(str(j))
        #     plt.show()
        #     if j>=9:
        #         break
        #     j = j+1     
        #choose VCC image
        idx = 0 

        vcc_img = vcc_list[idx]
        #Crop(p) image
        vcc_img = vcc_img[bounds['ystart']:bounds['yend'],bounds['xstart']:bounds['xend']]
        #Make mask
        mask = np.zeros([np.shape(vcc_img)[0],np.shape(vcc_img)[1],3])
        mask = cv2.ellipse(mask,(int(np.shape(vcc_img)[0]/2),int(np.shape(vcc_img)[1]/2)),(int(np.shape(vcc_img)[0]/2),int(np.shape(vcc_img)[1]/2)),0,0,360,(1,0,0), -1)
        mask = mask[:,:,0]  

        ## Manipulate image
        # Get background <br>
        # subtract background<br>
        # threshold<br>
        # mask<br>
        bg = np.mean(vcc_img[mask==0])
        vcc_img = (vcc_img-bg)*mask
        vcc_img[vcc_img<0] = 0
        # plt.imshow(vcc_img)
        vcc_img = np.flipud(np.fliplr(np.flipud(np.fliplr(np.flipud(vcc_img)))))
        np.save(dataset.save_loc + 'VCC.npy',vcc_img)
        im = Image.fromarray(vcc_img)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(dataset.save_loc + "VCC.jpeg")

    elif dataset.raw_vcc == 'included':
    # Load All Data/All Images
        all_images_screen = np.load(dataset.save_loc + 'all_images_stack_' + screen_nickname + '.npy')
        # bg_cropp = np.load(save_loc + 'background_' + screen_nickname + '.npy')
        all_data = pd.read_pickle(dataset.save_loc + 'all_data_stack_' + screen_nickname + '.pkl')
        moments_data = pd.read_hdf(dataset.save_loc + 'moments' + screen_nickname + '.h5')

        #Filter by which ones have a VCC image

        def filter_df(df, col):
            # Keep rows where the value is not None and not a string containing 'NaN'
            idx = (~df[col].isna()) & (df[col] != None) & (~df[col].astype(str).str.contains('NaN'))
            df_new = df[idx]
            return df_new, idx
            
        non_sync_data_screen, idx = filter_df(all_data,'CAMR:LT10:900:Image:ArrayData')
        print(len(non_sync_data_screen))

        # Remove all data rows and all image rows that do not have a VCC image
        if len(non_sync_data_screen)!=len(all_data):
            all_images_screen = all_images_screen[idx,:,:]
            all_data = all_data[idx]
            moments_data = moments_data[idx]

        # Remove badly sized images
        df_sizes = all_data[['CAMR:LT10:900:Image:ArrayData'.split('Image:')[0] + 'Image:ArraySize1_RBV','CAMR:LT10:900:Image:ArrayData'.split('Image:')[0] + 'Image:ArraySize0_RBV']]
        df_sizes.drop_duplicates(inplace=True)

        idx = []
        for i,row in df_sizes.iterrows():
            test = 0
            try:
                (int(row['CAMR:LT10:900:Image:ArrayData'.split('Image:')[0] + 'Image:ArraySize0_RBV']))
            except:
                test = test +1
            try:
                (int(row['CAMR:LT10:900:Image:ArrayData'.split('Image:')[0] + 'Image:ArraySize1_RBV']))
            except:test = test+1
            idx.append(test)
            
        idx=np.array(idx)
        idx = idx>=1
        idx
        df_sizes = df_sizes.loc[~idx]

        # Reshape all images
        def im_stack_from_df(df,col,ncolcol,nrowcol,fliplr=0,rot=0):
            ncols=df[ncolcol]
            nrows=df[nrowcol]
            if np.abs(np.sum(np.diff(ncols)))+np.abs(np.sum(np.diff(nrows)))==0:
                ncol=int(ncols.mean())
                nrow=int(nrows.mean())
                imgs=df[col]
                if rot:
                    imgArray=np.zeros([len(imgs),nrow,ncol])
                else:
                    imgArray=np.zeros([len(imgs),ncol,nrow])
                j=0
                
                for i,row in df.iterrows():
                    # print(i)
                    # print(j)
                    img=imgs.iloc[j]
                    # plt.imshow(img)
                    # print(j)
                    
                    img=np.reshape(img,(ncol,nrow))
                    if rot:
                        img=scipy.ndimage.rotate(img,90)
                    if fliplr:
                        img=np.fliplr(img)
                    imgArray[j,:,:]=img
                    j=j+1
            return imgArray

        # Make VCC Image stack
        imgArray_List = []
        df_list = []
        moments_list = []
        all_images_screen_list = []
        for i, row in df_sizes.iterrows():
            idx1 = all_data['CAMR:LT10:900:Image:ArrayData'.split('Image:')[0] + 'Image:ArraySize1_RBV']==row['CAMR:LT10:900:Image:ArrayData'.split('Image:')[0] + 'Image:ArraySize1_RBV']
            idx2 = all_data['CAMR:LT10:900:Image:ArrayData'.split('Image:')[0] + 'Image:ArraySize0_RBV']==row['CAMR:LT10:900:Image:ArrayData'.split('Image:')[0] + 'Image:ArraySize0_RBV']
            idx = idx1*idx2
            df_subset = all_data.loc[idx]
            moments_subset = moments_data.loc[idx]
            all_images_screen_subset = all_images_screen[idx,:,:]
            all_images_screen_list.append(all_images_screen_subset)
            moments_list.append(moments_subset)
            df_list.append(df_subset)
            imgArray=im_stack_from_df(df_subset,'CAMR:LT10:900:Image:ArrayData','CAMR:LT10:900:Image:ArrayData'.split('Image:')[0] + 'Image:ArraySize1_RBV','CAMR:LT10:900:Image:ArrayData'.split('Image:')[0] + 'Image:ArraySize0_RBV',fliplr=0,rot=0)
            imgArray_List.append(imgArray)
        # vcc_img_stack = im_stack_from_df(non_sync_data_screen,'CAMR:LT10:900:Image:ArrayData','CAMR:LT10:900:Image:ArraySize1_RBV','CAMR:LT10:900:Image:ArraySize0_RBV')
        # Flip all images in the stack vertically (along the second axis, i.e., axis=1)
        # vcc_img_stack = np.flip(vcc_img_stack, axis=1)
        for i in range(len(imgArray_List)):
            img_temp = np.sum(imgArray_List[i],0)
            # plt.figure()
            # plt.imshow(np.sum(imgArray_List[i],0))
            # plt.show()

        image_stack = []
        for i in range(len(bound_list)):
            imArray = imgArray_List[i]
            test = imArray[:,bound_list[i]['ystart']:bound_list[i]['yend'],bound_list[i]['xstart']:bound_list[i]['xend']]
            image_stack.append(test)

        xcs = []
        ycs = []

        for i in range(len(imgArray_List)):
            img_temp = np.sum(image_stack[i],0)
            # plt.figure()
            # plt.imshow(img_temp)
            # plt.show() 
            xfit,yfit,xfit45,yfit45 = IA.Gaussian_Fit_4_Dim(img_temp)
            IA.visualize_projections(img_temp)
            xcs.append(xfit.Center)
            ycs.append(yfit.Center)

        all_images = np.concatenate(image_stack)
        all_images_screen = np.concatenate(all_images_screen_list)
        plt.imshow(np.sum(all_images,0))
        imgs = []
        for i in range(np.shape(all_images)[0]):
            vcc_img = all_images[i,:,:]

            mask = np.zeros([np.shape(vcc_img)[0],np.shape(vcc_img)[1],3])
            mask = cv2.ellipse(mask,(int(np.shape(vcc_img)[0]/2),int(np.shape(vcc_img)[1]/2)),(int(np.shape(vcc_img)[0]/2),int(np.shape(vcc_img)[1]/2)),0,0,360,(1,0,0), -1)
            mask = mask[:,:,0]   
            bg = np.mean(vcc_img[mask==0])
            vcc_img = (vcc_img-bg)*mask
            vcc_img[vcc_img<0] = 0 
            imgs.append(vcc_img)
        all_images = np.array(imgs)

        plt.imshow(np.sum(all_images,0))
        os.makedirs(dataset.save_loc + 'Processed_Images/', exist_ok=True)
        plt.savefig(dataset.save_loc + 'Processed_Images/' + 'VCC_sum_' + screen_nickname + '.png')

        all_data = pd.concat(df_list)
        moments_data = pd.concat(moments_list)
        # plt.scatter(range(len(all_data)),all_data.index)
        # plt.scatter(range(len(moments_data)),moments_data.index)
       
        all_data.reset_index(drop=True,inplace=True)
        moments_data.reset_index(drop=True,inplace=True)

        plt.scatter(range(len(all_data)),all_data.index)
        plt.scatter(range(len(moments_data)),moments_data.index)
        plt.savefig(dataset.save_loc + 'Processed_Images/' + 'VCC_index_scatter_' + screen_nickname + '.png')
        plt.close()
        rot_image_stack = np.flip(all_images,axis=1)
        np.save(dataset.save_loc + 'VCC_stack_' + screen_nickname + '.npy',rot_image_stack)
        np.save(dataset.save_loc + 'total_images_stack_' + screen_nickname + '.npy',all_images_screen)
        all_data.to_pickle(dataset.save_loc + 'total_data_stack_' + screen_nickname + '.pkl')
        moments_data.to_hdf(dataset.save_loc + 'moments' + screen_nickname + '.h5','moments')



        # Do Analysis on VCC images
        # plt.imshow(np.mean(vcc_img_stack, axis=0))

        # Save VCC image stack
        # np.save(save_loc + 'vcc_image_stack_' + screen_nickname + '.npy', vcc_img_stack)



__all__ = [
    # Analysis parameters
    "AnalysisParameters",
    "validate_dataset",
    # Utilities
    "screen_nickname_finder",
    "second_parent",
    "fix_path",
    "read_hdf",
    "Merge",
    "im_stack_from_df",
    "get_border_mean",
    "get_proj_mean",
    "get_x_y_proj_mean",
    "safe_extract",
    "rms_extractor",
    "zscore",
    "outlier_detect",
    # Pipeline steps
    "Generic_Preprocessing",
    "Generic_DAQ_Preprocessing",
    "Generic_Data_Processing",
    "Generic_Image_Processing",
    "Background_Treatment",
    "filter_beams",
    "Generic_Moment_Calculation",
    "Generic_VCC_Analysis",
]
