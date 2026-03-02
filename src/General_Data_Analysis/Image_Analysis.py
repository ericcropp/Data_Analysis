import scipy.optimize 
import numpy as np
from PIL import Image,ImageFilter
from matplotlib import pyplot as plt
import copy
import cv2
from scipy import stats
from scipy.ndimage import median_filter

class GaussianParams:
    def __init__(self,*args):
        """
        Defines the gaussian parameters: center, amplitude, standard deviation and baseline.  The inputs can be specified as a dict
        with the following keys "Center", "Amplitude", "Sigma" and "Baseline", or as a list in that order.

        """
        try:
            if type(args[0])==dict:
                self.Center=args[0]['Center']
                self.Amplitude=args[0]['Amplitude']
                self.Sigma=args[0]['Sigma']
                self.Baseline=args[0]['Baseline']
            elif isinstance(args[0],list):
                self.Center=args[0][0]
                self.Amplitude=args[0][1]
                self.Sigma=args[0][2]
                self.Baseline=args[0][3]
            else:
                raise ValueError("The inputs are not properly specified")
        except:
            raise ValueError("Improperly Formatted Input")
    def evaluate_gaussian(self,x):
        """
        Method to evaluate a gaussian at each of the values provided in x
        Returns the values of the Gaussian evaluated at x.
        
        Argument:
        x -- np.array of values at which to evaluate the Gausssian

        """
        
        #Check inputs
        try:
            x = np.array(x)
        except:
            raise ValueError("x cannot be converted to a numpy array.")

        x = np.squeeze(x)
        assert len(np.shape(x)) == 1, "The shape of the image provided is not 1-D"
        
        #Define a Gaussian
        try:
            return self.Amplitude * np.exp(-(x - self.Center) ** 2 / (2 * self.Sigma ** 2)) + self.Baseline
        except:
            raise ValueError("Gaussian distribution not defined parameters")
    def print_params(self):
        return [self.Center, self.Amplitude, self.Sigma, self.Baseline]


def np_array_dim_checker(img, dim=2):
    """
    This function checks to make sure that the input is indeed a N-D numpy array.  Throws an error if not.  Returns image in a guaranteed numpy array
    
    Argument:
    img -- input to be tested
    dim -- optional int representing the dimensionalty of the desired numpy array.
    
    """
    # Check inputs 
    assert isinstance(dim,int)==True, "Dimension is not an integer!"
    
    # Convert img to np array 
    try:
        img=np.array(img)
    except:
        raise ValueError("img cannot be converted to a numpy array.")
    
    # Check dimensions
    img=np.squeeze(img)
    assert len(np.shape(img))==dim, "The shape of the image provided is not " + str(dim)+"-D"
    return img

def visualize_projections(img1,pixcal=1, units='pixels'):
    """
    Visualizes the Gaussian fits of an image, along with the projections
    Does not return any variables

    Argument:
    img1 -- 2-D np.array: image to bevisualized
    pxcal -- float: a conversion from pixels to physical units (Optional; default: 1)
    units -- string: a label for the plot that describes the physical units of the plot (Optional; default: 'pixels')

    """           
    # Check inputs
    img1 = np_array_dim_checker(img1)
    try: 
        pixcal = float(pixcal)
    except:
        raise ValueError("pixcal could not be converted to float")
    assert isinstance(units,str),"Units not a string"
    
    # x and y projections and fits
    xproj=np.sum(img1,0)
    yproj=np.sum(img1,1)
    xfit=fit_gauss(xproj)
    yfit=fit_gauss(yproj)

    # get background
    bg=(xfit.Baseline/np.shape(img1)[0]+yfit.Baseline/np.shape(img1)[1])/2
    # rotate image and fill background, get rotated projections
    img3=imrotate45(img1,bg)
    xproj45=np.sum(img3,0)
    yproj45=np.sum(img3,1)
    
    # rotated fits
    xfit45=fit_gauss(xproj45)
    yfit45=fit_gauss(yproj45)

    # Plot images
    plt.figure()
    fig,ax=plt.subplots(2,2)

    plt_x_axis=(range(len(xproj))-xfit.Center)*pixcal
    ax[0,0].plot(plt_x_axis,xproj)
    ax[0,0].plot(plt_x_axis,xfit.evaluate_gaussian(range(len(xproj))))
    ax[0,0].set_title('x')
    ax[0,0].set_xlabel(units)
    ax[0,0].set_ylabel('Int. Intensity')

    plt_x_axis=(range(len(yproj))-yfit.Center)*pixcal
    ax[0,1].plot(plt_x_axis,yproj)
    ax[0,1].plot(plt_x_axis,yfit.evaluate_gaussian(range(len(yproj))))
    ax[0,1].set_title('y')
    ax[0,1].set_xlabel(units)
    ax[0,1].set_ylabel('Int. Intensity')

    plt_x_axis=(range(len(xproj45))-xfit45.Center)*pixcal
    ax[1,0].plot(plt_x_axis,xproj45)
    ax[1,0].plot(plt_x_axis,xfit45.evaluate_gaussian(range(len(xproj45))))
    ax[1,0].set_title('x45')
    ax[1,0].set_xlabel(units)
    ax[1,0].set_ylabel('Int. Intensity')

    plt_x_axis=(range(len(yproj45))-yfit45.Center)*pixcal
    ax[1,1].plot(plt_x_axis,yproj45)
    ax[1,1].plot(plt_x_axis,yfit45.evaluate_gaussian(range(len(yproj45))))
    ax[1,1].set_title('y45')
    ax[1,1].set_xlabel(units)
    ax[1,1].set_ylabel('Int. Intensity')

    plt.tight_layout()
    plt.show()

               
               

def eval_gauss_baseline(p,x):  
    """
    This function returns the gaussian y values given x values and a set of gaussian parameters.
    
    Argument:
    p -- a 1-D np.array of parameters, [Center, Sigma, Amplitude, Baseline], all floats
    x -- a 1-D np.array of x values at which to evelauate p
    
    """
    # check inputs
    x = np_array_dim_checker(x,dim=1)
    p = np_array_dim_checker(p,dim=1)
    # print(p)
    zx = p[2]*np.exp(-0.5*(x-p[0])**2./(p[1]**2)) +p[3]
    
    return zx

def penalty_func(p,v,x):
    """
    Define a penalty function, returning the L2 norm between the proposed gaussian an dthe data.
    Returns the penalty function value
    
    Argument:
    p -- a 1-D np.array of parameters, [Center, Sigma, Amplitude, Baseline], all floats
    x -- a 1-D np.array of x values at which to evelauate p
    v -- a 1-D np.array of y values to compare with the output of the gaussian evaluated at x
    """
    # check inputs
    x = np_array_dim_checker(x,dim=1)
    v = np_array_dim_checker(v,dim=1)
    p = np_array_dim_checker(p,dim=1)
    assert len(x)==len(v), "lengths of the x and v are not the same!"
    
    # evaluate and subtract, return
    zx = eval_gauss_baseline(p,x) - v
    z = np.sum(zx**2)
    return z

def init_guess(xproj,lengthscale=25):
    """
    This function returns a GaussianParams object with intial guess Gaussian parameters given an image projection.
    
    Argument:
    
    xproj -- a 1-D numpy array representing the projection of an image
    lengthscale -- optional float a guess for the rms of the beam using the length of the image divided by this lengthscale
    
    """
    # check inputs
    xproj = np_array_dim_checker(xproj,dim=1)
    try:
        lengthscale = float(lengthscale)
    except: 
        raise ValueError("lengthscale could not be converted to float")
    assert isinstance(lengthscale,float)==True, "lengthscale is not float"
    
    #Estimate baseline from edge of projection
    base=np.mean(xproj[0:10])
    
    # Estimate center from location of maximum
    cx=np.argmax(xproj)
    
    # Estimate amplitude as maximum of xproj
    tempx=np.max(xproj)
    
    # Estimate sigma using lengthscale
    sx=len(xproj)/lengthscale
    
    # Return a GaussianParams object
    return GaussianParams([cx,tempx-base,sx,base])


def fit_gauss(xproj):
    """
    #returns a GaussianParams object of the parameters of a 1D gaussian fit with baseline
    
    Argument: 
    xproj -- 1-D numpy array representing the projection of an image 
    """
    # check inputs
    xproj = np_array_dim_checker(xproj,dim=1)
    
    # Get initial guess
    guess=init_guess(xproj)
    
    # Define x for fit
    jx=np.arange(0,len(xproj),1)
    
    # Extract list for gaussian evaluator
    p=np.array([guess.Center,guess.Sigma,guess.Amplitude,guess.Baseline])
    
    # Fit!
    res = scipy.optimize.minimize(penalty_func, p, args=(xproj,jx), method='nelder-mead', options={'xatol': 1e-5, 'disp': False})
    
    # Return Gaussian Params object
    out=GaussianParams([res.x[0],res.x[2],res.x[1],res.x[3]])
    return out

def imrotate45(img,bg):
    """
    This function rotates and returns a 45 degree rotated image.  
    This function expands the rectangular image such that the 45 degree rotated image fully fits within this box.  
    The new areas are filled with a uniform background of pixel value given by the inputs.  
    
    Argument:
    img -- 2-D numpy array 
    bg -- float or int background pixel 
    """
    #Check inputs
    try: bg = int(bg)
    except: raise ValueError("Background pixel could not be converted to integer!")
    img = np_array_dim_checker(img)
    
    # Use PIL to rotate image and return as np array
    img1=Image.fromarray(img)
    img2=img1.rotate(45,fillcolor=int(bg),expand=True)
    img3=np.array(img2)
    # print(img3.shape)
    return img3

# Runs through 4 image fits for a single image
def Gaussian_Fit_4_Dim(img1):
    """
    This function fits the 1-D projections of an image in x, y, x45 and y45, where the latter two are the 
    x and y projections after rotating the image 45 degrees CCW.  This returns 4 GaussianParams objects
    
    Argument: 
    img1 -- a 2-D numpy array 
    """
    # Check inputs
    img1 = np_array_dim_checker(img1)
    
    # 1-D fits, xy
    xproj = np.sum(img1, 0)
    yproj = np.sum(img1, 1)
    xfit = fit_gauss(xproj)
    yfit = fit_gauss(yproj)

    # Compute background and rotate image
    bg = (xfit.Baseline / np.shape(img1)[0] + yfit.Baseline / np.shape(img1)[1]) / 2
    img3 = imrotate45(img1, bg)

    # Compute 1-D fits in rotated frame
    xproj45 = np.sum(img3, 0)
    yproj45 = np.sum(img3, 1)
    xfit45 = fit_gauss(xproj45)
    yfit45 = fit_gauss(yproj45)

    # Return
    return xfit, yfit, xfit45, yfit45

def bg_thresh(bg):
    """
    remove outliers from background image
    """
    
    bg_flat =np.reshape(bg,-1)
    z = np.abs(stats.zscore(bg_flat))
    idx = z>3
    bg_outlier = bg_flat[~idx]
    # plt.hist(bg_outlier,bins=10)

    return np.std(bg_outlier)

def image_analysis_5(img,return_images=False,initial_Gauss=False):
    img = median_filter(img, size=3)
    if initial_Gauss==False:
        xfit, yfit,xfit45,yfit45=RMS_Image_Analysis(img)
    else:
        xfit, yfit,xfit45,yfit45=Gaussian_Fit_4_Dim(img)
    img2,bg1 = ellipse_crop_v2(img,sigmaThresh=3,xfit=xfit,yfit=yfit,x45fit=xfit45,y45fit=yfit45,return_bg=True)
    xfit, yfit,xfit45,yfit45 = RMS_Image_Analysis(img2)
    if return_images:
        return xfit, yfit,xfit45,yfit45,img2
    else:
        return xfit, yfit,xfit45,yfit45

def image_analysis_6(img,return_images=False,initial_Gauss=False):
    img = median_filter(img, size=3)
    if initial_Gauss==False:
        xfit, yfit,xfit45,yfit45=RMS_Image_Analysis(img)
    else:
        xfit, yfit,xfit45,yfit45=Gaussian_Fit_4_Dim(img)
    img2,bg1 = ellipse_crop_v3(img,sigmaThresh=3,xfit=xfit,yfit=yfit,x45fit=xfit45,y45fit=yfit45,return_bg=True)
    xfit, yfit,xfit45,yfit45 = RMS_Image_Analysis(img2)
    if return_images:
        return xfit, yfit,xfit45,yfit45,img2
    else:
        return xfit, yfit,xfit45,yfit45


def RMS_img_Analysis_thresh(img,n_sigma=0):
    """
    Takes background subtracted image
    Does ellipse cropping
    Any background pixels are assumed to be new background
    subtract (mean + n_sigma) of background
    threshold
    do RMS analysis
    """
    img_temp=copy.deepcopy(img)
    img_temp[img_temp<0]=0
    xfit, yfit,xfit45,yfit45=RMS_Image_Analysis(img_temp)
    img2=image_cropp_center(img,3,xfit=xfit,yfit=yfit)
    img2,bg1 = ellipse_crop_v2(img2,sigmaThresh=3,return_bg=True)
    bg1 = np.reshape(bg1,-1)
    if len(bg1)>1:
        
        thresh = n_sigma*np.abs(bg_thresh(bg1))+np.mean(bg1)
    elif len(bg1)==1:
        thresh = np.mean(bg1)
    else:
        thresh = 0
        
    img2 = img2-thresh
    img2[img2<0]=0
    xfit, yfit,xfit45,yfit45 = RMS_Image_Analysis(img2)
    return xfit, yfit,xfit45,yfit45

def RMS_Calc(data):
    """
    Calculates the RMS of a 1-D distribution

    Returns a GaussianParams object that with the calcuated sigma and centroid.  The baseline is set to zero and assumed to be zero throughout the calculation.

    Argument:
    data -- 1-D np.array: data on which to calculate RMS

    """
    #Check inputs
    data = np_array_dim_checker(data,dim=1)
    
    # Get coordinates
    x=np.linspace(0,len(data)-1,len(data))
    
    # Threshold negative values
    # data[data<0]=0
    
    #Calculate centroid
    cent=np.matmul(x,data)/np.sum(data)
    
    #Calculate RMS
    rms_calc=np.sqrt(np.matmul(data,(x-cent)**2)/np.sum(data))
    
    # Calculate RMS
    peak=np.interp(cent,x,data)
    
    # Return
    params=GaussianParams([cent,peak,rms_calc,0])
    return params 

def RMS_Image_Analysis(img):
    """
    Calculates the of the beam from an image in each dimension, and the two rotated dimensions

    Returns 4 GaussianParams objects.

    Argument:
    img -- 2-D np.array: data on which to calculate RMS in each dimension.  (Should be background subtracted for this to work well).

    """
    #Check input
    img = np_array_dim_checker(img)
    
    # Get x and y projections and do RMS calculation
    xproj=np.sum(img,0)
    yproj=np.sum(img,1)
    xfit=RMS_Calc(xproj)
    yfit=RMS_Calc(yproj)
    
    # Get the baseline and add it to the corners when rotating image 45 degrees
    bg=(xfit.Baseline/np.shape(img)[0]+yfit.Baseline/np.shape(img)[1])/2
    img3=imrotate45(img,bg)
    
    # Rotated projections and calculations
    xproj45=np.sum(img3,0)
    yproj45=np.sum(img3,1)
    xfit45=RMS_Calc(xproj45)
    yfit45=RMS_Calc(yproj45)
    
    # Return data
    return xfit, yfit,xfit45,yfit45

def image_cropp_center(img,num_std,xfit=None,yfit=None):
    
    """
    Crop(p)s the beam such that the beam is to within a pixel of the center of the image.  This is a rectangular cropping, so only xfit and yfit are taken into account.  This may fail for beams with a large ellipticity and the semi-major axes are not aligned with cropping axes.

    Returns a 2-D numpy array of a cropped and centered image

    Argument:
    img2 -- 2-D np.array: image to be centered
    num_std -- float: the number of stds to crop to
    xfit -- GaussianParams object for the fit in x (Optional), will be fit if not provided, but this is a relatively costly evaluation
    yfit -- GaussianParams object for the fit in y (Optional), will be fit if not provided, but this is a relatively costly evaluation

    """

    #check inputs
    if isinstance(xfit,GaussianParams): pass
    else: xfit=None
    
    if isinstance(yfit,GaussianParams): pass
    else: yfit=None
    
    img2 = np_array_dim_checker(img)
               
    #fit if the fits are not provided    
    if xfit==None or yfit==None:
        xfit, yfit,xfit45,yfit45=Gaussian_Fit_4_Dim(img)

    # Define size of rectangle to crop
    sz_x=num_std*xfit.Sigma
    sz_y=num_std*yfit.Sigma
               
    #find max edges by distance from center
    miny=max(int(np.floor(yfit.Center-sz_y)),0)
    maxy=min(int(np.ceil(yfit.Center+sz_y)),np.shape(img)[0])

    minx=max(int(np.floor(xfit.Center-sz_x)),0)
    maxx=min(int(np.ceil(xfit.Center+sz_x)),np.shape(img)[1])
    
    #Crop(p)
    img2=img[miny:maxy,minx:maxx]

    return img2



def ellipse_crop_v3(img2,sigmaThresh=5,xfit=None,yfit=None,x45fit=None,y45fit=None,return_bg=False):
    """
    
    This is much faster due to the OpenCV implementation of a mask
    
    Argument:
    img2 -- 2-D numpy array, image
    sigmaThesh -- float of the number of sigmas outside of which to set zero
    xfit -- GaussianParams object that describes the beam (x dimension). If any GaussianParams object is not speficied, then a simple RMS calculation will be done on the image projections.
    yfit -- GaussianParams object that describes the beam (y dimension). If any GaussianParams object is not speficied, then a simple RMS calculation will be done on the image projections.
    x45fit -- GaussianParams object that describes the beam (x45 dimension). If any GaussianParams object is not speficied, then a simple RMS calculation will be done on the image projections.
    y45fit -- GaussianParams object that describes the beam (y45 dimension). If any GaussianParams object is not speficied, then a simple RMS calculation will be done on the image projections.
    return_bg -- bool to indicate whether to return a list of pixels outside the beam (nominally used to determine a noise floor at which to threshold)
    
    """
    #Check inputs
    if isinstance(xfit,GaussianParams): pass
    else: xfit = None
    
    if isinstance(y45fit,GaussianParams): pass
    else: y45fit = None

    if isinstance(x45fit,GaussianParams): pass
    else: x45fit = None
    
    if isinstance(yfit,GaussianParams): pass
    else: yfit = None

    img2 = np_array_dim_checker(img2)
    
    
    # Copy image so we can edit it.
    img = copy.deepcopy(img2)

    # If moments are not specified, then calculate them
    if xfit == None or yfit == None or x45fit == None or y45fit == None:
         xfit, yfit, x45fit, y45fit = RMS_Image_Analysis(img)

    # Get second order moments from RMS quantities.  
    # Note that there are two ways to calculate the sigma_xy term.  Here, we just assume that it is the average of the two.  
    # See F. Cropp Ph.D. thesis (UCLA, 2023 for details)                
    sigma_xx = xfit.Sigma ** 2
    sigma_yy = yfit.Sigma ** 2
    sigma_uu = x45fit.Sigma ** 2
    sigma_vv = y45fit.Sigma ** 2

    sigma_xy1 = -0.5 * (2 * sigma_uu - sigma_xx - sigma_yy)
    sigma_xy2 = 0.5 * (2 * sigma_vv - sigma_xx - sigma_yy)

    sigma_xy = (sigma_xy1 + sigma_xy2) / 2

    # Determine angle and rotate ellipse to crop
    phi = -0.5 * np.arctan(2 * sigma_xy / (sigma_xx - sigma_yy))

    covmat = np.array([[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]])
    rotmat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    covmat_rot = rotmat @ covmat @ rotmat.T

    b = sigmaThresh ** 2 * covmat_rot[0, 0]
    a = sigmaThresh ** 2 * covmat_rot[1, 1]

    # sigma_uu_rot = np.sin(phi) * (np.sin(phi) * sigma_xx - np.cos(phi) * sigma_xy) + np.cos(phi) * (np.cos(phi) * sigma_yy - np.sin(phi) * sigma_xy)
    # sigma_vv_rot = np.cos(phi) * (np.cos(phi) * sigma_xx + np.sin(phi) * sigma_xy) + np.sin(phi) * (np.sin(phi) * sigma_yy + np.cos(phi) * sigma_xy)
    # a = sigmaThresh ** 2 * sigma_uu_rot
    # b = sigmaThresh ** 2 * sigma_vv_rot

    # Make elliptical mask
    center = (int(xfit.Center), int(yfit.Center))
    axes = (int(np.sqrt(b)), int(np.sqrt(a)))

    angle = phi * 180 / np.pi
    # print(angle)
    thickness = -1
    shift = (xfit.Center - center[0], yfit.Center - center[1])

    mask = np.zeros([np.shape(img)[0], np.shape(img)[1], 3])
    mask = cv2.ellipse(mask, center, axes, angle, 0, 360, (1, 0, 0), -1)
    mask = mask[:, :, 0]

    # Crop(p) outside the ellipse
    if return_bg:
        bg = img[mask == 0]
    img = img * mask

    # Return requested quantities
    if return_bg:
        return img, bg
    else:
        return img


def ellipse_crop_v2(img2, sigmaThresh=5, xfit=None, yfit=None, x45fit=None, y45fit=None, return_bg=False):
    """
    
    This is much faster due to the OpenCV implementation of a mask
    
    Argument:
    img2 -- 2-D numpy array, image
    sigmaThesh -- float of the number of sigmas outside of which to set zero
    xfit -- GaussianParams object that describes the beam (x dimension). If any GaussianParams object is not speficied, then a simple RMS calculation will be done on the image projections.
    yfit -- GaussianParams object that describes the beam (y dimension). If any GaussianParams object is not speficied, then a simple RMS calculation will be done on the image projections.
    x45fit -- GaussianParams object that describes the beam (x45 dimension). If any GaussianParams object is not speficied, then a simple RMS calculation will be done on the image projections.
    y45fit -- GaussianParams object that describes the beam (y45 dimension). If any GaussianParams object is not speficied, then a simple RMS calculation will be done on the image projections.
    return_bg -- bool to indicate whether to return a list of pixels outside the beam (nominally used to determine a noise floor at which to threshold)
    
    """
    #Check inputs
    if isinstance(xfit,GaussianParams): pass
    else: xfit=None
    
    if isinstance(y45fit,GaussianParams): pass
    else: y45fit=None
    
    if isinstance(x45fit,GaussianParams): pass
    else: x45fit=None
    
    if isinstance(yfit,GaussianParams): pass
    else: yfit=None
    
    img2 = np_array_dim_checker(img2)
    
    
    # Copy image so we can edit it.
    img=copy.deepcopy(img2)
    
    # If moments are not specified, then calculate them
    if xfit==None or yfit==None or x45fit==None or y45fit==None:
         xfit, yfit,x45fit,y45fit=RMS_Image_Analysis(img)
            
    # Get second order moments from RMS quantities.  
    # Note that there are two ways to calculate the sigma_xy term.  Here, we just assume that it is the average of the two.  
    # See F. Cropp Ph.D. thesis (UCLA, 2023 for details)                
    sigma_xx=xfit.Sigma**2
    sigma_yy=yfit.Sigma**2
    sigma_uu=x45fit.Sigma**2
    sigma_vv=y45fit.Sigma**2
    
    sigma_xy1=-0.5*(2*sigma_uu-sigma_xx-sigma_yy)
    sigma_xy2=0.5*(2*sigma_vv-sigma_xx-sigma_yy)
    
    sigma_xy=(sigma_xy1+sigma_xy2)/2
    
    # Determine angle and rotate ellipse to crop
    phi=-0.5*np.arctan(2*sigma_xy/(sigma_xx-sigma_yy))
    
    sigma_uu_rot=np.sin(phi)*(np.sin(phi)*sigma_xx - np.cos(phi)*sigma_xy) + np.cos(phi)*(np.cos(phi)*sigma_yy - np.sin(phi)*sigma_xy);
    sigma_vv_rot=np.cos(phi)*(np.cos(phi)*sigma_xx + np.sin(phi)*sigma_xy) + np.sin(phi)*(np.sin(phi)*sigma_yy + np.cos(phi)*sigma_xy);
    a=sigmaThresh**2*sigma_uu_rot;
    b=sigmaThresh**2*sigma_vv_rot;

    # Make elliptical mask
    center=(int(xfit.Center),int(yfit.Center))
    axes=(int(np.sqrt(b)),int(np.sqrt(a)))
    
    angle=phi*180/np.pi
    # print(angle)
    thickness=-1
    shift=(xfit.Center-center[0],yfit.Center-center[1])
    
    mask=np.zeros([np.shape(img)[0],np.shape(img)[1],3])
    mask=cv2.ellipse(mask,center,axes,angle,0,360,(1,0,0), -1)
    mask=mask[:,:,0]    
    
    # Crop(p) ouutside the ellipse
    if return_bg:
        bg=img[mask==0]
    img=img*mask

    # Return requested quantities
    if return_bg:
        return img,bg
    else:
        return img


__all__ = [
    # Core classes
    "GaussianParams",
    # Fitting
    "eval_gauss_baseline",
    "penalty_func",
    "init_guess",
    "fit_gauss",
    "Gaussian_Fit_4_Dim",
    # Image processing
    "imrotate45",
    "image_cropp_center",
    "ellipse_crop_v3",
    "bg_thresh",
    # RMS / moment analysis
    "RMS_Calc",
    "RMS_Image_Analysis",
    "RMS_img_Analysis_thresh",
    # Higher-level analysis
    "image_analysis_5",
    "image_analysis_6",
    # Visualisation
    "visualize_projections",
    "np_array_dim_checker",
]
