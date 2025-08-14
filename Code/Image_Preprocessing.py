import numpy as np
import cv2
import math
import scipy
from scipy import signal
from scipy import ndimage


''''''
# This functins contains all preprocessing functions neeed for fingerpring enchancment  #

def Normalize_image (Image):
    #Normalize the image by subtracting the mean and divide by image std
    #inputs img ,mean ,standard deviation

        Normalized_image = (Image - np.mean(Image)) / (np.std(Image))
        return (Normalized_image)



def Ridge_Segementaion(img,ridge_blocksize,Ridge_threshold ):
    ''''''
    # inputs : imge ,block size ,thrshold to identify the ridge 

    # ourputs  : 1- Renormaized_image : image  inwhich the ridges is normalized 
    # 2- mask : to identify the ridge regions as follows 0->not ridge ,1->ridge region

    # Workflow : the function will divide the image into blocks f sizes ridge_blocksize*ridge_blocksize
    # for each region ,it will calculate the std abd cmpare it to the Ridge_threshold
    # if it is > threshld -> part of the image (ridge part) ,vice versa

    ''''''
    Renormaized_image =[]
    Mask=[]
    r, c = img.shape
    Norm = Normalize_image(img)  # normalise to get zero mean and unit standard deviation


    #getting the new indiceis for dividing the image
    ''''''
   
    r_new = np.int(ridge_blocksize * np.ceil((np.float(r)) / (np.float(ridge_blocksize)))) 
    col_new = np.int(ridge_blocksize * np.ceil((np.float(c)) / (np.float(ridge_blocksize))))

    ''''''
    ##padding the immage with zeros
    padded_img = np.zeros((r_new, col_new))
    image_new = np.zeros((r_new, col_new))
    padded_img[0:r][:, 0:c] = Norm

    ##dividing into blocks of size ridge_blocksize
    for i in range(0, r_new, ridge_blocksize):
        for j in range(0, col_new,ridge_blocksize):
            block = padded_img[i:i + ridge_blocksize][:, j:j + ridge_blocksize]

            image_new[i:i + ridge_blocksize][:, j:j + ridge_blocksize] = np.std(block) * np.ones(block.shape)

    image_new = image_new[0:r][:, 0:c]
    ##check the trheshold
    Mask = image_new > Ridge_threshold

    ##Normalize the mask   
    Renormaized_image = (Norm - np.mean(Norm[Mask])) / (np.std(Norm[Mask]))
    return Renormaized_image ,Mask





def Ridge_orientain(Renormaized_image,sigma_gradient,sigma_block,sigma_smooth):
    
    ##inputs : Renormaized_image : image  inwhich the ridges is normalized 
    # sigma_gradient,sigma_block,sigma_smooth different sigma for different purposes of different filters : derivative of gaussian 
    # smoothing 
    #Calculate image gradients.
    #gaussian size ~ 6 or 5 * sigma value
    sze = 6*sigma_gradient  
    #odd size kernel
    if sze % 2 == 0:
        sze = sze+1

    gaussian = cv2.getGaussianKernel(np.int(sze),sigma_gradient)
    Kernel = gaussian * gaussian.T
    fy,fx = np.gradient(Kernel)                     #Gradients
    Gx = signal.convolve2d(Renormaized_image, fx, mode='same')
    Gy = signal.convolve2d(Renormaized_image, fy, mode='same')
    Gxx = np.power(Gx,2)
    Gyy = np.power(Gy,2)
    Gxy = Gx*Gy
    sze = np.fix(6*sigma_block)
    gaussian = cv2.getGaussianKernel(np.int(sze), sigma_block)
    Kernel = gaussian * gaussian.T

    Gxx = ndimage.convolve(Gxx,Kernel)
    Gyy = ndimage.convolve(Gyy,Kernel)
    Gxy = 2*ndimage.convolve(Gxy,Kernel)

    ##Estimating the orientiaon of the Neighbourd 
    Divider = np.sqrt(np.power(Gxy,2) + np.power((Gxx - Gyy),2)) + np.finfo(float).eps

    sin2theta = Gxy/Divider                   #sines of dubled the angels
    cos2theta = (Gxx-Gyy)/Divider             #cosines of dubled the angels
    ## smoothing to remove the noise from the noisy directional map 
    if sigma_smooth:
        sze = 6*sigma_smooth
    if sze % 2 == 0:
        sze = sze+1
    gaussian = cv2.getGaussianKernel(np.int(sze), sigma_smooth)
    Kernel = gaussian * gaussian.T
    cos2theta = ndimage.convolve(cos2theta,Kernel)                  
    sin2theta = ndimage.convolve(sin2theta,Kernel)                 
    image_orientian = np.pi/2 + np.arctan2(sin2theta,cos2theta)/2

    return image_orientian

def block_frequency(img_block,block_orientaion,winsize,minWlength,MaxWlength):
       ##this function estimates the mean freqency of a block 
       ##Achiieved by averaging the sines and cosines of doubled angels
        rows, cols = np.shape(img_block)
        cosorient = np.mean(np.cos(2 * block_orientaion))
        sinorient = np.mean(np.sin(2 * block_orientaion))
        orient = math.atan2(sinorient, cosorient) / 2
        Rotating_image = scipy.ndimage.rotate(img_block, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3,
                                     mode='nearest')

        # Cropping the image to avoid invalid regions 

        Cropsize = int(np.fix(rows / np.sqrt(2)))
        offset = int(np.fix((rows - Cropsize) / 2))
        Rotating_image = Rotating_image[offset:offset + Cropsize][:, offset:offset + Cropsize]

        ##Summation of the values to 
        proj = np.sum(Rotating_image, axis=0)
        dilation = scipy.ndimage.grey_dilation(proj,winsize, structure=np.ones(winsize))
        temp = np.abs(dilation - proj)
        peak_thresh = 2
        maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
        maxind = np.where(maxpts)
        rows_maxind, cols_maxind = np.shape(maxind)

        if (cols_maxind < 2):
            
            return(np.zeros(img_block.shape))
        else:
           
            NoOfPeaks = cols_maxind
            waveLength = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (NoOfPeaks - 1)
            if waveLength >= minWlength and waveLength <= MaxWlength:
                return(1 / np.double(waveLength) * np.ones(img_block.shape))
            else:
                return(np.zeros(img_block.shape))






def Ridge_frequencies (Renormaized_image,oriented_image,Mask,block_size,block_window,minWlength,MaxWlength):
        #print("ridge freq")
        Mean_fr=[]
        median_freq=[]
        Ridge_freq=[]
        rows, cols = Renormaized_image.shape
        freq = np.zeros((rows, cols))

        for r in range(0, rows - block_size, block_size):
            for c in range(0, cols -block_size,block_size):
                blkim = Renormaized_image[r:r + block_size][:, c:c + block_size]
                #print(blkim)
                blkor = oriented_image[r:r +block_size][:, c:c + block_size]

                freq[r:r + block_size][:, c:c + block_size] = block_frequency(blkim, blkor,block_window,minWlength,MaxWlength)
      
        Ridge_freq  = freq * Mask
        #print(Ridge_freq)
        freq_1d = np.reshape(Ridge_freq , (1, rows * cols))
        ind = np.where(freq_1d > 0)

        ind = np.array(ind)
        ind = ind[1, :]

        non_zero_elems_in_freq = freq_1d[0][ind]

        Mean_fr = np.mean(non_zero_elems_in_freq)
        median_freq = np.median(non_zero_elems_in_freq)  # does not work properly

        Ridge_freq = Mean_fr * Mask
        return  Ridge_freq



def gaborfilter (Renormaized_image,Ridge_freq,image_orientian) :
        ##This function  to implemnt gabor filter 
        ##it taekes normalized image  ,frequency maps ,directional map 
        ##Return Filtered image 
        kx=ky=.65
        angle=3
        threshold=-3
        binary_image=[]
        im = np.double(Renormaized_image)
        rows, cols = im.shape
        newim = np.zeros((rows, cols))

        freq_1d = np.reshape(Ridge_freq, (1, rows * cols))
        ind = np.where(freq_1d > 0)

        ind = np.array(ind)
        ind = ind[1, :]

        # rounding t nearst .01
        freq_values = freq_1d[0][ind]
        freq_values = np.double(np.round((freq_values * 100))) / 100

        Unique_frequiencies = np.unique(freq_values)
        ###Generating the filter based on the frequencies 
        sigmax = 1 / Unique_frequiencies[0] * kx
        sigmay = 1 / Unique_frequiencies[0] * ky
        size = np.int(np.round(3 * np.max([sigmax, sigmay])))
        x, y = np.meshgrid(np.linspace(-size, size, (2 * size + 1)), np.linspace(-size, size, (2 * size + 1)))
        Original_filter = np.exp(-(((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay)))) * np.cos(
            2 * np.pi * Unique_frequiencies[0] * x)        ###Orginal Filter

        filt_rows, filt_cols = Original_filter.shape

        angleRange = np.int(180 / angle)

        gabor_filter = np.array(np.zeros((angleRange, filt_rows, filt_cols)))

        for o in range(0, angleRange):
            ##Generating filer but rotated 

            rot_filt = scipy.ndimage.rotate(Original_filter, -(o * angle + 90), reshape=False)
            gabor_filter[o] = rot_filt

        # find the max points greater than  Max_size 
        Max_size = int(size)
        temp = Ridge_freq> 0
        validr, validc = np.where(temp)
        t1 = validr > Max_size
        t2 = validr < rows - Max_size
        t3 = validc > Max_size
        t4 = validc < cols - Max_size

        final_temp = t1 & t2 & t3 & t4

        finalind = np.where(final_temp)

      
        maxorientindex = np.round(180 / angle)
        orient_index = np.round(image_orientian/ np.pi * 180 / angle)
        ###Filtering
        for i in range(0, rows):
            for j in range(0, cols):
                if (orient_index[i][j] < 1):
                    orient_index[i][j] = orient_index[i][j] + maxorientindex
                if (orient_index[i][j] > maxorientindex):
                    orient_index[i][j] = orient_index[i][j] - maxorientindex
        finalind_rows, finalind_cols = np.shape(finalind)
        size = int(size)
        for k in range(0, finalind_cols):
            r = validr[finalind[0][k]]
            c = validc[finalind[0][k]]

            img_block = im[r - size:r + size + 1][:, c - size:c + size + 1]

            newim[r][c] = np.sum(img_block * gabor_filter[int(orient_index[r][c]) - 1])

        binary_image = newim < threshold
        return binary_image,newim 

    
    
