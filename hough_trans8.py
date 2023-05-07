import numpy as np
import cv2 as cv
from skimage import io
import os

import matplotlib.pyplot as plt
from skimage.filters import threshold_minimum
from scipy.stats import sem

# read image file names from directory and store in list to be called when
tiff_images = []
for file in os.listdir('Xray_CT_t4_0.1MCaCl2_made221122'):
    if file.endswith('.tiff'):
        tiff_images.append(file)
        #print(file)
    else:
        continue
tiff_images.sort()
#print(tiff_images)

def px_to_mm(pixels,image):
    '''
    Conversion function from pixels to millimeters with
    error in conversion.
        Error relates to image size. If conversion is
        nominally provided by CT scan equipment, then this
        error is false and can be commented out.
    NB. if mmerr is removed from return statement, adjustments
    need to be made to any instance calling function as it 
    will return float instead of list.
    '''
    mm_per_pixel = (1/25.6)
    mm = mm_per_pixel*pixels
    mmerr = mm/np.sqrt(image.shape[0]*image.shape[1])
    return mm,mmerr

def normalise8(image_sequence):
    '''
    Normalising function to normalise all .tiff pixel values
    to 8-bit image values.
    '''
    mn = 0
    mx = 0
    for image in image_sequence:
        if np.min(image) < mn:
            mn = np.min(image)
        if np.max(image) > mx:
            mx = np.max(image)
    image_sequence = ((image_sequence - mn)/(mx - mn)) * 255
    return image_sequence.astype(np.uint8)

img = io.imread('Xray_CT_t4_0.1MCaCl2_made221122/'+'slice1744.tiff',cv.IMREAD_UNCHANGED)
#cv.imshow('original',img)
#cv.waitKey(0)
#cv.destroyAllWindows
#print(f'dtype: {img.dtype}, shape: {img.shape}, min: {np.min(img)}, max: {np.max(img)}')
img = normalise8(img)
#cv.imshow('normalised',img)
#cv.waitKey(0)
#cv.destroyAllWindows
#print(f'dtype: {img.dtype}, shape: {img.shape}, min: {np.min(img)}, max: {np.max(img)}')

thresh_min = threshold_minimum(img)

img = cv.GaussianBlur(img,(3,3),0)
img = img[100:550,110:560]
ret,thresh1 = cv.threshold(img,thresh_min,255,cv.THRESH_BINARY)
cthresh1 = cv.cvtColor(255-thresh1,cv.COLOR_GRAY2BGR)
#cv.imshow('binary threshold',thresh1)
#cv.waitKey(0)
#cv.destroyAllWindows

#quit()
try:
    circles = cv.HoughCircles(255-thresh1,cv.HOUGH_GRADIENT,1,minDist=100,param1=30,param2=10,minRadius=15,maxRadius=25)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        #draw the circle
        cv.circle(cthresh1,(i[0],i[1]),i[2],(0,255,0),2)
        #draw the centre of the circle
        cv.circle(cthresh1,(i[0],i[1]),1,(0,0,255),3)

except TypeError:
    print('No circles identifiable')

flagged_slices = {}
circ_x = np.zeros((len(tiff_images),9))
circ_y = np.zeros((len(tiff_images),9))
circ_rhough = np.full((len(tiff_images),9),np.nan)

rss = np.full((len(tiff_images),9),np.nan)
rss_err = np.full((len(tiff_images),9),np.nan)
rps = np.full((len(tiff_images),9),np.nan)
rps_err = np.full((len(tiff_images),9),np.nan)
Cs = np.full((len(tiff_images),9),np.nan)
Cs_err = np.full((len(tiff_images),9),np.nan)

for k in range(len(tiff_images)):
    img_stack = io.imread('Xray_CT_t4_0.1MCaCl2_made221122/'+tiff_images[k],cv.IMREAD_UNCHANGED)
    cv.imwrite(f'r_orig_slice{k}.png',img_stack)
    img_stack = normalise8(img_stack)
    img_stack = cv.GaussianBlur(img_stack,(3,3),0)
    img_stack = img_stack[100:550,110:560]
    cv.imwrite(f'r_proc_slice{k}.png',img_stack)
    ret,thresh_stack1 = cv.threshold(img_stack,thresh_min,255,cv.THRESH_BINARY)
    cv.imwrite(f'r_thresh_slice{k}.png',thresh_stack1)
    edge_stack = cv.Canny(255-thresh_stack1,threshold1=thresh_min-12,threshold2=thresh_min+12)
    cv.imwrite(f'r_edge_slice{k}.png',edge_stack)
    j=0
    for i in circles[0,:]:
        # check to identify 
        # region of threshold image is selected 
        thresh_stack = 255-thresh_stack1[i[1]-27:i[1]+27, i[0]-27:i[0]+27]
        cthresh_stack = cv.cvtColor(thresh_stack,cv.COLOR_GRAY2BGR)
        circs_stack = cv.HoughCircles(thresh_stack,cv.HOUGH_GRADIENT,1,minDist=100,param1=30,param2=10,minRadius=15,maxRadius=25)
        try:
            circs_stack = np.uint16(np.around(circs_stack))
        except:
            # flagging slices that don't contain all 9 circles
            # store flagged slices in dictionary to count how many circles are undetected in each slice
            if tiff_images[k] not in flagged_slices:
                key_val = {tiff_images[k]:1}
                flagged_slices.update(key_val)
            elif tiff_images[k] in flagged_slices:
                flagged_slices[tiff_images[k]] += 1
            continue
        for c in circs_stack[0,:]:
            cv.circle(cthresh_stack,(c[0],c[1]),c[2],(0,255,0),2)
            cv.circle(cthresh_stack,(c[0],c[1]),1,(0,0,255),2)
        
        # error from resolution and Gaussian blurring of images
        resblur_error = 3/thresh_stack.shape[0]
        # region of original slice image is 
        circimg_stack = img_stack[i[1]-25:i[1]+25, i[0]-25:i[0]+25]
        # threshold image used to obtain surface area of circle
        circthresh_stack = 255-thresh_stack1[i[1]-25:i[1]+25, i[0]-25:i[0]+25]
        S = cv.countNonZero(circthresh_stack)
        rs = np.sqrt(S/np.pi)
        rserr = rs/np.sqrt(circimg_stack.shape[0]*circimg_stack.shape[1]) + rs*resblur_error
        # Canny edge image generated to obtain perimeter of circle
        circedge_stack = cv.Canny(circthresh_stack,threshold1=thresh_min-12,threshold2=thresh_min+12)
        P = cv.countNonZero(circedge_stack)
        rp = P/(2*np.pi)
        rperr = rp/np.sqrt(circedge_stack.shape[0]*circedge_stack.shape[1]) + rp*resblur_error
        try:
            # compute circularity of circle
            C = 4*np.pi*S/(P**2)
            # store values for radius and circularity
            circ_x[k,j],circ_y[k,j],circ_rhough[k,j] = i[0],i[1],i[2]
            rss[k,j] = rs
            rss_err[k,j] = rserr
            rps[k,j] = rp
            rps_err[k,j] = rperr
            if C <= 5:
                Cs[k,j] = C
                Cs_err[k,j] = (rss_err[k,j]/rss[k,j] + rps_err[k,j]/rps[k,j]) * Cs[k,j]
            j += 1
        except ZeroDivisionError:
            #print('circularity not computed')
            #circimgs_stack = cv.hconcat([circimg_stack,circthresh_stack,circedge_stack])
            #cv.imshow('concatenated images',circimgs_stack)
            #cv.waitKey(0)
            continue

slice_number = np.linspace(0,len(tiff_images)-1,len(tiff_images))
for iii in range(0,9):
    s1 = plt.scatter(slice_number/25.6,circ_rhough[:,iii]/25.6,marker='.',s=15,color='blue')
    s2 = plt.scatter(slice_number/25.6,rss[:,iii]/25.6,marker='^',s=15,color='red')
    plt.errorbar(slice_number/25.6,rss[:,iii]/25.6,yerr=(sem(~np.isnan(rss[:,iii]))+rss_err[:,iii])/25.6,xerr=None,fmt='none',ecolor='red')
    s3 = plt.scatter(slice_number/25.6,rps[:,iii]/25.6,marker='s',s=15,color='green')
    plt.errorbar(slice_number/25.6,rps[:,iii]/25.6,yerr=(sem(~np.isnan(rps[:,iii]))+rps_err[:,iii])/25.6,xerr=None,fmt='none',ecolor='green')
    plt.title(f'variation in radius of circle {iii+1}')
    plt.ylim(0,np.maximum(rss[:,iii].all(),rps[:,iii].all())*1.5)
    plt.xlabel('height (mm)')
    plt.ylabel('radius (mm)')
    plt.legend([s1,s2,s3],['Hough radius','Circle area radius','Circle perimeter radius'])
    plt.savefig(f'r_radius_plot{iii}.png')
    plt.close()

    s4 = plt.scatter(slice_number[40:-40]/25.6,circ_rhough[40:-40,iii]/25.6,marker='.',s=15,color='blue')
    s5 = plt.scatter(slice_number[40:-40]/25.6,rss[40:-40,iii]/25.6,marker='^',s=15,color='red')
    plt.errorbar(slice_number[40:-40]/25.6,rss[40:-40,iii]/25.6,yerr=(sem(~np.isnan(rss[:,iii]))+rss_err[40:-40,iii])/25.6,xerr=None,fmt='none',ecolor='red')
    s6 = plt.scatter(slice_number[40:-40]/25.6,rps[40:-40,iii]/25.6,marker='s',s=15,color='green')
    plt.errorbar(slice_number[40:-40]/25.6,rps[40:-40,iii]/25.6,yerr=(sem(~np.isnan(rps[:,iii]))+rps_err[40:-40,iii])/25.6,xerr=None,fmt='none',ecolor='green')
    plt.title(f'variation in radius of circle {iii+1} (selective region)')
    plt.ylim(0,np.maximum(rss[:,iii].all(),rps[:,iii].all())*1.5)
    plt.xlabel('height (mm)')
    plt.ylabel('radius (mm)')
    plt.legend([s4,s5,s6],['Hough radius','Circle area radius','Circle perimeter radius'])
    plt.savefig(f'r_selective_radius_plot{iii}.png')
    plt.close()
    
    s7 = plt.scatter(slice_number/25.6,Cs[:,iii],marker='.',s=15)
    plt.errorbar(slice_number/25.6,Cs[:,iii],yerr=sem(~np.isnan(Cs[:,iii]))+Cs_err[:,iii],xerr=None,fmt='none',ecolor='blue')
    l1, = plt.plot(slice_number/25.6,np.full(len(slice_number),np.nanmedian(Cs[:,iii])),color='blue')
    plt.plot(slice_number/25.6,np.full(len(slice_number),np.nanmedian(Cs[:,iii])+sem(~np.isnan(Cs[:,iii]),axis=None,ddof=0))+np.nanmean(Cs_err[:,iii]),color='blue',linestyle='--')
    plt.plot(slice_number/25.6,np.full(len(slice_number),np.nanmedian(Cs[:,iii])-sem(~np.isnan(Cs[:,iii]),axis=None,ddof=0))-np.nanmean(Cs_err[:,iii]),color='blue',linestyle='--')
    plt.title(f'variation in roundness of circle {iii+1}')
    plt.ylim(0,1.5)
    plt.xlabel('height (mm)')
    plt.ylabel('roundness')
    plt.legend([s7,l1],['roundness values','roundness median value'])
    plt.savefig(f'r_roundness_plot{iii}.png')
    plt.close()

    s8 = plt.scatter(slice_number[40:-40]/25.6,Cs[40:-40,iii],marker='.',s=15)
    plt.errorbar(slice_number[40:-40]/25.6,Cs[40:-40,iii],yerr=sem(~np.isnan(Cs[40:-40,iii]))+Cs_err[40:-40,iii],xerr=None,fmt='none',ecolor='blue')
    l2, = plt.plot(slice_number[40:-40]/25.6,np.full(len(slice_number[40:-40]),np.nanmedian(Cs[40:-40,iii])),color='blue')
    plt.plot(slice_number[40:-40]/25.6,np.full(len(slice_number[40:-40]),np.nanmedian(Cs[40:-40,iii])+sem(~np.isnan(Cs[40:-40,iii]),axis=None,ddof=0))+np.nanmean(Cs_err[40:-40,iii]),color='blue',linestyle='--')
    plt.plot(slice_number[40:-40]/25.6,np.full(len(slice_number[40:-40]),np.nanmedian(Cs[40:-40,iii])-sem(~np.isnan(Cs[40:-40,iii]),axis=None,ddof=0))-np.nanmean(Cs_err[40:-40,iii]),color='blue',linestyle='--')
    plt.title(f'variation in roundness of circle {iii+1} (selective region)')
    plt.ylim(0,1.5)
    plt.xlabel('height (mm)')
    plt.ylabel('roundness')
    plt.legend([s8,l2],['roundness values','roundness median value'])
    plt.savefig(f'r_selective_roundness_plot{iii}.png')
    plt.close()

    print(f'Hough radius {iii+1}:',px_to_mm(np.nanmedian(circ_rhough[:,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(circ_rhough[:,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmedian(circ_rhough[:,iii]),img)[1])
    print(f'Surface radius {iii+1}:',px_to_mm(np.nanmedian(rss[:,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(rss[:,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmean(rss_err[:,iii]),img)[0]+px_to_mm(np.nanmedian(rss[:,iii]),img)[1])
    print(f'Perimeter radius {iii+1}:',px_to_mm(np.nanmedian(rps[:,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(rps[:,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmean(rps_err[:,iii]),img)[0]+px_to_mm(np.nanmedian(rps[:,iii]),img)[1])
    print(f'Circularity {iii+1}:',np.nanmedian(Cs[:,iii]),'+/-',sem(~np.isnan(Cs[:,iii]),axis=None,ddof=0)+np.nanmean(Cs_err[:,iii]))
    print(f'Percentage of total slices discarded {iii+1}:',np.count_nonzero(np.isnan(circ_rhough[:,iii]))/len(circ_rhough[:,iii])*100,'\n')

    print(f'Hough radius (post-selection) {iii+1}:',px_to_mm(np.nanmedian(circ_rhough[40:-40,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(circ_rhough[40:-40,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmedian(circ_rhough[40:-40,iii]),img)[1])
    print(f'Surface radius (post-selection) {iii+1}:',px_to_mm(np.nanmedian(rss[40:-40,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(rss[40:-40,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmean(rss_err[40:-40,iii]),img)[0]+px_to_mm(np.nanmedian(rss[40:-40,iii]),img)[1])
    print(f'Perimeter radius (post-selection) {iii+1}:',px_to_mm(np.nanmedian(rps[40:-40,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(rps[40:-40,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmean(rps_err[40:-40,iii]),img)[0]+px_to_mm(np.nanmedian(rps[40:-40,iii]),img)[1])
    print(f'Circularity (post-selection) {iii+1}:',np.nanmedian(Cs[40:-40,iii]),'+/-',sem(~np.isnan(Cs[40:-40,iii]),axis=None,ddof=0)+np.nanmean(Cs_err[40:-40,iii]))
    print(f'Percentage of selected slices discarded {iii+1}:',(len(circ_rhough[:,iii])-len(circ_rhough[40:-40,iii])+np.count_nonzero(np.isnan(circ_rhough[40:-40,iii])))/len(circ_rhough[:,iii])*100,'\n')

failed_slices = 0
for flag in flagged_slices:
    if flagged_slices[flag] == 9:
        failed_slices += 1
print('total slices in stack',len(tiff_images))
print('total slices with all identifiable circles',len(tiff_images)-len(flagged_slices))
print('total slices with limited identifiable circles',len(flagged_slices)-failed_slices)
print('total slices with no identifiable circles',failed_slices)
cv.destroyAllWindows