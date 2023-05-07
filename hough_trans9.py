import numpy as np
import cv2 as cv
from skimage import io
import os

import matplotlib.pyplot as plt
from skimage.filters import threshold_minimum
from scipy.stats import sem

# read image file names from directory and store in list to be called when
tiff_images = []
for file in os.listdir('Halloysite_test_sample_E_latex_13Feb'):
    if file.endswith('.tiff'):
        tiff_images.append(file)
    else:
        continue
tiff_images.sort()

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
    
    Ensures that cv.threshold,cv.Canny,cv.HoughCircles are
    provided with images that have consistent pixel value ranges
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

# selected "good image", chosen from visual assessment of slices
img = io.imread('Halloysite_test_sample_E_latex_13Feb/'+'slice1739.tiff',cv.IMREAD_UNCHANGED)
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

img = img[135:600,145:610]
img = np.where(img>=170,0,img)
img = cv.GaussianBlur(img,(3,3),0)
ret,thresh1 = cv.threshold(img,thresh_min,255,cv.THRESH_BINARY)
cthresh1 = cv.cvtColor(255-thresh1,cv.COLOR_GRAY2BGR)

# identify circles corresponding to porous substrate inner wall
try:
    circles = cv.HoughCircles(255-thresh1,cv.HOUGH_GRADIENT,1,minDist=100,param1=30,param2=10,minRadius=15,maxRadius=25)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        #draw the outer circle
        cv.circle(cthresh1,(i[0],i[1]),i[2],(0,255,0),2)
        #draw the centre of the circle
        cv.circle(cthresh1,(i[0],i[1]),1,(0,0,255),3)
except TypeError:
    print('No circles identifiable')

flagged_slices = {}
circ_x = np.zeros((len(tiff_images),9))
circ_y = np.zeros((len(tiff_images),9))
circ_rhough = np.full((len(tiff_images),9),np.nan)

rss_stack = np.full((len(tiff_images),9),np.nan)
rss_stack_err = np.full((len(tiff_images),9),np.nan)
rps_stack = np.full((len(tiff_images),9),np.nan)
rps_stack_err = np.full((len(tiff_images),9),np.nan)
Cs_stack = np.full((len(tiff_images),9),np.nan)
Cs_stack_err = np.full((len(tiff_images),9),np.nan)

rss_coat = np.full((len(tiff_images),9),np.nan)
rss_coat_err = np.full((len(tiff_images),9),np.nan)
rps_coat = np.full((len(tiff_images),9),np.nan)
rps_coat_err = np.full((len(tiff_images),9),np.nan)
Cs_coat = np.full((len(tiff_images),9),np.nan)
Cs_coat_err = np.full((len(tiff_images),9),np.nan)

# iterating over all slices in stack
for k in range(len(tiff_images)):
    img_stack = io.imread('Halloysite_test_sample_E_latex_13Feb/'+tiff_images[k],cv.IMREAD_UNCHANGED)
    #cv.imwrite(f'h_orig_slice{k}.png',img_stack)
    img_stack = normalise8(img_stack)
    img_stack = img_stack[135:600,145:610]
    img_coat = img_stack
    img_coat = np.where(img_stack<170,0,img_coat)
    img_stack = np.where(img_stack>=170,0,img_stack)
    img_stack = cv.GaussianBlur(img_stack,(3,3),0)
    img_coat = cv.GaussianBlur(img_coat,(3,3),0)
    
    #cv.imwrite(f'h_proc_slice{k}.png',img_stack)
    #cv.imwrite(f'h_proc_coat{k}.png',img_coat)
    ret,thresh_stack1 = cv.threshold(img_stack,thresh_min,255,cv.THRESH_BINARY)
    ret,thresh_coat1 = cv.threshold(img_coat,163,255,cv.THRESH_BINARY)
    cv.imwrite(f'h_thresh_slice{k}.png',thresh_stack1)
    cv.imwrite(f'h_thresh_coat{k}.png',thresh_coat1)
    edge_stack = cv.Canny(255-thresh_stack1,threshold1=thresh_min-12,threshold2=thresh_min+12)
    edge_coat = cv.Canny(255-thresh_coat1,threshold1=165,threshold2=175)
    #cv.imwrite(f'h_edge_slice{k}.png',edge_stack)
    #cv.imwrite(f'h_edge_coat{k}.png',edge_coat)
    j=0
    for i in circles[0,:]:
        thresh_stack = 255-thresh_stack1[i[1]-27:i[1]+27, i[0]-27:i[0]+27]
        thresh_coat = 255-thresh_coat1[i[1]-27:i[1]+27, i[0]-27:i[0]+27]
        cthresh_stack = cv.cvtColor(thresh_stack,cv.COLOR_GRAY2BGR)
        cthresh_coat = cv.cvtColor(thresh_coat,cv.COLOR_GRAY2BGR)
        circs_stack = cv.HoughCircles(thresh_stack,cv.HOUGH_GRADIENT,1,minDist=100,param1=30,param2=10,minRadius=15,maxRadius=25)
        resblur_error = 3/thresh_stack.shape[0]
        try:
            circs_stack = np.uint16(np.around(circs_stack))
        except:
            #print('No circle identifiable')
            if tiff_images[k] not in flagged_slices:
                key_val = {tiff_images[k]:1}
                flagged_slices.update(key_val)
            elif tiff_images[k] in flagged_slices:
                flagged_slices[tiff_images[k]] += 1
            continue
        for c in circs_stack[0,:]:
            cv.circle(cthresh_stack,(c[0],c[1]),c[2],(0,255,0),2)
            cv.circle(cthresh_stack,(c[0],c[1]),1,(0,0,255),2)
        
        circimg_stack = img_stack[i[1]-25:i[1]+25, i[0]-25:i[0]+25]
        circimg_coat = img_coat[i[1]-25:i[1]+25, i[0]-25:i[0]+25]
        circthresh_stack = 255-thresh_stack1[i[1]-25:i[1]+25, i[0]-25:i[0]+25]
        circthresh_coat = thresh_coat1[i[1]-25:i[1]+25, i[0]-25:i[0]+25]
        
        S_stack = cv.countNonZero(circthresh_stack)
        S_coat = cv.countNonZero(circthresh_coat)
        
        rs_stack = np.sqrt(S_stack/np.pi)
        rs_coat = np.sqrt(S_coat/np.pi)
        
        rs_stack_err = rs_stack/np.sqrt(circimg_stack.shape[0]*circimg_stack.shape[1]) + rs_stack*resblur_error
        rs_coat_err = rs_coat/np.sqrt(circimg_coat.shape[0]*circimg_coat.shape[1]) + rs_coat*resblur_error
        
        circedge_stack = cv.Canny(circthresh_stack,threshold1=thresh_min-12,threshold2=thresh_min+12)
        circedge_coat = cv.Canny(circthresh_coat,threshold1=165,threshold2=175)
        
        P_stack = cv.countNonZero(circedge_stack)
        P_coat = cv.countNonZero(circedge_coat)
        
        rp_stack = P_stack/(2*np.pi)
        rp_coat = P_coat/(2*np.pi)
        
        rp_stack_err = rp_stack/np.sqrt(circedge_stack.shape[0]*circedge_stack.shape[1]) + rp_stack*resblur_error
        rp_coat_err = rp_coat/np.sqrt(circedge_coat.shape[0]*circedge_coat.shape[1]) + rp_coat*resblur_error
        try:
            C_stack = 4*np.pi*S_stack/(P_stack**2)
            C_coat = 4*np.pi*S_coat/(P_coat**2)
            
            circ_x[k,j],circ_y[k,j],circ_rhough[k,j] = i[0],i[1],i[2]
            
            rss_stack[k,j] = rs_stack
            rss_stack_err[k,j] = rs_stack_err
            
            rps_stack[k,j] = rp_stack
            rps_stack_err[k,j] = rp_stack_err
            
            if C_stack <= 5:
                Cs_stack[k,j] = C_stack
                Cs_stack_err[k,j] = (rss_stack_err[k,j]/rss_stack[k,j] + rps_stack_err[k,j]/rps_stack[k,j]) * Cs_stack[k,j]
            
            rss_coat[k,j] = rs_coat
            rss_coat_err[k,j] = rs_coat_err
            
            rps_coat[k,j] = rp_coat
            rps_coat_err[k,j] = rp_coat_err
            
            if C_coat <= 5:
                Cs_coat[k,j] = C_coat
                Cs_coat_err[k,j] = (rss_coat_err[k,j]/rss_coat[k,j] + rps_coat_err[k,j]/rps_coat[k,j]) * Cs_coat[k,j]
            j += 1
        except ZeroDivisionError:
            #print('circularity not computed')
            #circimgs_stack = cv.hconcat([circimg_stack,circthresh_stack,circedge_stack])
            #cv.imshow('concatenated images',circimgs_stack)
            #cv.waitKey(0)
            continue

selrange_start,selrange_finish = 30,-70
slice_number = np.linspace(0,len(tiff_images)-1,len(tiff_images))
for iii in range(0,9):
#STACK PLOTS
    s1 = plt.scatter(slice_number/25.6,circ_rhough[:,iii]/25.6,marker='.',s=15,color='blue')
    s2 = plt.scatter(slice_number/25.6,rss_stack[:,iii]/25.6,marker='^',s=15,color='red')
    plt.errorbar(slice_number/25.6,rss_stack[:,iii]/25.6,yerr=(sem(~np.isnan(rss_stack[:,iii]))+rss_stack_err[:,iii])/25.6,xerr=None,fmt='none',ecolor='red')
    s3 = plt.scatter(slice_number/25.6,rps_stack[:,iii]/25.6,marker='s',s=15,color='green')
    plt.errorbar(slice_number/25.6,rps_stack[:,iii]/25.6,yerr=(sem(~np.isnan(rps_stack[:,iii]))+rps_stack_err[:,iii])/25.6,xerr=None,fmt='none',ecolor='green')
    plt.title(f'variation in radius of circle {iii+1} - disk')
    plt.xlim(0,slice_number[-1]/25.6)
    plt.ylim(0,np.maximum(rss_stack[:,iii].all(),rps_stack[:,iii].all())*2.5)
    plt.xlabel('height (mm)')
    plt.ylabel('radius (mm)')
    plt.legend([s1,s2,s3],['Hough radius','Circle area radius','Circle perimeter radius'])
    #plt.savefig(f'h_stack_radius_plot{iii}.png')
    #plt.show()
    plt.close()

    s4 = plt.scatter(slice_number[selrange_start:selrange_finish]/25.6,circ_rhough[selrange_start:selrange_finish,iii]/25.6,marker='.',s=15,color='blue')
    s5 = plt.scatter(slice_number[selrange_start:selrange_finish]/25.6,rss_stack[selrange_start:selrange_finish,iii]/25.6,marker='^',s=15,color='red')
    plt.errorbar(slice_number[selrange_start:selrange_finish]/25.6,rss_stack[selrange_start:selrange_finish,iii]/25.6,yerr=(sem(~np.isnan(rss_stack[selrange_start:selrange_finish,iii]))+rss_stack_err[selrange_start:selrange_finish,iii])/25.6,xerr=None,fmt='none',ecolor='red')
    s6 = plt.scatter(slice_number[selrange_start:selrange_finish]/25.6,rps_stack[selrange_start:selrange_finish,iii]/25.6,marker='s',s=15,color='green')
    plt.errorbar(slice_number[selrange_start:selrange_finish]/25.6,rps_stack[selrange_start:selrange_finish,iii]/25.6,yerr=(sem(~np.isnan(rps_stack[selrange_start:selrange_finish,iii]))+rps_stack_err[selrange_start:selrange_finish,iii])/25.6,xerr=None,fmt='none',ecolor='green')
    plt.title(f'variation in radius of circle {iii+1} (selective region) - disk')
    plt.xlim(slice_number[selrange_start]/25.6,slice_number[selrange_finish]/25.6)
    plt.ylim(0,np.maximum(rss_stack[:,iii].all(),rps_stack[:,iii].all())*2.5)
    plt.xlabel('height (mm)')
    plt.ylabel('radius (mm)')
    plt.legend([s4,s5,s6],['Hough radius','Circle area radius','Circle perimeter radius'])
    #plt.savefig(f'h_stack_selective_radius_plot{iii}.png')
    #plt.show()
    plt.close()

    s7 = plt.scatter(slice_number/25.6,Cs_stack[:,iii],s=15)
    plt.errorbar(slice_number/25.6,Cs_stack[:,iii],yerr=sem(~np.isnan(Cs_stack[:,iii]))+Cs_stack_err[:,iii],xerr=None,fmt='none',ecolor='blue')
    l1, = plt.plot(slice_number/25.6,np.full(len(slice_number),np.nanmedian(Cs_stack[:,iii])),color='blue')
    plt.plot(slice_number/25.6,np.full(len(slice_number),np.nanmedian(Cs_stack[:,iii])+sem(~np.isnan(Cs_stack[:,iii]),axis=None,ddof=0))+np.nanmean(Cs_stack_err[:,iii]),color='blue',linestyle='--')
    plt.plot(slice_number/25.6,np.full(len(slice_number),np.nanmedian(Cs_stack[:,iii])-sem(~np.isnan(Cs_stack[:,iii]),axis=None,ddof=0))-np.nanmean(Cs_stack_err[:,iii]),color='blue',linestyle='--')
    plt.title(f'variation in roundness of circle {iii+1} - disk')
    plt.xlim(0,slice_number[-1]/25.6)
    plt.ylim(0,1.5)
    plt.xlabel('height (mm)')
    plt.ylabel('roundness')
    plt.legend([s7,l1],['roundness values','roundness median value'])
    #plt.savefig(f'h_stack_roundness_plot{iii}.png')
    #plt.show()
    plt.close()

    s8 = plt.scatter(slice_number[selrange_start:selrange_finish]/25.6,Cs_stack[selrange_start:selrange_finish,iii],s=15)
    plt.errorbar(slice_number[selrange_start:selrange_finish]/25.6,Cs_stack[selrange_start:selrange_finish,iii],yerr=sem(~np.isnan(Cs_stack[selrange_start:selrange_finish,iii]))+Cs_stack_err[selrange_start:selrange_finish,iii],xerr=None,fmt='none',ecolor='blue')
    l2, = plt.plot(slice_number[selrange_start:selrange_finish]/25.6,np.full(len(slice_number[selrange_start:selrange_finish]),np.nanmedian(Cs_stack[selrange_start:selrange_finish,iii])),color='blue')
    plt.plot(slice_number[selrange_start:selrange_finish]/25.6,np.full(len(slice_number[selrange_start:selrange_finish]),np.nanmedian(Cs_stack[selrange_start:selrange_finish,iii])+sem(~np.isnan(Cs_stack[selrange_start:selrange_finish,iii]),axis=None,ddof=0))+np.nanmean(Cs_stack_err[selrange_start:selrange_finish,iii]),color='blue',linestyle='--')
    plt.plot(slice_number[selrange_start:selrange_finish]/25.6,np.full(len(slice_number[selrange_start:selrange_finish]),np.nanmedian(Cs_stack[selrange_start:selrange_finish,iii])-sem(~np.isnan(Cs_stack[selrange_start:selrange_finish,iii]),axis=None,ddof=0))-np.nanmean(Cs_stack_err[selrange_start:selrange_finish,iii]),color='blue',linestyle='--')
    plt.title(f'variation in roundness of circle {iii+1} (selective region) - disk')
    plt.xlim(slice_number[selrange_start]/25.6,slice_number[selrange_finish]/25.6)
    plt.ylim(0,1.5)
    plt.xlabel('height (mm)')
    plt.ylabel('roundness')
    plt.legend([s8,l2],['roundness values','roundness median value'])
    #plt.savefig(f'h_stack_selective_roundness_plot{iii}.png')
    #plt.show()
    plt.close()

#COAT PLOTS
    s9 = plt.scatter(slice_number/25.6,circ_rhough[:,iii]/25.6,marker='.',s=15,color='blue')
    s10 = plt.scatter(slice_number/25.6,rss_coat[:,iii]/25.6,marker='^',s=15,color='red')
    plt.errorbar(slice_number/25.6,rss_coat[:,iii]/25.6,yerr=(sem(~np.isnan(rss_coat[:,iii]))+rss_coat_err[:,iii])/25.6,xerr=None,fmt='none',ecolor='red')
    s11 = plt.scatter(slice_number/25.6,rps_coat[:,iii]/25.6,marker='s',s=15,color='green')
    plt.errorbar(slice_number/25.6,rps_coat[:,iii]/25.6,yerr=(sem(~np.isnan(rps_coat[:,iii]))+rps_coat_err[:,iii])/25.6,xerr=None,fmt='none',ecolor='green')
    plt.title(f'variation in radius of circle {iii+1} - coating')
    plt.xlim(0,slice_number[-1]/25.6)
    plt.ylim(0,np.maximum(rss_coat[:,iii].all(),rps_coat[:,iii].all())*1.5)
    plt.xlabel('height (mm)')
    plt.ylabel('radius (mm)')
    plt.legend([s9,s10,s11],['Hough radius','Circle area radius','Circle perimeter radius'])
    #plt.savefig(f'h_coat_radius_plot{iii}.png')
    #plt.show()
    plt.close()

    s12 = plt.scatter(slice_number[selrange_start:selrange_finish]/25.6,circ_rhough[selrange_start:selrange_finish,iii]/25.6,marker='.',s=15,color='blue')
    s13 = plt.scatter(slice_number[selrange_start:selrange_finish]/25.6,rss_coat[selrange_start:selrange_finish,iii]/25.6,marker='^',s=15,color='red')
    plt.errorbar(slice_number[selrange_start:selrange_finish]/25.6,rss_coat[selrange_start:selrange_finish,iii]/25.6,yerr=(sem(~np.isnan(rss_coat[selrange_start:selrange_finish,iii]))+rss_coat_err[selrange_start:selrange_finish,iii])/25.6,xerr=None,fmt='none',ecolor='red')
    s14 = plt.scatter(slice_number[selrange_start:selrange_finish]/25.6,rps_coat[selrange_start:selrange_finish,iii]/25.6,marker='s',s=15,color='green')
    plt.errorbar(slice_number[selrange_start:selrange_finish]/25.6,rps_coat[selrange_start:selrange_finish,iii]/25.6,yerr=(sem(~np.isnan(rps_coat[selrange_start:selrange_finish,iii]))+rps_coat_err[selrange_start:selrange_finish,iii])/25.6,xerr=None,fmt='none',ecolor='green')
    plt.title(f'variation in radius of circle {iii+1} (selective region) - coating')
    plt.xlim(slice_number[selrange_start]/25.6,slice_number[selrange_finish]/25.6)
    plt.ylim(0,np.maximum(rss_coat[:,iii].all(),rps_coat[:,iii].all())*1.5)
    plt.xlabel('height (mm)')
    plt.ylabel('radius (mm)')
    plt.legend([s12,s13,s14],['Hough radius','Circle area radius','Circle perimeter radius'])
    #plt.savefig(f'h_coat_selective_radius_plot{iii}.png')
    #plt.show()
    plt.close()
    
    s15 = plt.scatter(slice_number/25.6,Cs_coat[:,iii],s=15)
    plt.errorbar(slice_number/25.6,Cs_coat[:,iii],yerr=sem(~np.isnan(Cs_coat[:,iii]))+Cs_coat_err[:,iii],xerr=None,fmt='none',ecolor='blue')
    l3, = plt.plot(slice_number/25.6,np.full(len(slice_number),np.nanmedian(Cs_coat[:,iii])),color='blue')
    plt.plot(slice_number/25.6,np.full(len(slice_number),np.nanmedian(Cs_coat[:,iii])+sem(~np.isnan(Cs_coat[:,iii]),axis=None,ddof=0))+np.nanmean(Cs_coat_err[:,iii]),color='blue',linestyle='--')
    plt.plot(slice_number/25.6,np.full(len(slice_number),np.nanmedian(Cs_coat[:,iii])-sem(~np.isnan(Cs_coat[:,iii]),axis=None,ddof=0))-np.nanmean(Cs_coat_err[:,iii]),color='blue',linestyle='--')
    plt.title(f'variation in roundness of circle {iii+1} - coating')
    plt.xlim(0,slice_number[-1]/25.6)
    plt.ylim(0,1.5)
    plt.xlabel('height (mm)')
    plt.ylabel('roundness')
    plt.legend([s15,l3],['roundness values','roundness median value'])
    #plt.show()
    #plt.savefig(f'h_coat_roundness_plot{iii}.png')
    plt.close()

    s16 = plt.scatter(slice_number[selrange_start:selrange_finish]/25.6,Cs_coat[selrange_start:selrange_finish,iii],s=15)
    plt.errorbar(slice_number[selrange_start:selrange_finish]/25.6,Cs_coat[selrange_start:selrange_finish,iii],yerr=sem(~np.isnan(Cs_coat[selrange_start:selrange_finish,iii]))+Cs_coat_err[selrange_start:selrange_finish,iii],xerr=None,fmt='none',ecolor='blue')
    l4, = plt.plot(slice_number[selrange_start:selrange_finish]/25.6,np.full(len(slice_number[selrange_start:selrange_finish]),np.nanmedian(Cs_coat[selrange_start:selrange_finish,iii])),color='blue')
    plt.plot(slice_number[selrange_start:selrange_finish]/25.6,np.full(len(slice_number[selrange_start:selrange_finish]),np.nanmedian(Cs_coat[selrange_start:selrange_finish,iii])+sem(~np.isnan(Cs_coat[selrange_start:selrange_finish,iii]),axis=None,ddof=0))+np.nanmean(Cs_coat_err[selrange_start:selrange_finish,iii]),color='blue',linestyle='--')
    plt.plot(slice_number[selrange_start:selrange_finish]/25.6,np.full(len(slice_number[selrange_start:selrange_finish]),np.nanmedian(Cs_coat[selrange_start:selrange_finish,iii])-sem(~np.isnan(Cs_coat[selrange_start:selrange_finish,iii]),axis=None,ddof=0))-np.nanmean(Cs_coat_err[selrange_start:selrange_finish,iii]),color='blue',linestyle='--')
    plt.title(f'variation in roundness of circle {iii+1} (selective region) - coating')
    plt.xlim(slice_number[selrange_start]/25.6,slice_number[selrange_finish]/25.6)
    plt.ylim(0,1.5)
    plt.xlabel('height (mm)')
    plt.ylabel('roundness')
    plt.legend([s16,l4],['roundness values','roundness median value'])
    #plt.savefig(f'h_coat_selective_roundness_plot{iii}.png')
    #plt.show()
    plt.close()

    print('Stack values')
    print(f'Hough radius {iii+1}:',px_to_mm(np.nanmedian(circ_rhough[:,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(circ_rhough[:,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmedian(circ_rhough[:,iii]),img)[1])
    print(f'Surface radius {iii+1}:',px_to_mm(np.nanmedian(rss_stack[:,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(rss_stack[:,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmean(rss_stack_err[:,iii]),img)[0]+px_to_mm(np.nanmedian(rss_stack[:,iii]),img)[1])
    print(f'Perimeter radius {iii+1}:',px_to_mm(np.nanmedian(rps_stack[:,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(rps_stack[:,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmean(rps_stack_err[:,iii]),img)[0]+px_to_mm(np.nanmedian(rps_stack[:,iii]),img)[1])
    print(f'Circularity {iii+1}:',np.nanmedian(Cs_stack[:,iii]),'+/-',sem(~np.isnan(Cs_stack[:,iii]),axis=None,ddof=0)+np.nanmean(Cs_stack_err[:,iii]),'\n')

    print(f'Hough radius (post-selection) {iii+1}:',px_to_mm(np.nanmedian(circ_rhough[selrange_start:selrange_finish,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(circ_rhough[selrange_start:selrange_finish,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmedian(circ_rhough[selrange_start:selrange_finish,iii]),img)[1])
    print(f'Surface radius (post-selection) {iii+1}:',px_to_mm(np.nanmedian(rss_stack[selrange_start:selrange_finish,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(rss_stack[selrange_start:selrange_finish,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmean(rss_stack_err[selrange_start:selrange_finish,iii]),img)[0]+px_to_mm(np.nanmedian(rss_stack[selrange_start:selrange_finish,iii]),img)[1])
    print(f'Perimeter radius (post-selection) {iii+1}:',px_to_mm(np.nanmedian(rps_stack[selrange_start:selrange_finish,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(rps_stack[selrange_start:selrange_finish,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmean(rps_stack_err[selrange_start:selrange_finish,iii]),img)[0]+px_to_mm(np.nanmedian(rps_stack[selrange_start:selrange_finish,iii]),img)[1])
    print(f'Circularity (post-selection) {iii+1}:',np.nanmedian(Cs_stack[selrange_start:selrange_finish,iii]),'+/-',sem(~np.isnan(Cs_stack[selrange_start:selrange_finish,iii]),axis=None,ddof=0)+np.nanmean(Cs_stack_err[selrange_start:selrange_finish,iii]),'\n')

    print('Coating values')
    print(f'Surface radius {iii+1}:',px_to_mm(np.nanmedian(rss_coat[:,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(rss_coat[:,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmean(rss_coat_err[:,iii]),img)[0]+px_to_mm(np.nanmedian(rss_coat[:,iii]),img)[1])
    print(f'Perimeter radius {iii+1}:',px_to_mm(np.nanmedian(rps_coat[:,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(rps_coat[:,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmean(rps_coat_err[:,iii]),img)[0]+px_to_mm(np.nanmedian(rps_coat[:,iii]),img)[1])
    print(f'Circularity {iii+1}:',np.nanmedian(Cs_coat[:,iii]),'+/-',sem(~np.isnan(Cs_coat[:,iii]),axis=None,ddof=0)+np.nanmean(Cs_coat_err[:,iii]),'\n')

    print(f'Surface radius (post-selection) {iii+1}:',px_to_mm(np.nanmedian(rss_coat[selrange_start:selrange_finish,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(rss_coat[selrange_start:selrange_finish,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmean(rss_coat_err[selrange_start:selrange_finish,iii]),img)[0]+px_to_mm(np.nanmedian(rss_coat[selrange_start:selrange_finish,iii]),img)[1])
    print(f'Perimeter radius (post-selection) {iii+1}:',px_to_mm(np.nanmedian(rps_coat[selrange_start:selrange_finish,iii]),img)[0],'+/-',px_to_mm(sem(~np.isnan(rps_coat[selrange_start:selrange_finish,iii]),axis=None,ddof=0),img)[0]+px_to_mm(np.nanmean(rps_coat_err[selrange_start:selrange_finish,iii]),img)[0]+px_to_mm(np.nanmedian(rps_coat[selrange_start:selrange_finish,iii]),img)[1])
    print(f'Circularity (post-selection) {iii+1}:',np.nanmedian(Cs_coat[selrange_start:selrange_finish,iii]),'+/-',sem(~np.isnan(Cs_coat[selrange_start:selrange_finish,iii]),axis=None,ddof=0)+np.nanmean(Cs_coat_err[selrange_start:selrange_finish,iii]),'\n')
    
    print(f'Percentage of total slices discarded {iii+1}:',np.count_nonzero(np.isnan(circ_rhough[:,iii]))/len(circ_rhough[:,iii])*100)
    print(f'Percentage of total slices discarded (post-selection) {iii+1}:',(len(circ_rhough[:,iii])-len(circ_rhough[selrange_start:selrange_finish,iii])+np.count_nonzero(np.isnan(circ_rhough[selrange_start:selrange_finish,iii])))/len(circ_rhough[:,iii])*100,'\n')

# additional statistics for circle detection throughout the stack
failed_slices = 0
for flag in flagged_slices:
    if flagged_slices[flag] == 9:
        failed_slices += 1
print('total slices in stack',len(tiff_images))
print('slices with all identified circles',len(tiff_images)-len(flagged_slices))
print('slices with limited identified circles',len(flagged_slices)-failed_slices)
print('slices with no identifiable circles',failed_slices)
print('slices within selected range',len(circ_rhough[selrange_start:selrange_finish,iii]))