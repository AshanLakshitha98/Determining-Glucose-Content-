import matplotlib.pyplot as plt
import numpy
import math
import pylab
from PIL import Image

#Function to read the image-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def closest(lst, K):
    lst = numpy.asarray(lst)
    idx = (numpy.abs(lst - K)).argmin()
    return lst[idx]

def spectrum(image_name):
    im = Image.open(image_name) # Can be many different formats.
    pix = im.load()
    im_width, im_height=im.size
    wl_offset=400.0 # wavelength start value for the spectra
    wl_range=300.0 # wavelength range for the spectra
    y_bin=30  # bin size for adding spectral intensity for each wavelength
    wl_p_pix=wl_range/im_width

    x_all=[]
    y_all=[]

    for i in range(0,im_width):      
        x_sum=0.0
        c=0;
        for j in range(0,im_height):
            s=pix[i,j]
            x_sum+=0.21*int(s[0]) + 0.72*int(s[1]) + 0.07*int(s[2])  #Luminosity Method 0.21 R + 0.72 G + 0.07 B
            c+=1.0
        x_all.append(i*wl_p_pix+wl_offset)
        y_all.append(x_sum/c)
    y_all=numpy.array(y_all,dtype='f')
    y_all = y_all[::-1]
    index_max=numpy.where(y_all==max(y_all))
    index_max=numpy.array(index_max,dtype='i')
    index_max=index_max*wl_p_pix+wl_offset
    final_index=len(index_max)
    peak_start_value=(index_max[0][0] - 1)*wl_p_pix+wl_offset
    peak_end_value=(index_max[final_index-1][0])*wl_p_pix+wl_offset
    error_wavelength=(peak_end_value - peak_start_value)/2
    c = (peak_end_value + peak_start_value) / 2
    return [x_all,y_all, c]

def draw_wavelength_reference_point():
    '''Blue_light_wavelength=450-495 nm range'''
    wave_required=463
    plt.axvline(x=wave_required,linestyle='--',label='%.1f nm'%wave_required)

#Function to compute the errors of absorbance

err_i=math.sqrt((0.21*0.21)+(0.72*0.72)+(0.07*0.07))
def err_absorbance(T,y):
    err_absorbance=(numpy.log10(math.e))*(numpy.sqrt((1/y)*2+(1/ybC)*2))*err_i
    return err_absorbance
    
#Control SETUP-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

xC,ybC,cC=spectrum('Ctrl.png')    
 
#plotting graphs for absorbance and intensity-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
x1,yb1,c1=spectrum('1.jpg')
x2,yb2,c2=spectrum('2.jpg')
x3,yb3,c3=spectrum('3.png')
x4,yb4,c4=spectrum('4.jpg')

a=plt.figure()
plt.plot(x1,yb1, label="1", color="red")
plt.plot(x2,yb2, label="2", color="blue")
plt.plot(x3,yb3, label="3",  color="green")
plt.plot(x4,yb4, label="4", color="yellow")
plt.legend(loc='upper right')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.title("The Graph of Intensity vs Wavelength  for range 0.6-1.0%")
a.show()
plt.savefig('Intensity_Spectrum_0.6-1.0_Range.png')

'''Calculating the absorbance for each case'''
'''Transmittance=I/I0
I-refers to the intensity of the considered spectrum
I0-refres to the intensity of the control setup'''


T1=yb1/ybC
T2=yb2/ybC
T3=yb3/ybC
T4=yb4/ybC

'''Absorbance=-log10(T)'''

A1 = -(numpy.log10(T1))
A2 = -(numpy.log10(T2))
A3 = -(numpy.log10(T3))
A4 = -(numpy.log10(T4))

'''plotting absorbance vs wavelength'''

b=plt.figure()
plt.plot(xC,A1, label="1", color="red")
plt.plot(xC,A2, label="2", color="blue")
plt.plot(xC,A3, label="3",  color="green")
plt.plot(xC,A4, label="4", color="yellow")
draw_wavelength_reference_point()
plt.legend(loc='upper right')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorption')
plt.title("The Graph of Absorption vs Wavelength  for range 0.6-1.0%")
b.show()
plt.savefig('Absorption_Spectrum_0.6-1.0_Range.png')

#Extracting the absorbance of blue light for each concentration of glucose---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ab_wl = 463  # selected wavelength

c_wl=closest(xC, ab_wl)
print(c_wl)
index=numpy.where(xC==c_wl)
print(index)

ab1=A1[index[0]][0]
ab2=A2[index[0]][0]
ab3=A3[index[0]][0]
ab4=A4[index[0]][0]

print('absorbance 1 : '+str(ab1))
print('absorbance 2: '+str(ab2))
print('absorbance 3 : '+str(ab3))
print('absorbance 4 : '+str(ab4))

err_a1=err_absorbance(T1,yb1)
err_a2=err_absorbance(T2,yb2)
err_a3=err_absorbance(T3,yb3)
err_a4=err_absorbance(T4,yb4)

err_ab1 = err_a1[index[0]][0]
err_ab2 = err_a2[index[0]][0]
err_ab3 = err_a3[index[0]][0]
err_ab4 = err_a4[index[0]][0]


print("The absorbance error of Sample 1:"+str(err_ab1))
print("The absorbance error of Sample 2:"+str(err_ab2))
print("The absorbance error of Sample 3:"+str(err_ab3))
print("The absorbance error of Sample 4:"+str(err_ab4))
