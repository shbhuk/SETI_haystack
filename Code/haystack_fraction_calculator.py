'''
To calculate the fraction scanned of the 8D haystack that has been scanned
The 9 dimensions are - 
1) Sensitivity (m^2/W)
2) 3 Distance dimensions (m^3)
5) Transmitter BW (Hz)
6) Transmitter Frequency (Hz)
7) Repetition Time (sec)
8) Polarization

9) Modulation - not covered

The user must enter the parameters of their observation. 
Also, they may change the default bounds of the haystack. 
'''

import numpy as np
from astropy import units as u
from astropy import constants as ac
import sys,os
import configparser
home = sys.path[0]

# Read in the Haystack and search parameters from the config files. 
# The config (input) files must be stored in the same directory as this script.
# Please use the default units for the observation parameters and haystack boundaries


haystack_boundary = os.path.join(home,'Haystack_boundaries.txt')
search_parameters = os.path.join(home,'Search_parameters.txt')
#search_parameters = os.path.join(home,'Tingay2018.txt')

haystack_config =  configparser.ConfigParser()
haystack_config.read(haystack_boundary)
search_config = configparser.ConfigParser()
search_config.read(search_parameters)


##############################
# User observation  parameters
##############################


# Total integration time for observation
total_scan = float(search_config['Search']['Time_on_Target']) * u.s # [seconds]

# Channel bandwidth (spectral resolution)
c = float(search_config['Search']['Channel_BW']) * u.Hz # [Hertz]

# Survey sensitivity 
Survey_sensitivity = float(search_config['Search']['Survey_sensitivity']) * u.Jy # [Jansky]

# The telescope diameter is used to calculate the beam width. Assuming uniformity and diffraction limited.
diam_telescope = float(search_config['Search']['Telescope_diameter']) * u.m # [meters]

# Max and minimum frequency in the instrument band
band_max = float(search_config['Search']['Max_receiver_band']) * u.Hz # [Hertz]
band_min = float(search_config['Search']['Min_receiver_band']) * u.Hz # [Hertz]

# Number of observations (targets)
n = float(search_config['Search']['No_of_targets'])

# Fraction of polarization. 0.5 or 1
fpol = float(search_config['Search']['Polarization_fraction'])

#############################
# Haystack Bounds / Constants
#############################


# Minimum EIRP to be sensitive to. Defines the bound for the sensitivity dimension.
P_min = float(haystack_config['Haystack']['Min_EIRP']) * u.W  # [Watts] 

# Bounds for the transmission frequency dimension
f_max = float(haystack_config['Haystack']['Max_central_freq']) * u.Hz # [Hertz] 
f_min = float(haystack_config['Haystack']['Min_central_freq']) * u.Hz # [Hertz]

# Maximum transmission bandwidth for haystack boundary. Cannot be greater than min_central_freq.  
tmax = float(haystack_config['Haystack']['Max_trans_BW']) * u.Hz #[Hz]


# Upper Bound for repetition rate dimension. Lower bound is assumed to be zero (continuous transmission).
t_upper = float(haystack_config['Haystack']['Max_rep_period']) * u.yr # [years]

# Upper Bound for distance dimension. Lower bound is assumed to be zero.
dmax = float(haystack_config['Haystack']['Max_distance']) * u.lyr # [light year]



#####################


# Central receiver frequency
mean_band_frequency = (band_max + band_min)/2 # Mean frequency of instrument band. Eg for GBT L Band it is about ~1.5 GHz
mean_band_wavelength = (ac.c/(mean_band_frequency)).to(u.m)

# Bandwidth of the entire instrument (receiver)
i = band_max - band_min

# Phi is flux (W/m2), whereas Survey_sensitivity is specific flux (W/m2/Hz)
phi = (Survey_sensitivity  * c).to(u.W/u.m**2)

# Using 1/phi and P_min/4pi for use of calculation
s = 1 / phi
P0 = P_min / (4*np.pi)

####################

# Conditional statements for the bounds of the 6d integral
tcrit_dmax3 = (c*P0*P0*s*s / (dmax**4)).to(u.Hz)
tcrit_dmax5 = (np.sqrt(c*i)*P0*s/(dmax*dmax)).to(u.Hz)

d1upper = min(dmax,np.sqrt(P0*s))
t1upper = min(tmax,c)

d2upper = max(dmax,np.sqrt(P0*s))
t2upper = min(tmax,c)

t3mid =  max(c,min(tcrit_dmax3,tmax,i))
t3upper = max(c,min(i,tmax))

t4mid = t3mid
t4upper = t3upper

t5mid = max(i,min(tcrit_dmax5,tmax))
t5upper = max(tmax,i) 

t6mid = t5mid
t6upper = t5upper


# Volume calculation for 6 dimensions => frequency (1), transmitter BW (1), distance (3) and sensitivity (1) dimensions (6d)
V1 =  - (((d1upper**5) * t1upper * ( -14*P0*i*s + 5*(d1upper**2)*t1upper - 
    7*P0*s*t1upper)) / (70*P0*P0*s))
V1 = V1.to(u.Hz**2 * u.m**5 / u.W)

V2 = (i*s*t2upper/3*((d2upper**3) - (np.sqrt(P0*s)**3))).to(u.Hz**2 * u.m**5 / u.W)
V2 = V2.to(u.Hz**2 * u.m**5 / u.W)

V3a = ((dmax**5)/(70*P0*P0*s)) * ( - 7*P0*s*(c*c - t3mid*t3mid) + 14*P0*i*s*(t3mid - c) + 4*dmax*dmax*(c*c - (t3mid**(5/2)/np.sqrt(c)))) 
V3b1 = (4*c**(5/4)*s*(P0*s)**(3/2))/(105*(t3mid*t3upper)**(1/4))
V3b2 = 21*i*(t3upper**(1/4) - t3mid**(1/4)) + 2*(t3mid*t3upper)**(1/4) * (t3upper**(3/4) - t3mid**(3/4))
V3 = V3a + V3b1*V3b2
V3 = V3.to(u.Hz**2 * u.m**5 / u.W)

V4 = ((np.sqrt(c)*i*s)/3) * ((c**(3/4)*(P0*s)**(3/2))*(-(4)/(t4mid**(1/4)) + 
    (4)/(t4upper**(1/4))) - 2*dmax**3 * (np.sqrt(t4mid) - np.sqrt(t4upper)) ) 
V4 = V4.to(u.Hz**2 * u.m**5 / u.W)

V5a = (1/210) * ((42*(dmax**5)*i*(t5mid-i))/P0 + (3*(5*(dmax**7)*np.sqrt(c*i) - 7*c*P0*s*dmax**5)*(i*i - t5mid*t5mid))/(c*P0*P0*s))
V5b = (1/210) * 4 * s * (c*i)**(5/4) * (((P0*s)/(t5mid*t5upper))**(3/2)) * (-2*i*t5mid**(3/2) - 21*t5mid**(3/2)*t5upper + 2*i*t5upper**(3/2) + 21*t5mid*t5upper**(3/2))
        
V5 = V5a + V5b
V5 = V5.to(u.Hz**2 * u.m**5 / u.W)

V6 = ((np.sqrt(c*i)*s)/3) * ((c*i)**(3/4) * (P0*s)**(3/2) * ((-2/np.sqrt(t6mid)) + (2/np.sqrt(t6upper))) + 
    dmax**3 * (t6upper - t6mid))
V6 = V6.to(u.Hz**2 * u.m**5 / u.W)



# Calculate ratio of solid angle of beam to 4pi
diff_angle = 1.22 * mean_band_wavelength/diam_telescope # Airy ring telescope
omega = 2*np.pi* (1-np.cos(diff_angle.value/2)) # Solid Angle

# Preventing outrageous values from hacking the fractions!
if omega > 4*np.pi:
    omega = 4*np.pi


f_sa = omega / (4*np.pi)



print('Observer Sensitivity = {:.3E}'.format(Survey_sensitivity ))
print('Channel BW = {:.3E} : Instrument BW = {:.3E}'.format(c,i))
print('Time on Target = {:.3E}'.format(total_scan))
print('Beam Width = {:.3E} arcminutes\n'.format(diff_angle * 180 * 60/np.pi))
print('d_max = {:.3E} : t_max = {:.3E} f_max = {:.3E} : f_min = {:.3E} : Max. Rep Time = {:.3E}'.format(dmax.to(u.lyr),tmax,f_max,f_min,t_upper))
print('EIRP for sensitivity max = {:.3E}\n'.format(P_min))


print('Region 1 = {:.5E}'.format(V1))
print('Region 2 = {:.5E}'.format(V2))
print('Region 3 = {:.5E}'.format(V3))
print('Region 4 = {:.5E}'.format(V4))
print('Region 5 = {:.5E}'.format(V5))
print('Region 6 = {:.5E}'.format(V6))


# Therefore V_scanned = summation (V1 - V6) times the solid angle
V_scanned_6d = ( V1 + V2 + V3 + V4 + V5 + V6 ) * omega # To account for solid angle fraction (implicit in distance dimension)

# In the frequency, sensitivity, transmitter BW and distance space  (6d); what is the total volume that can be covered?
V_total_6d = ((dmax**5 / (5 * P0)) * (tmax) * np.abs(f_max - f_min)).to(u.Hz**2 * u.m**5 / u.W)
V_total_6d *= 4 * np.pi # To account for solid angle fraction (implicit in distance dimension)


f_6d = V_scanned_6d / V_total_6d

if total_scan.to(u.s).value > t_upper.to(u.s).value:
    total_scan = t_upper

V_scanned_8d = V_scanned_6d * (total_scan.to(u.s)) * fpol
V_total_8d = V_total_6d * (t_upper.to(u.s))

f_8d = V_scanned_8d / V_total_8d
f_n8d = f_8d * n


print('Volume scanned in frequency (1), transmitter BW (1), distance (3), sensitivity (1), repetition rate (1) and polarization (1) dimensions (8d) for {} search/es = {:.5E}'.format(1,V_scanned_8d))
print('Volume scanned in frequency (1), transmitter BW (1), distance (3), sensitivity (1), repetition rate (1) and polarization (1) dimensions (8d) for {} search/es = {:.5E}'.format(int(n),V_scanned_8d*n))
print('Total haystack volume in this 8d space = {:.5E}\n'.format(V_total_8d))


#print('Fraction of 6D Volume = {:.5E}'.format(f_6d))
print('Total fraction of 8D volume for 1 search = {:.5E}'.format(f_8d))
print('Total fraction of 8D volume for {} search/es = {:.5E}'.format(n,f_n8d))
