#old imports
r"""
from FLASHtools.read_flash import Fields
import FLASHtools.aux_funcs.power_spectra_funcs as ps
import FLASHtools.aux_funcs.derived_var_funcs as dv
"""
from PLASMAtools.read_funcs.read import Fields
import PLASMAtools.aux_funcs.power_spectra_funcs as ps
import PLASMAtools.aux_funcs.derived_var_funcs as dv
import PLASMAanalysis.create_video as video

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from fractions import Fraction as frac
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

#CORES TO BE USED IN COMPUTATIONS
cores = 5

#DEFINE THE GLOBAL DERIVEDVARS OBJECT
dvar = dv.DerivedVars()

class DataLoader:
    def __init__(self, filepath, i, space=False):
        filename = 	str('Turb_hdf5_plt_cnt_') + str(str("%0" + str(4) +"d") % i)
        if space == True:
            filename = str(' Turb_hdf5_plt_cnt_') + str(str("%0" + str(4) +"d") % i)

        #read data
        self.turb = Fields(str(filepath) + filename, reformat = True)
        
def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def quartic(x,a,b,c,d,e):
    return(a*x**4 + b*x**3 + c*x**2 + d*x + e)

def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}
def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)
def energy_density_ratio(mag_vector, vel_vector, density,Mean = False):
    #compute magnetic energy
    mag_energy = (mag_vector[0]**2 + mag_vector[1]**2 + mag_vector[2]**2)/(8*np.pi)
    #compute kinetic energy
    kin_energy = (1/2)*(vel_vector[0]**2 + vel_vector[1]**2 + vel_vector[2]**2)*density
    
    if Mean == True:
        energy_ratio = np.mean(mag_energy)/np.mean(kin_energy)
        return energy_ratio
    else:
        energy_ratio = mag_energy/kin_energy
        return energy_ratio

def energy_evolution(max_timestep, directory, filename='none'):
    """Plots average ratio of mag to kin energy as a function of the timestep. Returns a plot and saves it
    as a pdf in the working directory.

    Parameters
    ----------
    max_timestep : int
        Maximum timestep value of data
    directory : str
        Location of Data. Please separate folders of the filepath with 2 slashes \\ (e.g. C:\\Users\\ etc.)
    """   
    #extract values from each datacube. 
    #for i in np.arange(0,max_timestep+1):
    def do_work(i):
        print(f"Analyzing timestep {i}...")  
        #read data
        data = DataLoader(directory,i)
        turb = data.turb
        turb.read('dens')
        turb.read('vel')
        turb.read('mag')
        mag_vector = turb.mag
        vel_vector = turb.vel
        #compute ave energy ratio 
        ratio = energy_density_ratio(mag_vector, vel_vector, turb.dens[0], Mean = True)
        return([i, ratio])

    #perform the parallelized computation
    result = Parallel(n_jobs=cores)(delayed(do_work)(i) for i in np.arange(1,max_timestep+1))  
    time_step = np.transpose(result)[0]
    energy_ratio = np.transpose(result)[1]  
    #produce plot
    plt.figure()
    plt.plot(time_step, energy_ratio)
    plt.xlabel('$t/t_0$')
    plt.ylabel('$\mathcal{E}_{mag} / \mathcal{E}_{kin} $')
    plt.yscale('log')
    plt.title('Energy Ratio Evolution')
    if filename != 'none':
        plt.savefig('C:\\Users\\shash\\Desktop\\Assignments\\AST425\\Plots\\' + filename + '.pdf')
    plt.show()
    #return data for further analysis
    return([time_step, energy_ratio])


def Animate_Energy2D(max_timestep, directory, slice_axis, slice_value, fps_value,box_size, quiver = False,quiver_density = 1):
    """Creates animation of the evolution of mag energy in the data cube through time along a particular slice, and plots
    streamlines of the magnetic field or quiverplot of J and B. Animation is saved
    as .mp4 in the working directory.

    Parameters
    ----------
    max_timestep : int
        Maximum timestep value of data
    directory : str
        Location of Data. Please separate folders of the filepath with 2 slashes \\ (e.g. C:\\Users\\ etc.)
    slice_axis : int
        0,1,2 for x,y,z respectively
    slice_value : int
        Index value to slice array at along specified axis. Must be in the bounds of the array 
    fps_value : int
        fps to render the animation videoa as.
    quiver: boolean
        If true, will plot quiver plot of J and B fields on top of energy density. False by default
    quiver_density: int
        Density of quiver plot. 1 by default (Outputs for every point in the array). Increase to reduce density
    """
    #for titles
    if slice_axis == '0':
        slice_title = "X"
    elif slice_axis == '1':
        slice_title = "Y"
    elif slice_axis == '2':
        slice_title = "Z"

    def length_axis_formatter(val, pos, denomlim=100, L=r'L'):
        """ REQUIRED BY Animate_data2D() FUNCTION
        format label properly with L values
        """
        new_val = val-box_size/2
        minus = "-" if new_val < 0 else ""
        new_val = abs(new_val)
        ratio = frac((new_val)/box_size).limit_denominator(denomlim)
        n, d = ratio.numerator, ratio.denominator
        
        fmt2 = "%s" % d 
        if n == 0:
            fmt1 = "0"
        elif n == 1:
            fmt1 = L
        else:
            fmt1 = r"%s%s" % (n,L)
            
        fmtstring = "$" + minus + (fmt1 if d == 1 else r"{%s}/{%s}" % (fmt1, fmt2)) + "$"
        
        return fmtstring 
    
    #initialize definitions
    fig,ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    ticklen = (box_size)/2

    #define slicer for quiverplot
    skip = slice(None, None, quiver_density)

    # setting ticks labels
    ax.xaxis.set_major_formatter(FuncFormatter(length_axis_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(length_axis_formatter))
    # setting ticks at proper numbers
    ax.xaxis.set_major_locator(MultipleLocator(base=ticklen))
    ax.yaxis.set_major_locator(MultipleLocator(base=ticklen))
    
    #find minimum and maximum of energy
    print("Finding max/min bounds for colorbar...")
    def find_minmax(i):
        #read data
        data = 	DataLoader(directory,i)
        #initialize class
        turb = data.turb
        turb.read('dens')                         #TEMP, edit once Fields class is edited to contain these values by default
        turb.read('vel')
        turb.read('mag')
        #compute energy ratio
        mag_vector = turb.mag
        vel_vector = turb.vel
        energy_ratio = energy_density_ratio(mag_vector, vel_vector, turb.dens[0])
        #find minimum and maximum
        return([np.min(energy_ratio), np.max(energy_ratio)])
    result = Parallel(n_jobs=cores)(delayed(find_minmax)(i) for i in np.arange(1,max_timestep+1))
    result = np.transpose(result)
    minimum = np.min(result[0])   #find the smallest minimum
    maximum = np.max(result[1])   #find the largest maximum
    print("Max/Min found!")
    print("Generating Plot...")
    #compute for each timestep  
    #for i in np.arange(0,max_timestep+1):
    def generate_plot(i):
        #read data
        data = 	DataLoader(directory,i)
        #initialize class
        turb = data.turb
        turb.read('dens')                       
        turb.read('vel')
        turb.read('mag')

        if quiver == True:
            turb.derived_var('cur')
            cur_vector = turb.cur
        mag_vector = turb.mag
        vel_vector = turb.vel
        #compute energy ratio
        energy_density = energy_density_ratio(mag_vector, vel_vector, turb.dens[0])

        #create slice of both energy as well as direction of B fields, assumes box is square
        #define direction arrays for streamplot/quiverplot
        direction1 = np.arange(0,np.shape(mag_vector[0])[0])
        direction2 = np.arange(0,np.shape(mag_vector[0])[1])

        
        #create the plot                               
        if slice_axis == '0':
            #create 2D energy array for background
            Slice = energy_density[slice_value,:,:]
            #plot either quiver or streamline
            if quiver == True:
                #normalize vectors being used
                mag_vector[1], mag_vector[2] = Normalize_2d(mag_vector[1], mag_vector[2])
                cur_vector[1], cur_vector[2] = Normalize_2d(cur_vector[1], cur_vector[2])
                #plot   
                ax.quiver(direction1[skip], direction2[skip], mag_vector[1][slice_value,skip,skip], mag_vector[2][slice_value,skip,skip], color = 'green', linewidth = 2)
                ax.quiver(direction1[skip], direction2[skip], cur_vector[1][slice_value,skip,skip], cur_vector[2][slice_value,skip,skip], color = 'blue', linewidth = 2)
            #elif quiver == False:
                
                #ax.streamplot(direction1, direction2, turb.magy[slice_value,:,:], turb.magz[slice_value,:,:], color = 'red', density = 1.5)
            
            ax.set_xlabel("Y")
            ax.set_ylabel("Z")
        elif slice_axis == '1':
            #create 2D energy array for background
            Slice = energy_density[:,slice_value,:]
            #plot either quiver or streamline
            if quiver == True:
                #normalize vectors being used
                mag_vector[0], mag_vector[2] = Normalize_2d(mag_vector[0], mag_vector[2])
                cur_vector[0], cur_vector[2] = Normalize_2d(cur_vector[0], cur_vector[2])
                #plot
                ax.quiver(direction1[skip], direction2[skip], mag_vector[0][slice_value,skip,skip], mag_vector[2][slice_value,skip,skip], color = 'green', linewidth = 2)
                ax.quiver(direction1[skip], direction2[skip], cur_vector[0][slice_value,skip,skip], cur_vector[2][slice_value,skip,skip], color = 'blue', linewidth = 2)
                #ax.streamplot(direction1, direction2, turb.magx[:,slice_value,:], turb.magz[:,slice_value,:], color = 'red', density = 1.5)
            
            ax.set_xlabel("X component")
            ax.set_ylabel("Z component")
        elif slice_axis == '2':
            #create 2D energy array for background
            Slice = energy_density[:,:,slice_value]
            #plot either quiver or steamline
            if quiver == True:
                #normalize vectors being used
                mag_vector[0], mag_vector[1] = Normalize_2d(mag_vector[0], mag_vector[1])
                cur_vector[0], cur_vector[1] = Normalize_2d(cur_vector[0], cur_vector[1])
                #plot
                ax.quiver(direction1[skip], direction2[skip], mag_vector[0][slice_value,skip,skip], mag_vector[1][slice_value,skip,skip], color = 'green', linewidth = 2)
                ax.quiver(direction1[skip], direction2[skip], cur_vector[0][slice_value,skip,skip], cur_vector[1][slice_value,skip,skip], color = 'blue', linewidth = 2)
            #elif quiver == False:
                #ax.streamplot(direction1, direction2, turb.magx[:,:,slice_value], turb.magy[:,:,slice_value], color = 'red', density = 1.5)
            
            ax.set_xlabel("X component")
            ax.set_ylabel("Y component")
        else:
            raise ValueError('Invalid axis. Please choose from 0,1,2 for x,y,z respectively.')
        
        #take snapshot for animation
        image = ax.imshow(np.log(Slice), cmap = 'inferno', origin = 'lower', vmin=np.log(minimum), vmax=np.log(maximum))
        cbar = fig.colorbar(image, cax = cax)
        cbar.set_label('$\log \mathcal{E}_{mag} / \mathcal{E}_{kin} $')
        if quiver==True:
            #title for quiver
            ax.text(0.02, 1.08, str(slice_title) + "="+ str(slice_value)+ " slice of $\mathcal{E}_{mag}/\mathcal{E}_{kin}$ with $J=$Blue, $B=$Green at $t = $" + str(i/100) + "$t_0$", transform=ax.transAxes, fontsize=13, verticalalignment='top')
        elif quiver == False:
            #title for streamline
            ax.text(0.02, 1.08, str(slice_title) + "="+ str(slice_value)+ " slice of $\mathcal{E}_{mag}/\mathcal{E}_{kin}$ at $t = $" + str(i/100) + "$t_0$", transform=ax.transAxes, fontsize=13, verticalalignment='top')
       
        #save the plot
        video.animation_frame(f"{directory}",i)

    #generate all the frames    
    Parallel(n_jobs=cores)(delayed(generate_plot)(i) for i in np.arange(1,max_timestep+1))

    #create the animation
    if quiver == True:
        title = 'quiver_animation'
    elif quiver == False:
        title = 'streamline_animation'

    video.create_video(f"{directory}", title, fps_value)

def compute_derivative(axis, data):
    """Will compute the derivative along the specified coordinate axis to the given data array.
        This function assumes the length scale of entire box is normalized to 1. 

    Parameters
    ----------
    axis : int
        0,1,2, corresponding to x,y,z axes respectively
    data : array
        3D spacial array of data to be differentiated
    """
    derivative  = (np.roll(data, +1, axis) - np.roll(data, -1, axis))/(1/np.shape(data)[0])
    return derivative


def compute_current_density(B_vector):
    """Will compute the current density given the magnetic field components. Returns the current vector

    Parameters
    ----------
    B_vector : array
        the components of B, given in a spacial 3D arrangement as [Bx,By,Bz]
    """
    #compute the derivative of the magnetic field components
    dBx_dx = compute_derivative(0, B_vector[0,:,:,:])
    dBx_dy = compute_derivative(1, B_vector[0,:,:,:])
    dBx_dz = compute_derivative(2, B_vector[0,:,:,:])
    dBy_dx = compute_derivative(0, B_vector[1,:,:,:])
    dBy_dy = compute_derivative(1, B_vector[1,:,:,:])
    dBy_dz = compute_derivative(2, B_vector[1,:,:,:])
    dBz_dx = compute_derivative(0, B_vector[2,:,:,:])
    dBz_dy = compute_derivative(1, B_vector[2,:,:,:])
    dBz_dz = compute_derivative(2, B_vector[2,:,:,:])

    #compute the current density components
    Jx = (1/(4*np.pi))*(dBz_dy - dBy_dz)
    Jy = (1/(4*np.pi))*(dBx_dz - dBz_dx)
    Jz = (1/(4*np.pi))*(dBy_dx - dBx_dy)

    return ([Jx,Jy,Jz])


def dot_product(vector_a, vector_b):
    """Will compute the dot product of two vectors

    Parameters
    ----------
    vector_a : array
        3D spacial array of vector components
    vector_b : array
        3D spacial array of vector components
    """
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return (np.sum(vector_a*vector_b, axis = 0))

def Normalize_2d(vec_comp1, vec_comp2):
    """Normalizes vector in 2d vector to a unit vector. Returns unit vector

    Parameters
    ----------
    vec_comp1 : array
        components of vector in first direction
    vec_comp2 : array
        components of vector in second direction
    """
    #compute magnitude of vector
    mag = np.sqrt(vec_comp1**2 + vec_comp2**2)
    #normalize vector
    vec_comp1 = vec_comp1/mag
    vec_comp2 = vec_comp2/mag
    return ([vec_comp1, vec_comp2])

def vector_angle(vector_a, vector_b):
    """Will compute the angle between two vectors

    Parameters
    ----------
    vector_a : array
        3D spacial array of vector components
    vector_b : array
        3D spacial array of vector components
    """
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return np.arccos(dot_product(vector_a, vector_b)/(np.sqrt(dot_product(vector_a, vector_a))*np.sqrt(dot_product(vector_b, vector_b))))

def pi_axis_formatter(val, pos, denomlim=10, pi=r'\pi'):
    """ REQUIRED BY angle_pdf() FUNCTION
    format label properly with pi values
    for example: 0.6666 pi --> 2π/3
               : 0      pi --> 0
               : 0.50   pi --> π/2  
    """
    minus = "-" if val < 0 else ""
    val = abs(val)
    ratio = frac(val/np.pi).limit_denominator(denomlim)
    n, d = ratio.numerator, ratio.denominator
    
    fmt2 = "%s" % d 
    if n == 0:
        fmt1 = "0"
    elif n == 1:
        fmt1 = pi
    else:
        fmt1 = r"%s%s" % (n,pi)
        
    fmtstring = "$" + minus + (fmt1 if d == 1 else r"{%s}/{%s}" % (fmt1, fmt2)) + "$"
    
    return fmtstring



def angle_pdf(filepath, timesteps, bins, animate = False):
    """Creates the probability distribution function of the angle between J and B for a list of timesteps

        Returns either a plot of all specified timesteps, or an animation that shows the evolution through the
        timesteps.

    Parameters
    ----------
    filepath : str
        string of filepath to data folder
    timesteps : array
        array containing the timesteps wished to be plotted
    bins : int
        number of bins to be used in the histogram
    save : bool, optional
        If True, will save the animation/plot as a .mp4//pdf file in the working directory. The default is False.
    animate : bool, optional
        If True, the output of the function will be an animated plot instead of a static plot. The default is False.
    """
    ticklen = np.pi/3
    #for i in timesteps:
    def analyze_frame(i):
        fig,ax = plt.subplots(figsize=(7,5))  
        ax.set_xlim(0,np.pi)
        ax.set_ylim(0,0.75)

        # setting ticks labels
        ax.xaxis.set_major_formatter(FuncFormatter(pi_axis_formatter))
        # setting ticks at proper numbers
        ax.xaxis.set_major_locator(MultipleLocator(base=ticklen))

        print(f"Analyzing frame {i}...")
        data = DataLoader(filepath,i)
        turb = data.turb
        turb.derived_var("cur")
        turb.read("mag")
        angle = vector_angle(turb.cur, turb.mag)

        #flatten the angle array
        angle = angle.flatten()

        #create histogram
        plt.title("Probability Density of $J$, $B$ Angle")
        plot = plt.hist(angle, bins = bins, density = True, histtype = 'step', color='black', label = "Timestep " + str(i))        
        plt.xlabel('Angle between $J$ and $B$ (radians)')
        plt.ylabel('Probability')
        plt.legend()
        if animate == True:
            #save this to the temp directory where videos are made
            video.animation_frame(f"{filepath}",i)
        else:
            #save it to the plots directory
            plt.savefig(f"{filepath}Plots\\angle_pdf_{i}.pdf")
        fig.clear()
        plt.close(fig)

    #perform the parallelized computation
    Parallel(n_jobs=cores)(delayed(analyze_frame)(i) for i in timesteps)
    if animate == True:
        video.create_video(f"{filepath}", 'angle_pdf', 10)

def abs_k_peak(k_array, intensity_array, gridsize=64):
    #DEFINES NUMBER OF POINTS TO CONSIDER ON EACH SIDE OF GLOBAL MAX
    N=2
    intensity_mag=list(intensity_array)
    peak_index = intensity_mag.index(np.max(intensity_mag))
    if peak_index >= gridsize/2 - N:
        peak_index = gridsize/2 - N
    if peak_index <= N:
        peak_index = N
    '''
    #interpolate about this maximum, then find the true maximum
    k_array.tolist()
    #interp_func = interp1d(k_array[peak_index-N:peak_index+N], intensity_mag[peak_index-N:peak_index+N], kind='quadratic',fill_value = 'extrapolate')
    interp_func = interp1d(k_array, intensity_mag, kind='quadratic',fill_value = 'extrapolate')
    #continuous_array = np.linspace(k_array[peak_index-N], k_array[peak_index+N-1], 100)
    continuous_array = np.linspace(k_array[0], k_array[-1], 1000)
    model = interp_func(continuous_array).tolist()
    #find where the maximum of the model is, then take the corresponding k value
    max_index = model.index(np.max(model))
    return continuous_array[max_index], [continuous_array, model]
    '''
    #perform quartic fit
    params, pcov = curve_fit(quartic, k_array[0:20], intensity_array[0:20])
    continuous_array = np.linspace(k_array[0], k_array[20], 1000)
    model = quartic(continuous_array, *params).tolist()
    max_index = model.index(np.max(model))
    return continuous_array[max_index], [continuous_array, model]

def new_power_spectra(filepath, timesteps, quantity, yrange, animate = False):
    """Creates the 1D power spectra of a quantity for a list of timesteps. If animate is false, result is a plot of the different
    timesteps. If animate is true, result is an animation of the different timesteps.
    If check_scaling = True, will plot power spectra divided by k^power to understand k scaling of the power spectrum

    Parameters
    ----------
    filepath : str
        location of data files
    timesteps : array
        timesteps to consider for power spectra
    quantity : str
        mag_energy, kin_energy, or both are the possible inputs
    save : bool, optional
        Saves file in working directory, by default False
    animate : bool, optional
        If true, will return as an animation, otherwise returns a simple plot with each timestep
         overlayed. by default False
    yrange : array
        yrange of plot

    """
    if animate:
        def create_frame(i):
            print(f"Analyzing frame {i}...")
            fig = plt.figure(figsize=(11,5))
            plt.xlabel(r"$$kL/2\pi$$")
            plt.ylabel("Power")
            plt.ylim(yrange[0],yrange[1])
            
            #load in data
            data = DataLoader(filepath,i)
            turb = data.turb
            
            if quantity == 'both':
                turb.read('mag')
                turb.read('vel')
                turb.read('dens')

                #generate the power spectra
                k_array_mag, intensity_mag = mag_spectrum(turb.mag)
                k_array_kin, intensity_kin = kin_spectrum(turb.vel, turb.dens[0])
                plt.loglog(k_array_mag, intensity_mag, label = "$\mathcal{E}_{\rm{mag},k}$")
                plt.loglog(k_array_kin, intensity_kin, label = "$\mathcal{E}_{\rm{kin},k}$")
                plt.legend()
            
            elif quantity == 'mag_energy':
                turb.read('mag')
                k_array, intensity = mag_spectrum(turb.mag)
                plt.loglog(k_array, intensity, label = "$\mathcal{E}_{\rm{mag},k}$")
            
            elif quantity == 'kin_energy':
                turb.read('vel')
                turb.read('dens')
                k_array, intensity = kin_spectrum(turb.vel, turb.dens[0])
                plt.loglog(k_array, intensity, label = "$\mathcal{E}_{\rm{kin},k}$")
            
            plt.title(f"Power Spectrum at $t = {i/100}t_0$")

            #save the frame
            video.animation_frame(f"{filepath}",i)
            #clear figures
            fig.clear()
            plt.close(fig)
        
        #clear temp directory
        video.clear_temp(base_directory=f"{filepath}")
        #generate the frames
        Parallel(n_jobs=cores)(delayed(create_frame)(i) for i in timesteps)
        #create the animation
        video.create_video(f"{filepath}", f'power_spectra_{quantity}', 10)
    
    elif not animate:
        plt.figure(figsize=(11,5))
        plt.xlabel(r"$$kL/2\pi$$")
        plt.ylabel("Power")
        plt.ylim(yrange[0],yrange[1])
        color_list_mag = linear_gradient("#85FFFF", "#1300FF", len(timesteps))['hex']
        color_list_kin = linear_gradient("#FF6A6A", "#FF0000", len(timesteps))['hex']
        for i in timesteps:
            print(f"Analyzing frame {i}")
            data = DataLoader(filepath,i)
            turb = data.turb

            if quantity == 'mag_energy':
                savestring = "$\mathcal{E}_{mag}$"
                turb.read('mag')
                k_array, intensity = mag_spectrum(turb.mag)
                plt.loglog(k_array, intensity, color = color_list_mag[timesteps.index(i)])
            
            elif quantity == 'kin_energy':
                savestring = "$\mathcal{E}_{kin}$"
                turb.read('vel')
                turb.read('dens')
                k_array, intensity = kin_spectrum(turb.vel, turb.dens[0])
                plt.loglog(k_array, intensity, color = color_list_kin[timesteps.index(i)])
            
            elif quantity == 'both':
                Warning("Function does not support plotting both due to clutter.")
        
        plt.title(f"Power Spectrum of {savestring}")
        plt.savefig(f"{filepath}Plots\\power_spectra_{quantity}.pdf")
        plt.show()

def mag_spectrum(mag_vector):
    """Returns the 1D magnetic energy spectrum. Done by computing the spectrum of each component square
    then adds to get total spectrum

    Parameters
    ----------
    mag_vector : vector
        Magnetic energy vector in 3D
    """
    #compute 3d spectrum
    mag_spectrum = ps.compute_power_spectrum_3D(mag_vector)
    #integrate over sphere
    k_mag, intensity_mag = ps.spherical_integrate(mag_spectrum)
    intensity = intensity_mag/(8*np.pi)   #compute mag energy
    return([k_mag, intensity])

def kin_spectrum(vel_vector, density):
    """Returns the 1D kinetic energy spectrum. Done by computing the spectrum of each component square. Assumes density is constant

    Parameters
    ----------
    vel_vector : vector
        velocity vector in 3D
    density : scalar
        density of gas for kinetic energy
    """
    #compute 3d spectrum
    kin_spectrum = ps.compute_power_spectrum_3D(vel_vector)
    #integrate over sphere
    k_kin, intensity_kin = ps.spherical_integrate(kin_spectrum)
    intensity = (intensity_kin)*np.mean(density)/2    #compute kinetic energy
    return([k_kin, intensity])

def normalize_spectrum(k_array, intensity_array):
    """Will normalize the power spectrum by dividing by integral of the power spectrum

    Parameters
    ----------
    k_array : array
        k_space array
    intensity_array : array
        power spectrum intensity array
    """
    #compute integral
    integral = np.trapz(intensity_array, k_array)
    #normalize
    intensity_array = intensity_array/integral
    return(intensity_array)

def parsevel_check(filepath, timesteps, quantity, save=False, rho = 1):
    """Creates plot that displays the difference of parsevels theorem over time
    i.e. shows the value of integral dk E(k) (- 1/2 rho_0 <v^2>_V for kinetic energy, -1/8pi <B^2>_V for magnetic energy)

    Parameters
    ----------
    filepath : string
        location of data folder
    timesteps : array
        timesteps you wish to check the theorem for
    quantity : string
        either 'kin_energy' or 'mag_energy'
    save : bool, optional
        saves the file to pdf, by default False
    rho : float, optional
        initial density used in calculation of kinetic spectrum. Assumes subsonic flow so that density is
        essentially constant throughout time. default is 1
    """
    def do_work(i):   #defines the function used in the parallelization
        print(f"Analyzing frame {i}")
        #read data
        data = DataLoader(filepath,i)
        turb = data.turb
        #compute the energy
        
        if quantity == 'kin_energy':
            turb.read('vel')

            #compute the kinetic energy spectrum
            k_array, intensity = kin_spectrum(turb.vel,rho)
            #compute the integral
            integral = np.trapz(intensity, k_array)
            #compute the mean velocity square
            mean_vel2 = np.mean(turb.vel[0]**2 + turb.vel[1]**2 + turb.vel[2]**2)
            rhs = 0.5 * rho * mean_vel2    #computes rhs
            result = integral - rhs
        elif quantity == 'mag_energy':
            turb.read('mag')

            #compute the magnetic energy spectrum
            k_array, intensity = mag_spectrum(turb.mag)
            #compute the integral
            integral = np.trapz(intensity, k_array)
            #compute the mean magnetic field square
            mean_mag2 = np.mean(turb.mag[0]**2 + turb.mag[1]**2 + turb.mag[2]**2)
            rhs = (1/(8*np.pi)) * mean_mag2
            result = integral - rhs
        return result
    
    result_array = Parallel(n_jobs=cores)(delayed(do_work)(i) for i in timesteps)    
    #plot results, should be very close to 0 for all time if power spectrum is correct
    plt.figure()
    plt.plot(timesteps/100, result_array)
    plt.xlabel('Time ($t_0$)')
    plt.ylabel("Difference from Parsevel's Theorem")
    if save == True:
        plt.savefig(f'{filepath}Plots\\parsevel_check_' + quantity + '.pdf')
    else:
        plt.show()

def bmag_theta_pdf(filepath, timesteps,x_limit,angle_quantity):
    """Creates a 2d pdf to investigate the relationship between the magnitude of B and the angle between a quantity and B
    The quantities are current and velocity

    Parameters
    ----------
    filepath : string
        string of filepath to data folder
    timesteps : array
        array containing the timesteps wished to be plotted
    x_limit : float
        upper limit of x axis
    angle_quantity : string
        either 'current' or 'velocity'
    save : bool, optional
        Saves to current working directory, by default False
    """
    def do_work(i):
        print(f"Analyzing frame {i}...")
        fig,ax = plt.subplots()
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        ax.set_xlim(9e-4,np.log(x_limit*100000)*10)
        ax.set_ylim(0,np.pi)
        ax.set_xlabel(r'$\ln(B/B_{\rm rms})$')
        if angle_quantity == 'current':        
            ax.set_ylabel(r'$\theta$ between $J$ and $B$ (rad)')
        if angle_quantity == 'velocity':
            ax.set_ylabel(r'$\theta$ between $V$ and $B$ (rad)')
        #load data
        data = DataLoader(filepath,i)
        turb = data.turb
        #compute angle
        turb.read('mag')
        if angle_quantity == 'current':
            turb.derived_var("cur")
            angle = vector_angle(turb.cur, turb.mag)
        elif angle_quantity == 'velocity':
            turb.read('vel')
            angle = vector_angle(turb.vel, turb.mag)
        else:
            raise ValueError("angle_quantity must be 'current' or 'velocity'")
        #flatten the angle into 1D array
        angle = angle.flatten()
        

        #compute magnetic field magnitude ratio
        bmag = np.sqrt(turb.mag[0]**2 + turb.mag[1]**2 + turb.mag[2]**2)
        bmag = bmag/np.sqrt(np.mean(bmag**2))

        #flatten the bmag into 1D array
        bmag = bmag.flatten()

        #plot the hexplot
        if angle_quantity == 'current':
            ax.text(0.02, 1.06, r"PDF of $\vec{J}$, $\vec{B}$ angle and $\vec{B}$ magnitude at " + str(i/100) + "$t_0$", transform=ax.transAxes)
        elif angle_quantity == 'velocity':
            ax.text(0.02, 1.06, r"PDF of $\vec{V}$, $\vec{B}$ angle and $\vec{B}$ magnitude at " + str(i/100) + "$t_0$", transform=ax.transAxes)
        image = ax.hexbin(bmag, angle, gridsize = (round(np.sqrt(3)*200),70), cmap = 'inferno',xscale='log', bins = 'log', mincnt = 1, vmin=1, vmax=2e2, extent=(-3,np.log(x_limit),0,np.pi))
        cbar = fig.colorbar(image, cax = cax)
        cbar.set_label('Counts')
        #generate the frame
        video.animation_frame(f"{filepath}",i)
        fig.clear()   #close fig to save memory
        plt.close(fig)
    #generate the frames for the video
    Parallel(n_jobs=cores)(delayed(do_work)(i) for i in timesteps)
    #construct the animation
    video.create_video(filepath, 'bmag_theta_' + str(angle_quantity), 7)

def taylor_microscale(filepath, timesteps, quantity, save=False,check_velocity=False, fit_endpoints = [0,0]):
    """Computes the taylor micrscale, defined as integral k^2 P_B(k)dk/ integral P_B(k)dk

    Parameters
    ----------
    filename: str
        locatoin of data files 
    quantity: str
        quantity to compute taylor microscale for
    timesteps: array
        array of timesteps to compute taylor microscale
    check_velocity: bool, optional
        If true, will perform a linear fit to the microscale after converting from k space to lengths
        from the values t_i to t_f.
    fit_endpoints: array, optional
        Needed if check_velocity = True. Tuple of the form (t_i, t_f) where t_i is the initial file index
        and t_f is the final file index to perform the linear fit over.
    """
    #needed for curve fitting to linear function
    def linear_func(x, m, b):
        return m*x + b
    
    plt.figure()
    #for i in timesteps:
    def do_work(i):
        print(f"Analyzing frame {i}...")
        data = DataLoader(filepath,i)
        turb = data.turb
        turb.read('mag')
        turb.read('dens')
        turb.read('vel')

        k_array = np.array([])
        spectrum = np.array([])
        #compute power spectra
        if quantity == 'mag_energy':
            k_array, spectrum = mag_spectrum(turb.mag)
        if quantity == 'kin_energy':
            k_array, spectrum = kin_spectrum(turb.vel, 1)
        
        #compute the taylor microscale
        numerator = np.trapz((k_array**2)*spectrum, k_array)
        denominator = np.trapz(spectrum, k_array)
        result = np.sqrt(numerator/denominator)
        return i/100, result
    output = np.transpose(Parallel(n_jobs=cores)(delayed(do_work)(i) for i in timesteps))
    time_array = output[0]
    scale_array = output[1]

    #compute the velocity scales if requested
    if check_velocity == True and quantity == 'mag_energy':
        #convert spectra to length space
        length_array = np.reciprocal(scale_array)
        #perform linear fit to the length array
        popt, pcov = curve_fit(linear_func, time_array[fit_endpoints[0]:fit_endpoints[1]], length_array[fit_endpoints[0]:fit_endpoints[1]])
        #plot the fit
        x_array = np.linspace(time_array[fit_endpoints[0]-5], time_array[fit_endpoints[1]+5], 100)
        plt.plot(x_array, linear_func(x_array, popt[0],popt[1]), linestyle = 'dashed', color = 'red')
        plt.plot(time_array, length_array, color = 'black')
        plt.ylabel(r'Taylor Microscale $l_u$')
        print("The slope is: " + str(popt[0]))
        plt.text(0.5,0.5, "Slope = " + str(round(popt[0],3)), transform=plt.gca().transAxes, verticalalignment='top')
    else:
        plt.plot(time_array, scale_array)
        plt.ylabel(r'Taylor Microscale $k_\mu$')
    plt.xlabel('Time ($t_0$)')
    
    title = ""
    if quantity == 'mag_energy':
        title = 'Taylor Microscale for Magnetic Energy'
    elif quantity == 'kin_energy':
        title = 'Taylor Microscale for Kinetic Energy'
    plt.title(title)
    if save == True:
        if check_velocity == True:
            plt.savefig(f'{filepath}Plots\\taylor_microscale' + str(quantity) + '_fit.pdf')
        else:
            plt.savefig(f'{filepath}Plots\\taylor_microscale' + str(quantity) + '.pdf')
    else:
        plt.show()

def stretching_power_spectrum(filepath, timesteps, save=False):
    """Generates the power spectrum of b_i b_j S_ij, the symmetric component Sij of the velocity gradient tensor with mag field
    unit vectors b_i b_j.

    Parameters
    ----------
    filepath : string
        Folder containing the data
    timesteps : array
        Time steps to compute the power spectrum for
    save : bool, optional
        If true, will save the output to the current working directory, by default False
    """

    def do_work(i):
        print(f"Analyzing frame {i}...")
        #initialize plot        
        fig = plt.figure()
        #compute the contraction
        contraction,_,_ = vel_grad_decomp(filepath, i)

        #compute the power spectrum of the contraction value
        spectrum = ps.compute_power_spectrum_3D(contraction)
        k_mode, radial_sum = ps.radial_integrate(spectrum, bins=32)

        #plot the power spectrum as an animation
        plt.loglog(k_mode, radial_sum, color='black')
        plt.ylim([1e-8,1e-1])
        plt.text(0.02, 1.08, r'Power Spectrum of $\hat{b}_i \hat{b}_j S_{ij}$ at $t = $ ' + str(i/100) + r"$t_0$", transform=plt.gca().transAxes, verticalalignment='top')
        plt.xlabel(r'$k$-mode')
        plt.ylabel("Power")
        #generate the frame
        video.animation_frame(f"{filepath}",i)
        #close figures to save RAM
        fig.clear()
        plt.close(fig)
    #generate the frames
    Parallel(n_jobs=cores)(delayed(do_work)(i) for i in timesteps)
    #save the animation
    video.create_video(filepath,'stretching_power_spectrum',10)

def vel_grad_decomp(filepath, i):
    #load data
    data = DataLoader(filepath,i)
    turb = data.turb
    #load the velocity field
    turb.read('vel')
    turb.read('mag')
    velocity_field = np.array(turb.vel)

    #compute the velocity gradient
    vel_gradient = dvar.gradient_tensor(velocity_field, order = 6) 

    #perform the decomposition into the bulk, symmetric, and antisymmetric tensors
    sym, anti, bulk = dvar.orthogonal_tensor_decomposition(vel_gradient, sym=False, all=True )

    #find the mag field unit vectors
    mag_field = np.array(turb.mag)
    mag_field_unit = mag_field/np.sqrt(dot_product(mag_field, mag_field))

    #next contract bibj
    outer_prod = np.einsum("i...,j...->ij...",mag_field,mag_field)
    contraction = np.einsum("ij...,ij...->...",outer_prod,sym)
    mean_mag_field = np.mean(dot_product(mag_field,mag_field))
    return contraction,mean_mag_field,outer_prod

def absolute_angle(angle):
    """Finds the reference angle for angles between 0 to pi

    Parameters
    ----------
    angle : np.ndarray
        Array of the angles that you wish to convert, between 0 to pi

    Returns
    -------
    np.ndarray
        Array of the reference angles, now betwewen 0 to pi/2
    """
    condition_list = [angle < np.pi/2, angle >= np.pi/2]  #do a vectorized conversion
    choice_list = [angle, np.pi - angle]
    new_angle = np.select(condition_list, choice_list)

    return new_angle

def solve_evp(directory,i):
    """Will solve for the eigenvalues and eigenvectors. Will not return eigenvectors as they are not needed. Can be modified
    if we do need the other eigenvectors.

    Parameters
    ----------
    directory : string
        path to the data, with double slashes (e.g. "C:\\Users\\shash\\Desktop\\Assignments\\AST425\\Data\\")
    i : int
        time step to solve the eigenvalue problem for
    
    Returns
    -------
    angle : np.ndarray
        Array of the angles between the stretching eigenvector and the magnetic field
    eig_compress : np.ndarray
        Array of the compressive eigenvalues
    eig_null : np.ndarray
        Array of the null eigenvalues
    eig_stretch : np.ndarray
        Array of the stretching eigenvalues
    """
    #load data
    data = DataLoader(directory, i)
    turb = data.turb
    turb.read('vel')
    turb.read('mag')

    #compute the velocity gradient tensor
    vel_gradient = dvar.gradient_tensor(np.array(turb.vel), order = 4)

    #define the magnetic vector field
    mag_vector = np.array(turb.mag)

    #compute the eigenvalues
    eigvals, eigvecs = dvar.symmetric_eigvals(mag_vector, vel_gradient, find_vectors=True)

    #separate eigenvalues into different arrays
    eig_compress = eigvals[0,:,:,:]
    eig_null = eigvals[1,:,:,:]
    eig_stretch = eigvals[2,:,:,:]

    #define the eigenvector with the largest eigenvalue to be the stretching eigenvector
    stretching_vector = eigvecs[2,:,:,:,:]
    
    #compute the angle between eigenvector and magnetic vector
    angle = vector_angle(stretching_vector, mag_vector)

    #take absolute cosine
    angle = np.abs(np.cos(angle))

    return angle, eig_compress, eig_null, eig_stretch

