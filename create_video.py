from subprocess import call
import shutil
import os
import matplotlib.pyplot as plt
def clear_temp(base_directory):
    """Clears the temporary directory where images are stored.

    Parameters
    ----------
    base_directory : string
        Full filepath directory where data folder is.
        Temp directory will be made here. (e.g. 'C:\\Users\\shash\\Data\\') Must have double back slashes
        like this. Image will be stored in ...\\temp_images\\
    """
    temp_directory = f'{base_directory}temp_images/'  #temporary directory to store images
    #check if the directory exists
    if os.path.exists(temp_directory):
        shutil.rmtree(temp_directory)

def animation_frame(base_directory, identifier):
    """Generates an image for use by create_video function.

    Parameters
    ----------
    base_directory : string
        Full filepath directory where data folder is.
        Temp directory will be made here. (e.g. 'C:\\Users\\shash\\Data\\') Must have double back slashes
        like this. Image will be stored in ...\\temp_images\\
    identifier : int
        Integer identifier for the image, telling the create_video function what frame the image is.

    """
    temp_directory = f'{base_directory}temp_images\\'  #temporary directory to store images    
    #check if the directory exists, and clear it
    if not os.path.exists(temp_directory):
        clear_temp(temp_directory)
        os.makedirs(temp_directory)

    #find the current active figure
    fig = plt.gcf()

    #save the figure to this temporary directory
    fig.savefig(f'{temp_directory}image{identifier}.png')

def create_video(base_dir, output_name, fps):
    """Creates a video from a series of images.

    Parameters
    ----------
    base_dir : string
        Full filepath to the directory where the video will be saved. (e.g. 'C:\\Users\\shash\\Desktop\\Assignments\\AST425\\')
    output_name : string
        Name of the output video file. (e.g. 'eigval_theta_composite')
    fps : int
        Frames per second of the video.
    """
    temporary_dir = f'{base_dir}temp_images\\'     
    #create 
    #generate the image names to make the video from
    call(['ffmpeg','-y','-framerate', f'{fps}', '-i', f"{temporary_dir}image%d.png", '-pix_fmt', 'yuv420p', f'{base_dir}{output_name}.mp4'])


    #once video is made, delete temp_directory
    shutil.rmtree(temporary_dir)

#EXAMPLE USAGE
#create_video('C:\\Users\\shash\\Desktop\\Assignments\\AST425\\','eigval_theta_composite',10)
