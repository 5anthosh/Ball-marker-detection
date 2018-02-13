import cv2
import numpy as np 

folder = "croped_renamed"
 
def convert_to_array(number):
    """ read the image name number and convert into numpy array
    
    Arguments:
        number {int} -- image number
    
    Returns:
        array_np {numpy.ndarray} -- numpy array for image
    """

    array_cv = cv2.imread(folder + "/" + str(number).zfill(3) + ".jpg")
    array_np = np.array([array_cv])
    return array_np


number_of_images = 47
training_image_numpy_array = convert_to_array(1)
for i in range(2, number_of_images+1):
    training_image_numpy_array =  np.vstack((training_image_numpy_array, convert_to_array(i)))

print(training_image_numpy_array.shape)
np.save("training_image",training_image_numpy_array)