from torchvision import transforms
from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
 
# load the image
#img_path = '/Users/manukastratta/Developer/CS329D/test-time-training-project/data/camelyon17_v1.0/patches/patient_087_node_0/patch_patient_087_node_0_x_37920_y_15008.png'
img_path = '/Users/manukastratta/Developer/CS329D/test-time-training-project/data/camelyon17_v1.0/patches/patient_096_node_0/patch_patient_096_node_0_x_33760_y_10016.png'
img = Image.open(img_path)

# Convert image to a 4-channel image
img = img.convert('RGBA')
# Convert image to a 3-channel image by discarding the alpha channel
img = img.convert('RGB')
plt.imshow(img)
plt.show()


# define custom transform
# here we are using our calculated
# mean & std

# mean = [0.720475903842406, 0.5598634658657207, 0.7148542202373653]
# std = [0.02169931605678509, 0.025169192291280777, 0.017899670079910467]

mean = [183.72132071, 142.7651698, 182.28779395]
std = [34.25966575, 39.11443915, 28.19893461]

mean = [x/255 for x in mean]
std = [x/255 for x in std]
print(mean)
print(std)

transform_norm = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
 
# get normalized image
img_normalized = transform_norm(img)
print("img_normalized.shape: ", img_normalized.shape)


# convert this image to numpy array
img_normalized = np.array(img_normalized)
 
# transpose from shape of (3,,) to shape of (,,3)
img_normalized = img_normalized.transpose(1, 2, 0)
 
# display the normalized image
plt.imshow(img_normalized)
plt.show()