import colortrans
import numpy as np
from PIL import Image

# Load data
content = np.array(Image.open('utils/unlabeled_target_img.png').convert('RGB'))
reference = np.array(Image.open('utils/labeled_source_img.png').convert('RGB'))

# Transfer colors using different algorithms
output_lhm = colortrans.transfer_lhm(content, reference)
output_pccm = colortrans.transfer_pccm(content, reference)
output_reinhard = colortrans.transfer_reinhard(content, reference)

# Save outputs
Image.fromarray(output_lhm).save('output1.jpg')
Image.fromarray(output_pccm).save('output2.jpg')
Image.fromarray(output_reinhard).save('output3.jpg')