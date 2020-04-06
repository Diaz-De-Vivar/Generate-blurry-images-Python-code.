import os
import cv2

# Images paths
path_originals='Path containing the original images'
path_edited='Path where the edited images will be stored'

for i in os.listdir(path_originals):

    img = cv2.imread(os.path.join(path_originals , i))

    blur = cv2.blur(img,(35,35))
    cv2.imwrite(os.path.join(path_edited , i), blur)
    
    # Process check
    print(i)