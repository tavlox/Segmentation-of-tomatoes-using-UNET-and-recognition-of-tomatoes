import albumentations as A
import cv2
import os 
import shutil
from PIL import Image

path_slike_pa = r'/home/ivae/Desktop/seminarska_ST/laboro_tomato_original/train/images/tomato'
path_maske_1 = r'/home/ivae/Desktop/seminarska_ST/laboro_tomato_original/train/masks/tomato'
#path_slike_pa = r'/home/modi1s9/Desktop/seminarska_ST/laboro_tomato/train/images/tomato'
#path_maske_1 = r'/home/modi1s9/Desktop/seminarska_ST/laboro_tomato/train/masks/tomato'
#path_slike_pa = r'C:\Users\cr008\OneDrive\Desktop\seminarska_ST_real\seminarska_ST\laboro_tomato_original\train\images\tomato'
#path_maske_1 = r'C:\Users\cr008\OneDrive\Desktop\seminarska_ST_real\seminarska_ST\laboro_tomato_original\train\masks\tomato'

transforms_real = [A.HorizontalFlip(p=0.5), A.Resize(p=1, height=512, width=384, interpolation=2),A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1)]

def load_images(path):
    images = []
    for i in sorted(os.listdir(path)):
        img = cv2.imread(os.path.join(path, i))
        images.append(img)
    return images

train_images = load_images(path_slike_pa)
train_masks = load_images(path_maske_1)




def augment(transforms, path_slike, path_maske, stevec_slike, stevec_maske):
    transformed_images = []
    transformed_images_mask = []
    for i in range(0, len(train_images)):
        transform = A.Compose(transforms)
        curr_image = train_images[i]
        curr_mask = train_masks[i]
        curr_image_RGB = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
        curr_image_RGB = cv2.rotate(curr_image_RGB, cv2.ROTATE_90_COUNTERCLOCKWISE)
        transformed = transform(image = curr_image_RGB, mask = curr_mask)
        transformed_image = transformed["image"]
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        transformed_mask = transformed["mask"]
        transformed_images.append(transformed_image)
        transformed_images_mask.append(transformed_mask)
    

    for j,k in enumerate(transformed_images):
        ime_slike = 'augmented' + str(stevec_slike) + '.png'
        os.chdir(path_slike)
        cv2.imwrite(ime_slike, k)
        stevec_slike+=1
    

    for j,k in enumerate(transformed_images_mask):
        ime_slike = 'augmented_maske' + str(stevec_maske) + '.png'
        os.chdir(path_maske)
        img_gray = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(ime_slike, img_gray)
        stevec_maske+=1

train_augmented = r'/home/ivae/Desktop/seminarska_ST/augmented_new/train/images'
train_mask_augmented = r'/home/ivae/Desktop/seminarska_ST/augmented_new/train/masks'
#train_augmented = r'C:\Users\cr008\OneDrive\Desktop\augmented_new\train\images'
#train_mask_augmented = r'C:\Users\cr008\OneDrive\Desktop\augmented_new\train\masks'


    


###################### augmentation on training images and masks ###############################

# horizontal flip 643 train images
#augment(transforms_real, train_augmented, train_mask_augmented,1,1)

transform_brigthness = [A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=1.0), A.Resize(p=1, height=512, width=384, interpolation=2), A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1)] 
# contrast changed 643 train images
#augment(transform_brigthness, train_augmented, train_mask_augmented,644,644)

# contrast changed and flipped 643 train images
transform_contrast_flipped = [A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=1.0),A.HorizontalFlip(p=0.5), A.Resize(p=1, height=512, width=384, interpolation=2), A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1)]
#augment(transform_contrast_flipped, train_augmented, train_mask_augmented,1287,1287)

# shift_scale_rotate train images
transform_shiftScaleRotate = [A.ShiftScaleRotate(p=0.5, shift_limit_x=(-0.06, 0.06), shift_limit_y = (-0.06, 0.06), scale_limit=(-0.03, 0.03),rotate_limit = (-20, 20), interpolation=2, rotate_method = 'largest_box'), A.Resize(p=1, height=512, width=384, interpolation=2), A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1)]
#augment(transform_shiftScaleRotate, train_augmented, train_mask_augmented,1930,1930)




## move augmented images to laboro_tomato original

def move_augmented(path_in, path_out):
    for i in os.listdir(path_in):
        path = os.path.join(path_in, i)
        shutil.copy2(path, path_out)

#move_augmented(test_mask_augmented, path_test_maske)