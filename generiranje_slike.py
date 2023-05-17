import json 
import os
from PIL import Image
import numpy as np
import random

file = open(r'/home/modi1s9/Desktop/laboro_tomato/annotations.json')
data = json.load(file)
path_train = r'/home/modi1s9/Desktop/laboro_tomato/train/images/tomato'
path_test = r'/home/modi1s9/Desktop/laboro_tomato/test/images/tomato'
train_slike = os.listdir(path_train)
tomato_train = []
tomato_test = []
## 7781 paradiznikov v train
## 1996 paradiznikov v test

# 804 je len(data)
#print(len(data))

############# cropped tomatoes ####################
path_cropped = r'/home/modi1s9/Desktop/cropped/train'
path_test_cropped = r'/home/modi1s9/Desktop/cropped/test'
n=1
n_test = 1
for ime_slike in data:
    if ime_slike in train_slike:
        path_train_real = os.path.join(path_train, ime_slike)
        x = data[ime_slike]
        tomato_train.append(len(x))
        #print(sum(tomato_train))
        img = Image.open(path_train_real)
        for j in range(0, len(x)):
                x_1 = x[j][0][0]
                y_1 = x[j][0][1]
                x_2 = x[j][0][2]
                y_2 = x[j][0][3]
                img_crop = img.crop((x_1, y_1, x_2, y_2))
                ime_slike = 'cropped' + str(n) + '.png'
                path_slike = os.path.join(path_cropped, ime_slike)
                img_crop.save(path_slike)
                n+=1
    else:
         x = data[ime_slike]
         tomato_test.append(len(x))
         #print(sum(tomato_test))
         path_test_real = os.path.join(path_test,ime_slike)
         img = Image.open(path_test_real)
         for j in range(0, len(x)):
                x_1 = x[j][0][0]
                y_1 = x[j][0][1]
                x_2 = x[j][0][2]
                y_2 = x[j][0][3]
                img_crop = img.crop((x_1, y_1, x_2, y_2))
                ime_slike = 'cropped_test' + str(n_test) + '.png'
                path_slike = os.path.join(path_test_cropped, ime_slike)
                img_crop.save(path_slike)
                n_test+=1
    
#######################  extract background from tomatoes ###############################
def load_masks(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = Image.open(os.path.join(folder,filename))
        images.append(img)
    return images    
train_masks = r'/home/modi1s9/Desktop/laboro_tomato/train/masks/tomato'
test_masks = r'/home/modi1s9/Desktop/laboro_tomato/test/masks/tomato'
maske_train = load_masks(train_masks)
images_train = load_masks(path_train)
maske_test = load_masks(test_masks)
images_test = load_masks(path_test)
lista_cropped_train = sorted(os.listdir(path_cropped))
lista_cropped_test = sorted(os.listdir(path_test_cropped))
width = 384
height = 512
cropped_background_train = r'/home/modi1s9/Desktop/backgrounds/train'
cropped_background_test = r'/home/modi1s9/Desktop/backgrounds/test'
n_cropped=1
for img in lista_cropped_test:
      for mask in range(0, len(maske_test)):
        cropped_train = Image.open(os.path.join(path_test_cropped,img))
        rect_width, rect_heigth = cropped_train.size
        maska = maske_test[mask]
        image = images_test[mask]
         
        # generate random x and y coordinates within image dimensions of tomatoes
        
        x = random.randint(0, width - rect_width)
        y = random.randint(0, height - rect_heigth)

        for i in range(x, x + rect_width):
            for j in range(y, y + rect_heigth):
                ### crop outside the box coordinates of tomatoes 
                if np.any(np.array(maska.crop((i, j, i+rect_width, j+rect_heigth))) == 1):         
                        x = random.randint(0, width - rect_width)
                        y = random.randint(0, height - rect_heigth)
                        ime = 'cropped' + '_' + str(n_cropped) + '.png'
                        path = os.path.join(cropped_background_test, ime)
                        i=x
                        j=y
                else:
                        ime = 'cropped' + '_' + str(n_cropped) + '.png'
                        path = os.path.join(cropped_background_test, ime)
                        x=i
                        y=j
                        break
            else:
                ime = 'cropped' + '_' + str(n_cropped) + '.png'
                path = os.path.join(cropped_background_test, ime)
                continue
            break
        n_cropped+=1

        cropped_background = image.crop((x, y, x + rect_width, y + rect_heigth))
        cropped_background.save(path)

cropped_remove= r'/home/modi1s9/Desktop/backgrounds/test'
#back_delete=os.listdir(cropped_remove)[2000:] 
#for file in back_delete:
      #os.remove(os.path.join(cropped_remove,file))

  
