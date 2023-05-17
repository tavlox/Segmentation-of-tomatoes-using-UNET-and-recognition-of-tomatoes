import torch
from network import U_Net
import pickle
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
#unet = U_Net().to('cuda:0')
path_load = r'/home/ivae/Desktop/seminarska_ST/outputs_UNET/unet_weights_original.pth'
path_load_home = r'C:\Users\cr008\OneDrive\Desktop\outputs_new\outputs_UNET/unet_weights_original.pth'
model = U_Net()

checkpoint = torch.load(path_load_home, map_location = 'cpu')
#unet.load_state_dict(checkpoint['state_dict'])
epoch = checkpoint['epoch']
#weights = checkpoint['state_dict']

##### za originalne slike najboljsa epoha 89
#print('Epoch: ',epoch)
#print('Weights: ',weights)
open_data = open(r'C:\Users\cr008\OneDrive\Desktop\results_new\results\Loss_acc.picl', 'rb')
open_data_augmented =  open(r'C:\Users\cr008\OneDrive\Desktop\results_UNET\results\Loss_acc.picl', 'rb')
data  = pickle.load(open_data)
best_acc = max(data['Average acc'])
print(best_acc)
epochs_best = data['Epoch_best_acc']
#print(epochs_best)
#print(data['Average acc'].index(best_acc))
#print(max((data['Best acc'])))
#plt.figure()
#plt.plot(epochs_best, data['Best acc'])
#plt.xlabel('epoch')
#plt.ylabel('Best accuracy')
#plt.title('Best accuracy on augmented data')
#plt.show()
# plot best acc
#plt.plot(epoch_list, data['Best acc'])
#plt.xlabel('epoch')
#plt.ylabel('Best accuracy')
#plt.title('Best accuracy on unaugmented data')
#plt.show()




path_original = r'C:\Users\cr008\OneDrive\Desktop\seminarska_ST_real\seminarska_ST\laboro_tomato_original\test\gt\tomato\*.png'
path_original = sorted(glob.glob(path_original))
path_gt_original = r'C:\Users\cr008\OneDrive\Desktop\seminarska_ST_real\seminarska_ST\laboro_tomato_original\test\masks\tomato\*.png'
path_gt_original = sorted(glob.glob(path_gt_original))
# v ta folder so rocno premaknjene slike iz 86 epoha (najboljsi acc)
path_outputs_original = r'C:\Users\cr008\OneDrive\Desktop\outputs_best\*.png'
path_outputs_original = sorted(glob.glob(path_outputs_original))



 
def tri_slike(index):
    original = cv2.imread(path_original[index])
    output = cv2.imread(path_outputs_original[index])
    maska = cv2.imread(path_gt_original[index])
    ime = path_original[index]
    return original, output, maska, ime

path_stacked = r'C:\Users\cr008\OneDrive\Desktop\outputs_best_stacked'

#for i in range(0,160):
    #original, output, maska, ime = tri_slike(i)
    #output_path = path_stacked  + '\\' + (ime.split('\\')[11]).split('.')[0] + '_stacked' + '.png'
    #numpy_horizontal = np.hstack((output, original, maska))
    #cv2.imwrite(output_path,numpy_horizontal)







