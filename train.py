from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from network import U_Net
from torch.nn import functional
from torch.optim import Adam
import torch
from torch import nn
import glob
from torch.utils.data.dataset import Dataset  # For custom data-sets
from mine_dataset import CustomDataset
from torchmetrics import JaccardIndex
from torchvision.utils import save_image
import pickle
import cv2
import matplotlib.pyplot as plt

train_directory = r'/home/ivae/Desktop/seminarska_ST/augmented_new/augmented_new/train/images/*.png'
train_data = sorted(glob.glob(train_directory))
train_masks = r'/home/ivae/Desktop/seminarska_ST/augmented_new/augmented_new/train/masks/*.png'
mask_data = sorted(glob.glob(train_masks))
test_directory = r'/home/ivae/Desktop/seminarska_ST/augmented_new/augmented_new/test/images/tomato/*.png'
test_masks = r'/home/ivae/Desktop/seminarska_ST/augmented_new/augmented_new/test/masks/tomato/*.png'
test_data = sorted(glob.glob(test_directory))
mask_test_data = sorted(glob.glob(test_masks))

# on original laboro_tomato
train_directory_original = r'/home/ivae/Desktop/seminarska_ST/laboro_tomato_original/train/images/tomato/*.png'
train_data_original = sorted(glob.glob(train_directory_original))
train_masks_original = r'/home/ivae/Desktop/seminarska_ST/laboro_tomato_original/train/masks/tomato/*.png'
mask_data_original = sorted(glob.glob(train_masks_original))
test_directory_original = r'/home/ivae/Desktop/seminarska_ST/laboro_tomato_original/test/images/tomato/*.png'
test_data_original = sorted(glob.glob(test_directory_original))
test_mask_original = r'/home/ivae/Desktop/seminarska_ST/laboro_tomato_original/test/masks/tomato/*.png'
mask_data_test_original = sorted(glob.glob(test_mask_original))
train_directory_true = r'/home/ivae/Desktop/seminarska_ST/laboro_tomato_original/train/images/tomato'
test_directory_true = r'/home/ivae/Desktop/seminarska_ST/laboro_tomato_original/test/images/tomato'
train_augmented_true = r'/home/ivae/Desktop/seminarska_ST/augmented_new/augmented_new/train/images'
test_augmented_true = r'/home/ivae/Desktop/seminarska_ST/augmented_new/augmented_new/test/images/tomato'



tf_images = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) #definiramo zaporedje funkcij, ki se izvedejo po vrsti
tf_masks = transforms.Compose([transforms.ToTensor()])
train_images = CustomDataset(train_augmented_true, train_data, mask_data, tf_images, tf_masks)
#Data loader
train_images_dl = DataLoader(train_images, batch_size=2, shuffle=True, drop_last=True)

test_images = CustomDataset(test_augmented_true,test_data, mask_test_data, tf_images, tf_masks)
test_images_dl = DataLoader(test_images, batch_size=2, shuffle=True, drop_last=True)





image_folder = r'/home/ivae/Desktop/seminarska_ST/outputs_UNET/masks/'
jaccard_metric = JaccardIndex(task = 'binary', num_classes=2).to('cuda:0')
#Model
unet = U_Net().to('cuda:0')


#Optimization algorithm
optimizer = Adam(unet.parameters(), lr=0.0001) #lr je parameter ki definira learning rate
#Loss function
loss_criterion = torch.nn.BCEWithLogitsLoss()
num_epochs = 100
best_acc = 0

#Initialize a dictionary to store loss from train, average acc and best acc
results = {"Loss_train": [], "Average acc": [], "Best acc": [], "Epoch_best_acc": []}


for epoch in range(num_epochs):
    # Set the UNet to train mode
    unet.train()
    loss_sum = 0
    for images_tr,masks_tr, batch_names in tqdm(train_images_dl): 
        images_tr = images_tr.to("cuda:0")
        masks_tr = masks_tr.to("cuda:0")
        optimizer.zero_grad()
        outputs = unet(images_tr)
        loss = loss_criterion(outputs, masks_tr)
        loss_sum += loss 
        loss.backward() 
        optimizer.step()
        
    avg_loss = loss_sum/len(train_images_dl)   
    print("Epoch_train {}/{}, Loss: {:.3f}".format(epoch+1,num_epochs, loss_sum/len(train_images_dl)))
    results["Loss_train"].append(avg_loss.cpu().detach().numpy())
    # test mode
    unet.eval()
    jaccard_sum=0
    c=1
    
    with torch.no_grad():
        for te_images, te_masks, batch_names in tqdm(test_images_dl):
                te_images = te_images.to("cuda:0")
                te_masks = te_masks.type(torch.int64).to("cuda:0")
                output_test = unet(te_images).to('cuda:0')
                output_test = torch.sigmoid(output_test)
                # show the name of images in this batch
                prva_slika = batch_names[0]
                prva_slika = prva_slika.split('.')[0]
                druga_slika = batch_names[1]
                druga_slika = druga_slika.split('.')[0]
                
                
                #print("Epoch {} Image Names: {}".format(epoch+1, batch_names))
                save_image(output_test[0], image_folder + 'epoch' + '_' + str(epoch) +  '_' + prva_slika + '_' + 'count' + '_' + str(c) + ".png")
                save_image(output_test[1], image_folder + 'epoch' + '_' + str(epoch) +  '_' + druga_slika + '_' + 'count' + '_' + str(c) + ".png")
               
                c+=1
                jaccard = jaccard_metric(output_test, te_masks).to('cuda:0')
                jaccard_sum += jaccard 
    
    avg_acc = jaccard_sum/len(test_images_dl)
    results["Average acc"].append(avg_acc.cpu().detach().numpy())
    
    
    if avg_acc > best_acc:
            #Save the model to disk
            torch.save({'epoch': epoch, 'state_dict': unet.state_dict()}, '/home/ivae/Desktop/seminarska_ST/outputs_UNET/unet_weights_original.pth')                
            best_acc = avg_acc
            results["Best acc"].append(best_acc.cpu().detach().numpy())
            results["Epoch_best_acc"].append(epoch+1)
    print("Epoch_test {}/{}, Average_Acc: {:.3f}, Best_acc: {:.3f}".format(epoch+1,num_epochs, avg_acc, best_acc))
        
    # Save the dictionary 
    results_pickle = open(r'/home/ivae/Desktop/seminarska_ST/results/Loss_acc.picl', "wb")
    pickle.dump(results, results_pickle)
    results_pickle.close()


     

        

    
    



        

            
       















