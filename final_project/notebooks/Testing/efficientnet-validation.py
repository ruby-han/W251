#!/usr/bin/env python
# coding: utf-8
# %%

# This is a very interesting task and it's very suitable for beginners, through this task you can learn about data analysis, data processing, model building, model training, parameter optimization and so on. When only use the images, with a common network, such as Resnet, Densenet, you can achieve a relatively good accuracy very easily. 
# 
# By analyzing the data, the basic information of the patient is also related to the classification of the diseased tissue. Therefore, if we can combine the case information to carry out the classification task, it will be a very meaningful work. Actually during clinical diagnosis, doctors will also combine different modal data to make comprehensive judgments.
# 
# Due to the urgency of time, my current method only uses image data, and then I will consider adding the patient's personal information to the classification task to train a more complete model. I will update my kernel immediately once I finished.
# 
# Before you really start, I strongly recommend you to read the material of pigmented lesions and dermatoscopic images[https://arxiv.org/abs/1803.10417]. After that, you can learn about the characteristics and distribution of the data from the task description and this kernel[https://www.kaggle.com/kmader/dermatology-mnist-loading-and-processing]
# 
# In this kernel I have followed following steps for model building and evaluation: 
# 
# > Step 1. Data analysis and preprocessing
# 
# > Step 2. Model building
# 
# > Step 3. Model training
# 
# > Step 4. Model evaluation
# 
# I used the pytorch framework to complete the entire task. The code contains several common networks, such as Resnet, VGG, Densenet, and Inception. You only need to make minor changes on the code to complete the network switch. Without the hyperparameter adjustment, I used **Densenet-121 to achieve an accuracy of more than 90% on the validation set in 10 epochs.**
# 
# 

#  ### First, import all libraries that used in this project

# %%


# # Before moving to the next step, pip install a few packages, and then restart for them to become active
# ! pip3 install ipywidgets
# ! pip3 install pandas
# ! pip3 install imbalanced-learn
# # ! pip3 install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip
# ! pip3 install efficientnet_pytorch


# %%


# ! ls /data/mnist_skin/mnist_skin


# %%


# ### %matplotlib inline
# python libraties
import os, cv2,itertools
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# import imblearn
import logging
from tqdm import tqdm
from glob import glob
from PIL import Image
# import ipywidgets
# pytorch libraries
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

from efficientnet_pytorch import EfficientNet



# from torchsampler import ImbalancedDatasetSampler

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# to make the results are reproducible
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# print(os.listdir("/data/mnist_skin/mnist_skin/skin-cancer-mnist-ham10000/"))


# ## Step 1. Data analysis and preprocessing

# Get the all image data paths， match the row information in HAM10000_metadata.csv with its corresponding image

# %%


data_dir = '/data/final_project/MNIST-Skin-Cancer-with-Jetson/skin-cancer-mnist-ham10000/'
all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# all_image_path
imageid_path_dict


# %%


# Check to see that there are 10,015 images split across the data
# !ls /data/final_project/MNIST-Skin-Cancer-with-Jetson/skin-cancer-mnist-ham10000/HAM10000_images_part_1/ | wc -l
# !ls /data/final_project/MNIST-Skin-Cancer-with-Jetson/skin-cancer-mnist-ham10000/HAM10000_images_part_2/ | wc -l


# This function is used to compute the mean and standard deviation on the whole dataset, will use for inputs normalization

# %%


def compute_img_mean_std(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """

    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs


# Return the mean and std of RGB channels

# %%


# norm_mean,norm_std = compute_img_mean_std(all_image_path)
norm_mean = [0.7628294, 0.5463282, 0.570702]
norm_std = [0.14125313, 0.15315892, 0.17052336]


# Add three columns to the original DataFrame, path (image path), cell_type (the whole name),cell_type_idx (the corresponding index  of cell type, as the image label )

# %%


# Metadata - legion_id, image_id, dx (label), dx_type, age, sex, localization
df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))

# Adding in image path
df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)

# Adding in full name of dx type
df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)

# Assigning code to categorical variable - cell_type_idx
df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
df_original.head()


# %%


# this will tell us how many images are associated with each lesion_id
df_undup = df_original.groupby('lesion_id').count()
# now we filter out lesion_id's that have only one image associated with it
df_undup = df_undup[df_undup['image_id'] == 1]
df_undup.reset_index(inplace=True)
df_undup.head()


# %%


df_dup = df_original.groupby('lesion_id').count()
df_dup = df_dup[df_dup['image_id'] > 1]
df_dup.reset_index(inplace = True)
print(f' {len(df_dup)} lesion_id\'s have two or more associated images')
df_dup.head()


# Examine two of the "duplicate" images. From documentation, and evident here - we can see these are not exact duplicates. Instead, they are essentially transforms of the same legion.

# %%


# from IPython.display import Image as Image2
# Image2(filename = df_original[df_original['lesion_id'] == 'HAM_0000000'].iloc[0][7])


# %%


# Image2(filename = df_original[df_original['lesion_id'] == 'HAM_0000000'].iloc[1][7])


# %%


# here we identify lesion_id's that have duplicate images and those that have only one image.
def get_duplicates(x):
    unique_list = list(df_undup['lesion_id'])
    if x in unique_list:
        return 'unduplicated'
    else:
        return 'duplicated'

# create a new colum that is a copy of the lesion_id column
df_original['duplicates'] = df_original['lesion_id']
# apply the function to this new column
df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
df_original.head()


# %%


df_original['duplicates'].value_counts()


# %%


# now we filter out images that don't have duplicates
df_undup = df_original[df_original['duplicates'] == 'unduplicated']
df_undup.shape


# %%


# now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set
y = df_undup['cell_type_idx']
_, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
df_val.shape
# Notes:
# 1) test_size = .2 ends up being closer to 10% of total images since the df_undup size is only half of overall
# 2) stratify = y retains the class ratios of the original (df_undup) dataset 


# %%


df_val['cell_type_idx'].value_counts()
df_val


# %%


# This set will be df_original excluding all rows that are in the val set
# This function identifies if an image is part of the train or val set.
def get_val_rows(x):
    # create a list of all the lesion_id's in the val set
    val_list = list(df_val['image_id'])
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'

# identify train and val rows
# create a new colum that is a copy of the image_id column
df_original['train_or_val'] = df_original['image_id']
# apply the function to this new column
df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
# filter out train rows
df_train = df_original[df_original['train_or_val'] == 'train']
print(len(df_train))
print(len(df_val))


# %%


df_train['cell_type_idx'].value_counts()


# %%


df_val['cell_type'].value_counts()


# **From From the above statistics of each category, we can see that there is a serious class imbalance in the training data. To solve this problem, I think we can start from two aspects, one is equalization sampling, and the other is a loss function that can be used to mitigate category imbalance during training, such as focal loss.**

# %%


df_train[['lesion_id','image_id','path','cell_type','cell_type_idx','duplicates']]


# %%


df_train['cell_type'].value_counts(normalize=False)


# %%


# x = df_train[['lesion_id','image_id','path','cell_type_idx','duplicates']]
# y = df_train['cell_type']
# over = imblearn.over_sampling.RandomOverSampler(sampling_strategy='auto',random_state=5)

# resampled = over.fit_resample(x,y)


# %% [markdown]
# ## Undersampling  & Oversampling

# %%
# x = df_train[['lesion_id','image_id','path','cell_type_idx','duplicates']]
# y = df_train['cell_type']
# under = imblearn.under_sampling.RandomUnderSampler(
#     sampling_strategy={
#     'Melanocytic nevi': 2000,
#     'dermatofibroma':  1067,
#     'Benign keratosis-like lesions ': 1011,
#     'Basal cell carcinoma': 479,
#     'Actinic keratoses': 297,
#     'Vascular lesions': 129,
#     'Dermatofibroma'  : 107,   
# },
#     random_state=5
#                                                   )
# resampled = under.fit_resample(x,y)

# df_train =resampled[0]
# df_train['cell_type'] =resampled[1].to_frame('cell_type')#.value_counts()
# df_train['cell_type'].value_counts()

# %%
# x = df_train[['lesion_id','image_id','path','cell_type_idx','duplicates']]
# y = df_train['cell_type']
# over = imblearn.over_sampling.RandomOverSampler(
#     sampling_strategy={
#     'Melanocytic nevi': 2000,
#     'dermatofibroma':  1067,
#     'Benign keratosis-like lesions ': 1011,
#     'Basal cell carcinoma': 479,
#     'Actinic keratoses': 400,
#     'Vascular lesions': 400,
#     'Dermatofibroma'  : 400,   
# },
#     random_state=5
#                                                   )
# resampled = over.fit_resample(x,y)

# df_train =resampled[0]
# df_train['cell_type'] =resampled[1].to_frame('cell_type')#.value_counts()
# df_train['cell_type'].value_counts()


# %%
df_train['cell_type'].value_counts(normalize=True)

# Copy fewer class to balance the number of 7 classes
data_aug_rate = [15,10,5,50,0,40,5]
for i in range(7):
    if data_aug_rate[i]:
        df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)
df_train['cell_type'].value_counts(normalize=True)


# At the beginning, I divided the data into three parts, training set, validation set and test set. 
# Considering the small amount of data, I did not further divide the validation set data in practice.

# %%


# We can split the test set again in a validation set and a true test set:
# df_val, df_test = train_test_split(df_val, test_size=0.5)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
# df_test = df_test.reset_index()


# %% [markdown]
# # ## Step 2. Model building

# %%
model_path = '/data/final_project/MNIST-Skin-Cancer-with-Jetson/notebooks/model_efficientnet_augment.pth'


# %%
def get_dict_model(model_path):
    state_dict = torch.load(model_path)
#     print(state_dict.keys())
    return state_dict
#     model.load_state_dict(state_dict)

# %%


# df_train.reset_index(drop=True)
logging.info("df train"+str(df_train.shape))
logging.info("df val"+str(df_val.shape))


# ## Step 2. Model building

# %%


# feature_extract is a boolean that defines if we are finetuning or feature extracting. 
# If feature_extract = False, the model is finetuned and all model parameters are updated. 
# If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# %%


def initialize_model(model_name, model_path, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18, resnet34, resnet50, resnet101
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "densenet":
        """ Densenet121
        """
#         model_ft = models.densenet121(pretrained=use_pretrained)
        model_ft = models.densenet201(pretrained=use_pretrained)
        print(type(model_ft))
        print(feature_extract)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
        
    elif model_name == 'efficientnet':
        model_ft = EfficientNet.from_pretrained('efficientnet-b7',num_classes=num_classes)
        set_parameter_requires_grad(model_ft, feature_extract)

#         # Handle the primary net
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 600

    else:
        print("Invalid model name, exiting...")
        exit()
    model_ft.load_state_dict(get_dict_model(model_path))
    return model_ft, input_size


# You can change your backbone network, here are 4 different networks, each network also has sevaral versions. Considering the limited training data, we used the ImageNet pre-training model for fine-tuning. This can speed up the convergence of the model and improve the accuracy.
# 
# There is one thing you need to pay attention to, the input size of Inception is different from the others (299x299), you need to change the setting of compute_img_mean_std() function 

# %%


# resnet,vgg,densenet,inception
model_name = 'efficientnet'
num_classes = 7
feature_extract = False
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, model_path, num_classes, feature_extract, use_pretrained=True)
# Define the device:
device = torch.device('cuda:0')
# Put the model on the device:
model = model_ft.to(device)


# %%
input_size

# %%


# norm_mean = (0.49139968, 0.48215827, 0.44653124)
# norm_std = (0.24703233, 0.24348505, 0.26158768)
# define the transformation of the train images.
train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                      transforms.RandomCrop(size=(input_size,input_size)),
#                                       transforms.RandomInvert(), transforms.RandomPosterize(bits=2),
#                                       transforms.RandomAdjustSharpness(sharpness_factor=2),
#                                       transforms.RandomSolarize(threshold=192.0),
#                                       transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
# define the transformation of the val images.
val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])


# %%


# Define a pytorch dataloader for this dataset
class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


# %%


# df_train = df_train[df_train.columns[[0,8,10]]]
df_train


# %%


# df_val = df_val[df_val.columns[[0,8,10]]]
df_val.columns


# %%
# df_train =df_train[df_train.columns[[0,8,10]]]
df_train =df_train[df_train.columns[[7,8,9]]]
# df_train.head()
# df_val = df_val[df_val.columns[[7,8,9]]].reset_index(drop = True)


# %%


# Define the training set using the table train_df and using our defined transitions (train_transform)
training_set = HAM10000(df_train, transform=train_transform)
train_loader = DataLoader(training_set, batch_size= 3, #64, 
                          shuffle=True, num_workers=4)
# Same for the validation set:
validation_set = HAM10000(df_val[df_val.columns[[7,8,9]]].reset_index(),
                          transform=val_transform)
val_loader = DataLoader(validation_set, batch_size=3,#64,
                        shuffle=False, num_workers=4)


# %%


# next(iter(train_loader))
validation_set.df.head()
df_train


# %%


# we use Adam optimizer, use cross entropy loss as our loss function
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.SGD(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss().to(device)


# ## Step 3. Model training

# %%


# this function is used during training process, to calculation the loss and accuracy
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# %%


total_loss_train, total_acc_train = [],[]
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        images, labels = data
        N = images.size(0)
        # print('image shape:',images.size(0), 'label shape',labels.size(0))
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())
        curr_iter += 1
        if (i + 1) % 100 == 0:
#         if (i + 1) % 1 == 0:
            logging.info(f'[epoch {epoch}], [iter {i+1} of {len(train_loader)}],[train loss {train_loss.avg:.5f}], [train acc {train_acc.avg:.5f}]')
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)
    return train_loss.avg, train_acc.avg


# %%


def validate(val_loader, model, criterion, optimizer, epoch):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)

            val_loss.update(criterion(outputs, labels).item())

    logging.info('------------------------------------------------------------')
    logging.info(f'[epoch {epoch}], [val loss {val_loss.avg:.5f}], [val acc {val_acc.avg:.5f}]')
    logging.info('------------------------------------------------------------')
    return val_loss.avg, val_acc.avg


# %%


epoch_num = 10
best_val_acc = 0
total_loss_val, total_acc_val = [],[]
logging.info("Starting Training")
for epoch in range(1, epoch_num+1):
#     loss_train, acc_train = train(train_loader, model, criterion, optimizer, epoch)
#     print('1')
    loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch)
#     print('2')
    total_loss_val.append(loss_val)
#     print('3')
    total_acc_val.append(acc_val)
#     print('4')
    if acc_val > best_val_acc:
#         print('5')
        best_val_acc = acc_val
#         torch.save(model.state_dict(), '/data/mnist_skin/model_efficientnet_augment.pth')
        logging.info('*****************************************************')
        logging.info(f'best record: [epoch {epoch}], [val loss {loss_val:.5f}], [val acc {acc_val:.5f}]')
        logging.info('*****************************************************')


# ## Step 4. Model evaluation

# %%


fig = plt.figure(num = 2)
fig1 = fig.add_subplot(2,1,1)
fig2 = fig.add_subplot(2,1,2)
# fig1.plot(total_loss_train, label = 'training loss')
# fig1.plot(total_acc_train, label = 'training accuracy')
fig2.plot(total_loss_val, label = 'validation loss')
fig2.plot(total_acc_val, label = 'validation accuracy')
plt.legend()
plt.show()


# %%


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# %%


model.eval()
y_label = []
y_predict = []
with torch.no_grad():
    for i, data in enumerate(val_loader):
        images, labels = data
        N = images.size(0)
        images = Variable(images).to(device)
        outputs = model(images)
        prediction = outputs.max(1, keepdim=True)[1]
        y_label.extend(labels.cpu().numpy())
        y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

# compute the confusion matrix
confusion_mtx = confusion_matrix(y_label, y_predict)
# plot the confusion matrix
plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc','mel']
plot_confusion_matrix(confusion_mtx, plot_labels)


# %%


# Generate a classification report
report = classification_report(y_label, y_predict, target_names=plot_labels)
print(report)


# %%


label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
plt.bar(np.arange(7),label_frac_error)
plt.xlabel('True Label')
plt.ylabel('Fraction classified incorrectly')


# ## Conclusion

# I tried to train with different network structures. When using Densenet-121, the average accuracy of 7 classes on the validation set can reach 92% in 10 epochs. We also calculated the confusion matrix for all classes and the F1-score for each class, which is a more comprehensive indicator that can take into account both the precision and recall of the classification model.Our model can achieve more than 90% on the F1-score indicator.
# 
# Due to limited time, we did not spend much time on model training. By increasing in training epochs, adjustmenting of model hyperparameters, and attempting at different networks may further enhance the performance of the model.

# ## Next plan

# How to use image data and patient case data at the same time, my plan is to use CNN to extract features from images, use xgboost to convert medical records into vectors and then concat them with CNN network full-layer features. Two branch networks are trained simultaneously using a loss function. We can refer to the methods used in the advertising CTR estimation task.

# %%
