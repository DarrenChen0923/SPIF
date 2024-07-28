import numpy as np
import torch
import os
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as fc
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from datetime import datetime
from utils.cli import get_parser
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

writer = SummaryWriter('runs/mnist_experiment_1')

# cml arguments
parser = get_parser()
args = parser.parse_args()

degrees = [0,90,180,270]
fums = [1,2,3]
grids = [args.grid]
version = 2

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential( nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels) )
   
    def forward(self, x):
        out = fc.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = fc.relu(out)
        return out

class HeatMapCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(HeatMapCNN, self).__init__()
        self.in_channels = 16  
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 1, stride=1)
        self.layer2 = self._make_layer(32, 1, stride=2)
        if args.grid !=5 :
            self.layer3 = self._make_layer(64, 1, stride=2)
            self.layer4 = self._make_layer(128, 1, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        if args.grid ==20:
            self.fc = nn.Linear(512, num_classes)  
        else :
            self.fc = nn.Linear(128, num_classes)
 
    def _make_layer(self, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = fc.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        if args.grid !=5 :
            x = self.layer3(x)
            x = self.layer4(x)
        x = fc.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def numeric_sort(file_name):
    file_name_parts = file_name.split('/')[-1].split('_')
    file_name_parts[-1] = file_name_parts[-1].split('.')[0]  
    return int(file_name_parts[0]), int(file_name_parts[1])

def read_data(train_or_test):
        # set Image file and label file path
    image_folder = args.project_root + f'/SPIF_DU/Croppings/version_{version}/{train_or_test}_dataset/{grids[0]}mm/images'
    label_folder = args.project_root + f'/SPIF_DU/Croppings/version_{version}/{train_or_test}_dataset/{grids[0]}mm/labels'
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]

    # Create empty list to store iamges and lables
    X = []  # store images
    y = []  # store lables

    # Iterate Images
    for image_path in image_files:

        # Get images file name
        image_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Build path for label file
        label_path = os.path.join(label_folder, f'{image_filename}.txt')

        # Read label file
        with open(label_path, 'r') as label_file:
            label = label_file.read().strip() 

        # Image preprocessing
        with Image.open(image_path) as img:
            img = np.array(img) 

        img = img.transpose((2, 0, 1))

        # put image and label into list
        X.append(img)
        y.append(label)

    return X,y

for fum in fums:
    for grid in grids:
        for degree in degrees:
            X_train,y_train = read_data("train")
            X_test,y_test = read_data("test")
           


# transfer X and y into numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)



# Normalise
X_train = X_train / 255.0
X_test = X_test/255.0

def normalize_mean_std(image):
    mean = np.mean(image)
    stddev = np.std(image)
    normalized_image = (image - mean) / stddev
    return normalized_image

X_train = normalize_mean_std(X_train)
X_test = normalize_mean_std(X_test)
batch = 64



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


# #tensor to numpy
X_train_tensor = torch.from_numpy(X_train)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

#  #gpu environment: transfer into cuda
if torch.cuda.is_available():
    X_train_tensor = X_train_tensor.cuda()
    X_test_tensor = X_test_tensor.cuda()
    y_train_tensor = y_train_tensor.cuda()
    y_test_tensor = y_test_tensor.cuda()

# #Cominbe dataset
train_dataset = Data.TensorDataset(X_train_tensor,y_train_tensor)
val_dataset = Data.TensorDataset(X_test_tensor,y_test_tensor)

# #Create dataset loader

train_data_loader = Data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_data_loader = Data.DataLoader(val_dataset, batch_size=batch, shuffle=False)

result_file_path = args.project_root + f'/SPIF_DU/Croppings/version_{version}/result/{grids[0]}mm/tencrosvalidationresult_validate_new.txt'


if torch.cuda.is_available():
    model = HeatMapCNN().cuda()
else:
    model = HeatMapCNN()
criterion = nn.MSELoss()
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr = learning_rate) #lr

# Training

num_epochs = 1000
for epoch in range(num_epochs):
    for images, labels in train_data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        al_label = labels.unsqueeze(1)
        loss = criterion(outputs, al_label)
        loss.backward()
        optimizer.step()
        writer.add_scalar('training loss', loss, epoch)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


# saving the model
os.makedirs("trained_models",exist_ok=True)
torch.save(model.state_dict(), f'trained_models/heatmap_cnn_{grids[0]}mm.pth')

print(f"Model saved in trained_models/heatmap_cnn_{grids[0]}mm.pth!")



writer.close()
# Validation on training performance
    
model.eval()
pre = model(X_train_tensor)
pre = pre.cpu().detach().numpy()
y_train_tensor = y_train_tensor.cpu().detach().numpy()

mae = mean_absolute_error(y_train_tensor,pre)
mse = mean_squared_error(y_train_tensor,pre)
rmse = mean_squared_error(y_train_tensor,pre,squared=False)
r2=r2_score(y_train_tensor,pre)

print("Training Performance:")
print("MAE",mae)
print("MSE",mse)
print("RMSE",rmse)
print("R2",r2)
