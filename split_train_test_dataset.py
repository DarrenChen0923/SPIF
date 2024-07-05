from sklearn.model_selection import train_test_split
from PIL import Image
import os
import numpy as np
import shutil
from utils.cli import get_parser
np.set_printoptions(20)

# cml arguments
parser = get_parser()
args = parser.parse_args()

degrees = [0,90,180,270]
fums = [1,2,3]
grids = [args.grid]
version = 2

def numeric_sort(file_name):
    file_name_parts = file_name.split('/')[-1].split('_')
    file_name_parts[-1] = file_name_parts[-1].split('.')[0] 
    return int(file_name_parts[0]), int(file_name_parts[1])


def read_data(f_num,d,degree):
    if degree == 0:
        # set Image file and label file path
        image_folder = args.project_root + f'/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/images'
        label_folder = args.project_root + f'/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/labels'
    else:
        image_folder = args.project_root + f'/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/rotate/{degree}/images'
        label_folder = args.project_root + f'/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/rotate/{degree}/labels'
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]

    image_files = sorted(image_files, key=numeric_sort)
    # image_files.sort(key=lambda x:int(x.split('.'[0])))


    # Create empty list to store iamges and lables
    X = []  # store images
    y = []  # store lables
    image_paths = []

    # Iterate Images
    count = 0
    for image_path in image_files:
        count+=1
        # Get images file name
        image_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Build path for label file
        label_path = os.path.join(label_folder, f'{image_filename}.txt')

        # Read label file
        with open(label_path, 'r') as label_file:
            label = label_file.read().strip()

        # Image preprocessing
        with Image.open(image_path) as img:
            img = np.array(img)  # 将图像转化为NumPy数组 transfer image to Numpy array
        img = img.transpose((2, 0, 1))

        # put image and label into list
        image_paths.append(image_path)
        X.append(img)
        y.append(label)
    return image_paths,X,y
  

def read_data_flip(f_num,d,degree):
    if degree == 0:
        # set Image file and label file path
        image_folder = args.project_root + f'/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/flip/images'
        label_folder = args.project_root + f'/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/flip/labels'
    else:
        image_folder = args.project_root + f'/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/flip/rotate/{degree}/images'
        label_folder = args.project_root + f'/SPIF_DU/Croppings/version_{version}/f{f_num}_out/{d}mm/flip/rotate/{degree}/labels'
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]
    

    image_files = sorted(image_files, key=numeric_sort)
    # image_files.sort(key=lambda x:int(x.split('.'[0])))


    # Create empty list to store iamges and lables
    X = []  # store images
    y = []  # store lables
    image_paths = []


    # Iterate Images
    count = 0
    for image_path in image_files:
        count+=1

        # Get images file name
        image_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Build path for label file
        label_path = os.path.join(label_folder, f'{image_filename}.txt')

        # Read label file
        with open(label_path, 'r') as label_file:
            label = label_file.read().strip()
        # Image preprocessing
        with Image.open(image_path) as img:
            img = np.array(img)  # 将图像转化为NumPy数组 transfer image to Numpy array
        img = img.transpose((2, 0, 1))

        # put image and label into list
        image_paths.append(image_path)
        X.append(img)
        y.append(label)
    return image_paths,X,y
  

# f_num = 3
# d = 5
# X,y = read_data(3,5)
X = [] 
y = []
image_paths = []

for fum in fums:
    for grid in grids:
        for degree in degrees:
            image_paths_grid_degree,X_fum_grid_degree,y_fum_grid_degree = read_data(fum,grid,degree)
            image_paths_grid_degree_flip,X_fum_grid_degree_flip,y_fum_grid_degree_flip = read_data_flip(fum,grid,degree)
            X += X_fum_grid_degree + X_fum_grid_degree_flip
            y += y_fum_grid_degree + y_fum_grid_degree_flip
            image_paths += image_paths_grid_degree + image_paths_grid_degree_flip

Image_train, Image_test, y_train, y_test = train_test_split(image_paths,y,test_size=0.28)

def save_images_and_labels(images, labels, output_dir, prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, (image_path, label) in enumerate(zip(images, labels)):
        # destination_path = os.path.join(output_dir, f"images/{prefix}_image_{i+1}.png")
        # shutil.copy2(image_path,destination_path)
        label_path = os.path.join(output_dir, f"labels/{prefix}_{i+1}.txt")
        with open(label_path, 'w') as label_file:
            label_file.write(str(label))
        destination_path = os.path.join(output_dir, f"images/{prefix}_{i+1}.jpg")
        shutil.copy(image_path,destination_path)
    

# saving img and labels
save_images_and_labels(Image_train, y_train, args.project_root + f'/SPIF_DU/Croppings/version_2/train_dataset/{grids[0]}mm/', 'train')
save_images_and_labels(Image_test, y_test, args.project_root + f'/SPIF_DU/Croppings/version_2/test_dataset/{grids[0]}mm/', 'test')

