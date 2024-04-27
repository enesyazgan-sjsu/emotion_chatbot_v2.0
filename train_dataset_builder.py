import os
import cv2
import shutil
import random
random.seed(25327) #everyone seed: 25327, v3,4 seed: 456
import numpy as np

#pytorch libraries
import torch
from torchvision import transforms, datasets
import torch.utils.data as data

##Face Detection Library
from facenet_pytorch import MTCNN

def mkdir_if_dne(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def calc_highest_image_num(image_folder):
    highest_num = 0
    for image_name in list(os.listdir(image_folder)):
        name,ext = image_name.split(".")
        if "_" in name:
            n, num = name.split("_")
            num = int(num)
            if num > highest_num:
                highest_num=num
                
    return highest_num
    
def copy_and_crop_images(image_path_list, emotion_folder, highest_num, use_fd = False, fd_model = None):
    for image_path in image_path_list:
        image_name = image_path.split("/")[-1]
        if os.path.exists(emotion_folder+image_name):
            name,ext = image_name.split(".")
            if "_" in name:
                n, num = name.split("_")
                num = int(num)+highest_num+1
                if num > highest_num:
                    highest_num=num
                name = n+"_"+str(num)
            image_name = name+"."+ext    
        if use_fd:
            frame = cv2.imread(image_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            #extract face from frame with MTCNN
            face = fd_model(frame)
            if face is not None:
                face = face.permute(1, 2, 0).numpy().astype(np.uint8)
                face = face[...,::-1]
                cv2.imwrite(emotion_folder+image_name,face)
        else:
            shutil.copy(image_path, emotion_folder+image_name)

def main():
    raw_data_folder = "./emotion_dataset/"
    train_data_folder = "./FER_Train_Dataset/"
    mkdir_if_dne(train_data_folder)
    
    use_fd = True
    
    if use_fd:
        #instantiate Face Detection Model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Loading FD Model")
        if device.type == 'cuda':
            fd_model = MTCNN(margin=40, select_largest=True, post_process=False, device = "cuda")
        else:
            fd_model = MTCNN(margin=40, select_largest=True, post_process=False)
        fd_model.to(device)
    else:
        fd_model = None
    

    
    data_dict = {}
    name_dict = {"enes":{},"gurpreet":{},"eric":{},"brandon":{},"other":{}}
    for data_subfolder in os.listdir(raw_data_folder):
        image_folder = raw_data_folder + data_subfolder + "/"
        
        emotion = data_subfolder.split("_")[0].lower()
        data_dict.setdefault(emotion,{"train":[],"val":[]})

        dataset_size = len(list(os.listdir(image_folder)))
        path_list = list(os.listdir(image_folder))
        random.shuffle(path_list)
        
        
        added = False
        for name in name_dict.keys():
            if name in data_subfolder.lower():
                name_dict[name].setdefault(emotion,[])
                name_dict[name][emotion].append(path_list)
                added=True
        if not added:
            name_dict["other"].setdefault(emotion,[])
            name_dict["other"][emotion].append(path_list)
        
        
        
        force_train = False
        """
        force_train = False
        if dataset_size > 70:
            reduced_list = list(os.listdir(image_folder))
            reduced_list = reduced_list[:int(len(reduced_list))]#*0.4)]
            path_list = reduced_list
            #force_train = True
        """
        
        """
        ran_num = random.randint(0,20)
        
        
        thresh_num = 2
        if emotion == "angry":
            thresh_num = 6
        if emotion == "disgust":
            thresh_num = 2
        if emotion == "fear":
            thresh_num = 4
        if emotion == "happy":
            thresh_num = 4
        if emotion == "neutral":
            thresh_num = 2
        if emotion == "sad":
            thresh_num = 3
        if emotion == "surprise":
            thresh_num = 6
        
        
        if force_train or ran_num > thresh_num:
            for image_file in path_list:
                if ".jpg" in image_file or ".png" in image_file:
                    data_dict[emotion]["train"].append(image_folder+image_file)
        else:
            for image_file in path_list:
                if ".jpg" in image_file or ".png" in image_file:
                    data_dict[emotion]["val"].append(image_folder+image_file)
        """
    for name in name_dict.keys():
        print("for name: " + str(name))
        emotions_dict = name_dict[name]
        total_samples = 0
        for emotion in emotions_dict.keys():
            print("**for emotion: " + str(emotion))
        
            list_of_lists = emotions_dict[emotion]
            num_samples = 0
            num_sets = len(list_of_lists)
            for sublist in list_of_lists:
                num_samples+=len(sublist)
            total_samples+=num_samples
            
            print("****num lists: " + str(num_sets))
            print("****num samples: " + str(num_samples))
            print("****TOTAL SAMPLES: " + str(total_samples))
    
    
    100/0
                
    
    """
    data_dict = {}
    for data_subfolder in os.listdir(raw_data_folder):
        image_folder = raw_data_folder + data_subfolder + "/"

        emotion = data_subfolder.split("_")[0].lower()
        
        data_dict.setdefault(emotion,[])

        for image_file in list(os.listdir(image_folder)):
            if ".jpg" in image_file or ".png" in image_file:
                data_dict[emotion].append(image_folder+image_file)
    """            
    
    #train_ratio = 0.8
    for key in data_dict.keys():
        """
        image_list = list(data_dict[key])
        random.shuffle(image_list)
        
        train = image_list[:int(train_ratio*len(image_list))]
        val = image_list[int(train_ratio*len(image_list)):]
        """
        train = data_dict[key]["train"]
        val = data_dict[key]["val"]
        print(key)
        print(len(train))
        print(len(val))
        
        #save paths to train & val to be able to properly evaluate results
        train_val_txt_folder = "./train_val_txt/"
        mkdir_if_dne(train_val_txt_folder)
        
        train_paths_txt = open(train_val_txt_folder+"train_paths_"+str(key)+".txt","w")
        for path in train:
            train_paths_txt.write(str(path)+"\n")
        train_paths_txt.close()
        
        val_paths_txt = open(train_val_txt_folder+"val_paths_"+str(key)+".txt","w")
        for path in val:
            val_paths_txt.write(str(path)+"\n")
        val_paths_txt.close()

        
        #begin saving images for FER training
        train_folder = train_data_folder+"train/"
        val_folder = train_data_folder+"val/"
        mkdir_if_dne(train_folder)
        mkdir_if_dne(val_folder)
        
        train_emotion_folder = train_folder+str(key)+"/"
        val_emotion_folder = val_folder+str(key)+"/"
        mkdir_if_dne(train_emotion_folder)
        mkdir_if_dne(val_emotion_folder)
        
        highest_train_num = calc_highest_image_num(train_emotion_folder)
        highest_val_num = calc_highest_image_num(val_emotion_folder)

        copy_and_crop_images(train, train_emotion_folder, highest_train_num, use_fd = use_fd, fd_model = fd_model)
        copy_and_crop_images(val, val_emotion_folder, highest_val_num, use_fd = use_fd, fd_model = fd_model)
        
        
        
if __name__ == "__main__":
    main()