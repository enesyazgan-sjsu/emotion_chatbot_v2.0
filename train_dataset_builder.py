import os
import cv2
import shutil
import random
random.seed(456) #v3,4 seed: 456
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
    
def copy_and_crop_images(image_path_list, emotion_folder, highest_num, use_fd = False, fd_model = None, num_digits_in_name=5):
    image_number = highest_num+1
    for image_path in image_path_list:
        orig_image_name = image_path.split("/")[-1]
        name,ext = orig_image_name.split(".")
        image_name = "0"*(num_digits_in_name-len(str(image_number))) + str(image_number)+"."+ext
        
        if use_fd:
            frame = cv2.imread(image_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            #extract face from frame with MTCNN
            face = fd_model(frame)
            if face is not None:
                face = face.permute(1, 2, 0).numpy().astype(np.uint8)
                face = face[...,::-1]
                cv2.imwrite(emotion_folder+image_name,face)
                image_number+=1
        else:
            shutil.copy(image_path, emotion_folder+image_name)
            image_number+=1

def main():
    raw_data_folder = "./emotion_dataset/"
    train_datasets_folder = "./Train_Datasets/"
    mkdir_if_dne(train_datasets_folder)
    
    main_names = ["enes", "eric", "gurpreet", "brandon"]
    use_fd = True
    
    train_ratio = 0.8
    
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
    

    
    data_dict = {"enes":{}, "eric":{}, "gurpreet":{}, "brandon":{}, "other":{}}
    for data_subfolder in os.listdir(raw_data_folder):
        data_elements = [e.lower() for e in data_subfolder.split("_")]
        
        emotion = data_elements[0]
        person = data_elements[1]
        
        name = "other"
        if person in main_names:
            name = person
            
        frames_path = raw_data_folder+data_subfolder+"/"
        frame_paths = [frames_path+f for f in list(os.listdir(frames_path))]
        frame_count = len(frame_paths)

        data_dict[name].setdefault(emotion,[])
        data_dict[name][emotion].append((frame_count,frame_paths))
        data_dict[name][emotion] = sorted(data_dict[name][emotion], reverse = True)

    
    train_val_data_dict = {}
    emotion_mins = {}
    for name in data_dict.keys():
        train_val_data_dict.setdefault(name, {})
        for emotion in data_dict[name]:
            total_frames = sum([count for count, paths in data_dict[name][emotion]])
            
            if name == "other":
                targ_train = total_frames #there isn't enough "other" data, so we simply add them to the trainset only (for variety)
            else:
                targ_train = total_frames*train_ratio
                

            train_val_data_dict[name].setdefault(emotion, {"train":[],"val":[]})
            
            train_size = 0
            val_size = 0
            for c, paths in data_dict[name][emotion]:
                if train_size < targ_train and train_size +c <= targ_train*1.1:
                    train_val_data_dict[name][emotion]["train"].extend(paths)
                    train_size+=c
                else:
                    train_val_data_dict[name][emotion]["val"].extend(paths)
                    val_size+=c
                    
            emotion_mins.setdefault(emotion, {"train":float("inf"),"val":float("inf")})
            if name in main_names:
                if train_size < emotion_mins[emotion]["train"]:
                    emotion_mins[emotion]["train"] = train_size
                if val_size < emotion_mins[emotion]["val"]:
                    emotion_mins[emotion]["val"] = val_size

    #create "all" dataset
    train_val_data_dict.setdefault("all", {})
    for name in data_dict.keys():
        for emotion in data_dict[name]:
            train_val_data_dict["all"].setdefault(emotion, {"train":[],"val":[]})
            
            train_paths = train_val_data_dict[name][emotion]["train"]
            if len(train_paths) > emotion_mins[emotion]["train"]:
                train_paths=train_paths[:emotion_mins[emotion]["train"]]
            train_val_data_dict["all"][emotion]["train"].extend(train_paths)
            
            val_paths = train_val_data_dict[name][emotion]["val"]
            if len(val_paths) > emotion_mins[emotion]["val"]:
                val_paths=train_paths[:emotion_mins[emotion]["val"]]
            train_val_data_dict["all"][emotion]["val"].extend(val_paths)    
        
    #merge "other" frames into named trainsets
    for name in train_val_data_dict.keys():
        total_train = 0
        total_val = 0
        if name != "other":
            print("-----")
            print("Dataset: " + str(name))
            for emotion in train_val_data_dict[name].keys():
                print("-Emotion: " + str(emotion))
                if name != "all":
                    if emotion in train_val_data_dict["other"].keys():
                        train_val_data_dict[name][emotion]["train"].extend(train_val_data_dict["other"][emotion]["train"])
                        train_val_data_dict[name][emotion]["val"].extend(train_val_data_dict["other"][emotion]["val"])
                print("--train size: " + str(len(train_val_data_dict[name][emotion]["train"])))
                print("--val size: " + str(len(train_val_data_dict[name][emotion]["val"])))
                
                total_train += len(train_val_data_dict[name][emotion]["train"])
                total_val += len(train_val_data_dict[name][emotion]["val"])
                
            print("total train: " + str(total_train))
            print("total val: " + str(total_val))

    print("___")
    print("Building datasets")
    for name in train_val_data_dict.keys():
        train_data_folder = train_datasets_folder+"/" + name +"_dataset/"
        mkdir_if_dne(train_data_folder)
        print("---")
        print(name)
        for emotion in train_val_data_dict[name].keys():
            train = train_val_data_dict[name][emotion]["train"]
            val = train_val_data_dict[name][emotion]["val"]
            print(emotion)
            print(len(train))
            print(len(val))
            
            train_val_txt_folder = train_datasets_folder + "/" + name + "_train_val_txt/"
            mkdir_if_dne(train_val_txt_folder)
            
            train_paths_txt = open(train_val_txt_folder+"train_paths_"+str(emotion)+".txt","w")
            for path in train:
                train_paths_txt.write(str(path)+"\n")
            train_paths_txt.close()
            
            val_paths_txt = open(train_val_txt_folder+"val_paths_"+str(emotion)+".txt","w")
            for path in val:
                val_paths_txt.write(str(path)+"\n")
            val_paths_txt.close()
            
            #begin saving images for FER training
            train_folder = train_data_folder+"train/"
            val_folder = train_data_folder+"val/"
            mkdir_if_dne(train_folder)
            mkdir_if_dne(val_folder)
            
            train_emotion_folder = train_folder+str(emotion)+"/"
            val_emotion_folder = val_folder+str(emotion)+"/"
            mkdir_if_dne(train_emotion_folder)
            mkdir_if_dne(val_emotion_folder)
            
            highest_train_num = calc_highest_image_num(train_emotion_folder)
            highest_val_num = calc_highest_image_num(val_emotion_folder)

            copy_and_crop_images(train, train_emotion_folder, highest_train_num, use_fd = use_fd, fd_model = fd_model)
            copy_and_crop_images(val, val_emotion_folder, highest_val_num, use_fd = use_fd, fd_model = fd_model)
        
if __name__ == "__main__":
    main()