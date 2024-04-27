import cv2
import numpy as np
import os  
import sys
import argparse
from PIL import Image as Img
import matplotlib.pyplot as plt

#pytorch libraries
import torch
from torchvision import transforms, datasets
import torch.utils.data as data

##Face Detection Library
from facenet_pytorch import MTCNN

## [FER] DDAMFN Libraries
#obtain fer_model from: https://github.com/simon20010923/DDAMFN
from networks.DDAM import DDAMNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/data/rafdb/', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    parser.add_argument('--model_path', default = './checkpoints_ver2.0/FER_epoch53_acc0.8959_bacc0.8906.pth')
    return parser.parse_args()


def preprocess_webcam_image(device, image, width, height):
    image = Img.fromarray(image)
    data_transforms = transforms.Compose([
        transforms.Resize((width, height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) 
                                 
    image = data_transforms(image)  
    image = image.unsqueeze(0)
    image = image.float()
    image = image.to(device)
    return image

def load_and_evaluate_frames(fer_database, fd_model, fer_model, device, width=112, height=112, use_fd = True, paths_to_eval=None):
    class_names = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'angry']  
    class_result_dict = {}
    for emotion_subfolder in os.listdir(fer_database):
        image_folder = fer_database + emotion_subfolder + "/"
        print("Evaluating " + str(emotion_subfolder) + " images from FER database")
        
        emotion = emotion_subfolder.split("_")[0].lower()
        
        num_correct = 0
        num_incorrect = 0
        for i, image in enumerate(list(os.listdir(image_folder))):
            image_path = image_folder + image
            eval_image = True
            if paths_to_eval is not None:
                target_txt = paths_to_eval+"val_paths_"+emotion+".txt"
                paths_file = open(target_txt,"r")
                val_paths = [p[:-1] for p in list(paths_file.readlines())] #strip /n from lines
                if image_path not in val_paths:
                    eval_image = False
            
            
            if eval_image:
                try:
                    frame = cv2.imread(image_path)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    #extract face from frame with MTCNN
                    if use_fd:
                        face = fd_model(frame)
                    else:
                        face = frame
                    if face is not None:
                        if use_fd:
                            face = face.permute(1, 2, 0).numpy().astype(np.uint8)
                            face = face[...,::-1]

                        #make prediction with DDAMFN
                        input_tensor = preprocess_webcam_image(device, face, width, height)
                        out,feat,heads = fer_model(input_tensor)

                        _, prediction = torch.max(out, 1)
                        pred_class = class_names[prediction[0].tolist()]
                    else:
                        prediction = -1
                        pred_class=None
                    
                    if pred_class != None:
                        print("pred: " + str(prediction))
                        print("truth: " + str(class_names.index(emotion.lower())))
                        if pred_class.lower() != emotion.lower():
                            num_incorrect += 1
                        else:
                            num_correct += 1
                            
                            
                        
                    print("Completed " + str(i) + " evaluations.")
                    print("evaluated: " + str(image_path))
                except Exception as e:
                    print(e)
                    
        if num_correct+num_incorrect > 0:
            print("Num correct: " + str(num_correct))
            print("Num incorrect: " + str(num_incorrect))

            class_result_dict.setdefault(emotion,[0,0])
            class_result_dict[emotion][0]+=num_correct
            class_result_dict[emotion][1]+=num_incorrect
    
    print(class_result_dict)
    return class_result_dict

def mkdir_if_dne(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)    

# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
        
def generate_bargraph(result_dict, result_dir, eval_name):
    x = list(result_dict.keys())
    y = [float('{:,.3f}'.format(y)) for y in list(result_dict.values())]
    
    # setting figure size by using figure() function 
    plt.figure(figsize = (10, 5))
     
    # making the bar chart on the data
    plt.bar(x, y)
     
    # calling the function to add value labels
    addlabels(x, y)
     
    # giving title to the plot
    plt.title("FER Results: " + str(eval_name))
     
    # giving X and Y labels
    plt.xlabel("Emotions")
    plt.ylabel("Accuracy")
    
    plt.savefig(result_dir+ eval_name+".png")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #instantiate Face Detection Model
    print("Loading FD Model")
    if device.type == 'cuda':
        fd_model = MTCNN(margin=40, select_largest=True, post_process=False, device = "cuda")
    else:
        fd_model = MTCNN(margin=40, select_largest=True, post_process=False)
    fd_model.to(device)
    
    #instantiate FER Model
    print("Loading FER Model")
    args = parse_args()
    fer_model = DDAMNet(num_class=7,num_head=args.num_head)
    checkpoint = torch.load(args.model_path, map_location=device)
    fer_model.load_state_dict(checkpoint['model_state_dict'])
    fer_model.to(device)
    fer_model.eval() 
    
    #FER Database
    fer_database = "./FER_Train_Dataset/val/"
    
    valiation_paths_folder = None #"./train_val_txt/" #if none, evaluates on entire dataset

    #Running capture/eval loop    
    result_dict = load_and_evaluate_frames(fer_database, fd_model, fer_model, device, use_fd = True, paths_to_eval = valiation_paths_folder)
    
    result_dir = "./result_charts/"
    mkdir_if_dne(result_dir)
    eval_name = "FER_test_eval"
    
    #generate_bargraph(result_dict, result_dir, eval_name)
    
if __name__ == "__main__":
    main()