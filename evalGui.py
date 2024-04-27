#import time
#import socket
#import openai
#from chatHandler import ChatHandler
#from DoSpeech import DoSpeech
#from PIL import Image, ImageTk
#from multiprocessing.pool import ThreadPool

import os
import sys
import time
import socket
import random
from tkinter import *
from tkvideo import tkvideo
from ttkwidgets import TickScale
from tkvideoutils import VideoPlayer
from tkinter import filedialog, messagebox
import tkinter.messagebox
from PIL import Image, ImageTk
from configparser import ConfigParser 

def mkdir_if_dne(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)  
       
# GUI_EVAL class for evaluation
class GUI_EVAL:
    def __init__(self, eval_list, eval_data_path, chatWinWidth = 400, chatWinHeight = None, minHeight = 10, ratingScale = 5, \
                 dataPath = None, observerDataPath = None):
        
        self.eval_list = eval_list
        self.current_eval_index = 0
        self.current_eval_entry = self.eval_list[self.current_eval_index]
        
        self.observerDataPath = eval_data_path
        
        ## Window Attributes ###
        if chatWinHeight == None:
            chatWinHeight = chatWinWidth * .75
        if chatWinHeight <= 0:
            chatWinHeight = minHeight
        chatWinHeight = int(chatWinHeight)
        chatWinWidth = int(chatWinWidth)
        self.playerSizeX = 1280
        self.playerSizeY = 720 
        self.playerButtonSizeY = 0.5 
        self.bgColor = "#17202A"
        self.relTopOfWindow = 0.01
        self.buttonHeight = 0.05
        self.smallButtonHeight = 0.02
        self.ratingButtonWidth= 0.05
        self.symStartHeight = 0.72
        self.buffer = 0.01
        
        ## Query Display Options
        self.randomizeResponse = False #True: observer sees a random mix of augmented and baseline, False: Observer sees only aug queries
        self.seeAugResp = True
        self.applyRandomization()
        
        ###################################
        #   BEGIN WINDOW CONSTRUCTION
        ###################################
        self.Window = Tk()
        self.Window.configure(width=chatWinWidth, height=chatWinHeight, bg=self.bgColor)
        
        # center it
        screenWidth = self.Window.winfo_screenwidth()
        screenHeight = self.Window.winfo_screenheight()
        winXpos = int((screenWidth-chatWinWidth)/2)
        winYpos = int(((screenHeight-chatWinHeight)/2)-50)#subtract a little for quick start bar
        if winYpos < 0:
            winYpos = 0
        if winXpos < 0:
            winXpos = 0
        geoString = str(chatWinWidth)+"x"+str(chatWinHeight)+ \
                        "+"+str(winXpos)+"+"+str(winYpos)
        self.Window.geometry(geoString)
        self.Window.state("zoomed") # opens maximized.......
        #self.Window.attributes("-topmost",True)
        self.Window.grab_set()
        self.Window.focus()         
        self.Window.deiconify()
        self.Window.title("EVALUATION")
        self.Window.resizable(width=True,
                              height=True)
         

        
        
        # video and playing labels
        self.videoLabelHeight = self.relTopOfWindow
        self.videoLabel = Label(self.Window, text="watch the following video and rate the response to the query",\
                                justify=CENTER)
        self.videoLabel.place(relwidth=0.98, relheight=self.buttonHeight,\
                              relx=0.01, rely=self.videoLabelHeight)                      
        
        
        # video player label
        self.playerLabelHeight = self.videoLabelHeight + self.buttonHeight + self.buffer
        self.playerLabel = Label(self.Window,bg=self.bgColor,justify=CENTER)
        self.playerLabel.place(relwidth=0.56, relheight=self.playerButtonSizeY*0.8, \
                              relx=0.22, rely=self.playerLabelHeight)
                              
        self.playerControlsLabelHeight = self.playerLabelHeight + self.playerButtonSizeY*0.8
        self.playerControlsLabel = Label(self.Window,bg=self.bgColor,justify="left")
        self.playerControlsLabel.place(relwidth=0.3, relheight=self.playerButtonSizeY*0.2, \
                              relx=0.36, rely= self.playerControlsLabelHeight)
       
        # video player window
        self.makeVideoWindow()

        
        # re/play button
        self.prevButton = Button(self.Window, text='previous video',\
                                    justify=LEFT,command=lambda: self.prevVideo()) # left of video
        self.prevButton.place(relwidth=0.2, relheight=self.playerButtonSizeY, \
                              relx=0.01, rely=self.playerLabelHeight)

        # next video button
        self.nextButton = Button(self.Window, text='next video',\
                                    justify=RIGHT,command=lambda: self.nextVideo()) # left of video
        self.nextButton.place(relwidth=0.2, relheight=self.playerButtonSizeY, \
                              relx=0.79, rely=self.playerLabelHeight)
                              
                              
        # responses and button labels
        self.queryLabelHeight = self.playerLabelHeight + self.playerButtonSizeY + self.buffer 
        text_query = self.current_eval_entry["user_query"] #self.data.dataDict[self.currentDataTS]['origQuery']
        self.queryLabel = Label(self.Window, text=text_query,\
                                justify=CENTER)
        self.queryLabel.place(relwidth=0.98, relheight=self.buttonHeight, \
                              relx=0.01, rely=self.queryLabelHeight)
                              
                              
        self.responseLabelHeight = self.queryLabelHeight +self.buttonHeight + self.buffer
        text_response = self.current_eval_entry["response"] #self.data.dataDict[self.currentDataTS]['augResponse']
        self.responseLabel = Label(self.Window, text=text_response,\
                                justify=CENTER)
        self.responseLabel.place(relwidth=0.98, relheight=self.buttonHeight, \
                              relx=0.01, rely=self.responseLabelHeight)                      
                              
        ### Build Radio Buttons

        # radio buttons for sympathy
        self.symButList = []
        self.sympathyVar = StringVar()
        self.sympathyLabel = Label(self.Window, text="sympathy",\
                                   justify=CENTER)
        self.sympathyLabel.place(relwidth=0.98, relheight=self.smallButtonHeight,\
                relx=0.01, rely=self.symStartHeight)
        for each in range(ratingScale):
            x = Radiobutton(self.Window, text=str(each+1),\
                            variable=self.sympathyVar,\
                        value=str(each+1),tristatevalue=0,\
                            command=self.recordSympathyValue)
            x.place(relwidth=self.ratingButtonWidth,relheight=self.smallButtonHeight,\
                    relx=.01+(each*(1/ratingScale)),\
                    rely=self.symStartHeight+self.smallButtonHeight+.01)
            self.symButList.append(x)
            
        self.appStartHeight = self.symStartHeight + (self.smallButtonHeight*2) + (self.buffer*2)
        # radio buttons for appropriateness
        self.appButList = []
        self.appropriatenessVar = StringVar()
        self.appropriatenessLabel = Label(self.Window, text="appropriateness",\
                                   justify=CENTER)
        self.appropriatenessLabel.place(relwidth=0.98, relheight=self.smallButtonHeight,\
                relx=0.01, rely=self.appStartHeight)
        for each in range(ratingScale):
            x = Radiobutton(self.Window, text=str(each+1),\
                            variable=self.appropriatenessVar,\
                        value=str(each+1),tristatevalue=0,\
                            command=self.recordAppropriatenessValue)
            x.place(relwidth=self.ratingButtonWidth,relheight=self.smallButtonHeight,\
                    relx=.01+(each*(1/ratingScale)),\
                    rely=self.appStartHeight+self.smallButtonHeight+.01)
            self.appButList.append(x)
            
        self.undStartHeight = self.appStartHeight +  \
                              (self.smallButtonHeight*2) + \
                              (self.buffer*2)
        # radio buttons for understanding evident
        self.undButList = []
        self.understandVar = StringVar()
        self.understandingLabel = Label(self.Window, text="understanding",\
                                   justify=CENTER)
        self.understandingLabel.place(relwidth=0.98, relheight=self.smallButtonHeight,\
                relx=0.01, rely=self.undStartHeight)
        for each in range(ratingScale):
            x = Radiobutton(self.Window, text=str(each+1),\
                            variable=self.understandVar,\
                        value=str(each+1),tristatevalue=0,\
                            command=self.recordUnderstandValue)
            x.place(relwidth=self.ratingButtonWidth,relheight=self.smallButtonHeight,\
                    relx=.01+(each*(1/ratingScale)),\
                    rely=self.undStartHeight+self.smallButtonHeight+.01)
            self.undButList.append(x)

        self.ovlStartHeight = self.undStartHeight + \
                              (self.smallButtonHeight*2) + \
                              (self.buffer*2)
        #self.undStartHeight = self.appStartHeight +  (self.smallButtonHeight*2) + (self.buffer*2)
        # radio buttons for overall evident
        self.ovlButList = []
        self.overallVar = StringVar()
        self.overallLabel = Label(self.Window, text="overall experience",\
                                   justify=CENTER)
        self.overallLabel.place(relwidth=0.98, relheight=self.smallButtonHeight,\
                relx=0.01, rely=self.ovlStartHeight)
        for each in range(ratingScale):
            x = Radiobutton(self.Window, text=str(each+1),\
                            variable=self.overallVar,\
                        value=str(each+1),tristatevalue=0,\
                            command=self.recordOverallValue)
            x.place(relwidth=self.ratingButtonWidth,relheight=self.smallButtonHeight,\
                    relx=.01+(each*(1/ratingScale)),\
                    rely=self.ovlStartHeight+self.smallButtonHeight+.01)
            self.ovlButList.append(x)

        self.makeTextWindow()

        self.Window.bind("<Escape>",self.closeEvalGui)
        self.Window.mainloop()
    
    def close_video_player(self):
        self.playpause_button.destroy()
        self.slider.destroy()
        self.videoplayer.close()
    
    def closeEvalGui(self,event):
        self.close_video_player()
        self.videoWindow.destroy()
        self.Window.destroy()


    def makeTextWindow(self):
        # window for full query and response
        self.textWindow = Toplevel()
        self.textWindow.resizable(width=True, height=True)
        self.textWindow.attributes("-topmost",True)

        geoString = "300x400+10+10" # "200x150+100+100"
        self.textWindow.geometry(geoString)
        self.textWindow.configure(width=300, height=350)

        t = self.current_eval_entry["user_query"]
        self.fullQueryLabel = Label(self.textWindow,\
                                 justify=CENTER, text=t)
        self.fullQueryLabel.place(relwidth=0.98, relheight=0.25, \
                              relx=0.01, rely=0.1)
        self.changeQuery(t)
        
        t = self.current_eval_entry["response"]
        self.fullResponseLabel = Label(self.textWindow,\
                                 justify=CENTER, text=t)
        self.fullResponseLabel.place(relwidth=0.98, relheight=0.25, \
                              relx=0.01, rely=0.5)
        self.changeResponse(t)

    def makeVideoWindow(self, pathToVideo = None, chatWinWidth = 600, chatWinHeight = 300, \
                        minHeight = 10, ratingScale = 10, winXpos = None, winYpos = None):
        # load images
        pause_image = PhotoImage(file=r'./icons/pause.png')
        play_image = PhotoImage(file=r'./icons/play.png')

        # create user interface
        self.playpause_button = Button(self.playerControlsLabel, image=play_image)
        video_path = self.current_eval_entry["video_path"]
        audio_path = "temp.wav"
        slider_var = IntVar(self.playerLabel)
        self.slider = TickScale(self.playerControlsLabel, orient="horizontal", variable=slider_var)
        self.playpause_button.pack( side = LEFT)
        self.slider.pack( side = RIGHT)

        self.videoplayer = VideoPlayer(self.playerLabel, video_path, audio_path, self.playerLabel, loading_gif=r'./icons/loading.gif', size=(chatWinWidth, chatWinHeight),
                             play_button=self.playpause_button, play_image=play_image, pause_image=pause_image,
                             slider=self.slider, slider_var=slider_var, keep_ratio=True, cleanup_audio=True)
        
    
    def changeQuery(self, newText = "changed"):
        self.queryLabel["text"]=newText
        self.queryLabel.update()

        reformated = self.formatLongText(newText)
        self.fullQueryLabel["text"]="USER QUERY\n\n" + reformated
        self.fullQueryLabel.update()
        
    def changeResponse(self, newText = "changed"):
        self.responseLabel["text"]=newText
        self.responseLabel.update()
        reformated = self.formatLongText(newText)
        self.fullResponseLabel["text"]="CHAT RESPONSE\n\n" + reformated
        self.fullResponseLabel.update()
        
    def resetButtons(self):
        for each in self.symButList:
            each.deselect()
        for each in self.appButList:
            each.deselect()
        for each in self.undButList:
            each.deselect()
        for each in self.ovlButList:
            each.deselect()
        print(self.sympathyVar.get(), self.appropriatenessVar.get(), \
              self.understandVar.get(), self.overallVar.get())

    def recordObserverData(self):
        if self.seeAugResp:
            print("used augmented response...")
        else:
            print("used original response...")
        print("recording observer ratings and appending it to: ", self.observerDataPath)
        
        # timestampStart|+|45||vidPath|+|./test.mp4||origQuery|+|hello||augQuery|+|hello(happy)||origResponse|+|yes?||augResponse|+|you seem happy!\n
        kvDelim = '|+|'
        elDelim = '||'
        snippet_name = self.current_eval_entry["snippet_name"]#str(self.currentDataTS)
        symResp = self.sympathyVar.get()
        appResp = self.appropriatenessVar.get()
        undResp = self.understandVar.get()
        ovlResp = self.overallVar.get()
        
        dataString = 'chat_name' + kvDelim + snippet_name +\
                     elDelim + 'symResp' + kvDelim + symResp +\
                     elDelim + 'appResp' + kvDelim + appResp +\
                     elDelim + 'undResp' + kvDelim + undResp +\
                     elDelim + 'ovlResp' + kvDelim + ovlResp +\
                     elDelim + 'seeAugResp' + kvDelim + str(self.current_eval_entry["augmented"]) + '\n'

        # if file exists, append data to it
        if not os.path.isfile(self.observerDataPath):
            with open(self.observerDataPath,'w') as f:
                f.write(dataString)
        else:
            #replace rating if changef 
            with open(self.observerDataPath, 'r') as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                entries = line.split(elDelim)
                name_entry = entries[0]
                _,snippet = name_entry.split(kvDelim)
                if snippet != snippet_name:
                    new_lines.append(line)
            new_lines.append(dataString)
            
        
            with open(self.observerDataPath, 'w') as f:
                for line in new_lines:
                    f.write(line)
                #f.write(dataString)

    def applyRandomization(self):
        if self.randomizeResponse:
            #shuffle list so that aug & non aug responses are dispersed randomly throughout eval list
            random.shuffle(self.eval_list)
        else:
            #retain only augmented responses:
            aug_only_list = []
            for entry in self.eval_list:
                if entry["augmented"]:
                    aug_only_list.append(entry)
            self.eval_list = aug_only_list
            
        self.current_eval_index = 0
        self.current_eval_entry = self.eval_list[self.current_eval_index]

    def prevVideo(self):
        av = self.appropriatenessVar.get()
        if av=='':
            av=0
        sv = self.sympathyVar.get()
        if sv=='':
            sv=0
        uv = self.understandVar.get()
        if uv=='':
            uv=0
        ov = self.overallVar.get()
        if ov=='':
            ov=0
        if (int(av) > 0 and int(sv) > 0 and int(uv) > 0 and int(ov)):
            self.videoLabel.config(text="watch the following video and rate the response to the query")
            print("measured scores:")
            print(self.sympathyVar.get(), self.appropriatenessVar.get(),\
                      self.understandVar.get(), self.overallVar.get())
            self.recordObserverData()          
                      
            self.current_eval_index-=1
            if self.current_eval_index<0:
                self.current_eval_index=len(self.eval_list)-1
            self.current_eval_entry=self.eval_list[self.current_eval_index]
            
            self.close_video_player()
            self.makeVideoWindow()
            self.update_query_and_response()
            self.resetButtons() # clear user's choices off radio buttons
        else:
            self.showWarning()

    def nextVideo(self):
        av = self.appropriatenessVar.get()
        if av=='':
            av=0
        sv = self.sympathyVar.get()
        if sv=='':
            sv=0
        uv = self.understandVar.get()
        if uv=='':
            uv=0
        ov = self.overallVar.get()
        if ov=='':
            ov=0
        if (int(av) > 0 and int(sv) > 0 and int(uv) > 0 and int(ov)):
            self.videoLabel.config(text="watch the following video and rate the response to the query")
            print("measured scores:")
            print(self.sympathyVar.get(), self.appropriatenessVar.get(),\
                      self.understandVar.get(), self.overallVar.get())
            self.recordObserverData()          
                      
            self.current_eval_index+=1
            if self.current_eval_index>=len(self.eval_list):
                self.current_eval_index=0
            self.current_eval_entry=self.eval_list[self.current_eval_index]
            
            self.close_video_player()
            self.makeVideoWindow()
            self.update_query_and_response()
            self.resetButtons() # clear user's choices off radio buttons
        else:
            self.showWarning()
        

    def showWarning(self, msg = "WARNING! you must rate the video for all aspects \nbefore moving on to the next video"):
        self.videoLabel.config(text=msg)
        """
        from tkinter import messagebox
        xpos, ypos, xwidth, yheight = self.getVideoWindowPositions()
        geoString = str(xwidth)+"x"+str('10')+ \
                        "+"+str(xpos)+"+"+str('0')
        self.playerLabel.geometry(geoString)

        eraseMessages = messagebox.showinfo(message=msg, title="WARNING")

        geoString = str(xwidth)+"x"+str(yheight)+ \
                        "+"+str(xpos)+"+"+str(ypos)
        self.playerLabel.geometry(geoString)
        """
    
    def update_query_and_response(self):
        self.queryLabel["text"]=self.current_eval_entry["user_query"]
        self.queryLabel.update()

        reformated = self.formatLongText(self.current_eval_entry["user_query"])
        self.fullQueryLabel["text"]="USER QUERY\n\n" + reformated
        self.fullQueryLabel.update()
        
        self.responseLabel["text"]=self.current_eval_entry["response"]
        self.responseLabel.update()
        
        reformated = self.formatLongText(self.current_eval_entry["response"])
        self.fullResponseLabel["text"]="CHAT RESPONSE\n\n" + reformated
        self.fullResponseLabel.update()

    def formatLongText(self,t):
        # formats longer queries and responses into 50 letter max wide
        if len(t) > 50:
            temp = ""
            for each in range(int(len(t)/50)+1):
                start = each*50
                stop = (each+1)*50
                if stop > len(t):
                    stop = len(t)
                add = t[start:stop] + "\n"
                temp = temp + add
            return temp
        else:
            return t
        
    def recordSympathyValue(self):
        pass
    def recordAppropriatenessValue(self):
        pass
    def recordUnderstandValue(self):
        pass
    def recordOverallValue(self):
        pass
 

def generate_eval_list(user_name, omit_self_from_eval = True):
    eval_data_folder = "./conversation_logs/"
    
    eval_list = []
    include_conversation = True
    for conversation_folder in os.listdir(eval_data_folder):
        name,_,_ = conversation_folder.split("_")
        if name.lower() == user_name.lower() and omit_self_from_eval:
            include_conversation = False
            
        if include_conversation:
            conversation_path = eval_data_folder+conversation_folder+"/"
            for snippet_folder in os.listdir(conversation_path):
                snippet_dict_aug = {"snippet_name": conversation_folder+"/"+snippet_folder,"video_path":None, "query_augmentation":None, "user_query":None, "response":None, "augmented":True}
                snippet_dict_non_aug = {"snippet_name": conversation_folder+"/"+snippet_folder,"video_path":None, "query_augmentation":None, "user_query":None, "response":None, "augmented":False}
            
                snippet_path = conversation_path+snippet_folder+"/"
                
                """
                video_folder = snippet_path+"video_data/"
                video_path = video_folder+"video.mp4"
                if os.path.exists(video_path):
                    snippet_dict_aug["video_path"] = video_path
                    snippet_dict_non_aug["video_path"] = video_path
                """
                video_folder = snippet_path+"video_data/"
                video_paths = [video_folder+"video.mp4", video_folder+"audio_video.mp4"]
                for video_path in video_paths:
                    if os.path.exists(video_path):
                        snippet_dict_aug["video_path"] = video_path
                        snippet_dict_non_aug["video_path"] = video_path    
                
                """
                audio_folder = snippet_path+"audio_files/"
                audio_path = audio_folder+"query_audio.wav"
                if os.path.exists(audio_path):
                    snippet_dict_aug["audio_path"] = audio_path
                    snippet_dict_non_aug["audio_path"] = audio_path
                """
                
                chatlog_file = snippet_path+"chatlog_files/conversation_log.txt"
                if os.path.exists(chatlog_file):
                    with open(chatlog_file, "r") as f:
                        lines = f.readlines()
                        data_line = lines[1].strip()
                        query_augmentation, user_query, augmented_response, non_augmented_response = data_line.split("|")
                        
                        snippet_dict_aug["query_augmentation"]=query_augmentation
                        snippet_dict_aug["user_query"]=user_query
                        snippet_dict_aug["response"]=augmented_response
                        
                        snippet_dict_non_aug["query_augmentation"]=query_augmentation
                        snippet_dict_non_aug["user_query"]=user_query
                        snippet_dict_non_aug["response"]=non_augmented_response
    
                eval_list.append(snippet_dict_aug)
                eval_list.append(snippet_dict_non_aug)
    
    #random.shuffle(eval_list)
    return eval_list
 
def main():
    omit_self_from_eval = False

    config_filename = 'user_preferences.ini'
    config = ConfigParser()
    config.read(config_filename)
    user_name = config.get("user_info","user_name")
    
    eval_list = generate_eval_list(user_name, omit_self_from_eval=omit_self_from_eval)
    
    eval_data_folder = "./eval_data/"
    mkdir_if_dne(eval_data_folder)
    eval_path= './eval_data/'+user_name+'_observerData.txt' # appends observer judgements to this file

      
    
    evalWin = GUI_EVAL(eval_list, eval_path)

if __name__ == "__main__":
    main()


"""

OLD FUNCTIONS
    def setVideo(self, newVideo = None):
        pathToVideo = self.current_eval_entry["video_path"]
        self.videoplayer.load(pathToVideo)
        self.videoplayer.pack(expand=True, fill="both")
        #self.player = tkvideo(newVideo, self.playerLabel, loop = 1, size = (self.playerSizeX,self.playerSizeY))
        
    def playVideo(self):
        
        if os.path.isfile(self.current_eval_entry["video_path"]):
            self.player.play()
        else:
            print("Can't find video...")
            self.playerLabel['text'] = 'trouble finding video - press next video'
        
        self.videoplayer.play()

    
        
    def rePlayVideo(self, remakeVidPath=None):
        self.resetButtons()
        xpos, ypos, xwidth, yheight = self.getVideoWindowPositions()
        self.videoWindow.destroy()
        if remakeVidPath != None:
            self.makeVideoWindow(pathToVideo=remakeVidPath, chatWinWidth = xwidth, \
                                 chatWinHeight = yheight, winXpos = xpos, winYpos = ypos)
        else:
            self.makeVideoWindow(chatWinWidth = xwidth, chatWinHeight = yheight, \
                                 winXpos = xpos, winYpos = ypos)

        #self.makeVideoWindow()
        self.playVideo()
        
    def getVideoWindowPositions(self):
        # winXpos, winYpos, chatWinWidth, chatWinHeight
        xpos = self.playerLabel.winfo_x()
        ypos = self.playerLabel.winfo_y()
        xwidth = self.playerLabel.winfo_width()
        yheight = self.playerLabel.winfo_height()

        #print(xpos, ypos, xwidth, yheight)
        return xpos, ypos, xwidth, yheight
     
    
    

"""