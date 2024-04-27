# GUI implemented using: https://www.geeksforgeeks.org/gui-chat-application-using-tkinter-in-python/
import os
import sys
import time
import socket
import subprocess
from tkinter import *
from threading import *
from PIL import Image, ImageTk
from collections import Counter
from configparser import ConfigParser 


#LLM Imports
from chatHandler import ChatHandler


#Speech to Text Imports
from speech_handler import speechHandler

#convert recorded frames to video
from frame_converter import convert_frames_to_videos

"""
Version 2.0 of Gui application.

Widget to Display current detected FER result: Complete
Text entry & display: Complete

QUery Augmentation dict: Skeleton Function Created: getQueryAugmentation()
LLM Response Function: Skeleton Function Created: getLLMResponse()

Microphone button/speech to text entry: Not Yet Implemented
"""



########################################
################ GUI ###################
########################################

# GUI class for the chat
class GUI:
    # constructor method
    def __init__(self, client, config, config_filename, conversation_folder):
        ###additional variables####
        self.client = client
        self.terminator = "#"
        self.chat_started = False
        self.fer_result = "None"
        self.fer_result_list = []
        
        self.chatHandler = ChatHandler() #openAI Chat Class
        self.msg = ""
        self.reply = ""
        self.query_augmentation = ""
        self.speakTrigger = [250,150] 
        self.user_audio = None
        self.user_spoken_query = None
        self.audio_thread = None

        self.config_filename = config_filename
        self.config = config
        
        
        self.show_record_option = False
        self.record_conversation = False
        self.is_creating_query = False
        #self.start_time = time.time()
        self.conversation_folder = conversation_folder #if recording, all conversation metadata will be stored here
        if self.conversation_folder is not None and os.path.exists(self.conversation_folder):
            #self.record_conversation = True
            self.show_record_option = True
        self.chat_folder = None
        
        
        self.icon_folder = "./icons/"
        self.icon_dict = {"none": self.icon_folder+'none.png',
            'neutral':self.icon_folder+'neutral.png',
            'happy':self.icon_folder+ 'happy.png', 
            'sad':self.icon_folder+ 'sad.png',
            'surprise':self.icon_folder+'surprise.png',
            'fear':self.icon_folder+'fear.png',
            'disgust':self.icon_folder+'disgust.png',
            'angry':self.icon_folder+'angry.png'}
        self.fer_path = self.icon_dict["none"]
        self.fer_image = None
        
        
        ###########################
 
        # chat window which is currently hidden
        self.Window = Tk()
        self.Window.withdraw()
 
        # login window
        self.login = Toplevel()
        # set the title
        self.login.title("Login")
        self.login.resizable(width=True,
                             height=True)
        self.login.configure(width=400,
                             height=300)
                             
        #check if username already exists in ini file
        found_user_name = False
        user_name = self.config.get('user_info', 'user_name')
        if user_name != "":
            found_user_name = True
        
        #check if api_key is already set
        found_api_key = False
        try:
            self.api_key = os.environ["OPENAI_API_KEY"]
            found_api_key = True
        except:
            print("api_key not set as environment variable. Checking user_preferences.ini")
        
        if not found_api_key:
            self.api_key = self.config.get('user_info', 'api_key')
            if self.api_key != "":
                found_api_key = True
        
        
        if found_user_name and found_api_key:
            self.beginChat(user_name, None, skip_login=True)
        else:
            # create a Label
            self.pls = Label(self.login,
                             text="Please login to continue",
                             justify=CENTER,
                             font="Helvetica 14 bold")
     
            self.pls.place(relheight=0.15,
                           relx=0.2,
                           rely=0.07)
            # create a Label
            self.labelName = Label(self.login,
                                   text="Name: ",
                                   font="Helvetica 12")
     
            self.labelName.place(relheight=0.2,
                                 relx=0.1,
                                 rely=0.2)
     
            # create a entry box for
            # typing the message
            self.entryName = Entry(self.login,
                                   font="Helvetica 14")
     
            self.entryName.place(relwidth=0.4,
                                 relheight=0.12,
                                 relx=0.35,
                                 rely=0.2)
     
            # set the focus of the cursor
            self.entryName.focus()
            
            ###if no api key found, add field for api key##
            self.entryKey = None
            if not found_api_key:
                self.labelKey = Label(self.login,
                                   text="Api_key: ",
                                   font="Helvetica 12")
     
                self.labelKey.place(relheight=0.2,
                                 relx=0.1,
                                 rely=0.4)
                                 
                self.entryKey = Entry(self.login,
                                   font="Helvetica 14")
     
                self.entryKey.place(relwidth=0.4,
                                 relheight=0.12,
                                 relx=0.35,
                                 rely=0.4)                 

     
            # create a Continue Button
            # along with action
            self.go = Button(self.login,
                             text="CONTINUE",
                             font="Helvetica 14 bold",
                             command=lambda: self.beginChat(self.entryName.get(), self.entryKey))
     
            self.go.place(relx=0.4,
                          rely=0.55)
        
        self.Window.after(10, self.getCurrentFER)
        self.Window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.Window.mainloop()
    
    def set_fer_image(self, first=False):
        fer_key = str(self.fer_result).lower()
        if fer_key in list(self.icon_dict.keys()):
            self.fer_path = self.icon_dict[fer_key]
        fer_image = Image.open(self.fer_path)
        self.fer_image = ImageTk.PhotoImage(fer_image)
        
        if not first:
            self.imageLabel.config(image=self.fer_image)
    
    def on_closing(self):
        print("Converting conversation snippet frames to videos!")
        print("Please wait for script to finish!")
        self.send_videostream_off_command()
        if self.conversation_folder is not None:
            convert_frames_to_videos(self.conversation_folder)
        self.Window.destroy()
        
    def beginChat(self, name, api_entry, skip_login=False):
        self.config.set('user_info', 'user_name',name)
        if api_entry is not None:
            self.api_key = api_entry.get()
            self.config.set('user_info', 'api_key',self.api_key)
        with open(self.config_filename, 'w') as configfile:
            self.config.write(configfile)
        if skip_login:
            self.login.destroy()
        self.layout(name)
        self.chat_started = True
        
        self.chatHandler.initializeAPI(self.api_key)
        
    def getCurrentFER(self, delay = 10):
        if self.chat_started:
            try:
                raw_message = str(self.client.recv(1024).decode('utf-8'))
                fer_list = [result for result in raw_message.split(self.terminator) if result != '']
                self.fer_result = fer_list[-1]
                self.fer_result_list.append(self.fer_result)
                self.labelHead.config(text="Detected Emotion: " + str(self.fer_result))
            except Exception as e:
                print(e)
        
        self.set_fer_image()
        self.Window.after(10, self.getCurrentFER)  # reschedule event in 2 seconds
 
    
 
    # The main layout of the chat
    def layout(self, name):
 
        self.name = name
        # to show chat window
        self.Window.deiconify()
        self.Window.title("CHATROOM")
        self.Window.resizable(width=True,
                             height=True)
        self.Window.configure(width=720,
                              height=640,
                              bg="#17202A")
        self.labelHead = Label(self.Window,
                               bg="#17202A",
                               fg="#EAECEE",
                               text="Starting Camera...",
                               font="Helvetica 13 bold",
                               pady=5)
 
        self.labelHead.place(relwidth=1,relheight=.1)
        self.line = Label(self.Window,
                          width=450,
                          bg="#ABB2B9")
 
        self.line.place(relwidth=1,
                        rely=0.07,
                        relheight=0.012)

        self.set_fer_image(first = True)
        self.imageLabel = Label(self.Window,
                               bg="#237828",
                               fg="#EAECEE",
                               image = self.fer_image
                               )
        self.imageLabel['bg']=self.imageLabel.master['bg']
        self.imageLabel.place(relwidth=1,
                             relx=0.0,
                             rely=0.09, relheight = .1)
        
 
        self.textCons = Text(self.Window,
                             width=20,
                             height=2,
                             bg="#17202A",
                             fg="#EAECEE",
                             font="Helvetica 14",
                             padx=5,
                             pady=5)
 
        self.textCons.place(relheight=0.60,
                            relwidth=1,
                            rely=0.2)
 
        self.labelBottom = Label(self.Window,
                                 bg="#ABB2B9",
                                 height=80)
 
        self.labelBottom.place(relwidth=1,
                               rely=0.825)
 
       
        # make a recording checkbox
        if self.show_record_option:
            self.recordVar = IntVar()
            self.recordCheckbutton = Checkbutton(self.labelBottom,text = 'record/add data to: '+self.conversation_folder,variable=self.recordVar, onvalue=1, offval=0,command=self.startStopRecording)
            self.recordCheckbutton.place(relx=0.01,
                                 rely=0.076,
                                 relheight=0.02,
                                 relwidth=0.98)

        
        
        self.entry_text = StringVar()
        self.entry_text.trace("w", lambda name, index,mode, var=self.entry_text: self.handle_entryMsg_typed())
        self.entryMsg = Entry(self.labelBottom,
                              bg="#2C3E50",
                              fg="#EAECEE",
                              font="Helvetica 13", textvariable=self.entry_text)
 
        # place the given widget
        # into the gui window
        self.entryMsg.place(relwidth=0.65,
                            relheight=0.06,
                            rely=0.008,
                            relx=0.011)
        
        self.entryMsg.bind("<1>", self.handle_entryMesg_clicked)
        self.entryMsg.bind('<Return>', self.handle_return_pressed)
        self.entryMsg.focus()
        
        #create a record button
        self.buttonRec = Button(self.labelBottom,
                                text="Record\n Audio",
                                font="Helvetica 10 bold",
                                width=20,
                                bg="#ABB2B9",
                                command=lambda: self.threaded_recordMessage()) #command=lambda: self.recordMessage())
 
        self.buttonRec.place(relx=0.67,
                             rely=0.008,
                             relheight=0.06,
                             relwidth=0.09)
 
        # create a Send Button
        self.buttonMsg = Button(self.labelBottom,
                                text="Send",
                                font="Helvetica 10 bold",
                                width=20,
                                bg="#ABB2B9",
                                command=lambda: self.sendButton(self.entryMsg.get()))
 
        self.buttonMsg.place(relx=0.77,
                             rely=0.008,
                             relheight=0.06,
                             relwidth=0.22)
 
        self.textCons.config(cursor="arrow")
 
        # create a scroll bar
        scrollbar = Scrollbar(self.textCons)
 
        # place the scroll bar
        # into the gui window
        scrollbar.place(relheight=1,
                        relx=0.974)
 
        scrollbar.config(command=self.textCons.yview)
 
        self.textCons.config(state=DISABLED)
    
    def startStopRecording(self):
        if self.recordVar.get() == 1:
            self.record_conversation = True
        if self.recordVar.get() == 0:
            self.record_conversation = False
 
    def create_chat_subfolder(self):
        chat_number = 0
        for chat_folder in list(os.listdir(self.conversation_folder)):
            name,num = chat_folder.split("_")
            num = int(num)+1
            if num>chat_number:
                chat_number = num
                
        self.chat_folder=self.conversation_folder+"chat_"+str(chat_number)+"/"
        mkdir_if_dne(self.chat_folder)
 
    def handle_query_event(self, function = None):
        if function == "query_start":
            if not self.is_creating_query:
                self.is_creating_query = True
                self.fer_result_list = []
                if self.record_conversation:
                    print("Began recording conversation snippet")
                    self.create_chat_subfolder()
                    self.send_videostream_record_command(command = function)

        if function == "query_end":
            if self.is_creating_query:
                self.is_creating_query = False
                self.fer_result_list = []
                if self.record_conversation:
                    print("Stopped recording conversation snippet")
                    self.send_videostream_record_command(command = function)
                    self.save_conversation_snippet()
                    
                self.user_audio = None
                self.msg = ""
                self.reply = ""
                self.query_augmentation = ""
        
    def send_videostream_off_command(self):
        message = "terminate" + self.terminator
        try:
            #print(self.client)
            self.client.send(message.encode("utf-8"))
            print("sent: " + str(message))
        except Exception as e:
            print(e)
        
    def send_videostream_record_command(self, command = None):
        print("sending " + str(command) + " to videostream loop.")
        message = command + self.terminator
        try:
            #print(self.client)
            self.client.send(message.encode("utf-8"))
            print("sent: " + str(message))
        except Exception as e:
            print(e)
 
    def save_conversation_snippet(self, delim = "|"):
        augmented_response = self.reply
        fer_aug = self.query_augmentation
        
        print("data to save from gui.py")
        print("user query: " + self.msg)
        print("fer aug: " + fer_aug)
        print("aug reply: " + augmented_response)
        print("query audio: " + str(self.user_audio))
        
        #current_time = time.time()
        #elapsed_time = current_time - self.start_time 
        
        if self.user_audio is not None:
            audio_savedir = self.chat_folder + "audio_files/"
            mkdir_if_dne(audio_savedir)
            
            audio_filename = "query_audio.wav"
            with open(audio_savedir+audio_filename, "wb") as f:
                f.write(self.user_audio.get_wav_data())
            f.close()
            
        
        
        self.getLLMResponse(use_aug = True)
        non_augmented_response = self.reply
        print("non aug reply: " + non_augmented_response)
        
        
        chat_savedir = self.chat_folder + "chatlog_files/"
        mkdir_if_dne(chat_savedir)
        
        conversation_log = chat_savedir + "conversation_log.txt"
        if not os.path.exists(conversation_log):
            with open(conversation_log, 'a+', encoding="utf-8") as f:
                #f.write("elapsed_time|query_augmentation|user_query|augmented_response|non_augmented_response\n")
                f.write("query_augmentation|user_query|augmented_response|non_augmented_response\n")
            f.close()
                
        
        with open(conversation_log, 'a+', encoding="utf-8") as f:
            #conversation_data = [elapsed_time, self.query_augmentation, self.msg, augmented_response, non_augmented_response]
            conversation_data = [fer_aug, self.msg, augmented_response, non_augmented_response]
            conversation_string = ""
            for entry in conversation_data:
                conversation_string+= str(entry)+delim
            conversation_string=conversation_string[:-1]
            conversation_string+="\n"
            f.write(str(conversation_string))
        f.close()
        
 
    def handle_entryMsg_typed(self):
        if len(self.entryMsg.get()) > 0:
            self.handle_query_event(function = "query_start")
            #print("user typed")
    
    def handle_entryMesg_clicked(self, event):
        self.handle_query_event(function = "query_start")
        #print("user clicked entrybox")
        
    def handle_return_pressed(self, event):
        self.sendButton(self.entryMsg.get())
        self.handle_query_event(function = "query_end")
 
    #sets self.query_augmentation based on FER result
    def setQueryAugmentation(self, fer_percent_thresh = 0.1):
        fer_counter = Counter(self.fer_result_list)
        fer_ordered_list = sorted(fer_counter, key=fer_counter.get)
        fer_ordered_list.reverse()
        num_samples = sum(fer_counter.values())
        
        chosen_fer = "None"
        for fer in fer_ordered_list:
            if chosen_fer == "None" or chosen_fer == "Neutral" and fer_counter[fer]/num_samples >= fer_percent_thresh:
                chosen_fer = fer

        aug_dict = {'None':[""],
                    'Neutral':[""],
                    'Happy':["(Reply as if I am really happy!) "], 
                    'Sad':["(Reply as if I am really sad!) "],
                    'Surprise':["(Reply as if I am really surprised!) "],
                    'Fear':["(Reply as if I am really scared!) "],
                    'Disgust':["(Reply as if I am really disgusted!) "],
                    'Angry':["(Reply as if I am really angry!) "]}
        
        self.query_augmentation = aug_dict[chosen_fer][0]

 
    #Get LLM reply here. Apply emotion-based query augmentation here as well.
    def getLLMResponse(self, use_aug = True):
        user_query = self.msg
        self.setQueryAugmentation()
        
        if use_aug:
            augmented_query =  self.query_augmentation+user_query
        else:
            augmented_query = user_query
            
        try:
            self.chatHandler.defineMessage(message=augmented_query)
            self.chatHandler.sendMessage()
            LLM_response = self.chatHandler.returnReply()
        except Exception as e:
            print(e)
            LLM_response = "Problem with LLM handler: "
        
        self.reply = LLM_response
     
    
    def threaded_recordMessage(self):
        self.audio_thread = Thread(target=self.recordMessage)
        self.audio_thread.start()
    
    def analyzeSpeech(self):
        print("running analyze speech")
        speech_handler = speechHandler(verbose = False)
        
        if self.record_conversation:
            #current_time = time.time()
            #elapsed_time = current_time - self.start_time 
            
            user_speech,audio = speech_handler.recognizeSpeechFromMic(self.speakTrigger, return_audio=True)
            self.user_audio = audio
            
        else:
            user_speech = speech_handler.recognizeSpeechFromMic(self.speakTrigger)

        self.user_spoken_query = user_speech

    # function to begin recording message
    def recordMessage(self):
        self.handle_query_event(function = "query_start")
        self.analyzeSpeech()
        self.entryMsg.insert("end",str(self.user_spoken_query))

    # function to basically start the thread for sending messages
    def sendButton(self, msg):
        self.textCons.config(state=DISABLED)
        self.msg = msg
        self.entryMsg.delete(0, END)
        self.getLLMResponse()
        
        
        
        
        # insert messages to text box
        show_query_aug = self.config.getboolean("debug_options","show_query_augmentation")
        self.textCons.config(state=NORMAL)

        if show_query_aug:
            self.textCons.insert(END, self.name+": "+self.query_augmentation+self.msg+"\n" + "Chatbot: " + self.reply + "\n")
        else:
            self.textCons.insert(END, self.name+": "+self.msg+"\n" + "Chatbot: " + self.reply + "\n")
        
        self.textCons.config(state=DISABLED)
        self.textCons.see(END)
        
        self.handle_query_event(function = "query_end")
 
def mkdir_if_dne(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)    

def main():
    config_filename = 'user_preferences.ini'
    config = ConfigParser()
    config.read(config_filename)
    user_name = config.get("user_info","user_name")
    record_conversation = config.getboolean("debug_options","record_conversation")
    
    if record_conversation:
        metadata_main_dir = "./conversation_logs/"
        mkdir_if_dne(metadata_main_dir)
        conversation_number = 0
        for conversation_folder in os.listdir(metadata_main_dir):
            _,_,num_str=conversation_folder.split("_")
            
            if int(num_str) > conversation_number:
                conversation_number = int(num_str)
        conversation_name = user_name+"_conversation_"+str(conversation_number+1)
        conversation_folder = metadata_main_dir+conversation_name+"/"
        mkdir_if_dne(conversation_folder)
    else:
        conversation_folder = None
        
    
    port = 400
    host = socket.gethostname()
    
    run_videostream = True
    if run_videostream:
        print("starting server...")
        serverProcess = subprocess.Popen('start python ./videostream_loop.py --conversation_path ' +str(conversation_folder) + "\n exit", stdout=subprocess.PIPE, shell=True) 
        time.sleep(1)
        
    
    
    connection_attempts = 5000
    connect_success = False
    for i in range(connection_attempts):
        if not connect_success:
            try:
                client = socket.socket(socket.AF_INET,
                                   socket.SOCK_STREAM)
                client.connect((host, port))

                chat_app = GUI(client, config, config_filename, conversation_folder)
                connect_success = True
                
            except Exception as e:
                print(e)
                if i%(connection_attempts//10)==0:
                    print("Failed to connect, retrying")
                continue
    
    if not connect_success: 
        print("Failed to connect to Videostream server! Please ensure it is running first.")
    else:
        print("Thanks for using this app.")
    

    
if __name__ == "__main__":
    main()
