Emotion Chatbot v2.0

Supported OS: Windows 10 & 11

Installation steps:

- Install Conda: https://docs.anaconda.com/free/miniconda/
- conda create -n fer_gui python=3.8
- conda activate fer_gui
- install pytorch. View pytorch_installation.txt for options
- pip install -r requirements.txt


Running instructions:


Prerequisite:
- Obtain openAI API Key: https://openai.com/index/openai-api/
- Save key to System environment variables or save to api_key in user_preferences.ini


User Preferences INI:
- set user_name field to preferred name
- show_query_augmentation. Values: True, False. Toggles whether emotion based query augmentations are displayed or not.
- record_conversation. Values: True, False. If true, clicking the checkbox at the bottom of the gui will record conversation data, until checkbox is toggled off.

Usage:
python gui.py

-----------------
Updated gui.py
- Begins recording conversation as soon as typing or speaking begins
- Records observed FER values into list as query is being written. Majority emotion is picked when query is sent (unless majority emotion is Neutral, in which case it checks for any significant other FER and sends that if so)
- FER now calculated even when audio is being recorded (audio recording on separate thread)
- Command to record frames sent to videostream loop via socket command
- frames are converted to .mp4 when application is closed. Audio is added to .mp4 file
- conversations are stored in ./conversation_logs (created if doesnt exist)
- conversations are split into query-response pairs (so we can eval individual query response pairs)

Updated evalGui.py
- allows playing/pausing of video & allows scrolling through video
- plays audio alongside video
- same function as v.10 evalGui, but allows for navigation to previous video
- saves observerData in same format, but replaces timestamp with conversation_name+query-response-name
- saves observerData in ./eval_data/ (created if doesnt exist)
- option to hide query/response pairs created by self
- eval_data files are distibguished by name of the reviewer/observer to prevent mixup

user_preferences.ini
- allows setting of user name, api key, and toggle record conversation on or off

Link to old Repository (v1.0): https://github.com/enesyazgan-sjsu/emotion-chatbot