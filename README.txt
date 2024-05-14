Emotion Chatbot v2.0

install requirements.txt

Run gui.py



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

