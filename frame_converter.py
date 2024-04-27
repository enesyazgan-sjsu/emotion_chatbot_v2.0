import os
import cv2
import shutil
import ffmpeg
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

def mkdir_if_dne(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)    

def convert_frames_to_videos(source_dir, video_name = "video.mp4"):
    snippet_folders = list(os.listdir(source_dir))
    for i, snippet_folder in enumerate(snippet_folders):
        print("Converting snippet: " + str(i+1) + " out of " + str(len(snippet_folders)))
        
        snippet_path = source_dir+snippet_folder+"/"
        frames_folder = snippet_path+"frame_data/"
        video_folder = snippet_path+"video_data/"
        audio_folder = snippet_path+"audio_files/"
       
        frames_list = []
        if os.path.exists(frames_folder):
            for frame in os.listdir(frames_folder):
                if ".png" in frame:
                    frames_list.append(frame)
        
            frames_list.sort()
            mkdir_if_dne(video_folder)
            video_file = video_folder+video_name
            first_frame = cv2.imread(frames_folder+frames_list[0])
            height, width, layers = first_frame.shape
            
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video = cv2.VideoWriter(video_file, fourcc, 10, (width,height))
            
            for frame in frames_list:
                video.write(cv2.imread(frames_folder+frame))

            video.release()
            

        #add audio to video if audio exists
        if os.path.exists(audio_folder):
            video_clip = VideoFileClip(video_folder+video_name)
            audio_clip = AudioFileClip(audio_folder+"query_audio.wav")
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(video_folder+"audio_video.mp4")	


def main():
    source_dir = "./conversation_logs/Enes_conversation_2/"
    convert_frames_to_videos(source_dir)

if __name__ == "__main__":
    main()