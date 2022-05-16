# coding=utf-8
from gtts import gTTS 
import os
from pathlib import Path
from omegaconf import OmegaConf 
import pandas as pd

config_path = "../config.yaml"
conf = OmegaConf.load(config_path)



# def create_audiofiles(query_audio_dir, output_path):

if __name__ == "__main__":
    # Directory where audio files will be stored
    query_audio_dir = os.path.join("../", conf.query_audio_dir)
    target_dir = os.path.join("../", conf.dataset_path, 'topic_task')
    topics_train_path = os.path.join(target_dir, 'podcasts_2020_topics_train.xml')

    # Text to speech options
    language = "en"
    slow_options = [True, False]
    query_options = ['query', 'description']
    
    # Read the topics
    topics_train = pd.read_xml(topics_train_path)
    
    # For all TTS options, create output files for query and topic description
    for idx, row in topics_train.iterrows():
        
        query_num = row.num
        query = row.query
        description = row.description

        for slow_option in slow_options:
            for query_option in query_options:

                if slow_option:
                    slow_name = 'slow'
                else:
                    slow_name = 'fast'

                if query_option == 'query':
                    text = query
                else:
                    text = description

                current_dir = os.path.join(query_audio_dir, query_option, slow_name)
                Path(current_dir).mkdir(parents=True, exist_ok=True)
                filename = os.path.join(current_dir, str(query_num) + '.mp3')
                print(query, filename)

                speech = gTTS(text=text, lang=language, slow=slow_option)
                speech.save(filename)





        
