"""
# To speed up the validation task, this file stores all yamnet embeddings from the test set to local storage

"""

from omegaconf import OmegaConf 
import os
from pathlib import Path
import shutil

from prepare_index_sentencelevel2 import read_metadata_subset
from src.data import find_paths



if __name__ == "__main__":
    # Load static configuration variables. 
    config_path = "./config.yaml"
    conf = OmegaConf.load(config_path)

    # Read metadata
    metadata_subset, topics_df, topics_df_targets = read_metadata_subset(conf, traintest='test')

    # Drop duplicate episodes
    metadata_subset = metadata_subset.drop_duplicates(subset=['episode_filename_prefix']).sort_values('episode_filename_prefix')
    print(len(metadata_subset))

    # Get input and output folders
    base_folder = conf.yamnet_embed_dir_cloud
    new_base_folder = conf.yamnet_embed_dir
    paths = find_paths(metadata_subset, base_folder, file_extension='.h5')
    

    for path in paths:
        # Create output filename for the current file
        p = os.path.normpath(path)
        p = p.split(os.sep)
        fname = os.path.join(new_base_folder, p[-4],p[-3],p[-2],p[-1])
        
        file_exists = os.path.isfile(fname) 
        if not file_exists:
            # create directory if it does not exist
            new_directory = os.path.split(fname)[0]
            Path(new_directory).mkdir(parents=True, exist_ok=True)
            
            # Copy the file from cloud to local storage
            shutil.copyfile(path, fname)
            print("Copied ", fname)
        else:
            print("Already existed ", fname)




                



