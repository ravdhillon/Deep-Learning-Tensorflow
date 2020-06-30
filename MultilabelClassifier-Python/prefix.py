import os
from shutil import copyfile

dict = {'damage': 'dmg', 'whole': 'whl', 'minor': 'mnr', 'moderate': 'mod', 'severe': 'svr', 'front': 'frt', 'rear': 'rr', 'side': 'sd'}
def read_files(dir):
    folders = dir.split(os.path.sep)
    prefix = dict.get(folders[-1])
    if prefix is None:
        return
    files = os.listdir(dir)
    count = 0
    for file in files:
        src = os.path.join(dir, file)
        filename = prefix + '_' + file
        dest = os.path.join(DEST_DATASET_PATH, filename)
        copyfile(src, dest)
        count = count + 1
    print(f'Copied {count} files from {dir}')

SRC_DATASET_PATH = '/Users/ravisher/Development/FatehLabs/AIDD-Project/car-dataset'
DEST_DATASET_PATH = '/Users/ravisher/Development/FatehLabs/AIDD-Project/multi-label-dataset'
SUB_FOLDERS = ['cars', 'severity', 'parts']
CAR_TYPES = ['damage', 'whole', 'front', 'rear', 'side', 'minor', 'moderate', 'severe']

for folder in SUB_FOLDERS:
    for c_type in CAR_TYPES:
        directory = os.path.join(SRC_DATASET_PATH, folder, 'training', c_type)
        if (os.path.exists(directory)):
            read_files(directory)