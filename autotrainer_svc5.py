# Automates so-vits-svc 5.0 training
# Designed to be AFK for 2 weeks.
CHARACTERS_TO_TRAIN = [
    'Twilight',
    # 'Fluttershy',
    #'Rarity',
    # 'Pinkie',
    'Applejack',
    'Rainbow',
    'Celestia',
    'Luna',
    'Starlight',
    'Apple Bloom',
    'Scootaloo',
    'Sweetie Belle',
    'Spike',
]
TEST_RUN = False # Does a minimal run-through for testing
FEATURES_ONLY = False # Does data preprocessing but no training, just features
SLICED_DIALOGUE = r"D:\MLP_Samples\AIData\Master file\Sliced Dialogue"
SONGS = r"D:\MLP_Samples\AIData\Songs"
SVC5_INSTALL = r"D:\Code\sovits5\so-vits-svc"
#DATASET_DIR = r"D:\MLP_Samples\AIData\Rainbow Dash Alt"
DATASET_DIR = SLICED_DIALOGUE
EARLY_RESTART = True

from ppp import PPPDataset
from pathlib import Path
import os
import math
import re
import subprocess

def longpath(path):
    import platform
    path = os.path.abspath(path)
    if 'Windows' in platform.system() and not path.startswith('\\\\?\\'):
        path = u'\\\\?\\'+path.replace('/','\\')
        return path
    else:
        return path

os.chdir(SVC5_INSTALL)
for c in CHARACTERS_TO_TRAIN:
    model_name = c

    # -1: Check for pre-existing checkpoints
    CHKPT = os.path.join("chkpt",c)
    max_name = None
    max_num = 0
    DATASET_RAW = "dataset_raw_"+c
    DATASET = "data_svc_"+c
    if os.path.exists(CHKPT):
        for name in os.listdir(CHKPT):
            match = re.search(c+'_(\d+)\.pt', name)
            if match and (int(match.group(1)) > max_num):
                max_num = int(match.group(1))
                max_name = name
    if (not FEATURES_ONLY) and (max_name is not None) and EARLY_RESTART:
        print("Pre-existing checkpoint detected, resuming training from",
             max_name)
        PREEXIST_CHKPT = os.path.join(CHKPT, max_name)

        # reset data
        subprocess.run(["python", "prepare/preprocess_train.py",
            "-d", DATASET,
            "-r", DATASET_RAW], env=os.environ)

        # resume training
        subprocess.run(["python", "svc_trainer.py",
            "-c", "configs/base.yaml",
            "-n", model_name,
            "-p", PREEXIST_CHKPT], env=os.environ)

        # ah... this is what happens when you have no goto...
        subprocess.run(["python", "svc_train_retrieval.py",
            "--base-path", DATASET, "--prefix", model_name], env=os.environ)
        continue

    print("Processing "+c)
    dataset = PPPDataset.collect([c],
        sliced_dialogue = DATASET_DIR,
        ignore_text=True,
    )
    #dataset = PPPDataset.dummy("D:/DataAugmentation/luna_singing", c)
    print("Collected "+str(len(dataset[c]))+" audio files")
    print("First audio file: "+str(dataset[c][0]['file']))
    dataset_length = len(dataset[c])
    batch_size = 16
    target_epochs = 42

    if not FEATURES_ONLY:
        print("Target epochs: ",target_epochs)

    # 0: adjust config 
    import yaml

    cfg_path = os.path.join("configs","base.yaml")
    with open(cfg_path) as f:
        data = yaml.safe_load(f)

    if TEST_RUN:
        data['train']['epochs'] = 2
        data['log']['eval_interval'] = 1
        data['log']['save_interval'] = 1
    else:
        data['train']['epochs'] = target_epochs
        data['log']['eval_interval'] = 5
        data['log']['save_interval'] = 5
    data['train']['batch_size'] = batch_size
    data['log']['keep_ckpts'] = 2

    new_yaml = yaml.dump(data)

    with open(cfg_path, 'w') as f:
        yaml.dump(data, f)

    # 1: in dataset_raw <-- character files, transcode to wav
    import ffmpeg
    os.makedirs(os.path.join(DATASET_RAW,c), exist_ok=True)
    for char, files in dataset.file_dict.items():
        for i,x in enumerate(files):
            # 1. Convert to wav
            out_path = os.path.join(DATASET_RAW,c,Path(x['file']).stem+'.wav')
            if not os.path.exists(longpath(out_path)):
                ffmpeg.input(x['file']).output(out_path).run()
            else:
                #print('Skipping existing file '+out_path)
                pass

    # Resampling
    subprocess.run(["python", "prepare/preprocess_a.py", "-w",
        DATASET_RAW, "-o", os.path.join(DATASET, "waves-16k"), "-s", "16000"],
        env=os.environ)
    subprocess.run(["python", "prepare/preprocess_a.py", "-w",
        DATASET_RAW, "-o", os.path.join(DATASET, "waves-32k"), "-s", "32000"],
        env=os.environ)

    # pitch extraction
    if not os.path.exists(os.path.join(DATASET, "whisper")):
        subprocess.run(["python", "prepare/preprocess_rmvpe.py",
            "-w", os.path.join(DATASET, "waves-16k"),
            "-p", os.path.join(DATASET, "pitch")], env=os.environ)

    # ppg extraction
    if not os.path.exists(os.path.join(DATASET, "hubert")):
        subprocess.run(["python", "prepare/preprocess_ppg.py",
            "-w", os.path.join(DATASET, "waves-16k"),
            "-p", os.path.join(DATASET, "whisper")], env=os.environ)

    # hubert extraction
    if not os.path.exists(os.path.join(DATASET, "speaker")):
        subprocess.run(["python", "prepare/preprocess_hubert.py",
            "-w", os.path.join(DATASET, "waves-16k"),
            "-v", os.path.join(DATASET, "hubert")], env=os.environ)

    # timbre code extraction
    if not os.path.exists(os.path.join(DATASET, "singer")):
        subprocess.run(["python", "prepare/preprocess_speaker.py",
            os.path.join(DATASET, "waves-16k"),
            os.path.join(DATASET, "speaker")], env=os.environ)

    # timbre code average
    if not os.path.exists(os.path.join(DATASET, "specs")):
        subprocess.run(["python", "prepare/preprocess_speaker_ave.py",
            os.path.join(DATASET, "speaker"),
            os.path.join(DATASET, "singer")], env=os.environ)

    # spec extraction
    subprocess.run(["python", "prepare/preprocess_spec.py",
        "-w", os.path.join(DATASET, "waves-32k"),
        "-s", os.path.join(DATASET, "specs")], env=os.environ)

    # training index
    subprocess.run(["python", "prepare/preprocess_train.py",
        "-d", DATASET,
        "-r", DATASET_RAW], env=os.environ)

    # final checks
    subprocess.run(["python", "prepare/preprocess_zzz.py"], env=os.environ)

    # late restart
    if (not FEATURES_ONLY) and (max_name is not None) and (not EARLY_RESTART):
        print("Pre-existing checkpoint detected, resuming training from",
             max_name)
        PREEXIST_CHKPT = os.path.join(CHKPT, max_name)
        # resume training
        subprocess.run(["python", "svc_trainer.py",
            "-c", "configs/base.yaml",
            "-n", model_name,
            "-p", PREEXIST_CHKPT], env=os.environ)
    # fresh train
    elif not FEATURES_ONLY:
        print("Fresh train on ",c)
        subprocess.run(["python", "svc_trainer.py",
            "-c", "configs/base.yaml",
            "-n", model_name], env=os.environ)

    subprocess.run(["python", "svc_train_retrieval.py",
        "--base-path", DATASET, "--prefix", model_name], env=os.environ)

    # rosebud peas. full of country goodness and green pea-ness.
    # wait, that's terrible.

    # export
    # I don't remember how the models format here, so we are going to need to
    # take a look at what actually happens...

    # Not strictly necessary; this can be done manually and is probably
    # easier to do manually

    # cleanup
    # Not necessary; thanks whoever made this repo for making the logs directory
    # actually only hold logs!
