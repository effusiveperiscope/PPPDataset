# Automates so-vits-svc 5.0 training
# Designed to be AFK for 2 weeks.
CHARACTERS_TO_TRAIN = [
    'Fluttershy',
    'Rarity',
    'Applejack',
    'Rainbow',
    'Celestia',
    'Luna',
    'Starlight',
    'Apple Bloom',
    'Sweetie Belle',
    'Scootaloo',
    'Trixie',
    'Pinkie'
]
TEST_RUN = False # Does a minimal run-through for testing
SLICED_DIALOGUE = r"D:\MLP_Samples\AIData\Master file\Sliced Dialogue"
SONGS = r"D:\MLP_Samples\AIData\Songs"
SVC5_INSTALL = r"D:\Code\sovits5\so-vits-svc"

from ppp import PPPDataset
from pathlib import Path
import os
import math
for c in CHARACTERS_TO_TRAIN:

    print("Processing "+c)
    if c == 'Trixie': # Use speaking lines for Trixie
        dataset = PPPDataset.collect([c],
        sliced_dialogue = SLICED_DIALOGUE)
    else:
        dataset = PPPDataset.collect([c],
        sliced_dialogue = SONGS)
    print("Collected "+str(len(dataset[c]))+" audio files")
    print("First audio file: "+str(dataset[c][0]['file']))
    dataset_length = len(dataset[c])
    batch_size = 16
    target_epochs = 2000
    model_name = c

    print("Target epochs: ",target_epochs)

    # 0: adjust config
    import yaml
    os.chdir(SVC5_INSTALL)
    cfg_path = os.path.join("configs","base.yaml")
    with open(cfg_path) as f:
        data = yaml.safe_load(f)

    if TEST_RUN:
        data['train']['epochs'] = 2
        data['log']['eval_interval'] = 1
        data['log']['save_interval'] = 1
    else:
        data['train']['epochs'] = target_epochs
        data['log']['eval_interval'] = 10
        data['log']['save_interval'] = 20
    data['train']['batch_size'] = batch_size
    data['log']['keep_ckpts'] = 2

    new_yaml = yaml.dump(data)

    with open(cfg_path, 'w') as f:
        yaml.dump(data, f)

    # 1: in dataset_raw <-- character files, transcode to wav
    import ffmpeg
    DATASET_RAW = "dataset_raw_"+c
    DATASET = "data_svc_"+c
    os.makedirs(os.path.join(DATASET_RAW,c), exist_ok=True)
    for char, files in dataset.file_dict.items():
        for i,x in enumerate(files):
            # 1. Convert to wav
            out_path = os.path.join(DATASET_RAW,c,Path(x['file']).stem+'.wav')
            if not os.path.exists(out_path):
                ffmpeg.input(x['file']).output(out_path).run()
            else:
                #print('Skipping existing file '+out_path)
                pass

    # Resampling
    import subprocess
    subprocess.run(["python", "prepare/preprocess_a.py", "-w",
        DATASET_RAW, "-o", os.path.join(DATASET, "waves-16k"), "-s", "16000"],
        env=os.environ)
    subprocess.run(["python", "prepare/preprocess_a.py", "-w",
        DATASET_RAW, "-o", os.path.join(DATASET, "waves-32k"), "-s", "32000"],
        env=os.environ)

    # pitch extraction
    subprocess.run(["python", "prepare/preprocess_crepe.py",
        "-w", os.path.join(DATASET, "waves-16k"),
        "-p", os.path.join(DATASET, "pitch")], env=os.environ)

    # ppg extraction
    subprocess.run(["python", "prepare/preprocess_ppg.py",
        "-w", os.path.join(DATASET, "waves-16k"),
        "-p", os.path.join(DATASET, "whisper")], env=os.environ)

    # hubert extraction
    subprocess.run(["python", "prepare/preprocess_hubert.py",
        "-w", os.path.join(DATASET, "waves-16k"),
        "-v", os.path.join(DATASET, "hubert")], env=os.environ)

    # timbre code extraction
    subprocess.run(["python", "prepare/preprocess_speaker.py",
        os.path.join(DATASET, "waves-16k"),
        os.path.join(DATASET, "speaker")], env=os.environ)

    # timbre code average
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

    # start training
    subprocess.run(["python", "svc_trainer.py",
        "-c", "configs/base.yaml",
        "-n", model_name], env=os.environ)

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
