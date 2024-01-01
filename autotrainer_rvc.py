CHARACTERS_TO_TRAIN = [
    'Scootaloo'
]
TEST_RUN = False
SLICED_DIALOGUE = r"D:\MLP_Samples\AIData\Master file\Sliced Dialogue"
SONGS = r"D:\MLP_Samples\AIData\Songs"
RVC_INSTALL = r"D:\Code\Retrieval-based-Voice-Conversion-WebUI"
DATASET_DIR = SLICED_DIALOGUE

BATCH_SIZE = 24
TOTAL_EPOCH = 300
SAVE_EVERY_EPOCHS = 50
CACHE_SETS = 1 # Cache training sets to GPU memory
N_CPU = 16
PRETRAINED_G = "assets/pretrained_v2/f0G48k.pth"
PRETRAINED_D = "assets/pretrained_v2/f0D48k.pth"
GPUS = "0" # GPU indices separated by -
N_GPU = 1

# Assume: Pitch guided, version v2
# 2b. CPU pitch extraction (use harvest)

from ppp import PPPDataset
import os
import shutil
import faiss
from sklearn.cluster import MiniBatchKMeans
import subprocess
import numpy as np

os.chdir(RVC_INSTALL)
for c in CHARACTERS_TO_TRAIN:
    model_name = c.replace(' ','_') # To avoid weirdness

    # Create experiment directory
    exp_dir = os.path.join("logs", model_name)
    base_dir = os.path.abspath(".")
    exp_dir_abs = os.path.abspath(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    # Gather files into training directory
    # gt_gathered
    trainset_dir = os.path.join(exp_dir, "ORIGINAL_GATHERED")
    os.makedirs(trainset_dir, exist_ok=True)

    dataset = PPPDataset.collect([c],
        sliced_dialogue = DATASET_DIR)
    print("Collected "+str(len(dataset[c]))+" audio files")
    print("First audio file: "+str(dataset[c][0]['file']))
    for char, files in dataset.file_dict.items():
        for x in files:
            shutil.copy(x['file'], trainset_dir)

    # do we need to do this?
    #f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    #f.close()

    # Preprocessing
    print("Running preprocessing...")
    subprocess.run(["python", 
        os.path.join("infer","modules","train","preprocess.py"),
        trainset_dir, str(48000), str(N_CPU),
        exp_dir_abs, "False", # Half precision
        "3.7"],
        env=os.environ)
    print("Done")

    f = open(os.path.join(exp_dir_abs,"extract_f0_feature.log"), "w")
    f.close()

    print("Running f0 extraction...")
    subprocess.run(["python",
        os.path.join("infer","modules","train","extract","extract_f0_rmvpe.py"),
        "1", # Total num GPUs
        "0", # GPU index 1
        "0", # GPU index 2
        exp_dir_abs,
        "False" # Half precision
        ])
    print("Done")

    print("Running feature extraction...")
    subprocess.run(["python",
        os.path.join("infer","modules","train","extract_feature_print.py"),
        "cuda:0", # config.device
        "1", # leng = Total num GPUs
        "0", # idx = GPU index 1
        "0", # n_g = GPU index 2
        exp_dir_abs, # Exp dir
        "v2" # Version
        ])
    print("Done")

    print("Generating filelist...")
    gt_wavs_dir = os.path.join(exp_dir_abs, "0_gt_wavs")
    feature_dir = os.path.join(exp_dir_abs, "3_feature768")
    f0_dir = os.path.join(exp_dir_abs, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir_abs, "2b-f0nsf")
    names = (
        set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
        & set([name.split(".")[0] for name in os.listdir(feature_dir)])
        & set([name.split(".")[0] for name in os.listdir(f0_dir)])
        & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
    ) # Kind of scary set intersection

    opt = []
    for name in names:
        opt.append(
            os.path.join(gt_wavs_dir.replace("\\","/"),name)+".wav|"+
            os.path.join(feature_dir.replace("\\","/"),name)+".npy|"+
            os.path.join(f0_dir.replace("\\","/"),name)+".wav.npy|"+
            os.path.join(f0nsf_dir.replace("\\","/"),name)+".wav.npy|"+
            "0") # speaker ID which is always 0 because we have only 1 spk

    # INTERESTING - they put mute items into the filelist randomly
    for _ in range(2):
        opt.append(
            os.path.join(base_dir,"logs","mute","0_gt_wavs","mute48k.wav")+"|"+
            os.path.join(base_dir,"logs","mute","3_feature768","mute.npy")+"|"+
            os.path.join(base_dir,"logs","mute","2a_f0","mute.wav.npy")+"|"+
            os.path.join(base_dir,"logs","mute","2b-f0nsf","mute.wav.npy")+"|"+
            "0")
    np.random.shuffle(opt)
    with open(os.path.join(exp_dir_abs,"filelist.txt"),"w") as f:
        f.write("\n".join(opt))
    print("Done")

    # Copy config
    shutil.copy(os.path.join("configs","v2","48k.json"),
        os.path.join(exp_dir_abs,"config.json"))

    print("Running training...")
    subprocess.run(["python",
        os.path.join("infer","modules","train","train.py"),
        #"-e",exp_dir_abs,
        "-e",model_name, # EXPORT DIR (actually character name)
        "-sr","48k","-f0","1","-bs",str(BATCH_SIZE),
        "-g",str(GPUS),"-te",str(TOTAL_EPOCH),"-se",str(SAVE_EVERY_EPOCHS),
        "-pg",PRETRAINED_G,"-pd",PRETRAINED_D,"-l","1",
        "-c",str(CACHE_SETS),"-sw","0","-v","v2"])

    # Create feature index
    npys = []
    listdir_res = list(os.listdir(feature_dir))
    for name in sorted(listdir_res):
        phone = np.load(feature_dir+"/"+name)
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5:
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * N_CPU,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            print(info)

    np.save(os.path.join(exp_dir_abs,"total_fea.npy"), big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(768, "IVF%s,Flat" % n_ivf)
    index_ivf = faiss.extract_index_ivf(index) 
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        os.path.join(exp_dir_abs,
        "trained_IVF"+str(n_ivf)+"_Flat_nprobe_"+str(index_ivf.nprobe)+
        "_"+c+"_v2.index")
    )
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        os.path.join(exp_dir_abs,
        "added_IVF"+str(n_ivf)+"_Flat_nprobe_"+str(index_ivf.nprobe)+
        "_"+c+"_v2.index")
    )

