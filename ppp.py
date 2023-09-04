HORSEWORDS_DICTIONARY = "./horsewords.clean"
CMU_DICTIONARY = "./cmudict-0.7b.txt"
import os
if os.name == "nt":
    SLICED_DIALOGUE = r"D:\MLP_Samples\AIData\Master file\Sliced Dialogue"
    SONGS = r"D:\MLP_Samples\AI Data\Songs"
else:
    SLICED_DIALOGUE = r"/mnt/nvme1n1p2/MLP_Samples/AI Data/Master file/Sliced Dialogue"
    SONGS = r"/mnt/nvme1n1p2/MLP_Samples/AI Data/Songs"
import random
import re
import itertools
from pathlib import Path

# Remove nums from ARPAbet dictionary
def load_dictionary(dict_path, remove_nums=True):
    arpadict = dict()
    with open(dict_path, "r") as f:
        for line in f.readlines():
            word = line.split("  ")
            assert len(word) == 2
            maps_to = word[1].strip()
            if remove_nums:
                maps_to = re.sub(r'\d+', '', maps_to)
            arpadict[word[0].strip().upper()] = maps_to
    return arpadict

def dict_replace(tx, dictionary):
    regex = re.findall(r"[\w'-]+|[^\w'-]", tx)
    assert tx == "".join(regex)
    for i in range(len(regex)):
        word = regex[i].upper()
        if word in dictionary.keys():
            regex[i] = "{" + dictionary[word] + "}"
        elif any(c.isalpha() for c in word):
            print("Note - "+word+" not in dictionary keys")
    return "".join(regex)

class PPPDataset:
    def character_parse(fname):
        ret = {}
        split = os.path.basename(fname).split('_')
        ret['hour'] = split[0]
        ret['min'] = split[1]
        ret['sec'] = split[2]
        ret['char'] = split[3]
        ret['emotion'] = split[4]
        ret['noise'] = split[5]
        return ret

    def __init__(self):
        self.file_dict = {}

    def collect(characters : list,
            max_noise = 1,
            sliced_dialogue = SLICED_DIALOGUE):
        dataset = PPPDataset()

        if not len(characters):
            return
        for (root,_,files) in os.walk(sliced_dialogue):
            for f in files:
                if not f.endswith('.flac'):
                    continue
                f = os.path.join(root,f)

                parse = PPPDataset.character_parse(f)
                if parse['char'] not in characters:
                    continue
                if max_noise == 0:
                    if parse['noise'] in ['Noisy','Very Noisy']:
                        continue
                elif max_noise == 1:
                    if parse['noise'] == 'Very Noisy':
                        continue
                txt = str(Path(f.removesuffix('..flac').removesuffix('.flac')
                    ))+'.txt'
                if not os.path.exists(txt):
                    txt = txt[:-4]+'..txt'
                    assert os.path.exists(txt) 
                parse['file'] = os.path.abspath(f)
                parse['txt'] = os.path.abspath(txt)
                with open(parse['txt'], 'r', encoding='utf8') as f:
                    parse['line'] = f.read()

                if not parse['char'] in dataset.file_dict:
                    dataset.file_dict[parse['char']] = []
                dataset.file_dict[parse['char']].append(parse)

        print("Finished collection")
        return dataset

    def __getitem__(self, c):
        return self.file_dict[c]

    def pits(self, data_path : str, training_list : str, validation_list : str,
            sr=22050, val_frac=.05):
        print("Processing for pits")
        data_path = os.path.abspath(data_path)
        if os.path.exists(data_path) and not os.path.isdir(data_path):
            raise ValueError(data_path + ' points to an existing file!')
        os.makedirs(data_path, exist_ok=True)

        arpa_dictionary = (load_dictionary(HORSEWORDS_DICTIONARY) |
            load_dictionary(CMU_DICTIONARY))
        val_file_data = []
        train_file_data = []

        import ffmpeg
        for char,files in self.file_dict.items():
            random.shuffle(files)
            val_partition = max(
                1,int(val_frac*len(files))) # min 1 val file per speaker
            for i,x in enumerate(files):
                # 1. Resample and convert to wav
                out_path = os.path.join(data_path,Path(x['file']).stem+'.wav')
                if not os.path.exists(out_path):
                    ffmpeg.input(x['file']).output(out_path, **{'ar':sr}).run()
                else:
                    #print('Skipping existing file '+out_path)
                    pass

                # 2. Create ARPAbet transcription
                arpa = dict_replace(x['line'], arpa_dictionary)

                # Separate into validation/training files
                if i < val_partition:
                    val_file_data.append({'out_path': out_path,
                        'stem_wav': Path(x['file']).stem+'.wav',
                        'arpa': arpa, 'char': char})
                else:
                    train_file_data.append({'out_path': out_path,
                        'stem_wav': Path(x['file']).stem+'.wav',
                         'arpa': arpa, 'char': char})

        with open(validation_list, 'w') as f:
            for d in val_file_data:
                f.write(d['stem_wav']+'|'+d['arpa']+'|'+d['char']+'\n')
        with open(training_list, 'w') as f:
            for d in train_file_data:
                f.write(d['stem_wav']+'|'+d['arpa']+'|'+d['char']+'\n')

    def vits2(self, data_path : str, training_list : str,
            validation_list : str,
            sr = 22050, val_frac = .05):
        print("Processing for vits2")

        data_path = os.path.abspath(data_path)
        if os.path.exists(data_path) and not os.path.isdir(data_path):
            raise ValueError(data_path + ' points to an existing file!')
        os.makedirs(data_path, exist_ok=True)

        val_file_data = []
        train_file_data = []

        import ffmpeg
        if len(self.file_dict) > 1:
            print("Multispeaker training detected")
            sid = 0
            for char,files in self.file_dict.items():
                random.shuffle(files)
                val_partition = max(1,int(val_frac*len(files)))
                for i,x in enumerate(files):
                    # 1. Resample and convert to wav
                    out_path = os.path.join(
                        data_path,Path(x['file']).stem+'.wav')
                    if not os.path.exists(out_path):
                        ffmpeg.input(x['file']).output(
                            out_path, **{'ar':sr}).run()

                    # TODO do we need to convert to ASCII?
                    # 2. Separate into validation/training files
                    if i < val_partition:
                        val_file_data.append(out_path+"|"+sid+"|"+x['line']+'\n')
                    else:
                        train_file_data.append(out_path+"|"+sid+"|"+x['line']+'\n')
                sid += 1

            # config considered out of scope
            # (if you are the one collecting the dataset you should know
            # how many speakers are in it.)
        else:
            for char,files in self.file_dict.items():
                random.shuffle(files)
                val_partition = max(1,int(val_frac*len(files)))
                for i,x in enumerate(files):
                    # 1. Resample and convert to wav
                    out_path = os.path.join(
                        data_path,Path(x['file']).stem+'.wav')
                    if not os.path.exists(out_path):
                        ffmpeg.input(x['file']).output(
                            out_path, **{'ar':sr}).run()

                    # 2. Separate into validation/training files
                    if i < val_partition:
                        val_file_data.append(out_path+"|"+x['line']+'\n')
                    else:
                        train_file_data.append(out_path+"|"+x['line']+'\n')

        # Write filelists
        with open(validation_list, 'w') as f:
            for d in val_file_data:
                f.write(d)
        with open(training_list, 'w') as f:
            for d in train_file_data:
                f.write(d)
        pass


PPPDataset.collect(['Twilight']).vits2(
        'D:/Code/vits2_pytorch/twilight_data',
        'D:/Code/vits2_pytorch/filelists/training_filelist.txt',
        'D:/Code/vits2_pytorch/filelists/validation_filelist.txt', val_frac=.02)
print("Done")
