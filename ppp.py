SLICED_DIALOGUE = r"D:\MLP_Samples\AI Data\Master file\Sliced Dialogue"
ARPABET_DICTIONARY = "./horsewords.clean"
import os
import random
import re
from pathlib import Path

def load_dictionary(dict_path):
    arpadict = dict()
    with open(dict_path, "r", encoding="utf8") as f:
        for line in f.readlines():
            word = line.split("  ")
            assert len(word) == 2
            arpadict[word[0].strip().upper()] = word[1].strip()
    return arpadict

def dict_replace(tx, dictionary):
    regex = re.findall(r"[\w'-]+|[^\w'-]", tx)
    assert tx == "".join(regex)
    for i in range(len(regex)):
        word = regex[i].upper()
        if word in dictionary.keys():
            regex[i] = "{" + dictionary[word] + "}"
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

    def pits(self, data_path : str, training_list : str, validation_list : str,
            sr=22050, val_frac=.05):
        print("Processing for pits")
        data_path = os.path.abspath(data_path)
        if os.path.exists(data_path) and not os.path.isdir(data_path):
            raise ValueError(data_path + ' points to an existing file!')
        os.makedirs(data_path, exist_ok=True)

        arpa_dictionary = load_dictionary(ARPABET_DICTIONARY)
        val_file_data = []
        train_file_data = []

        import ffmpeg
        for char,files in self.file_dict.items():
            random.shuffle(files)
            val_partition = min(
                1,int(val_frac*len(files))) # min 1 val file per speaker
            for i,x in enumerate(files):
                # 1. Resample and convert to wav
                out_path = os.path.join(data_path,Path(x['file']).stem+'.wav')
                if not os.path.exists(out_path):
                    ffmpeg.input(x['file']).output(out_path, **{'ar':sr}).run()
                else:
                    print('Skipping existing file '+out_path)

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

PPPDataset.collect(['Twilight']).pits(
        'D:/Code/pits/twilight_data',
        'D:/Code/pits/training_filelist.txt',
        'D:/Code/pits/validation_filelist.txt')
print("Done")
