HORSEWORDS_DICTIONARY = "./horsewords.clean"
CMU_DICTIONARY = "./cmudict-0.7b.txt"
import os
if os.name == "nt":
    SLICED_DIALOGUE = r"X:"
    SONGS = r"D:\MLP_Samples\AI Data\Songs"
else:
    SLICED_DIALOGUE = r"/mnt/nvme1n1p2/MLP_Samples/AI Data/Master file/Sliced Dialogue"
    SONGS = r"/mnt/nvme1n1p2/MLP_Samples/AI Data/Songs"
import random
import re
import itertools
from pathlib import Path
from util import check_file_dur_ms
import logging
import pickle
from unidecode import unidecode
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        try:
            ret['hour'] = split[0]
            ret['min'] = split[1]
            ret['sec'] = split[2]
            ret['char'] = split[3]
            ret['emotion'] = split[4]
            ret['noise'] = split[5]
        except IndexError as e:
            print("Failed parse: "+fname)
            return None
        return ret

    def __init__(self):
        self.file_dict = {}

    def __len__(self):
        return sum(len(lst) for lst in self.file_dict.values())

    def save_to_pickle(self, pkl_path):
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.file_dict, f)

    def from_pickle(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            self.file_dict = pickle.load(f)

    def search(self, substr):
        for char, files in self.file_dict.items():
            for x in files:
                if substr in x['line']:
                    yield self.obj_to_info1(x)

    def obj_to_info1(self, obj):
        ep = Path(obj['file']).parent.name
        return (f"ep:{ep}|h:{obj['hour']}"
            f"|m:obj['min']|s:obj['s']|char:{obj['char']}|line:{obj['line']}")

    def collect(characters : list,
            max_noise = 1,
            sliced_dialogue = SLICED_DIALOGUE,
            ignore_text = False,
            no_parse = False,
            audio_input_format = '.flac',
            force_character = '',
            emotions : list = []):
        dataset = PPPDataset()

        if len(characters):
            print(f"Collecting data for {characters}")
        else:
            print(f"Collecting data for all characters")

        for (root,_,files) in tqdm(os.walk(sliced_dialogue)):
            for f in files:
                if not f.endswith(audio_input_format):
                    continue
                f = os.path.join(root,f)

                if no_parse:
                    parse = {}
                    parse['char'] = force_character
                    parse['emotion'] = ''
                    parse['txt'] = ''
                    parse['line'] = ''
                    parse['noise'] = 'Clean'
                    parse['file'] = os.path.abspath(f)
                else:
                    parse = PPPDataset.character_parse(f)
                    if parse is None:
                        continue
                    # empty characters array = collect for all characters
                    if len(characters) and (parse['char'] not in characters):
                        continue
                    if len(emotions) and parse['emotion'] not in emotions:
                        continue
                    if max_noise == 0:
                        if parse['noise'] in ['Noisy','Very Noisy']:
                            continue
                    elif max_noise == 1:
                        if parse['noise'] == 'Very Noisy':
                            continue
                    parse['file'] = os.path.abspath(f)
                    if not ignore_text:
                        txt = str(Path(f.removesuffix('..flac').removesuffix('.flac')
                            ))+'.txt'
                        if not os.path.exists(txt):
                            print(txt)
                            txt = txt[:-4]+'..txt'
                            print(txt)
                            assert os.path.exists(txt) 
                        parse['txt'] = os.path.abspath(txt)
                        with open(parse['txt'], 'r', encoding='utf8') as f:
                            parse['line'] = f.read()

                if not parse['char'] in dataset.file_dict:
                    dataset.file_dict[parse['char']] = []
                dataset.file_dict[parse['char']].append(parse)

        if len(characters):
            print(f"Finished collecting data for {characters}")
        else:
            print(f"Finished collecting data for all characters")
        return dataset

    def all_dialogue_paths(self):
        paths = []
        for char,files in self.file_dict.items():
            for f in files:
                paths.append(f['file'])
        return paths

    def stats(self):
        from pydub import AudioSegment
        print("Collecting stats...")
        min_audio_ms = 0
        max_audio_ms = 0
        total_audio_ms = 0
        for char,files in self.file_dict.items():
            first_file = files[0]
            audio = AudioSegment.from_file(first_file['file'])
            min_audio_ms = len(audio)
            for f in tqdm(files, "Files for character "+char):
                audio = AudioSegment.from_file(f['file'])
                audio_length_ms = len(audio)
                min_audio_ms = min(min_audio_ms, audio_length_ms)
                max_audio_ms = max(max_audio_ms, audio_length_ms)
                total_audio_ms += audio_length_ms
            print(f"Character: {char} ({len(files)} lines)\n"
                f"Min: {min_audio_ms/1000} s, Max: {max_audio_ms/1000} s\n",
                f"Total: {total_audio_ms/1000} s")

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

    # vits2 OR generic ljspeech with ARPAbet
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

        ff_opts = {'ar':sr, 'ac':1}

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
                                out_path, **ff_opts).run()

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
                            out_path, **ff_opts).run()

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

    # coqui
    def coqui(self, data_path : str, training_list : str,
            validation_list : str,
            sr = 22050, val_frac = .05):
        print("Processing for coqui")

        data_path = os.path.abspath(data_path)
        if os.path.exists(data_path) and not os.path.isdir(data_path):
            raise ValueError(data_path + ' points to an existing file!')
        os.makedirs(data_path, exist_ok=True)

        val_file_data = []
        train_file_data = []

        ff_opts = {'ar':sr, 'ac':1}

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
                                out_path, **ff_opts).run()

                    # 2. Separate into validation/training files
                    if i < val_partition:
                        val_file_data.append(out_path+"|"+x['line']+"|"
                            +x['line']+'\n')
                    else:
                        train_file_data.append(out_path+"|"+x['line']+"|"
                            +x['line']+'\n')
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
                            out_path, **ff_opts).run()

                    if i < val_partition:
                        val_file_data.append(out_path+"|"+x['line']+"|"
                            +x['line']+'\n')
                    else:
                        train_file_data.append(out_path+"|"+x['line']+"|"
                            +x['line']+'\n')

        # Write filelists
        with open(validation_list, 'w') as f:
            for d in val_file_data:
                f.write(d)
        with open(training_list, 'w') as f:
            for d in train_file_data:
                f.write(d)
        pass

    def styletts2(self, data_path : str, training_list : str,
            validation_list : str,
            sr = 24000, val_frac = .05,
            min_audio_ms = 900):
        from g2p_utils import conv_to_ipa
        print("Processing for styletts2")

        data_path = os.path.abspath(data_path)
        if os.path.exists(data_path) and not os.path.isdir(data_path):
            raise ValueError(data_path + ' points to an existing file!')
        os.makedirs(data_path, exist_ok=True)

        val_file_data = []
        train_file_data = []

        ff_opts = {'ar':sr, 'ac':1}

        import ffmpeg
        if len(self.file_dict) > 1:
            print("Multispeaker training detected")
            sid = 0
            for char,files in self.file_dict.items():
                random.shuffle(files)
                val_partition = max(1,int(val_frac*len(files)))
                for i,x in tqdm(enumerate(files), desc="StyleTTS2"):
                    # 0. Check file length and reject if below min_audio_ms
                    file_ms = check_file_dur_ms(x['file'])
                    if file_ms < min_audio_ms:
                        logger.info(f"Rejected file {x['file']} with dur below min\n")
                        continue

                    # 1. Resample and convert to wav
                    out_path = os.path.join(
                        data_path,Path(x['file']).stem+'.wav')
                    rel_path = Path(x['file']).stem+'.wav'
                    if not os.path.exists(out_path):
                        ffmpeg.input(x['file']).filter('apad',pad_dur=0.1).output(
                                out_path, **ff_opts).global_args(
                                "-hide_banner").global_args("-loglevel","error").run()
                    ipa_line = conv_to_ipa(x['line'])

                    # -- Use $ as a stop token
                    # 2. Separate into validation/training files
                    if i < val_partition:
                        val_file_data.append(
                            rel_path+"|"+ipa_line+"$|"+str(sid)+'\n')
                    else:
                        train_file_data.append(
                            rel_path+"|"+ipa_line+"$|"+str(sid)+'\n')
                sid += 1

            # config considered out of scope
            # (if you are the one collecting the dataset you should know
            # how many speakers are in it.)
        else:
            for char,files in self.file_dict.items():
                random.shuffle(files)
                val_partition = max(1,int(val_frac*len(files)))
                for i,x in tqdm(enumerate(files), desc="StyleTTS2"):
                    # 0. Check file length and reject if below min_audio_ms
                    file_ms = check_file_dur_ms(x['file'])
                    if file_ms < min_audio_ms:
                        logger.info(f"Rejected file {x['file']} with dur below min\n")
                        continue
                    # 1. Resample and convert to wav
                    out_path = os.path.join(
                        data_path,Path(x['file']).stem+'.wav')
                    rel_path = Path(x['file']).stem+'.wav'
                    if not os.path.exists(out_path):
                        ffmpeg.input(x['file']).filter('apad',pad_dur=0.1).output(
                                out_path, **ff_opts).global_args(
                                "-hide_banner").global_args("-loglevel","error").run()
                    ipa_line = conv_to_ipa(x['line'])

                    # 2. Separate into validation/training files
                    if i < val_partition:
                        val_file_data.append(
                            rel_path+"|"+ipa_line+"$|0\n")
                    else:
                        train_file_data.append(
                            rel_path+"|"+ipa_line+"$|0\n")

        # Write filelists
        with open(validation_list, 'w', encoding='utf-8') as f:
            for d in val_file_data:
                f.write(d)
        with open(training_list, 'w', encoding='utf-8') as f:
            for d in train_file_data:
                f.write(d)
        pass

    def xtts2(self, data_path : str,
            training_list : str = "metadata_train.csv",
            validation_list : str = "metadata_eval.csv",
            sr = 48000, val_frac = .05):
        print("Processing for xtts2")

        data_path = os.path.abspath(data_path)
        if os.path.exists(data_path) and not os.path.isdir(data_path):
            raise ValueError(data_path + ' points to an existing file!')
        os.makedirs(data_path, exist_ok=True)

        val_file_data = []
        train_file_data = []

        # Audio needs to be loudness normalized for XTTS training
        ff_opts = {'ar':sr, 'ac':1, 'af':'loudnorm=TP=-1.5:linear=True'}

        import ffmpeg
        for char,files in self.file_dict.items():
            random.shuffle(files)
            val_partition = max(1,int(val_frac*len(files)))
            for i,x in enumerate(files):
                # 1. Resample and convert to wav
                out_path = os.path.join(
                    data_path,unidecode(Path(x['file']).stem)+'.wav')
                if not os.path.exists(out_path):
                    ffmpeg.input(x['file']).output(
                            out_path, **ff_opts).run()

                # 2. Separate into validation/training files
                if i < val_partition:
                    val_file_data.append(out_path+"|"
                        +unidecode(x['line']).lower()+"|"
                        +x['char']+'\n')
                else:
                    train_file_data.append(out_path+"|"
                        +unidecode(x['line']).lower()+"|"
                        +x['char']+'\n')

        # Write filelists
        with open(validation_list, 'w') as f:
            f.write('audio_file|text|speaker_name\n')
            for d in val_file_data:
                f.write(d)
        with open(training_list, 'w') as f:
            f.write('audio_file|text|speaker_name\n')
            for d in train_file_data:
                f.write(d)
        pass

    def styletts2_portable(self,
            data_path : str,
            training_list : str,
            validation_list : str,
            sr = 24000, val_frac = .05,
            min_audio_ms = 900):
        from g2p_utils import conv_to_ipa
        print("Processing for styletts2 low bandwidth")

        data_path = os.path.abspath(data_path)
        if os.path.exists(data_path) and not os.path.isdir(data_path):
            raise ValueError(data_path + ' points to an existing file!')
        os.makedirs(data_path, exist_ok=True)

        val_file_data = []
        train_file_data = []

        ff_opts = {'ar':sr, 'ac':1, 'b:a':'160k'}

        import ffmpeg
        if len(self.file_dict) > 1:
            print("Multispeaker training detected")
            sid = 0
            for char,files in self.file_dict.items():
                random.shuffle(files)
                val_partition = max(1,int(val_frac*len(files)))
                for i,x in tqdm(enumerate(files), desc="StyleTTS2"):
                    # 0. Check file length and reject if below min_audio_ms
                    file_ms = check_file_dur_ms(x['file'])
                    if file_ms < min_audio_ms:
                        logger.info(f"Rejected file {x['file']} with dur below min\n")
                        continue

                    # 1. Resample and convert to wav
                    out_path = os.path.join(
                        data_path,Path(x['file']).stem+'.opus')
                    rel_path = Path(x['file']).stem+'.wav'
                    if not os.path.exists(out_path):
                        ffmpeg.input(x['file']).filter('apad',pad_dur=0.1).output(
                                out_path, **ff_opts).global_args(
                                "-hide_banner").global_args("-loglevel","error").run()
                    ipa_line = conv_to_ipa(x['line'])

                    # 2. Separate into validation/training files
                    if i < val_partition:
                        val_file_data.append(
                            rel_path+"|"+ipa_line+"$|"+str(sid)+'\n')
                    else:
                        train_file_data.append(
                            rel_path+"|"+ipa_line+"$|"+str(sid)+'\n')
                sid += 1

            # config considered out of scope
            # (if you are the one collecting the dataset you should know
            # how many speakers are in it.)
        else:
            for char,files in self.file_dict.items():
                random.shuffle(files)
                val_partition = max(1,int(val_frac*len(files)))
                for i,x in tqdm(enumerate(files), desc="StyleTTS2"):
                    # 0. Check file length and reject if below min_audio_ms
                    file_ms = check_file_dur_ms(x['file'])
                    if file_ms < min_audio_ms:
                        logger.info(f"Rejected file {x['file']} with dur below min\n")
                        continue
                    # 1. Resample and convert to wav
                    out_path = os.path.join(
                        data_path,Path(x['file']).stem+'.opus')
                    rel_path = Path(x['file']).stem+'.wav'
                    if not os.path.exists(out_path):
                        ffmpeg.input(x['file']).filter('apad',pad_dur=0.1).output(
                            out_path, **ff_opts).global_args(
                                "-hide_banner").global_args("-loglevel","error").run()
                    ipa_line = conv_to_ipa(x['line'])

                    # 2. Separate into validation/training files
                    if i < val_partition:
                        val_file_data.append(
                            rel_path+"|"+ipa_line+"$|0\n")
                    else:
                        train_file_data.append(
                            rel_path+"|"+ipa_line+"$|0\n")

        # Write filelists
        with open(validation_list, 'w', encoding='utf-8') as f:
            for d in val_file_data:
                f.write(d)
        with open(training_list, 'w', encoding='utf-8') as f:
            for d in train_file_data:
                f.write(d)
        pass

    def gpt_sovits(self, data_path : str,
            training_list : str = "train.list",
            sr = 48000):
        print("Processing for gpt-sovits")

        data_path = os.path.abspath(data_path)
        if os.path.exists(data_path) and not os.path.isdir(data_path):
            raise ValueError(data_path + ' points to an existing file!')
        os.makedirs(data_path, exist_ok=True)

        val_file_data = []
        train_file_data = []

        ff_opts = {'ar':sr, 'ac':1, 'af':'loudnorm=TP=-1.5:linear=True'}

        import ffmpeg
        for char,files in self.file_dict.items():
            random.shuffle(files)
            for i,x in enumerate(files):
                # 1. Resample and convert to wav
                out_path = os.path.join(
                    data_path,unidecode(Path(x['file']).stem)+'.wav')
                if not os.path.exists(out_path):
                    ffmpeg.input(x['file']).output(
                            out_path, **ff_opts).run()

                #vocal_path|speaker_name|language|text
                train_file_data.append(Path(out_path).name+"|"+x['char']+"|en|"
                    +unidecode(x['line'])+"\n")

        # Write filelists
        with open(training_list, 'w') as f:
            for d in train_file_data:
                f.write(d)
        pass


#PPPDataset.collect(['Rarity']).xtts2(
#    'D:/MLP_Samples/AIData/XTTS_Data/Rarity',
#    'D:/MLP_Samples/AIData/XTTS_Data/Rarity_train.csv',
#    'D:/MLP_Samples/AIData/XTTS_Data/Rarity_eval.csv')
#PPPDataset.collect(['Pinkie']).styletts2(
#            'D:/MLP_Samples/AIData/StyleTTS2Pinkie',
#            'D:/MLP_Samples/AIData/StyleTTS2Pinkie/train_list.txt',
#            'D:/MLP_Samples/AIData/StyleTTS2Pinkie/val_list.txt')

#PPPDataset.collect(['Twilight', 'Rainbow', 'Pinkie', 'Fluttershy', 'Rarity',
#    'Applejack', 'Celestia', 'Luna', 'Cadance', 'Apple Bloom', 'Sweetie Belle',
#    'Scootaloo', 'Starlight', 'Trixie', 'Spike', 'Big Mac', 'Zecora',
#    'Cozy Glow', 'Cheerilee', 'Shining Armor', 'Discord', 'Diamond Tiara',
#    'Gilda', 'Chrysalis', 'Tirek', 'Flim', 'Flam']).styletts2_portable(
#            'D:/MLP_Samples/AIData/StyleTTS2Omnidata',
#            'D:/MLP_Samples/AIData/StyleTTS2Omnidata/train_list.txt',
#            'D:/MLP_Samples/AIData/StyleTTS2Omnidata/val_list.txt')
#PPPDataset.collect(
#    ['Twilight', 'Rainbow', 'Pinkie', 'Fluttershy', 'Rarity', 'Applejack']).gpt_sovits(
#        'D:/Code/GPT-SoVITS-beta/GPT-SoVITS/mane6_ft',
#        'D:/Code/GPT-SoVITS-beta/GPT-SoVITS/mane6_ft.list',
#    )
#print("Done")
