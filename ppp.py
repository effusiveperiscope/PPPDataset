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
from util import check_file_dur_ms, longpath
import logging
import pickle
from unidecode import unidecode
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Unfinished: label mappings for the label files to locations in the master files
# Unfinished because I don't actually have the original audios for most of these
SPECIAL_LABEL_MAPPINGS = {
    'mobile game_aj': 'MASTER_FILE_2/Other sources/Mobile Game/Applejack',
    'mobile game_fs': 'MASTER_FILE_2/Other sources/Mobile Game/Fluttershy',
    'mobile game_nmm': 'MASTER_FILE_2/Other sources/Mobile Game/Nightmare Moon',
    'mobile game_pp': 'MASTER_FILE_2/Other sources/Mobile Game/Pinkie Pie',
    'mobile game_ra': 'MASTER_FILE_2/Other sources/Mobile Game/Rarity',
    'mobile game_rd': 'MASTER_FILE_2/Other sources/Mobile Game/Rainbow Dash',
    'mobile game_spike': 'MASTER_FILE_2/Other sources/Mobile Game/Spike',
    'songs': 'MASTER_FILE_2/Songs',
    'eqg_dance magic': 'MASTER_FILE_1/Sliced Dialogue/EQG/EQG Dance Magic',
    'eqg_forgotten friendship': 'MASTER_FILE_1/Sliced Dialogue/EQG/EQG Forgotten Friendship',
    'eqg_friendship_games': 'MASTER_FILE_1/Sliced Dialogue/EQG/EQG Friendship Games',
    'eqg_original_movie': 'MASTER_FILE_1/Sliced Dialogue/Special source/EQG Original',
    'eqg_legend_of_everfree': 'MASTER_FILE_1/Sliced Dialogue/EQG/EQG Legend of Everfree',
    'eqg_mirror magic': 'MASTER_FILE_1/Sliced Dialogue/EQG/EQG Mirror Magic',
    'eqg_movie magic': 'MASTER_FILE_1/Sliced Dialogue/EQG/EQG Movie Magic',
    'eqg_holidays unwrapped': 'MASTER_FILE_1/Sliced Dialogue/Special source/EQG Holidays Unwrapped',
    'eqg_original_movie_special source': 'MASTER_FILE_1/Sliced Dialogue/Special source/EQG Original',
    'eqg_rainbow rocks_special source': 'MASTER_FILE_1/Sliced Dialogue/Special source/EQG Rainbow Rocks',
    'eqg_rollercoaster of friendship': 'MASTER_FILE_1/Sliced Dialogue/EQG/EQG Roller Coaster of Friendship',
    'eqg_rollercoaster of friendship_special source': 'MASTER_FILE_1/Sliced Dialogue/Special source/EQG Roller Coaster of Freindship Special Source',
    'eqg_better together_s02e04': 'MASTER_FILE_1/Sliced Dialogue/Special source/EQG Shorts/Better Together/S2/s2e4_Street Magic With Trixie',
    'eqg_better together_s02e05': 'MASTER_FILE_1/Sliced Dialogue/Special source/EQG Shorts/Better Together/S2/s2e5_Sic Skateboard',
    'eqg_better together_s02e06': 'MASTER_FILE_1/Sliced Dialogue/Special source/EQG Shorts/Better Together/S2/s2e6_Street Chic',
    'eqg_better together_s02e07': 'MASTER_FILE_1/Sliced Dialogue/Special source/EQG Shorts/Better Together/S2/s2e7_Game Stream',
    'eqg_better together_s02e08': 'MASTER_FILE_1/Sliced Dialogue/Special source/EQG Shorts/Better Together/S2/s2e8_Best in Show The Preshow',
    'fim_s09e23_special source': 'MASTER_FILE_1/Sliced Dialogue/Special source/s9e23 [CAUTION - REVERB]',
    'fim_s09e25_special source': 'MASTER_FILE_1/Sliced Dialogue/Special source/s9e25 [CAUTION - REVERB]',
    'fim_s09e26_special source': 'MASTER_FILE_1/Sliced Dialogue/Special source/s9e26 [CAUTION - REVERB]',
    'fim_movie': "MASTER_FILE_2/MLP Movie (still has music, don't use this in any training)/MLP Movie",
    'fim_rainbow roadtrip': "MASTER_FILE_1/Sliced Dialogue/FiM/Rainbow Roadtrip",
}
SPECIAL_LABEL_MAPPINGS_SPECIFIER = {
    'fim_s09e23_special source': 's9e23',
    'fim_s09e25_special source': 's9e25',
    'fim_s09e26_special source': 's9e26',
    'eqg_dance magic' : 'eqg_dance magic',
    'eqg_forgotten friendship' : 'eqg_forgotten friendship',
    'eqg_friendship_games' : 'eqg_friendship_games',
    'eqg_legend_of_everfree' : 'eqg_legend_of_everfree',
    'eqg_mirror magic' : 'eqg_mirror magic',
    'eqg_movie magic' : 'eqg_movie magic',
    'eqg_rollercoaster of friendship' : 'eqg_rollercoaster of friendship',
    'eqg_better together_s02e04' : 'eqg_better together_s02e04',
    'eqg_better together_s02e05' : 'eqg_better together_s02e05',
    'eqg_better together_s02e06' : 'eqg_better together_s02e06',
    'eqg_better together_s02e07' : 'eqg_better together_s02e07',
    'eqg_better together_s02e08' : 'eqg_better together_s02e08',
    'fim_movie' : 'fim_movie',
    'fim_rainbow roadtrip' : 'fim_rainbow roadtrip',
}

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

    # Generates a "dummy" dataset based on a walk of an existing folder.
    # This is useful for untagged datasets.
    def dummy(base_folder, char=''):
        dataset = PPPDataset()
        for f in os.listdir(base_folder):
            parse = {}
            parse['char'] = char
            parse['emotion'] = ''
            parse['txt'] = ''
            parse['line'] = ''
            parse['noise'] = ''
            parse['file'] = os.path.abspath(os.path.join(base_folder,f))
            if not parse['char'] in dataset.file_dict:
                dataset.file_dict[char] = []
            dataset.file_dict[char].append(parse)
        return dataset

    def collect(characters : list,
            max_noise = 1,
            sliced_dialogue = SLICED_DIALOGUE,
            ignore_text = False,
            no_parse = False,
            audio_input_format = '.flac',
            force_character = '',
            emotions : list = [],
            do_rev_index = False):
        dataset = PPPDataset()

        if len(characters):
            print(f"Collecting data for {characters}")
        else:
            print(f"Collecting data for all characters")

        if do_rev_index:
            assert not ignore_text

        for (root,_,files) in tqdm(os.walk(sliced_dialogue)):
            for f in files:
                if not f.endswith(audio_input_format):
                    continue
                f_basename = f.removesuffix('.flac')
                f = os.path.join(root,f)

                if no_parse:
                    parse = {}
                    parse['char'] = force_character
                    parse['emotion'] = ''
                    parse['txt'] = ''
                    parse['line'] = ''
                    parse['noise'] = ''
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
                    elif max_noise == -1:
                        if parse['noise'] in ['Clean','Noisy']:
                            continue
                    parse['file'] = os.path.abspath(f)
                    if not ignore_text:
                        txt = str(Path(f.removesuffix('..flac').removesuffix('.flac')
                            ))+'.txt'
                        if not os.path.exists(txt):
                            if os.path.exists(txt[:-4]+'..txt'):
                                txt = txt[:-4]+'..txt'
                            elif os.path.exists(txt[:-5]+'.txt'):
                                txt = txt[:-5]+'.txt'
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

    def lookup_by_substr(self, st):
        return [x for x in self.file_dict.keys() if st in x]

    def label_mapping(label_basename):
        if label_basename in SPECIAL_LABEL_MAPPINGS:
            return (SPECIAL_LABEL_MAPPINGS[label_basename], 
                SPECIAL_LABEL_MAPPINGS_SPECIFIER.get(label_basename), True)
        mapping = ''
        sp = label_basename.split('_')
        if len(sp) >= 3 and sp[2] == 'special source':
            mapping = 'MASTER_FILE_1/Sliced Dialogue/Special source'
            assert sp[0] == 'fim'
            specifier = ''.join(re.findall(r'\d+', sp[1]))[1:]
            specifier = 's' + specifier[0] + 'e' + str(int(specifier[1:]))
            mapping += '/' + specifier
            return mapping, specifier, True
        elif len(sp) >= 3 and sp[2] == 'outtakes':
            mapping = 'MASTER_FILE_1/Sliced Dialogue/Special source/Outtakes'
            specifier = ''.join(re.findall(r'\d+', sp[1]))[1:]
            mapping += '/' + specifier + ' outtakes'
            specifier = 's' + specifier[0] + 'e' + str(int(specifier[1:]))
            return mapping, specifier, True
        elif sp[0] == 'fim':
            mapping = 'MASTER_FILE_1/Sliced Dialogue/FiM/'
            specifier = ''.join(re.findall(r'\d+', sp[1]))[1:] 
            se_specifier = 's' + specifier[0] + 'e' + str(int(specifier[1:]))
            mapping += 'S' + specifier[0] + '/s' + specifier[0] + 'e' + str(int(specifier[1:]))
            return mapping, se_specifier, False

    #def specifier_to_num(sp):
        #return sp[1], sp[3:]

    def generate_fim_episodes_labels_index(
            labels_dir = SLICED_DIALOGUE + "/Label files",
            master_file_1 = 'D:/MLP_Samples/AIData/Master file',
            master_file_2 = 'D:/MEGASyncDownloads/Master file 2',
            max_noise = 1,
            ignore_text = False,
            no_parse = False,
            audio_input_format = '.flac',
            force_character = '',
            emotions : list = [],
            override_select : list = [],
            special_source_handling : bool = True):
            index = {}
            for (root,_,files) in os.walk(labels_dir):
                for f in tqdm(files, desc="Label files"):
                    # 'Other' not in scope for now
                    if 'Other' in root:
                        continue
                    # Process fim episodes only if no override
                    if len(override_select) == 0:
                        if 'fim' not in f or 's' not in f:
                            continue
                    # Ignore original/izo text lists, we just care about overall
                    if f.endswith('_original.txt') or f.endswith('_izo.txt') or f.endswith('_unmix.txt'):
                        continue
                    f_basename = f.removesuffix('.txt')
                    f_basename_true = f_basename.removesuffix(
                        '_special source').removesuffix('_outtakes')
                    is_special = f_basename.endswith('_special source')
                    is_outtake = f_basename.endswith('_outtakes')
                    # Outtake is priority 1
                    # Special source is priority 2
                    special_source_path = os.path.join(root, f_basename_true+'_special source.txt')
                    outtake_path = os.path.join(root, f_basename_true+'_outtakes.txt')
                    #print(f_basename, f_basename_true, is_special, is_outtake, special_source_path)
                    if special_source_handling:
                        if not is_special and not is_outtake:
                            if os.path.exists(special_source_path):
                                #print(f"base skip {f_basename_true} for outtake")
                                continue
                            if os.path.exists(outtake_path):
                                #print(f"base skip {f_basename_true} for outtake")
                                continue
                        if is_special:
                            if os.path.exists(outtake_path):
                                #print(f"special skip {f_basename_true} for outtake")
                                continue

                    mapping, specifier, special_source = PPPDataset.label_mapping(f_basename)
                    if len(override_select) != 0 and specifier not in override_select:
                        continue
                    print(f"specifier for {f_basename_true}: {specifier}")
                    placeholder_mapping = mapping
                    mapping = mapping.replace('MASTER_FILE_1', master_file_1)
                    mapping = mapping.replace('MASTER_FILE_2', master_file_2)
                    assert os.path.exists(mapping), mapping

                    index[specifier] = {
                        'lines': []
                    }
                    if len(override_select) == 0:
                        index[specifier]['season'] = specifier[1]
                        index[specifier]['episode'] = specifier[3:]
                    with open(os.path.join(root,f), encoding='utf-8') as f2:
                        line = f2.readline()
                        while line:
                            sp = [x.strip() for x in line.split('\t')]
                            sig = sp[2]
                            sig = sig.replace('?','_')
                            parse = PPPDataset.character_parse(sig)
                            filepath = os.path.join(mapping, sig+'.flac')
                            if not os.path.exists(longpath(filepath)):
                                alt_filepath = os.path.join(mapping, sig.rstrip('.').rstrip()+'.flac')
                                if os.path.exists(longpath(alt_filepath)):
                                    filepath = alt_filepath
                            placeholder_filepath = os.path.join(placeholder_mapping, sig+'.flac')
                            if not os.path.exists(longpath(filepath)):
                                print(f"Warning: {filepath} not found")
                                # Ignore lines for which the original file does not exist
                                line = f2.readline()
                                continue
                            #assert os.path.exists(longpath(filepath)), filepath
                            index[specifier]['lines'].append({
                                'ts': sp[0],
                                'te': sp[1],
                                'label': sp[2],
                                'orig_file': placeholder_filepath.replace('\\','/'),
                                'parse': parse
                            })
                            line = f2.readline()
            return index

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
        char_dict = {}
        for char,files in self.file_dict.items():
            first_file = files[0]
            audio = AudioSegment.from_file(first_file['file'])
            min_audio_ms = len(audio)
            char_audio_ms = 0
            for f in tqdm(files, "Files for character "+char):
                audio = AudioSegment.from_file(f['file'])
                audio_length_ms = len(audio)
                min_audio_ms = min(min_audio_ms, audio_length_ms)
                max_audio_ms = max(max_audio_ms, audio_length_ms)
                char_audio_ms += audio_length_ms
                total_audio_ms += audio_length_ms
            print(f"Character: {char} ({len(files)} lines {char_audio_ms/1000} s)\n"
                f"Min: {min_audio_ms/1000} s, Max: {max_audio_ms/1000} s\n",
                f"Total: {total_audio_ms/1000} s")
            char_dict[char] = char_audio_ms/1000
        return char_dict

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

idx = PPPDataset.generate_fim_episodes_labels_index(
    override_select = [
        'eqg_dance magic',
        'eqg_forgotten friendship',
        'eqg_friendship_games',
        'eqg_legend_of_everfree',
        'eqg_mirror magic',
        'eqg_movie magic',
        'eqg_rollercoaster of friendship',
        'eqg_better together_s02e04',
        'eqg_better together_s02e05',
        'eqg_better together_s02e06',
        'eqg_better together_s02e07',
        'eqg_better together_s02e08',
        'fim_movie',
        'fim_rainbow roadtrip',
    ],
    special_source_handling = False
)
import json
with open('extras_labels_index.json','w',encoding='utf-8') as f:
    json.dump(idx, f, ensure_ascii=False)
print("Done")

# There are two main episode specifications:
# Pony.Tube: S08E01
# YayPonies: 08x06