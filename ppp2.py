from util import check_file_dur_ms, longpath, character_parse, test_extensions
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
import os
import pickle
import logging
import math
import numpy as np
import ffmpeg
import random
from logging import warn, info, error

logging.basicConfig(level=logging.INFO)

@dataclass
class FolderSpec:
    """
    `path` is the path to the folder to search

    `parse` is a bool indicating whether you want the filename to be parsed
    according to common PPP filename rules

    `character_override` is a string which can be used to label the character if the
    filename is not parsed according to `parse`
    """
    path: Path
    parse: bool = True
    character_override: str = ""

@dataclass
class ExportSpec:
    """
    `export_path` is the export path

    `list_path` optionally specifies a path for a filelist to be created

    `split_frac` specifies the split fraction; by default this is applied
    evenly across all characters (I haven't encountered a situation where
    this is )

    `sr` = 0 indicates that the original sample rate is used (no resampling)
    """
    export_path : str = ''
    list_path : str = ''
    split_frac : float = 1.0
    sr : int = 0
    userdata : dict = field(default_factory=dict)

class PPPDataset2:
    @dataclass
    class Parse:
        hour: int = 0
        mins: int = 0
        sec: int = 0
        char: str = ''
        emotion: str = ''
        txt: str = ''
        line: str = ''
        noise: str = ''
        file: str = ''
        out_path: str = ''
        process_idx : int = -1

    def __init__(self):
        # Split by characters
        self.file_dict = {}

    def __len__(self):
        return sum(len(lst) for lst in self.file_dict.values())

    def save_to_pickle(self, pkl_path):
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.file_dict, f)

    def from_pickle(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            self.file_dict = pickle.load(f)

    def from_file(
        folder_specs: list[FolderSpec] = [],
        characters: list[str] = [],
        default_character : str = 'Audio',
        emotions: list[str] = [],
        noise: list[str] = ['', 'Noisy'],
        allow_extensions: list[str] = ['.wav', '.flac'],
        ignore_txt: bool = False
        ):
        """
        `folder_specs` is a list of FolderSpec indicating what directories to
        search

        `characters` is a list of character names indicating what character
        names to filter by (only works if character name parsing is used).
        If it is empty then the audio will not be filtered by character

        `default_character` is used if parse is disabled but no character name
        is provided

        `emotions` filters by emotion similarly to `characters`

        `noise` filters by noise similarly to `characters`

        `allow_extensions` is a list of audio formats to allow in filtering

        `ignore_txt`: Normally transcripts are obtained from a text file
        corresponding to each audio file; however this will disable that option
        """
        dataset = PPPDataset2()

        def collect_from_spec(spec):
            for (root,_,files) in os.walk(spec.path):
                for file in files:
                    data = process_file(root, spec, file)
                    if data is None:
                        continue
                    if not data.char in dataset.file_dict:
                        dataset.file_dict[data.char] = []
                    dataset.file_dict[data.char].append(data)

        def process_file(root, spec, file):
            data = PPPDataset2.Parse()

            ext = test_extensions(file, allow_extensions)
            if ext is None:
                return None # Non-matching extension
            f = os.path.join(root, file)
            data.file = os.path.abspath(f)
            if spec.parse == False:
                if len(spec.character_older):
                    data.char = spec.character_older
                else:
                    data.char = default_character
            else:
                parse = character_parse(f)
                if parse is None:
                    warn(f'Failed parse for {f}')
                    return None

                if len(characters) and (parse['char'] not in characters):
                    return None
                if len(emotions) and (parse['emotion'] not in emotions):
                    return None
                if len(noise) and (parse['noise'] not in noise):
                    return None
                data.hour = parse['hour']
                data.mins = parse['min']
                data.sec = parse['sec']
                data.char = parse['char']
                data.emotion = parse['emotion']
                data.noise = parse['noise']

            if not ignore_txt:
                txt = str(Path(f.removesuffix('..flac').removesuffix('.flac')
                    ))+'.txt'
                orig_txt = txt
                if not os.path.exists(longpath(txt)):
                    if os.path.exists(longpath(txt[:-4]+'..txt')):
                        txt = txt[:-4]+'..txt'
                    elif os.path.exists(longpath(txt[:-5]+'.txt')):
                        txt = txt[:-5]+'.txt'
                if not os.path.exists(longpath(txt)):
                    txt = orig_txt+'.txt'
                    if os.path.exists(longpath(txt[:-4]+'..txt')):
                        txt = txt[:-4]+'..txt'
                    elif os.path.exists(longpath(txt[:-5]+'.txt')):
                        txt = txt[:-5]+'.txt'
                if not os.path.exists(longpath(txt)):
                    warn(f'Could not find associated text file for {f}')
                data.txt = os.path.abspath(txt)
                with open(longpath(txt), 'r', encoding='utf8') as f:
                    data.line = f.read()
            return data

        all_no_parse = all([s.parse == False for s in folder_specs])
        for spec in folder_specs:
            collect_from_spec(spec)

        if len(characters):
            info(f"Finished collecting data for {characters} "
                f"(len {len(dataset)})")
        else:
            info(f"Finished collecting data for all characters "
                f"(len {len(dataset)})")
        return dataset

    def stats(self):
        from pydub import AudioSegment
        info("Collecting stats...")
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
            info(f"Character: {char} ({len(files)} lines {char_audio_ms/1000} s)\n"
                f"Min: {min_audio_ms/1000} s, Max: {max_audio_ms/1000} s\n",
                f"Total: {total_audio_ms/1000} s")
            char_dict[char] = char_audio_ms/1000
        return char_dict

    def export(
        self,
        specs : list[ExportSpec] = [],
        filename_formatter = lambda parse: Path(parse.file).stem+'.wav',
        fileline_formatter = lambda parse: parse.txt,
        extra_process = lambda spec, parse: None,
        do_shuffle : bool = False,
        random_seed : int = 0):
        """
        `specs` is a list of export specs.
        One spec corresponds to one data split.
        """

        # Resampling/reformatting task
        def process_audio(
            idx : int,
            spec : ExportSpec,
            data : PPPDataset2.Parse):
            if spec.export_path is not None:
                export_path = os.path.abspath(spec.export_path)
                os.makedirs(export_path, exist_ok=True)

                data.process_idx = idx
                out_path = os.path.join(export_path,
                    filename_formatter(data))
                data.out_path = out_path
                ffmpeg_opts = {'ac':1}
                if spec.sr != 0:
                    ffmpeg_opts['ar'] = spec.sr
                ffmpeg.input(data.file).output(out_path, **ffmpeg_opts).run(
                    overwrite_output=True
                )

            extra_process(spec, data)
            return data

        splits = [[] for s in specs]
        if len(specs) > 1:
            split_frac_sum = sum([s.split_frac for s in specs])
            assert math.isclose(split_frac_sum, 1.0), \
                "Split fractions do not add to 1.0"

        filelines = []

        idx : int = 0
        with tqdm(total=len(self)) as pbar:
            for char, data in self.file_dict.items():
                data_shuffled = data
                random.seed(random_seed)
                if do_shuffle:
                    random.shuffle(data_shuffled)

                split_frac_sum = 0.0
                split_begin = 0
                split_end = 0

                if len(specs) > len(data):
                    error(f"More splits ({len(specs)}) than"
                    f" files ({len(data)} in character "
                        f"{char}")
                    raise ValueError("More splits than files for character")
                for i, spec in enumerate(specs):
                    split_begin = split_end
                    split_begin = min(split_begin, len(data))
                    split_frac_sum += spec.split_frac

                    split_end += int(split_frac_sum * len(data))
                    split_end = np.clip(split_end, 1, len(data))
                    for d in data_shuffled[split_begin:split_end]:
                        newd = process_audio(idx, spec, d)
                        filelines.append(fileline_formatter(newd))
                        idx += 1
                        pbar.update(1)
                        splits[i].append(d)

        # Generate
        for i, spec in enumerate(specs):
            if len(spec.list_path):
                with open(spec.list_path, 'w', encoding='utf-8') as f:
                    for l in filelines:
                        f.write(l+'\n')