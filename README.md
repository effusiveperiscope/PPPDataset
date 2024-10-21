# What is this
Tools for manipulating PPP voice and text data and an attempt to make combined
synthetic training data for pony-related LLMs, made by a codelet.

# ppp2.py
New and only slightly improved version of `ppp.py`
Example:
```py
from ppp2 import PPPDataset2, FolderSpec, ExportSpec
dataset = PPPDataset2.from_file(
    folder_specs = [FolderSpec(
        path = r'D:\MLP_Samples\AIData\Master file'
    )],
    characters=['Twilight'])
dataset.export(
    specs = [
        ExportSpec(
            export_path = r'D:\Code\GPT-SoVITS\Twilight_data',
            list_path = r'D:\Code\GPT-SoVITS\filelist_Twilight.list'
        )
    ],
    filename_formatter = lambda parse: f'{parse.process_idx}.wav',
    fileline_formatter = lambda parse: f'{parse.out_path}|{parse.char}|en|{parse.line}'
)
```

# ppp.py
Point it at your Sliced Dialogue directory. Can be used to reformat datasets for
training voice models (example is for PITS). Also has an updated version of
`horsewords.clean` for ARPAbet substitutions.

# FiMFiction Tools
Various tools for trimming/sampling the fimficOmegaV3 dataset.

# fimfarchive.py
A tool for accessing and manipulating a downloaded copy of the fimfarchive archives.

# Text training data
Tools for scraping sources like the FiMFiction wiki for episode summaries,
episode titles/transcripts, Wikipedia episode synopsis, as well as experiments
for using LLMs via oobabooga api to create synthetic training data.
