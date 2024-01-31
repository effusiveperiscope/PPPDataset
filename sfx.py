from pydub import AudioSegment
from tqdm import tqdm
import os
import random
import pickle
SFX_DIRECTORY = r"D:\SFX and Music\SFX"
BGM_DIRECTORY = r"D:\MEGASyncDownloads\Master file 2\SFX and Music\Music"

VOCAL_TAGS = {
    "arguing", "agree", "agreement", "annoyed", "beatbox", "beatboxing", 
    "big gasp", "big sneeze", "bird call", "bleh", "burp",
    "chatter", "clear throat", "crying", "exertion", "gasp", "groan", "grunt",
    "hit", "hmmm", "laugh", "mutter", "panting", "anxious laugh", "uh huh",
    "whoa", "scream", "evil laugh", "nervous laugh", "nervous laughter",
    "overly dramatic crying", "ow", "ooh", "oof", "fun", "roar", "shout",
    "sigh", "wahoo", "yelp", "cheer",
    # some of these have no voices...
    "twilight", "pinkie", "applejack", "rainbow", "fluttershy", "rarity",
    "spike", "granny smith", "trixie", "cozy glow", "starlight", "mane-iac",
    "big mac", "scootaloo", "sweetie belle", "apple bloom", "applebloom",
    "caballeron", "cadance", "chrysalis", "tirek", "cmc", "discord",
    "nightmare moon", "sombra", "steven magnet", "mane 6", "minuette",
    "braeburn", "sunburst", "tree hugger", "pinkie swarm", "autumn blaze", 
    "octavia"
}
MOVE_TAGS = {
    "galloping", "hoofsteps", "jump"
}

class PPPSFXDataset:
    def __init__(self,
        sfx_directory = SFX_DIRECTORY,
        bgm_directory = BGM_DIRECTORY):
        print("Collecting SFX data")
        self.sfx_dict = {}
        self.nonvocal_dict = {}
        self.move_dict = {}
        self.crowd_dict = {}
        for root, _, files in os.walk(SFX_DIRECTORY):
            for f in files:
                if not f.endswith('.flac'):
                    continue
                path_f = os.path.join(root, f)
                tags = [t.strip().lower() for t in f.split('~')[0].split(',')]
                self.sfx_dict[path_f] = {'tags': tags}
                # faster like this
                if all(t not in VOCAL_TAGS for t in tags):
                    self.nonvocal_dict[path_f] = {'tags': tags}
                if any(t in MOVE_TAGS for t in tags):
                    self.move_dict[path_f] = {'tags' : tags}
                if ("crowd" in tags):
                    self.crowd_dict[path_f] = {'tags' : tags}

        print("Collecting music data")
        self.bgm_dict = {}
        for root, _, files in os.walk(BGM_DIRECTORY):
            for f in files:
                if not f.endswith('.flac'):
                    continue
                path_f = os.path.join(root, f)
                tags = [t.strip() for t in f.split('~')[0].split(',')]
                self.bgm_dict[path_f] = {'tags': tags}
                #print(tags)

    def fill_layer(
        self,
        target_segment,
        target_duration,
        source_list,
        chance = 1.0,
        gap_ms = 0):
        working_segment = target_segment
        cur_duration = 0
        dirty = False
        while cur_duration < target_duration:
            to_overlay = random.choice(source_list)
            to_overlay = AudioSegment.from_file(to_overlay, channels=1)
            # select a random segment from the file if it is longer than 10 secs
            if len(to_overlay) > target_duration:
                max_start = len(to_overlay) - target_duration
                rand_start = random.randint(0, max_start)
                to_overlay = to_overlay[rand_start:rand_start+target_duration-1]
            if random.random() <= chance:
                working_segment = working_segment.overlay(
                    to_overlay, position=cur_duration)
                dirty = True
            cur_duration += len(to_overlay) + gap_ms
        return working_segment, dirty

    # MUSDB format
    def synthetic_mixdown(
        self,
        pppdataset_paths,
        out_path='sfx_synthdata',
        n_samples=5,
        sample_duration=10, # sample length for demucs
        seed=42,
        max_sfx_layers=3,
        crowd_chance=0.02,
        move_chance=0.1,
        bgm_chance=0.3,
        bgm_volume=0.2,
        sfx_gainred_min=-6,
        sfx_gainred_max=-8):

        random.seed(seed)

        dialogue_paths = pppdataset_paths
        sfx_paths = list(self.sfx_dict.keys())
        nonvocal_paths = list(self.nonvocal_dict.keys())
        move_paths = list(self.move_dict.keys())
        crowd_paths = list(self.crowd_dict.keys())
        bgm_paths = list(self.bgm_dict.keys())
        print(f"nv: {len(nonvocal_paths)}, mv: {len(move_paths)}, "
            f"crowd: {len(crowd_paths)}")

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

        for n in tqdm(range(n_samples), desc="Generating samples"):
            sample_folder = os.path.join(out_path,str(n))
            if not os.path.exists(sample_folder):
                os.makedirs(sample_folder, exist_ok=True)

            vocal_base = AudioSegment.silent(duration=sample_duration*1000) # ms
            sfx_base = AudioSegment.silent(duration=sample_duration*1000)

            # Add dialogue layer
            vocal_base, _ = self.fill_layer(
                vocal_base, sample_duration*1000, dialogue_paths, gap_ms = 500)
            vocal_base = vocal_base.set_sample_width(2).set_frame_rate(48000)
            vocal_base.export(os.path.join(sample_folder,f"vocals.wav"),
                format='wav')

            dirty = False

            # Overlay a crowd layer
            if random.random() < crowd_chance:
                sfx_base, d = self.fill_layer(
                    sfx_base, sample_duration*1000, crowd_paths, 0.5)
                sfx_base = sfx_base.apply_gain(random.randint(
                    sfx_gainred_max, sfx_gainred_min+1))
                dirty = dirty or d

            # Overlay a BGM layer
            if random.random() < bgm_chance:
                sfx_base, d = self.fill_layer(
                    sfx_base, sample_duration*1000, bgm_paths)
                sfx_base = sfx_base.apply_gain(random.randint(
                    sfx_gainred_max, sfx_gainred_min+1))
                dirty = dirty or d

            # Overlay a move layer
            if random.random() < move_chance:
                sfx_base, d = self.fill_layer(
                    sfx_base, sample_duration*1000, move_paths, 0.5)
                dirty = dirty or d

            # Add extra sfx layers
            n_sfx_layers = random.randint(1,max_sfx_layers)
            for l in range(max_sfx_layers):
                sfx_base, _ = self.fill_layer(
                    sfx_base, sample_duration*1000, nonvocal_paths, 
                    0.5 if dirty else 1.0) # 100% chance for a SFX if not dirty yet

            sfx_base = sfx_base.apply_gain(random.randint(
                sfx_gainred_max, sfx_gainred_min+1))
            sfx_base = sfx_base.set_sample_width(2).set_frame_rate(48000)
            
            sfx_base.export(os.path.join(sample_folder,f"other.wav"),
                format='wav')

            mixed = vocal_base.overlay(sfx_base)
            mixed = mixed.set_sample_width(2).set_frame_rate(48000)
            mixed.export(os.path.join(sample_folder,f"mixture.wav"),
                format='wav')
    pass

from ppp import PPPDataset
import pickle

# Collect CLEAN dialogue for all characters
#dialogue_dataset = PPPDataset.collect(
#    characters=[], max_noise=0)
#with open('all_paths_pickle.pkl','wb') as f:
#    pickle.dump(dialogue_dataset.all_dialogue_paths(), f)

# Cached this in a pickle to save time
with open('all_paths_pickle.pkl','rb') as f:
    pppdataset_paths = pickle.load(f)
sfx_dataset = PPPSFXDataset()
#sfx_dataset.synthetic_mixdown(pppdataset_paths, n_samples=200)
# test set
sfx_dataset.synthetic_mixdown(pppdataset_paths, seed=50, n_samples=2000)