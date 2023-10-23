import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import json
import os
import logging
import time

logging.basicConfig(level=logging.INFO)

ARCH_PATH = r"D:\Archival\fimfarchive-20230901"

class Story:
    def __init__(self, id, story_index : dict):
        self.id = id
        self.index = story_index

    def compare_tags(self, tags : set, negative_tags : set = set()):
        if not hasattr(self, 'tags_set'):
            self.tags_set = set()
            for t in self.index.get("tags",[]):
                self.tags_set.add(t["name"])
        return (tags.issubset(self.tags_set)
            and negative_tags.isdisjoint(self.tags_set))

    def score(self):
        return self.index["num_likes"] - self.index["num_dislikes"]

    def ratio(self):
        return self.index["num_likes"] / (
            self.index["num_likes"] + self.index["num_dislikes"])

    def path(self):
        return self.index["archive"]["path"]

    def desc_text(self):
        return BeautifulSoup(self.index["description_html"]).get_text()

    def desc_intersect(self, words : set):
        from util import remove_punc
        desc_set = set(remove_punc(self.desc_text().lower()).split(' '))
        return words.intersection(desc_set)

    def write_full_text(self, stream, paragraph_filter = None):
        epub_path = os.path.join(ARCH_PATH, self.path())
        book = epub.read_epub(epub_path)
        for chapter in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(chapter.content, 'html.parser')
            content_div = soup.find('div', id='content')
            if content_div:
                paras = content_div.find_all('p')
                for p in paras:
                    stream.write(p.get_text())
                    stream.write('\n\n')
                stream.write('\n') 

class MatchedStories:
    def __init__(self, matched_list : list, matched_ids_list : list):
        self.matched_list = matched_list
        self.matched_ids_list = matched_ids_list

    def print_titles(self):
        for story in self.matched_list:
            print(story.index["title"])

    def print_descs(self):
        for story in self.matched_list:
            print(story.desc_text())

    def print_ids(self):
        print(self.matched_ids_list)

    def dump_to_file(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for story in self.matched_list:
                story.write_full_text(f)

class FIMFarchive:
    def __init__(self):

        with open(os.path.join(ARCH_PATH, "index.json"),
            "r", encoding="utf-8") as f:
            start_time = time.time()
            self.index = json.loads(f.read())

            self.stories = [Story(id,item) for (id,item) in self.index.items()]
            end_time = time.time()

            logging.info('Loaded index in '+str(end_time - start_time)+' s')

    def filter(self, comparator):
        matched_stories = []
        matched_ids = []
        for story in self.stories:
            if comparator(story):
                matched_stories.append(story)
                matched_ids.append(story.id)
        return MatchedStories(matched_stories, matched_ids)

    def from_ids(self, ids):
        matched_stories = []
        for id in ids:
            matched_stories.append(Story(id, self.index[id]))
        return MatchedStories(matched_stories, ids)

def example_comparator(story):
    return (story.compare_tags({"Sex", "Second Person"},
        {"Anthro","My Little Pony: Equestria Girls"})
            and story.ratio() > 0.8
            and not story.desc_intersect(
                {"futa", "pegging", "oviposition", "shrinking", "macro"}))

archive = FIMFarchive()
match1 = archive.filter(example_comparator)
match1.print_titles()
match1.print_descs()
match1.dump_to_file('match3.txt')
