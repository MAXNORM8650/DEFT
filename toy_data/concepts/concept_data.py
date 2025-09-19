#!/usr/bin/env python
# file: build_personalization_eval_jsonl.py
import os, json, csv, textwrap, argparse, itertools, pathlib

# ----------------------------------------------------------------------
# 1.  DATA SOURCES  ────────────────────────────────────────────────────
# ----------------------------------------------------------------------
CLASSES_CSV = """
subject_name,class
backpack,backpack
backpack_dog,backpack
bear_plushie,stuffed animal
berry_bowl,bowl
can,can
candle,candle
cat,cat
cat2,cat
clock,clock
colorful_sneaker,sneaker
dog,dog
dog2,dog
dog3,dog
dog5,dog
dog6,dog
dog7,dog
dog8,dog
duck_toy,toy
fancy_boot,boot
grey_sloth_plushie,stuffed animal
monster_toy,toy
pink_sunglasses,glasses
poop_emoji,toy
rc_car,toy
red_cartoon,cartoon
robot_toy,toy
shiny_sneaker,sneaker
teapot,teapot
vase,vase
wolf_plushie,stuffed animal
""".strip()

OBJECT_PROMPTS = [
    'a {0} {1} in the jungle',
    'a {0} {1} in the snow',
    'a {0} {1} on the beach',
    'a {0} {1} on a cobblestone street',
    'a {0} {1} on top of pink fabric',
    'a {0} {1} on top of a wooden floor',
    'a {0} {1} with a city in the background',
    'a {0} {1} with a mountain in the background',
    'a {0} {1} with a blue house in the background',
    'a {0} {1} on top of a purple rug in a forest',
    'a {0} {1} with a wheat field in the background',
    'a {0} {1} with a tree and autumn leaves in the background',
    'a {0} {1} with the Eiffel Tower in the background',
    'a {0} {1} floating on top of water',
    'a {0} {1} floating in an ocean of milk',
    'a {0} {1} on top of green grass with sunflowers around it',
    'a {0} {1} on top of a mirror',
    'a {0} {1} on top of the sidewalk in a crowded street',
    'a {0} {1} on top of a dirt road',
    'a {0} {1} on top of a white rug',
    'a red {0} {1}',
    'a purple {0} {1}',
    'a shiny {0} {1}',
    'a wet {0} {1}',
    'a cube shaped {0} {1}',
]

LIVE_PROMPTS = OBJECT_PROMPTS[:10] + [                 # keep the first 10 “scene” prompts
    'a {0} {1} wearing a red hat',
    'a {0} {1} wearing a santa hat',
    'a {0} {1} wearing a rainbow scarf',
    'a {0} {1} wearing a black top hat and a monocle',
    'a {0} {1} in a chef outfit',
    'a {0} {1} in a firefighter outfit',
    'a {0} {1} in a police outfit',
    'a {0} {1} wearing pink glasses',
    'a {0} {1} wearing a yellow shirt',
    'a {0} {1} in a purple wizard outfit',
] + OBJECT_PROMPTS[-5:]                                # colour / material / shape prompts

LIVE_CLASS_TOKENS = {'dog', 'cat', 'person', 'human'}   # extend if needed
UNIQUE_TOKEN = 'sks'                                    # stays the same as in your early examples
# ----------------------------------------------------------------------

def read_class_map(csv_text: str) -> dict[str, str]:
    rdr = csv.DictReader(textwrap.dedent(csv_text).splitlines())
    return {row['subject_name']: row['class'] for row in rdr}

def build_jsonl(root: pathlib.Path, out_file: pathlib.Path):
    class_map = read_class_map(CLASSES_CSV)
    lines_out = []
    for subject, class_token in class_map.items():
        subject_dir = root / subject
        if not subject_dir.is_dir():
            print(f"⚠️  Folder missing: {subject_dir} – skipping")
            continue
        img_files = sorted(f for f in subject_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not img_files:
            print(f"⚠️  No images inside {subject_dir}")
            continue

        prompts = LIVE_PROMPTS if class_token in LIVE_CLASS_TOKENS else OBJECT_PROMPTS
        for idx, img_path in enumerate(img_files):
            prompt = prompts[idx % len(prompts)].format(UNIQUE_TOKEN, class_token)
            lines_out.append({
                "task_type": "text_to_image",
                "instruction": prompt,
                "input_images": [],
                "output_image": img_path.as_posix(),   # e.g. "dog/00.jpg"
            })

    out_file.write_text("".join(json.dumps(l) + "\n" for l in lines_out))
    print(f"✅ Wrote {len(lines_out):,} lines to {out_file}")

# ----------------------------------------------------------------------
# 2.  CLI  ─────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate personalization-eval JSONL")
    parser.add_argument("dataset_root", type=pathlib.Path, help="Folder that contains backpack/, dog/, … sub-folders")
    parser.add_argument("-o", "--out", default="personalization_eval.jsonl", type=pathlib.Path)
    args = parser.parse_args()
    build_jsonl(args.dataset_root, args.out)