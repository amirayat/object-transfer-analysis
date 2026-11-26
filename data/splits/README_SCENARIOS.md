This folder contains `manifest.csv` with 17 filenames from the screenshot. Its initial
`development`/`test` assignments are **PROVISIONAL** because filenames cannot
reveal the scenario. The split is an index, not a physical copy of videos.

1. Edit the `scenario` column for each actual video based on viewing the clip.
2. Run `python src/dataset/prepare_dataset.py --resplit` to select one held-out
   test clip from every scenario. With six scenario groups with at least two
   videos each, this produces **11 development** and **6 test** clips.
3. Do not change hyperparameters based on the final six test recordings.

If you later record the missing 18th clip, rerun `prepare_dataset.py` and
revisit the split without leaking test clips into model/rule development.
