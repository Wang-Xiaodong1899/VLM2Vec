import json

json_path = "video_itg_data_single_clip_55884.json"

with open(json_path, "r") as f:
    data = json.load(f)

# save 2k data
with open("video_itg_data_single_clip_55884_processed_16k.json", "w") as f:
    json.dump(data[:16000], f, indent=4)
        