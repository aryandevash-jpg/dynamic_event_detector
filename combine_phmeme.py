import os
import json
import pandas as pd

rows = []
base_path = "/Users/aryanjangde/Downloads/6392078/all-rnr-annotated-threads"  # your root directory

for event in os.listdir(base_path):
    event_path = os.path.join(base_path, event)
    if not os.path.isdir(event_path):
        continue  # skip .DS_Store and other non-directory entries

    for rumour_type in ["rumours", "non-rumours"]:
        type_path = os.path.join(event_path, rumour_type)
        if not os.path.isdir(type_path):
            continue

        for tweet_id in os.listdir(type_path):
            # PHEME uses "source-tweets" (plural); fall back to singular just in case.
            source_path = os.path.join(type_path, tweet_id, "source-tweets")
            if not os.path.isdir(source_path):
                source_path = os.path.join(type_path, tweet_id, "source-tweet")
            if not os.path.isdir(source_path):
                continue

            for f in os.listdir(source_path):
                if not f.endswith(".json"):
                    continue  # skip .DS_Store etc.
                with open(os.path.join(source_path, f)) as file:
                    tweet = json.load(file)
                    rows.append({
                        "tweet_id": tweet.get("id_str"),
                        "tweet_text": tweet.get("text"),
                        "timestamp": tweet.get("created_at"),
                        "event": event,
                        "rumour_type": rumour_type
                    })

df = pd.DataFrame(rows)
df.to_csv("pheme_all.csv", index=False)