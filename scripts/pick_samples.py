import pandas as pd

manifest = pd.read_csv("data/processed/manifests/dataset_manifest.csv")
an = manifest[manifest["token"] == "a_n"]

for pathology in ["healthy", "Rekurrensparese", "Stimmlippenpolyp", "Laryngitis"]:
    subset = an[an["pathology_de"] == pathology].head(2)
    for _, row in subset.iterrows():
        print(
            f"{pathology} | {row['sample_key']} | {row['wav_path']} | speaker={row['speaker_id']}"
        )
