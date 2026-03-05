"""Quick dataset statistics."""

import pandas as pd

manifest = pd.read_csv("data/processed/manifests/dataset_manifest.csv")
print("=== MANIFEST SHAPE ===")
print(manifest.shape)
print()
print("=== COLUMNS ===")
print(list(manifest.columns))
print()
print("=== PATHOLOGY DISTRIBUTION ===")
print(manifest["pathology_de"].value_counts().to_string())
print()
print("=== TOKEN DISTRIBUTION ===")
if "token" in manifest.columns:
    print(manifest["token"].value_counts().to_string())
print()
print("=== UNIQUE SPEAKERS ===")
if "speaker_id" in manifest.columns:
    n_speakers = manifest["speaker_id"].nunique()
    print(f"Total unique speakers: {n_speakers}")
    print()
    print("Speakers per pathology:")
    print(manifest.groupby("pathology_de")["speaker_id"].nunique().to_string())
print()
print("=== SEX DISTRIBUTION ===")
if "sex" in manifest.columns:
    print(manifest["sex"].value_counts().to_string())

# Also check feature counts
print()
print("=== FEATURE TABLE SHAPES ===")
for name in [
    "sample_core",
    "acoustic_features",
    "multifractal_features",
    "opensmile_features",
    "neurokit2_features",
]:
    try:
        df = pd.read_csv(f"data/processed/features/{name}.csv")
        print(f"{name}: {df.shape}")
        if name != "sample_core":
            # Count feature columns (excluding status/error/key columns)
            feat_cols = [
                c
                for c in df.columns
                if c
                not in [
                    "sample_key",
                    "acoustic_status",
                    "acoustic_error",
                    "mf_status",
                    "mf_error",
                    "opensmile_status",
                    "opensmile_error",
                    "nk_status",
                    "nk_error",
                    "mf_num_scales",
                    "mf_num_q",
                ]
            ]
            print(f"  Feature columns: {len(feat_cols)}")
    except Exception as e:
        print(f"{name}: ERROR - {e}")

# a_n token only stats
print()
print("=== a_n TOKEN ONLY ===")
if "token" in manifest.columns:
    an_only = manifest[manifest["token"] == "a_n"]
    print(f"Samples: {len(an_only)}")
    print(f"Unique speakers: {an_only['speaker_id'].nunique()}")
    print()
    print("Per pathology:")
    print(an_only["pathology_de"].value_counts().to_string())
