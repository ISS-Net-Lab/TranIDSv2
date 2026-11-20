# Tran-IDS v2

## Model training and adaptation

1. [transformer_AEViT_1D.py](adaption/transformer_AEViT_1D.py): Pretrains the model.
2. [transformer_AEViT_ft.py](adaption/transformer_AEViT_ft.py): Fine-tunes the pretrained model.
3. [transformer_AEViT_1D_parquet_tabular.py](adaption/transformer_AEViT_1D_parquet_tabular.py): Pretrains the tabular model.
4. [transformer_AEViT_ft_tabular.py](adaption/transformer_AEViT_ft_tabular.py): Fine-tunes the pretrained tabular model.
5. [transformer_AEViT_ft_attack_opposite.py](adaption/transformer_AEViT_ft_attack_opposite.py): Runs the source-to-target fine-tuning experiment with source attack samples.

## PCAP extraction

1. [edit_pcaps_relative.ps1](extractor/edit_pcaps_relative.ps1): Converts raw network capture files into standardized PCAP format using EditCap.
2. [split_pcaps_relative.ps1](extractor/split_pcaps_relative.ps1): Splits large PCAP files into individual network flows using SplitCap.
3. [pack_flows_relative.ps1](extractor/pack_flows_relative.ps1): Compresses extracted flow PCAP files into multiple 7z archives for efficient storage and transfer.
4. [run_unpack_feature_cleanup.sh](extractor/run_unpack_feature_cleanup.sh): Automates NB15 archive extraction, feature extraction using get_featureNB15.py, and temporary file cleanup.
5. [run_unpack_feature_cleanup_toniot.sh](extractor/run_unpack_feature_cleanup_toniot.sh): Automates TON_IoT archive extraction, feature extraction using get_featureTONIoT.py, and temporary file cleanup.
6. [get_featureNB15.py](extractor/get_featureNB15.py): Extracts 480-byte flow features (5 packets × 96 bytes), assigns attack labels using NB15 Ground Truth rules, and saves the results as NPY files.
7. [get_featureTONIoT.py](extractor/get_featureTONIoT.py): Extracts 480-byte flow features (5 packets × 96 bytes), assigns attack labels using TON_IoT Ground Truth matching, and saves the results as NPY files.
8. [merge_npy_across_parts.py](extractor/merge_npy_across_parts.py): Merges class-specific NPY files from multiple NB15 data partitions into unified datasets.

models:https://drive.google.com/drive/folders/1Lh4iAV98fvGBDTLf2i6tOZL2BZLPIX2P?usp=sharing
