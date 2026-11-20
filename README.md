# Tran-IDS v2

## Model training and adaptation

1. [transformer_AEViT_1D.py](adaption/transformer_AEViT_1D.py): Pretrains the model.
2. [transformer_AEViT_ft.py](adaption/transformer_AEViT_ft.py): Fine-tunes the pretrained model.
3. [transformer_AEViT_1D_parquet_tabular.py](adaption/transformer_AEViT_1D_parquet_tabular.py): Pretrains the tabular model.
4. [transformer_AEViT_ft_tabular.py](adaption/transformer_AEViT_ft_tabular.py): Fine-tunes the pretrained tabular model.
5. [transformer_AEViT_ft_attack_opposite.py](adaption/transformer_AEViT_ft_attack_opposite.py): Runs the source-to-target fine-tuning experiment with source attack samples.

### Evaluation protocol

The adaptation scripts calibrate the anomaly threshold from benign validation reconstruction errors. The default target validation false-positive rate is 5%. Samples with scores at or above the threshold are classified as attacks. Test metrics use the frozen threshold.

Pretraining ranks Optuna trials by validation MSE and evaluates only the selected checkpoint on the test split.

## Experiment configurations

### Manuscript settings

| Component | Configuration |
| --- | --- |
| Datasets and transfer | CIC-IDS-2018 and UNSW-NB15, evaluated in both source-to-target directions. |
| Flow-prefix input | Group both directions of a flow and order packets by capture time. Exclude TCP handshake packets. Keep up to 5 packets and read at most 120 bytes per packet. Remove 24 bytes of MAC addresses, IPv4 addresses, and ports. Zero-pad short prefixes and scale bytes by 255 to obtain a `5 × 96` input. |
| Flow-prefix AE-ViT | Patch size 12, 40 patch tokens, 6 encoder blocks, 2 decoder blocks, embedding dimension 384, bottleneck dimension 32. |
| Training | AdamW with learning rate `2e-4`, zero weight decay, cosine annealing, up to 50 epochs, and patience 10 on validation reconstruction loss. Source training and source validation use benign flows. |
| Target protocol | Split target data `70/30` into an adaptation pool and a held-out test pool. Draw validation samples from the benign subset of the adaptation pool. Compare the source checkpoint and adapted model on the same held-out target test split. |
| Reporting | AUROC is the primary metric. Report the median over five runs and the observed minimum and maximum for error bars. Report F1, precision, recall, and FPR at the chosen operating point. |

### Experiment matrix

| Paper reference | Configuration and run notes |
| --- | --- |
| Table 3 and Figs. 4–6 | Compare both transfer directions before and after adaptation with the default depth `B=3`. |
| Fig. 7 | Compare sanitized flow prefixes with NF-CIC-IDS-2018 and NF-UNSW-NB15 statistical flow records in both directions. |
| Fig. 8 | Sweep target buffer sizes of 150,000, 270,000, 350,000, 500,000, 700,000, and 1,000,000 flows. |
| Figs. 9–10 | Sweep adaptation depth `B=1–3` with a buffer of 1,000,000 flows. |
| Fig. 11 | Evaluate both directions with 500, 1,000, 2,500, 5,000, and 10,000 benchmark attack samples in the adaptation buffer. |

The bottleneck, patch head, and decoder normalization are trainable at every depth. The block indices below are zero-based.

| Depth | Updated transformer blocks | Trainable parameters |
| --- | --- | ---: |
| 1 | `decoder[1]` | 1,528,028 |
| 2 | `decoder[0]`, `decoder[1]` | 3,007,196 |
| 3 (default) | `encoder[5]`, `decoder[0]`, `decoder[1]` | 4,486,364 |

### Random seeds

The fine-tuning scripts use seeds `[7, 17, 27, 37, 47]` and split seed `42`.

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
