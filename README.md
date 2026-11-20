# Tran-IDS v2

## Model training and adaptation

1. [transformer_AEViT_1D.py](adaption/transformer_AEViT_1D.py): Pretrains the model.
2. [transformer_AEViT_ft.py](adaption/transformer_AEViT_ft.py): Fine-tunes the pretrained model.
3. [transformer_AEViT_1D_parquet_tabular.py](adaption/transformer_AEViT_1D_parquet_tabular.py): Pretrains the tabular model.
4. [transformer_AEViT_ft_tabular.py](adaption/transformer_AEViT_ft_tabular.py): Fine-tunes the pretrained tabular model.
5. [transformer_AEViT_ft_attack_opposite.py](adaption/transformer_AEViT_ft_attack_opposite.py): Runs the source-to-target fine-tuning experiment with source attack samples.

### Evaluation protocol

The adaptation scripts calibrate the anomaly threshold from benign validation reconstruction errors. The default target validation false-positive rate is 5%. Samples with scores at or above the threshold are classified as attacks. Test metrics use the frozen threshold.

Pretraining ranks Optuna trials by validation MSE and evaluates only the selected checkpoint on the test split. New pretraining artifacts are written under `optuna_validation_mse/`. Set `TRANIDS_PRETRAIN_ROOT` to that directory when using a new checkpoint with either standard fine-tuning script.

## Experiment configurations

The settings below are documented in the manuscript, the experiment results slides, and the release scripts. Run-specific values are listed under [Run records needed for exact reproduction](#run-records-needed-for-exact-reproduction).

### Manuscript settings

| Component | Configuration |
| --- | --- |
| Datasets and transfer | CIC-IDS-2018 and UNSW-NB15, evaluated in both source-to-target directions. |
| Flow-prefix input | Group both directions of a flow and order packets by capture time. Exclude TCP handshake packets. Keep up to 5 packets and read at most 120 bytes per packet. Remove 24 bytes of MAC addresses, IPv4 addresses, and ports. Zero-pad short prefixes and scale bytes by 255 to obtain a `5 × 96` input. |
| Flow-prefix AE-ViT | Patch size 12, 40 patch tokens, 6 encoder blocks, 2 decoder blocks, embedding dimension 384, bottleneck dimension 32. |
| Training | AdamW with learning rate `2e-4`, zero weight decay, cosine annealing, up to 50 epochs, and patience 10 on validation reconstruction loss. Source training and source validation use benign flows. |
| Target protocol | Target adaptation/validation/test pools use a `70/10/20` split. Sample the adaptation buffer from the target adaptation pool without labels, following the manuscript protocol. Compare the source checkpoint and adapted model on the same held-out target test split. |
| Reporting | AUROC is the primary metric. Report the median over five runs and the observed minimum and maximum for error bars. Report F1, precision, recall, and FPR at the chosen operating point. |

### Experiment matrix

| Paper reference | Configuration and run notes |
| --- | --- |
| Table 3 and Figs. 4–6 | Compare both transfer directions before and after adaptation with the default depth `B=3`. The slides' optimized 500,000-flow runs match the main AUROC medians of 0.9348 (2018 → NB15) and 0.9194 (NB15 → 2018). The original run records are needed to confirm the exact checkpoint and split. |
| Fig. 7 | Compare sanitized flow prefixes with NF-CIC-IDS-2018 and NF-UNSW-NB15 statistical flow records in both directions. |
| Fig. 8 | Sweep target buffer sizes of 150,000, 270,000, 350,000, 500,000, 700,000, and 1,000,000 flows. The slides contain standard and optimized 500,000-flow runs. The final plotting input is needed to resolve the variant and timings. |
| Figs. 9–10 | Sweep adaptation depth `B=1–5` with a buffer of 1,000,000 flows. The reverse-transfer depth sweep needs verification against the final plotting data. |
| Fig. 11 | Evaluate both directions with 500, 1,000, 2,500, 5,000, and 10,000 benchmark attack samples in the adaptation buffer. Reverse-transfer results in the slides differ from the manuscript at 500 and 1,000 samples. |

The bottleneck, patch head, and decoder normalization are trainable at every depth. The block indices below are zero-based.

| Depth | Updated transformer blocks | Trainable parameters |
| --- | --- | ---: |
| 1 | `decoder[1]` | 1,528,028 |
| 2 | `decoder[0]`, `decoder[1]` | 3,007,196 |
| 3 (default) | `encoder[5]`, `decoder[0]`, `decoder[1]` | 4,486,364 |
| 4 | `encoder[4:6]`, both decoder blocks | 5,965,532 |
| 5 | `encoder[3:6]`, both decoder blocks | 7,444,700 |

### Release script defaults

- The fine-tuning scripts list training seeds `[7, 17, 27, 37, 47]` and split seed `42`. The published results still need a run-to-seed mapping.
- The standard fine-tuning scripts draw validation samples from the benign training subset. This differs from the manuscript's target pool allocation and should be checked against the original split manifests.
- The flow-prefix pretraining entry point defaults to 100 epochs and the tabular fine-tuning entry point to 3 epochs. The manuscript describes a 50-epoch limit.
- The contamination script lists source trial 6 for CIC-IDS-2018 and trial 2 for UNSW-NB15. Its active defaults use percentage-based contamination for 2018 → NB15. The script currently imports a cache helper absent from the checkout, which blocks execution. See [contamination script](adaption/transformer_AEViT_ft_attack_opposite.py).
- The release evaluation uses a 5% target false-positive rate on benign validation scores. Recompute threshold-dependent paper results under this calibration protocol. If validation-MSE selection changes the checkpoint, recompute AUROC as well.

### Run records needed for exact reproduction

- Dataset versions, capture manifests, extractor commands, split indices, per-seed adaptation sample IDs, and whether labels were used to construct the adaptation buffer.
- Source checkpoint filenames or checksums, Optuna trial records, batch sizes, and exact training arguments for each reported run.
- NetFlow feature order, padding, and checkpoint mapping for Fig. 7. Final input CSVs and plotting scripts for Figs. 8, 9, and 11. For contamination runs, record the sample source, whether samples were added or replaced, and the resulting buffer counts.
- Validation and test scores, frozen thresholds, and per-seed metrics for Table 3 and Fig. 4. For timing comparisons, record the measured stages and software versions. The manuscript lists PyTorch on an AMD EPYC 9684X and AMD Instinct MI300X.
- Baseline versions and configurations for AE, CNN, XGBoost, DANN, DI-NIDS, and TRANSFA-IDS v1.

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
