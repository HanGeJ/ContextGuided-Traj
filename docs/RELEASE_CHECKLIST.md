# Release checklist (do not mark complete without evidence)

- [x] Identify and hash stable Stage3E / Stage SDDS source modules
- [x] Exclude raw dataset archives, video frames and private checkpoint files
- [x] Include paper evaluation code, explicit ETH–UCY train-group split rule, Table IV reproduction runner
- [x] Supply synthetic CPU model-forward smoke (not a paper accuracy reproduction)
- [x] Inventory all 51 frozen AutoDL paper checkpoints, sizes and SHA256 (manifest-only)
- [ ] Export all 15 `multihyp_social` ETH–UCY checkpoints and their source SHA
- [ ] Export 3 SDD generator + geometry checkpoint pairs and locked test evidence
- [ ] Confirm data licenses and dataset preparation checksums against the original training runs
- [ ] Verify ETH–UCY full 5 folds x 3 seeds on a fresh compatible environment
- [ ] Verify locked SDD pixel-space result using the frozen model on the intended split
- [ ] Verify Fig. 6 video/frame/body-anchor registration and figure usage rights
- [ ] Verify paper figure 7 measurement rounding and original source results
- [ ] Validate frozen Table IV configuration and published aggregate against raw 60 run logs
- [ ] Approve copyright/affiliation release of source code, model weights and visual frames
- [x] Add MIT LICENSE and consistent README / CITATION.cff license metadata
- [x] Repair .gitignore to ensure source `data/*.py` is staged; add Git-index checker
- [x] Publish the source repository and inspect its tracked files and history
- [ ] Tag a fixed paper-associated source version and archive release checksum
