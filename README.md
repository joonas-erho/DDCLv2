# DDCL

Automatic chart generation for .sm files. See [DDCL](https://arxiv.org/abs/2507.01644).

---

# Acknowledgements

We reuse a significant portion of code from [DDC](https://github.com/chrisdonahue/ddc/tree/master), in particular for the loading of .sm files. We include BPM detection algorithms described by Bram van de Wetering and implemented in [ArrowVortex](https://arrowvortex.ddrnl.com/), as well as an adapted version from [SMEditor](https://github.com/tillvit/smeditor). 

---

# Attribution

If you use this code for research, please cite the following:

```
@misc{omalley2025dancedanceconvlstm,
      title={Dance Dance ConvLSTM}, 
      author={Miguel O'Malley},
      year={2025},
      eprint={2507.01644},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.01644}, 
}
```

---

##  Quick Start

### Requirements

- Tensorflow > v2
- Essentia
- TQDM
- If you want to take advantage of GPU acceleration, make sure you have appropriate CUDNN and CUDA versions installed.

###  Installation

```bash
# Clone the repository
git clone https://github.com/miguelomalley/DDCL.git

# Navigate into the folder
cd DDCL
```

### Generating charts

```bash
# First, place your songs in the input folder. Then run the chart generator.
python generate_charts.py

# If you want stamina charts, specify the stamina models.
python generate_charts.py --onset_model_fp='trained_models/onset_model_stamina.keras' --sym_model_fp='trained_models/sym_model_stamina.keras'

# You can change the BPM detection method with the bpm_method option.
python generate_charts.py --bpm_method='SMEdit'
```

### Training models

```bash
# Place your song packs in the raw/songs folder, just like for stepmania or ITGmania. Then, run the following.
python smfiler.py

# Train an onset model
python train_onset_model.py

# Train a symbolic model (you need both.)
python train_sym_model.py
```



