# GCT-Plus


## About the Project
In this project, we used conditional variational autoencoder (CVAE) based Transformer on multiple tasks, including unconditioned, property-based, structure-based, and property-structure-based generation, as well as molecular interpolation. The model architecture was sourced from GCT, trained on a dataset of approximately 1.58 million neutral molecules provided by the MOSES benchmarking platform.
More updated information can be seen from [GCT-Plus](https://github.com/chaoting-sun/GCT-Plus).

## Getting Started

(1) Clone the repository:
```bash
git clone https://github.com/chaoting-sun/GCT-Plus.git
```

(2) Create an environment:
```bash
cd GCT-Plus
conda env create -n gct-plus -f ./env.yml # create a new environment named gct-plus
conda activate gct-plus
```

(3) Download the Models:
```bash
# 1. unconditioned GCT
gdown https://drive.google.com/uc?id=1k8HxI-h3Z9ZfJM4HZMFfZEw8Rh8bMElf -O ./Weights/vaetf/vaetf1.pt

# 2. property-based GCT
gdown https://drive.google.com/uc?id=1D5g3TF3-eFB34SXpylERSa-6L1u_SR5d -O ./Weights/pvaetf/pvaetf1.pt

# 3. structure-based GCT
gdown https://drive.google.com/uc?id=1emVfSViCVWugPda1utYaIBenbRucH_j1 -O ./Weights/scavaetf/scavaetf1.pt

# 4. property-structure-based GCT

# selected properties: logP, tPSA, QED
gdown https://drive.google.com/uc?id=10ojI90-Wrc0RTWUgOfAea6VjRk_GIPVH -O ./Weights/pscavaetf/pscavaetf1.pt

# selected properties: logP, tPSA, SAS
gdown https://drive.google.com/1gA-woAsdYpUsDo_jQAO1n3Nf7WJS6g-D -O ./Weights/pscavaetf/pscavaetf1_molgpt.pt

# 5. property-based Transformer
gdown https://drive.google.com/uc?id=1ICK-p9p3WA4eOZfw0zPkPCP2LRks9hEg -O ./Weights/pscavaetf/pscavaetf1.pt
```

(4) Run Multiple Tasks
```bash
# unconditioned generation
Bashscript/infer/uc_sampling.sh

# property-based generation
Bashscript/infer/p_sampling.sh

# structure-based generation
Bashscript/infer/sca_sampling.sh

# property-structure-based generation
Bashscript/infer/psca_sampling.sh

# molecular interpolation
Bashscript/infer/mol_interpolation.sh

# visualize attention
Bashscript/infer/visualize_attention.sh
```

## Implementation
(1) Preprocess the data
```bash
Bashscript/preprocess/preprocess.sh
```

(2) Re-train Models

```bash
# train a model for unconditioned generation
Bashscript/train/train_vaetf.sh

# train a model for property-based generation
Bashscript/train/train_pvaetf.sh

# train a model for structure-based generation
Bashscript/train/train_scavaetf.sh

# train a model for property-structure-based generation
Bashscript/train/train_pscavaetf.sh
```

(3) Model Selection

The model for unconditioned generation (vaetf) can be selected the best epochs.
```bash
Bashscript/infer/model_selection.sh
```