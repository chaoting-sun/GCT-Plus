# Multi-Task Molecular Design by CVAE based Tranformer


## About the Project


## Implementation


### Preprocessing
```bash
Bashscript/preprocess/preprocess.sh
```

### Training

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

### Model Selection

```bash
Bashscript/infer/model_selection.sh
```

### Inference

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


