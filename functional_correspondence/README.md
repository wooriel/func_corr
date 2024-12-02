### Training from scratch

To train the models, use

```python
python functional_correspondence.py --train_dataset=faust --input_features=xyz
```
or, with heat kernel signature features
```python
python functional_correspondence.py --train_dataset=faust --input_features=hks
```

Passing `--train_dataset=scape` trains on the SCAPE dataset instead.

After training, the fitted model will be saved in `saved_models/[dataset]_[features].pth`.

During training, geodesic error metrics are computed on the test set after each iteration; see the note below.

### Pretrained models and evaluating geodesic accuracy

> **NOTE:** Geodesic error metrics are computed via the all-pairs geodesic distance matrix, which is giant and expensive to compute! These distances will be computed and cached the first time evaluation is run, which may take several minutes per model.

Compute geodesic accuracy of a trained model by running with the `--evaluate` flag. Load any pretrained model from disk with `--load_model=PATH_TO_MODEL.pth`. In this case, the `train_dataset` argument is ignored.

```python
python functional_correspondence.py --test_dataset=faust --input_features=xyz --load_model=pretrained_models/faust_xyz.pth
```

We include several pretrained models in the `pretrained_models/` directory for both FAUST and SCAPE, as well as `xyz` and `hks` features.
