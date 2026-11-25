# Reproduce the CCD80 lunar composition study

This is the companion to Inho Yoo’s *Automated Model Selection and Mapping of Six Major Oxide Abundances on the Lunar Surface*. It contains the Lunar AutoLab numerical code, the study’s sample-to-window assignments, the model search, the fitted models, and the data behind the validation tables and maps. Start with the [version 1.0.0 release](https://github.com/inhotwin-png/ccd80-thesis-reproduction/releases/tag/v1.0.0). Download `ccd80-thesis-reproduction-v1.0.0.zip`, extract it, and open a terminal in the directory containing `reproduce.py`.

## Reproduce the tables and figures

Use Python 3.12. The release was tested on 64-bit Windows with Python 3.12.14. Create a virtual environment with `python -m venv .venv`. Activate it with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on Linux and macOS. Then run:

```console
python -m pip install -r requirements.txt
python reproduce.py --maps
```

The program checks the bundled files against their SHA-256 hashes and writes the following files under `results/`. Omit `--maps` to generate just the tables. No orbital downloads or GPU are needed for this step.

| Thesis result | Output | Calculation |
| --- | --- | --- |
| Model selection | `selected_models.csv` | Selects six mapping and six ranking models from the archived search scores and checks all twelve against the frozen model identities. |
| Table 6 | `table6.csv` | Calculates site-held-out MAE, R² and pairwise accuracy from the fold predictions. Cross-site pairs use predictions from the fold that excluded both sites. |
| Table 7 | `table7.csv` | Calculates the area-weighted trimmed mean, valid area and priority score using the original 62 m predictions within 2 km of each first-ranked centre. Al₂O₃ and CaO share a centre. |
| Table 8 | `table8.csv`, `heldout_windows.csv` | Recomputes the errors of CCD80, Yang’s OneDCNN and Shi’s ResDCNN from individual held-out windows. The fixed FeO and TiO₂ equations use the MI reflectances directly. |
| Appendix C | `appendix_C.csv` | Recalculates the scores and order of 20 regions per oxide from the archived regional statistics. |
| Figures 2 and 4–7 | `figure*.png` | Redraws the global maps, site-held-out validation plots, candidate locations, native-grid cutouts and orbital input comparisons. |

A quick numerical check is the CCD80 FeO MAE of **0.555 wt%p** in Table 8. Abundances are in weight percent (wt%). MAE measures a difference in weight percentage points (wt%p). Coordinates use signed degrees north and east. CSVs retain precision for checking calculations. Round the reported results to three decimal places for comparison with the thesis.

## Retrain models or apply their fitted weights

To retrain the FeO mapping model on the eight leave-one-landing-site-out folds, run the first two commands below. The third command applies all twelve full-data fitted models to the 26 windows and writes 312 predictions to `results/full_fit_windows.csv`.

```console
python -m pip install -r requirements-models.txt
python refit.py
python predict_windows.py
```

Use `python refit.py --family ccd80 --oxide all` for all mapping models. For ranking models and all eight one-site plus 28 two-site folds, use `python refit.py --family ccd80 --oxide all --goal RANK --two-sites`. Each fold saves independently. Restart the same command to resume. Use a new `--out` directory for a different environment.

The neural comparisons use this study’s training windows and the same held-out landing sites. Yang’s OneDCNN uses the architecture from DL-IM with the settings in [yang.json](config/yang.json). After installing the model requirements, use:

```console
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
python refit.py --family yang --oxide all
```

For Shi’s ResDCNN, create a **separate Python 3.12 environment**, activate it, and run:

```console
python -m pip install -r requirements-shi.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
python refit.py --family shi --oxide all
```

Shi’s published network definition is included at its recorded revision. Each fold fits its own input and target scaling. The study’s eight-site comparison uses the settings in [shi.json](config/shi.json). Both CNN commands run on a single CPU thread. Recalculation from the archived predictions recovers all Table 8 entries at their printed precision. In a second CPU training run, the largest absolute difference from the original Yang oxide-level MAEs was below 0.032 wt%p. The CSV keeps the original Yang and replay columns separate. See the [per-oxide refit check](data/retraining_check.json).

## Locate the source data and mapping settings

- [Samples](data/samples.csv), [training inputs](assets/dataset.json) and [validation splits](data/folds.json) connect 28 returned samples to 26 observation windows at eight landing sites. The held-out unit is the landing site. The 62 m grid is the spatial unit of prediction. Each evaluated oxide has 25 or 26 windows with a laboratory value.
- [Search snapshot](config/search_snapshot.json), [model recipes](config/search_recipes.json.gz) and [oxide-specific scores](data/search_scores.csv.gz) document the search at model selection. Lunar AutoLab Next 1.0.54 recorded 1,614 completed configurations. [Selected models](config/selected_models.json) identifies their recipes and weights. The portable refit uses the numerical modules from version 1.0.73. [Provenance](config/provenance.json) records source and data hashes.
- [Mapping settings](config/map.json), [candidate thresholds and weights](config/selection_policy.json) and [mapping source](mapping_source) describe the original 496 m screening and regional refinement. The downloadable global display is a 720 × 360 grid of aggregated predictions. The five distinct first-ranked cutouts retain native 62 m cells. The complete planet-wide prediction database is outside the scope of this release. Figure 7 uses a 4 × 4-cell mean for the global CLASS ratio panel. Maps are rendered from the released values and may have different colour limits or layout from the printed figures.
- [Orbital source manifest](data/orbital_sources.json) lists retrievable products and their checksums. Run `python fetch_input.py` to list the 39 Kaguya MI sample-area tiles and ten CLASS products. To download and check one file, run `python fetch_input.py 296s/CaKa_Si`. See [data sources and literature](data/sources.csv) for the sample, MI, CLASS, WAC and comparator references.

The permanent archive for version 1.0.0 is [doi:10.5281/zenodo.24537736](https://doi.org/10.5281/zenodo.24537736). Cite that version using the repository’s “Cite this repository” menu or [CITATION.cff](CITATION.cff). Attribution details are in [NOTICES.md](NOTICES.md).
