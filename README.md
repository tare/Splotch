# Splotch

Splotch is a hierarchical generative probabilistic model for analyzing Spatial Transcriptomics (ST) [[1]](#references) data.

## Features

- Supports complex hierarchical experimental designs and model-based analysis of replicates
- Full Bayesian inference with Hamiltonian Monte Carlo (HMC) using the adaptive HMC sampler as implemented in NumPyro [[2]](#references)
  - CPU, GPU, and TPU support
- Analysis of expression differences between anatomical regions and conditions using posterior samples
- Different anatomical annotated regions (AARs) are modeled using a linear model
- Zero-inflated Poisson or Poisson likelihood
- Gaussian Process prior for spatial random effect

The Splotch code in this repository supports single-, two-, and three-level experimental designs.

## Installation

### PyPI

```console
$ pip install Splotch
```

### GitHub

```console
$ pip install git+https://git@github.com/tare/Splotch.git
```

#### CUDA

To install JAX with NVIDIA support, please see [this](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-via-pip-easier) page for instructions.

## Usage

The main steps of Splotch analysis are the following:

1. [Preparation of count files](#preparation-of-count-files)
2. [Annotation of ST spots](#annotation-of-st-spots)
3. [Preparation of metadata table](#preparation-of-metadata-table)
4. [Splotch analysis](#splotch-analysis)

### Preparation of count files

The count files have the following tab-separated values (TSV) file format

|               | 32.06_2.04 | 31.16_2.04 | 14.07_2.1 | …   | 28.16_33.01 |
| ------------- | ---------- | ---------- | --------- | --- | ----------- |
| A130010J15Rik | 0          | 0          | 0         | …   | 0           |
| A230046K03Rik | 0          | 0          | 0         | …   | 0           |
| A230050P20Rik | 0          | 0          | 0         | …   | 0           |
| A2m           | 0          | 1          | 0         | …   | 0           |
| ⋮             | ⋮          | ⋮          | ⋮         | ⋱   | ⋮           |
| Zzz3          | 0          | 1          | 0         | …   | 0           |

The rows and columns have gene identifiers and ST spot coordinates (X and Y coordinates are separated by an underscore), respectively.

### Annotation of ST spots

To get the most out of the statistical model of Splotch one has to annotate the ST spots based on their tissue context. These annotations will allow the model to share information across tissue sections, resulting in more robust conclusions.

To make the annotation step slightly less tedious, we have implemented a light-weight javascript tool called [Span](https://github.com/tare/Span).

The annotation files have the following TSV file format

|                | 32.06_2.04 | 31.16_2.04 | 14.07_2.1 | …   | 28.16_33.01 |
| -------------- | ---------- | ---------- | --------- | --- | ----------- |
| Vent_Med_White | 0          | 0          | 0         | …   | 0           |
| Vent_Horn      | 1          | 1          | 0         | …   | 0           |
| Vent_Lat_White | 0          | 0          | 0         | …   | 0           |
| Med_Grey       | 0          | 0          | 0         | …   | 0           |
| Dors_Horn      | 0          | 0          | 0         | …   | 0           |
| Dors_Edge      | 0          | 0          | 0         | …   | 1           |
| Med_Lat_White  | 0          | 0          | 0         | …   | 0           |
| Vent_Edge      | 0          | 0          | 1         | …   | 0           |
| Dors_Med_White | 0          | 0          | 0         | …   | 0           |
| Cent_Can       | 0          | 0          | 0         | …   | 0           |
| Lat_Edge       | 0          | 0          | 0         | …   | 0           |

The rows and columns correspond to the user-define anatomical annotation regions (AAR) and ST spot coordinates (X and Y coordinates are separated by an underscore), respectively. For instance, the spot 32.06_2.04 has the Vent_Horn annotation (i.e. located in ventral horn). The annotation category of each ST spot is **one-hot encoded** and we do not currently support more than one annotation category per ST spot.

ST spots without annotation categories are discarded in the analysis. This behaviour can be useful when you want to discard some ST spots from the analysis based on the tissue histology.

### Preparation of metadata table

The metadata table contains information about the samples (i.e. count files). Additionally, the metadata table is used for matching count and annotation files.

The metadata table has the following TSV file format

| name      | level_1   | level_2 | level_3 | count_file                                                       | annotation_file           | image_file              |
| --------- | --------- | ------- | ------- | ---------------------------------------------------------------- | ------------------------- | ----------------------- |
| L7CN36_C1 | G93A p120 | F       | 1394    | count_tables/L7CN36_C1_stdata_aligned_counts_IDs.txt.unified.tsv | annotations/L7CN36_C1.tsv | images/L7CN36_C1_HE.jpg |
| L7CN36_C2 | G93A p120 | F       | 1394    | count_tables/L7CN36_C2_stdata_aligned_counts_IDs.txt.unified.tsv | annotations/L7CN36_C2.tsv | images/L7CN36_C2_HE.jpg |
| L7CN30_C1 | WT p120   | M       | 2967    | count_tables/L7CN30_C1_stdata_aligned_counts_IDs.txt.unified.tsv | annotations/L7CN30_C1.tsv | images/L7CN30_C1_HE.jpg |
| L7CN30_C2 | WT p120   | M       | 2967    | count_tables/L7CN30_C2_stdata_aligned_counts_IDs.txt.unified.tsv | annotations/L7CN30_C2.tsv | images/L7CN30_C2_HE.jpg |
| L7CN69_D1 | WT p120   | M       | 1310    | count_tables/L7CN69_D1_stdata_aligned_counts_IDs.txt.unified.tsv | annotations/L7CN69_D1.tsv | images/L7CN69_D1_HE.jpg |
| L7CN69_D2 | WT p120   | M       | 1310    | count_tables/L7CN69_D2_stdata_aligned_counts_IDs.txt.unified.tsv | annotations/L7CN69_D2.tsv | images/L7CN69_D2_HE.jpg |
| CN96_E1   | WT p120   | F       | 1040    | count_tables/CN96_E1_stdata_aligned_counts_IDs.txt.unified.tsv   | annotations/CN96_E1.tsv   | images/CN96_E1_HE.jpg   |
| CN96_E2   | WT p120   | F       | 1040    | count_tables/CN96_E2_stdata_aligned_counts_IDs.txt.unified.tsv   | annotations/CN96_E2.tsv   | images/CN96_E2_HE.jpg   |
| CN93_E1   | G93A p120 | M       | 975     | count_tables/CN93_E1_stdata_aligned_counts_IDs.txt.unified.tsv   | annotations/CN93_E1.tsv   | images/CN93_E1_HE.jpg   |
| CN93_E2   | G93A p120 | M       | 975     | count_tables/CN93_E2_stdata_aligned_counts_IDs.txt.unified.tsv   | annotations/CN93_E2.tsv   | images/CN93_E2_HE.jpg   |

Each sample (i.e. slide) has its own row in the metadata table. The columns `level_1`, `level_2`, and `level_3` define how the samples are analyzed using the linear hierarchical AAR model. The columns `level_1`, `count_file`, and `annotation_file` are mandatory. The column `level_2` is mandatory when using the two-level model. Similarly, the columns `level_2` and `level_3` are mandatory when using the three-level model. At the moment we only support categorical variables.

If a given slide contains tissue sections from multiple biological conditions in terms of the explanatory variables, then it is recommended to split the tissue sections into multiple count files so that the design matrix can be defined accordingly.

The user can include additional columns at their own discretion. For instance, we will use the column `image_file` in the [tutorials](tutorials/).

### Example data

In the `tutorials` directory, we have two example ST data sets

1. [ALS](tutorials/als_st/) [[3]](#references)
2. [Olfactory Bulb](tutorials/olfactory_bulb_st/) [[1]](#references)

### Splotch analysis

Please see the [ALS](tutorials/als_st/Tutorial.ipynb) and [Olfactory Bulb](tutorials/olfactory_bulb_st/Tutorial.ipynb) tutorials.

In the simplest setting, the following lines would be enough to run Splotch on a single gene

```python
# read input data
splotch_input_data = get_input_data("metadata.tsv")

# run Splotch on the Gfap gene
key = random.PRNGKey(0)
key, key_ = random.split(key)
splotch_result_nuts = run_nuts(key_, ["Gfap"], splotch_input_data)
```

### References

[1] Ståhl, Patrik L., et al. ["Visualization and analysis of gene expression in tissue sections by spatial transcriptomics."](https://science.sciencemag.org/content/353/6294/78) _Science_ 353.6294 (2016): 78-82.

[2] Phan, Du, et al. ["Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro."](https://www.jstatsoft.org/article/view/v076i01) _arXiv preprint_ 1912.11554 (2019).

[3] Maniatis, Silas, et al. ["Spatiotemporal dynamics of molecular pathology in amyotrophic lateral sclerosis."](https://science.sciencemag.org/content/364/6435/89) _Science_ 364.6435 (2019): 89-93.
