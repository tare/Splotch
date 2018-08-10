Splotch
===================

Overview
-------------
This implementation has a three-level hierarchical model representing the hierarchical experimental design

![Three-level hierarchical model](https://raw.githubusercontent.com/tare/Splotch/master/images/three-level_hierarchical_model.png)

Example data
-------------
Here we provide an example input data (``data_Gfap.R``) to be used with the code.
Specifically, the input data file is consisted of 76,136 spatially-resolved measurements for the *Gfap* gene.
The ST measurements are from 1,165 tissue sections collected from 67 mice.
There are three genotypes and variable number of time points per genotype

| Genotype   | Time points              |
|------------|--------------------------|
| SOD1-WT    | P30, P70, P100, and P120 |
| SOD1-G93A  | P30, P70, P100, and P120 |
| *Atg7* cKO | P100 and P120            |

Below we will describe how the input has been generated and how the output should be interpreted. 

Installation
-------------
To get the latest version of Splotch from our GitHub repository
```console
cd $HOME
git clone https://github.com/tare/Splotch.git
```

Here we assume that CmdStan [2] has been installed successfully
```console
VERSION="2.17.1"
cd $HOME
wget https://github.com/stan-dev/cmdstan/releases/download/v"$VERSION"/cmdstan-"$VERSION".tar.gz
tar -xzvf cmdstan-"$VERSION".tar.gz
cd cmdstan-"$VERSION"
make build -j4
make $HOME/Splotch/mouse/splotch_mouse
```

Usage
-------------
To sample four chains using the example data set
```bash
# set a few variables
GENE="Gfap"
NUM_SAMPLES=500
NUM_WARMUP=500
N_CHAINS=4
BINARY="splotch_mouse"

cd $HOME/Splotch/mouse

# run four chains
for CHAIN in `seq 1 $N_CHAINS`
do
  ./"$BINARY" sample num_samples="$NUM_SAMPLES" num_warmup="$NUM_WARMUP" random id="$CHAIN" \
              data file=data_"$GENE".R output file=output_"$GENE"_"$CHAIN".csv refresh=50 &
done
```

Alternatively, user may also use PyStan [3] or RStan [4] interfaces, which might make the postprocessing of the samples easier. 
However, we have observed that other interfaces than CmdStan might have issues dealing with the scale of the output data.

After the sampling is finished, you can merge the chains into a single CSV file

```bash
GENE="Gfap"

cd $HOME/Splotch/mouse

# get variable names from the output file of the first chain
grep lp__ output_"$GENE"_1.csv > combined_"$GENE".csv

# concatenate the samples from all the chains
sed '/^[#l]/d' output_"$GENE"_*.csv >> combined_"$GENE".csv
```

Then user may read and parse the output file using their favourite programming language.


Input variables
-------------
The full list of input variables is

| Input variable | Type          | Dimensions   | Description                                                                                         |
|----------------|---------------|--------------|-----------------------------------------------------------------------------------------------------|
| ``N_tissues``      | int           | 1            | Number of tissue sections                                                                           |
| ``N_spots``        | int           | ``N_tissues``    | Number of spots per tissue section                                                                  |
| ``N_covariates``   | int           | 1            | Number of AARs                                                                                      |
| ``N_genotypes``    | int           | 1            | Number of genotypes                                                                                 |
| ``N_timepoints``   | int           | ``N_genotypes``  | Number of time points per genotype                                                                  |
| ``N_sexes``         | int           | 1            | Number of sexes                                                                                      |
| ``N_mice``         | int           | 1            | Number of mice                                                                                      |
| ``mouse_mapping``  | int           | ``N_mice``       | Origin of each mouse (see below)                                                                    |
| ``tissue_mapping`` | int           | ``N_tissues``       | Origin of each tissue section (see below)                                                           |
| ``counts``         | int           | Σ``N_spots`` | Counts per spot                                                                                     |
| ``size_factors``   | positive real | Σ``N_spots`` | Size factor for each spot                                                                           |
| ``D``              | int           | Σ``N_spots`` | AAR tag for each spot                                                                               |
| ``W_n``            | int           | 1            | Number of neighbor spot pairs                                                                       |
| ``W_sparse``       | int           | ``W_n`` x 2      | Adjacency pairs                                                                                     |
| ``D_sparse``       | real          | Σ``N_spots`` | Number of neighbors for each spot                                                                   |
| ``eig_values``     | real          | Σ``N_spots`` | Eigenvalues of matrix D<sup>-0.5</sup>W<sup>-0.5</sup>                                              |

Counts from all the considered tissue sections are stored in ``counts`` by concatenating the counts from each tissue section together.
Naturally, the order of elements in ``counts``, ``size_factors``, and ``D`` has to match.

Each considered spot has been linked to exactly to one anatomical annotation region. 
This information is encoded in ``D`` using the following coding
| Anatomical annotion region | Identifier |
|----------------------------|------------|
| Ventral medial white       | 1          |
| Ventral horn               | 2          |
| Ventral lateral white      | 3          |
| Medial grey                | 4          |
| Dorsal horn                | 5          |
| Dorsal edge                | 6          |
| Medial lateral white       | 7          |
| Ventral edge               | 8          |
| Dorsal medial white        | 9          |
| Central canal              | 10         |
| Lateral edge               | 11         |

For instance, if the 154<sup>th</sup> spot has been annotated to be in ventral horn, then the 154<sup>th</sup> element of ``D`` is 2.

Next we will cover how ``mouse_mapping`` and ``tissue_mapping`` have been constructed.
Each mouse has a combination of genotype, time point, and sex associated with it.
To incorporate that information into the model, we use ``mouse_mapping`` and the following mapping
| Genotype   | Time point | Sex    | Identifier |
|------------|------------|--------|------------|
| SOD1-WT    | P30        | Male   | 1          |
| SOD1-WT    | P70        | Male   | 2          |
| SOD1-WT    | P100       | Male   | 3          |
| SOD1-WT    | P120       | Male   | 4          |
| SOD1-WT    | P30        | Female | 5          |
| SOD1-WT    | P70        | Female | 6          |
| SOD1-WT    | P100       | Female | 7          |
| SOD1-WT    | P120       | Female | 8          |
| SOD1-G93A  | P30        | Male   | 9          |
| SOD1-G93A  | P70        | Male   | 10         |
| SOD1-G93A  | P100       | Male   | 11         |
| SOD1-G93A  | P120       | Male   | 12         |
| SOD1-G93A  | P30        | Female | 13         |
| SOD1-G93A  | P70        | Female | 14         |
| SOD1-G93A  | P100       | Female | 15         |
| SOD1-G93A  | P120       | Female | 16         |
| *Atg7* cKO | P100       | Male   | 17         |
| *Atg7* cKO | P120       | Male   | 18         |
| *Atg7* cKO | P100       | Female | 19         |
| *Atg7* cKO | P120       | Female | 20         |

For instance, if the first mouse (i.e. corresponds to the first element of ``mouse_mapping``) was male and had the SOD1-WT genotype and the tissue sections were collected at P100, then the value of the first element of ``mouse_mapping`` is 3.
Note that the order of the mice in ``mouse_mapping`` is arbitratory but it will impact the construction of ``tissue_mapping`` as described next. 

Similarly, each tissue section has been taken from one of the ``N_mice`` mice; moreover, we have taken multiple tissue sections for each mouse . 
This information is incorporated into the model through ``tissue_mapping`` by linking the tissue sections to mice using the ordering used in ``mouse_mapping``.
For instance, if the second tissue section has been sampled from the third mice (the mouse represented by the third element of ``mouse_mapping``), then the second element of ``tissue_mapping`` is 3.
Note that the order of tissue sections in ``tissue_mapping`` has to match with the concatenation order of tissue sections of ``counts``.

We use the exact sparse CAR representation described by Max Joseph in [Exact Sparse CAR Models in Stan](http://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html).
In summary, the input variables ``W_n``, ``W_sparse``, ``D_sparse``, and ``eig_values`` define the CAR prior and these input variables can be derived from the adjacency matrix.
Note that in our implementation, in addition to ``W`` and ``W_n``, we supply ``W_sparse``, ``D_sparse``, and ``eig_values`` as input data.

Output variables
-------------
Most of the users are interested in the following output variables
| Output variable           | Type                 | Dimensions                           | Description                              |
|---------------------------|----------------------|--------------------------------------|------------------------------------------|
| ``lambda``                    | positive real        | Σ``N_spots``                         | Rate parameter for each spot             |
| ``beta_genotype_timepoint`` | real                 | Σ``N_timepoints)`` x ``N_covariates``   | Genotype / time point level coefficients |
| ``beta_sex``                  | real                 | 2\*Σ``N_timepoints`` x ``N_covariates`` | Sex level coefficients                   |
| ``beta_mouse``                 | real                 | ``N_mice`` x ``N_covariates``              | Mouse level coefficients                 |
| ``theta``                     | real between 0 and 1 | 1                                    | Probability of extra zeros               |
| ``a``                         | real between 0 and 1 | 1                                    | Spatial autocorrelation                  |
| ``tau``                       | positive real        | 1                                    | Conditional precision                    |

The order of elements in ``lambda`` corresponds to the order of the spots in ``counts``.

The mapping from the rows of ``beta_genotype_timepoint`` to the genotype and time point is given in the following table
| Genotype   | Time point | Row index |
|------------|------------|------------|
| SOD1-WT    | P30        | 1          |
| SOD1-WT    | P70        | 2          |
| SOD1-WT    | P100       | 3          |
| SOD1-WT    | P120       | 4          |
| SOD1-G93A  | P30        | 5          |
| SOD1-G93A  | P70        | 6          |
| SOD1-G93A  | P100       | 7          |
| SOD1-G93A  | P120       | 8          |
| *Atg7* cKO | P100       | 9          |
| *Atg7* cKO | P120       | 10         |

Whereas, the mapping from the rows of ``beta_sex`` to the genotype, time point, and sex is given in the following table
| Genotype   | Time point | Sex    | Row index |
|------------|------------|--------|------------|
| SOD1-WT    | P30        | Male   | 1          |
| SOD1-WT    | P70        | Male   | 2          |
| SOD1-WT    | P100       | Male   | 3          |
| SOD1-WT    | P120       | Male   | 4          |
| SOD1-WT    | P30        | Female | 5          |
| SOD1-WT    | P70        | Female | 6          |
| SOD1-WT    | P100       | Female | 7          |
| SOD1-WT    | P120       | Female | 8          |
| SOD1-G93A  | P30        | Male   | 9          |
| SOD1-G93A  | P70        | Male   | 10         |
| SOD1-G93A  | P100       | Male   | 11         |
| SOD1-G93A  | P120       | Male   | 12         |
| SOD1-G93A  | P30        | Female | 13         |
| SOD1-G93A  | P70        | Female | 14         |
| SOD1-G93A  | P100       | Female | 15         |
| SOD1-G93A  | P120       | Female | 16         |
| *Atg7* cKO | P100       | Male   | 17         |
| *Atg7* cKO | P120       | Male   | 18         |
| *Atg7* cKO | P100       | Female | 19         |
| *Atg7* cKO | P120       | Female | 20         |

The order of the rows of ``beta_mouse`` matches with order of mice used in ``mouse_mapping``.

The mapping from the columns of ``beta_genotype_timepoint``, ``beta_sex``, and ``beta_mouse`` to the AARs is given in the following table
| Anatomical annotion region | Column index |
|----------------------------|------------|
| Ventral medial white       | 1          |
| Ventral horn               | 2          |
| Ventral lateral white      | 3          |
| Medial grey                | 4          |
| Dorsal horn                | 5          |
| Dorsal edge                | 6          |
| Medial lateral white       | 7          |
| Ventral edge               | 8          |
| Dorsal medial white        | 9          |
| Central canal              | 10         |
| Lateral edge               | 11         |

### References
[1] Ståhl, Patrik L., et al. "Visualization and analysis of gene expression in tissue sections by spatial transcriptomics." Science 353.6294 (2016): 78-82.

[2] Stan Development Team. 2017. CmdStan: the command-line interface to Stan, Version 2.17.0. <http://mc-stan.org>.

[3] Stan Development Team. 2017. PyStan: the Python interface to Stan, Version 2.16.0.0. <http://mc-stan.org>.

[4] Stan Development Team. 2017. RStan: the R interface to Stan. R package version 2.16.2. <http://mc-stan.org>.
