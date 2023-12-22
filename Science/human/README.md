Splotch
===================

Overview
-------------
This implementation has a three-level hierarchical model representing the hierarchical experimental design

![Two-level hierarchical model](https://raw.githubusercontent.com/tare/Splotch/master/images/two-level_hierarchical_model.png)

Example data
-------------
Here we provide an example input data (``data_SNAP25.R``) to be used with the code.
Specifically, the input data file is consisted of 61,031 spatially-resolved measurements for the *SNAP25* gene.
The ST measurements are from 80 tissue sections collected from 7 ALS donors.
Every donor had either lumbar or bulbar onset and the tissue samples were taken either from the lumbar or bulbar section of the post-mortem spinal cord.

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
make $HOME/Splotch/mouse/splotch_human
```

Usage
-------------
To sample four chains using the example data set
```bash
# set a few variables
GENE="SNAP25"
NUM_SAMPLES=500
NUM_WARMUP=500
N_CHAINS=4
BINARY="splotch_human"

cd $HOME/Splotch/human

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
GENE="SNAP25"

$HOME/Splotch/human

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
| ``N_onsets_tissue_locations``    | int           | 1            | Number of onset and tissue location combinations                                                                                 |
| ``N_donors``         | int           | 1            | Number of donor and tissue location combinations                                                                                      |
| ``donor_mapping``  | int           | ``N_donors``       | Origin of each donor sample (see below)                                                                    |
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

For instance, if the 98<sup>th</sup> spot has been annotated to be in dorsal horn, then the 98<sup>th</sup> element of ``D`` is 5.

Next we will cover how ``donor_mapping`` and ``tissue_mapping`` have been constructed.
Each donor sample is associated with an onset location and a tissue location.
As described above, we have two onset locations (lumbar and bulbar) and two tissue locations (lumbar and cervical).
To incorporate this information into the model through ``donor_mapping``, we use the following mapping

| Onset location   | Tissue location |  Identifier |
|------------------|-----------------|-------------|
| Lumbar onset     | Lumbar          | 1           |
| Lumbar onset     | Cervical        | 2           |
| Bulbar onset     | Cervical        | 3           |
| Bulbar onset     | Lumbar          | 4           |

That is, if the first donor sample (i.e. corresponds to the first element of ``donor_mapping``) corresponds to a donor with a lumbar onset and the sample was taken from the lumbar section of their spinal cord, then the value of the first element of ``donor_mapping`` is 1.
Note that the order of the donors in ``donor_mapping`` is arbitratory but it will impact the construction of ``tissue_mapping`` as described next. 

Similarly, each tissue section has been taken from one of the ``N_donors`` donors. 
This information is incorporated into the model through ``tissue_mapping`` by linking the tissue sections to donors and and sampling locations

| Donor identifier | Onset location | Tissue location | Identifier |
|------------|------------------|-----------------|-----------|
| D1 | Lumbar | Lumbar   | 1 |
| D2 | Lumbar | Lumbar   | 2 |
| D1 | Lumbar | Cervical | 3 |
| D3 | Lumbar | Cervical | 4 |
| D2 | Lumbar | Cervical | 5 |
| D4 | Bulbar | Cervical | 6 |
| D5 | Bulbar | Cervical | 7 |
| D6 | Bulbar | Cervical | 8 |
| D7 | Bulbar | Lumbar | 9 |
| D5 | Bulbar | Lumbar | 10 |
| D6 | Bulbar | Lumbar | 11 |

For instance, if the second tissue section has been sampled from the lumbar section of the fifth donor (D5), then the second element of ``tissue_mapping`` is 10.
For four of the donors (D1, D2, D5, and D6) we have tissue sections from both regions (lumbar and cervical), whereas, for the donors D3, D4, and D7 we only have tissue sections from one of the regions (either lumbar or cervical).
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
| ``beta_onset_tissue`` | real                 | ``N_onsets_locations`` x ``N_covariates``   | Onset/tissue level coefficients |
| ``beta_donor_onset_tissue``                 | real                 | ``N_donors`` x ``N_covariates``              | Donor/onset/tissue level coefficients                 |
| ``theta``                     | real between 0 and 1 | 1                                    | Probability of extra zeros               |
| ``a``                         | real between 0 and 1 | 1                                    | Spatial autocorrelation                  |
| ``tau``                       | positive real        | 1                                    | Conditional precision                    |

The order of elements in ``lambda`` corresponds to the order of the spots in ``counts``.

The mapping from the rows of ``beta_onset_tissue`` to the onset and tissue location is given in the following table

| Onset location   | Tissue location |  Row index |
|------------------|-----------------|------------|
| Lumbar onset     | Lumbar          | 1          |
| Lumbar onset     | Cervical        | 2          |
| Bulbar onset     | Cervical        | 3          |
| Bulbar onset     | Lumbar          | 4          |

Whereas, the mapping from the rows of ``beta_donor_onset_tissue`` to the donor and tissue location is given in the following table

| Donor identifier | Onset location | Tissue location | Row index |
|------------|------------------|-----------------|-----------|
| D1 | Lumbar | Lumbar   | 1 |
| D2 | Lumbar | Lumbar   | 2 |
| D1 | Lumbar | Cervical | 3 |
| D3 | Lumbar | Cervical | 4 |
| D2 | Lumbar | Cervical | 5 |
| D4 | Bulbar | Cervical | 6 |
| D5 | Bulbar | Cervical | 7 |
| D6 | Bulbar | Cervical | 8 |
| D7 | Bulbar | Lumbar | 9 |
| D5 | Bulbar | Lumbar | 10 |
| D6 | Bulbar | Lumbar | 11 |

The mapping from the columns of ``beta_onset_tissue`` and ``beta_donor_onset_tissue`` to the AARs is given in the following table

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
