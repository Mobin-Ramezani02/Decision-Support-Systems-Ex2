# ALS Implementation for Rating Matrix Completion

This project implements the **Alternating Least Squares (ALS)** algorithm to complete a sparse user–item rating matrix.  
The goal is to predict missing ratings (zeros in the input matrix) and reconstruct a full matrix of user–item preferences.

---

## Problem Description

The input is a `1000 × 200` rating matrix stored in a CSV file:

- Each **row** corresponds to a user.
- Each **column** corresponds to an item (e.g., a movie).
- Values **1 to 5** represent real ratings given by users.
- Value **0** means *no rating* (a missing value, not a true zero rating).

We want to factorize the rating matrix \( R \) as:

\[
R \approx U V^T
\]

where:

- \( U \in \mathbb{R}^{m \times k} \): latent feature matrix for users  
- \( V \in \mathbb{R}^{n \times k} \): latent feature matrix for items  
- \( k \): number of latent factors (dimensionality of the hidden feature space)

Once \( U \) and \( V \) are learned, we can predict ratings for all user–item pairs, including entries that were originally zero.

---

## Files and Paths

- Input rating matrix (CSV):  
  `dataset/matrix_1000x200_sparse40.csv`

- Output completed rating matrix (Excel):  
  `output/R_completed_round.xlsx`

> Make sure the `dataset/` folder contains the CSV file and the `output/` folder exists before running the script.

---

## Requirements

The code is written in Python and uses the following libraries:

- Python 3.x  
- `numpy`  
- `pandas`  
- `openpyxl` (for writing the Excel file)

Install them with:

```bash
pip install numpy pandas openpyxl
