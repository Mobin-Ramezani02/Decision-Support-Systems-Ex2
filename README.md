# ALS Implementation for Rating Matrix Completion

This project implements the **Alternating Least Squares (ALS)** algorithm to complete a sparse user–item rating matrix.  
The goal is to predict missing ratings (zeros in the input matrix) and reconstruct a full matrix of user–item preferences.

---

## Problem Description

The input is a rating matrix stored in a CSV file with 1000 rows and 200 columns:

- Each **row** corresponds to a user.
- Each **column** corresponds to an item (for example, a movie).
- Values from **1 to 5** are real ratings given by users.
- Value **0** means the user did not rate that item (missing data, not a real rating of zero).

The idea is to represent this large rating matrix as the product of two smaller matrices:

- A **user matrix** `U`, where each row is a latent feature vector describing a user.
- An **item matrix** `V`, where each row is a latent feature vector describing an item.

Multiplying `U` by the transpose of `V` gives an approximation of the original rating matrix.  
Using this approximation, we can predict ratings for entries that were originally zero.

---

## Files and Paths

- Input rating matrix (CSV):  
  `dataset/matrix_1000x200_sparse40.csv`

- Output completed rating matrix (Excel):  
  `output/R_completed_round.xlsx`

> Make sure the `dataset/` folder contains the CSV file and that the `output/` folder exists before running the script.

---

## Requirements

The code is written in Python and uses the following libraries:

- Python 3.x  
- `numpy`  
- `pandas`  
- `openpyxl` (needed for writing the Excel file)

Install them with:

```bash
pip install numpy pandas openpyxl
