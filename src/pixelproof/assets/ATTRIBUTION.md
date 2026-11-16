# Model training data attribution

This model was trained on a fixed subset of scikit-learn's bundled `load_digits` data, a copy of the original UCI test set. Original dataset: E. Alpaydin and C. Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning Repository, https://doi.org/10.24432/C50P49. Dataset license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.

PixelProof changes: fixed stratified seed-42 75/25 split of the 1,797 bundled samples, normalization by 16, and fitting logistic regression on the 1,347-sample training partition. The JSON artifact contains learned coefficients and training metadata. This is not the official UCI benchmark split. MIT licensing of project code does not replace dataset attribution or licensing.
