import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import time


# do not modify this
np.random.seed(89802024)


gene_exp = pd.read_csv('C:/Users/rocca/LLM/project6/normalized_gene_expression.txt',sep='\t')
# gene_exp.head()

diagnosis_map = {'AD': 0, 'PSP': 1}

labels = pd.read_csv('C:/Users/rocca/LLM/project6/labels.csv')
print(labels.groupby('Diagnosis').size())
labels = labels[['ID', 'Diagnosis']]
labels = labels[labels['Diagnosis'].isin(['AD', 'PSP'])]
labels['Diagnosis'] = labels['Diagnosis'].map(diagnosis_map)
# print(labels.groupby('Diagnosis').size())

df = pd.merge(gene_exp, labels, on='ID')
df.drop('ID', axis=1, inplace=True)
df['Diagnosis'].value_counts()

X = df.drop('Diagnosis', axis=1).values
y = df['Diagnosis'].values

assert(np.all((y == 0) | (y == 1)))






# Select top 1% the features with the most variance
from sklearn.feature_selection import VarianceThreshold
variances = np.var(X, axis=0)

num_features = len(variances)
k = int(num_features * 0.01)

variance_threshold = np.sort(variances)[-k]

selector = VarianceThreshold(threshold=variance_threshold)

X_var = selector.fit_transform(X)

X_var.shape


selected_indices = selector.get_support(indices=True)

print(f"Original features: {X.shape[1]}")
print(f"Selected features: {X_var.shape[1]}")
print(f"Selected indices: {selected_indices}")

X = X_var

alphas = np.array([0.1, 1, 10, 100, 1000])
lasso_cv_acc = []
lasso_non_zero_coeff = []
ridge_non_zero_coeff = []
ridge_cv_acc = []

# Create StratifiedKFold validation of 10 splits on 'X'
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Start timer
start_time = time.time()
print(f"Start time: {time.ctime(start_time)}")

# Lasso loop
for a in alphas:
    C = 1 / a
    model = LogisticRegression(penalty='l1', C=C, solver='saga', random_state=42, max_iter=10000)

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring='accuracy',
        return_estimator=True
    )

    mean_accuracy = np.mean(cv_results['test_score'])
    lasso_cv_acc.append(mean_accuracy)

    non_zero_counts = [np.sum(estimator.coef_[0] != 0) for estimator in cv_results['estimator']]
    lasso_non_zero_coeff.append(np.mean(non_zero_counts))

# Ridge loop
for a in alphas:
    C = 1 / a
    model = LogisticRegression(penalty='l2', C=C, solver='lbfgs', random_state=42, max_iter=10000)

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring='accuracy',
        return_estimator=True
    )

    mean_accuracy = np.mean(cv_results['test_score'])
    ridge_cv_acc.append(mean_accuracy)

    non_zero_counts = [np.sum(np.abs(estimator.coef_[0]) > 1e-6) for estimator in cv_results['estimator']]
    ridge_non_zero_coeff.append(np.mean(non_zero_counts))

# End timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Stop time: {time.ctime(end_time)}")
print(f"Total time: {elapsed_time:.2f} seconds")

# Output results
print("\nLasso Regularization (L1):")
for i, a in enumerate(alphas):
    print(f"Alpha = {a}: Accuracy = {lasso_cv_acc[i]:.4f}, Non-zero coefficients = {lasso_non_zero_coeff[i]:.1f}")

print("\nRidge Regularization (L2):")
for i, a in enumerate(alphas):
    print(f"Alpha = {a}: Accuracy = {ridge_cv_acc[i]:.4f}, Non-zero coefficients = {ridge_non_zero_coeff[i]:.1f}")
