Eigenbrains
----------
This exercise will provide a baseline for detecting anomalies in brain MRI images.
Makes uses of Scikit-learn's Isolation Forest capabilities to detect outliers. Also, one-class SVM for comparison.
Note: dataset removed due to privacy. 

- Currently only works for 'grid' images. Extending to full-sized images.
- This exercise will also be repeated with intensity-normalised scans and results compared.
- Further, project test-brains onto 'healthy space' to highlight anomalies (rather than external library tools)

Results:

6 top eigebrains:

![eigenbrains](/results/eigenbrains.png)


outlier detection:

![abnormalities](/results/anomalies.png)

sample of highlighted outlier images:

![abnormalities](/results/abnormalities.png)

