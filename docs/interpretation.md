# Code Snippet for Interpretation using SHAP

```python
# Interpretasi Model RandomForest Balanced (score yang paling tinggi diantara 3 model di atas dari Cross Validation Score)
import shap

explainer = shap.TreeExplainer(rf_balanced)
shap_values = explainer.shap_values(X_test)

# For binary classification, some SHAP versions return a list:
# [SHAP values for class 0, SHAP values for class 1]
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]

else:
    shap_values_class1 = np.asarray(shap_values)

    # Some versions return shape: (samples, features, classes)
    if shap_values_class1.ndim == 3:
        shap_values_class1 = shap_values_class1[:, :, 1]

shap.summary_plot(
    shap_values_class1,
    X_test,
    max_display = 20
)
```