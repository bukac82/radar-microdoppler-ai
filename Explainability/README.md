# Explainability Module

Tools to explain and interpret model predictions on micro-Doppler signals.

## Methods

| File | Method | Model Type | Description |
|------|--------|-----------|-------------|
| `shap_explainer.py` | SHAP | Classical ML | Feature importance via SHapley Additive exPlanations |
| `gradcam.py` | Grad-CAM | CNN (DL) | Gradient-weighted Class Activation Maps on spectrograms |
| `lime_explainer.py` | LIME | Any | Local Interpretable Model-agnostic Explanations |
| `feature_importance.py` | RF Importance | Random Forest | Built-in impurity-based feature importance |

## Usage

```python
# SHAP for classical ML
from Explainability.shap_explainer import explain_with_shap
shap_values = explain_with_shap(rf_model, X_test_features)

# Grad-CAM for CNN
from Explainability.gradcam import GradCAM
cam = GradCAM(cnn_model, target_layer=cnn_model.layer3)
heatmap = cam(spectrogram_tensor, class_idx=1)
```
