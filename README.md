# Evaluating-COPD-Based-on-IOS
Enhancing accuracy of chronic obstructive pulmonary disease screening through a multimodal information fusion strategy based on impulse oscillometry

# Project Overview
Impulse Oscillometry (IOS) provides a low-cooperation alternative to spirometry for screening Chronic Obstructive Pulmonary Disease (COPD).  
We propose a multimodal fusion model combining IOS parameters, IOS curves, and demographic features for accurate COPD detection.

  
# Dataset
The dataset used in this study includes:
- IOS-derived parameters (e.g., R5, R20, Fres, X5)
- Three waveform curves (flow, pressure, volume)
- Demographic information (height, age, weight, gender)  
⚠ **Note**: The datasets used during the current study are available from the corresponding author on reasonable request.

  
# Dependencies
- Python 3.8+
- PyTorch >= 1.11
- numpy, pandas, matplotlib
- scikit-learn

  
Install dependencies via:

```bash
pip install -r requirements.txt
```
  
  
  
# 🚀 Usage
## Clone the repository
git clone https://github.com/your-username/copd-ios-model.git
cd copd-ios-model
  
  
  
# 🧠 Model Architecture
[Flow, Pressure, Volume] ---> CNN ---\  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--> Fusion --> Classifier --> COPD Prediction  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/   
[Static features] -------> MLP ---------/
  

  
# 🔗 Load Model Weights

You can download the pretrained weights from:

- [`model.pth`](model.pth) – stored locally in the directory

To use the pretrained model for inference:

```python
import torch
from model import COPDModel

model = COPDModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

