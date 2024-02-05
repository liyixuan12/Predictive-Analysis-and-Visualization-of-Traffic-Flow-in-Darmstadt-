# Predictive Analysis and Visualization of Traffic Flow in Darmstadt (PAaV)
![image](https://github.com/liyixuan12/TrafficPredictionFlow/assets/98014565/a42560f7-3d1d-4972-bc51-b4794b18554f)
### Abstract:
Our work: PAaV - TF Darmstadt, analyzes traffic data for Darmstadt from the world's largest city comparison study UTD19.
Based on sensor-based traffic flow and geospatial data, we present time series prediction methods as well as visualization and analysis of geospatial data to evaluate tools that can help urban planners to solve road capacity planning problems while maintaining cities as livable spaces.
## 1. Introduction to the project structure
### Detailed Structure
Predictive Analysis and Visualization of Traffic Flow in Darmstadt/  
├── data/  
│   ├── Filtering_Pipeline.ipynb: Notebook containing the data filtering and preprocessing pipeline.  
│   ├── Processed/  
│   │   ├── final_darmstadt_data.csv: The final processed dataset for Darmstadt city.  
│   │   ├── K1D43/  
│   │   │   ├── K1D43_data.csv: Data specifically related to the K1D43 sensor location.  
│   │   │   ├── K1D43_data_period_1.csv: K1D43 data for a specific time period.  
│   │   │   ├── K1D43_data_period_2.csv: K1D43 data for a specific time period.    
│   │   │   └── K1D43_data_period_3.csv: K1D43 data for a specific time period.  
│   │   └── GUI/  
│   │   │   ├── processed_location_data.csv: Processed data used in the GUI for location-based analysis.  
│   │   │   ├── Darmstadt_Roads_Detid_Data.csv: Detailed road data for Darmstadt city.   
│   └── raw/    
│       ├── detectors_darmstadt_data.csv: Raw data from traffic detectors in Darmstadt.  
│       └── darmstadt_data.csv: Comprehensive raw dataset for Darmstadt city.  
├── Predictive_Models/  
│   ├── CNN.py: Python script for the Convolutional Neural Network model.    
│   ├── LSTM_tuner.py: Script for tuning LSTM model parameters.    
│   ├── LSTM.py: Script implementing the Long Short-Term Memory model.    
│   └── Arima.py: Implementation of the ARIMA model for time-series prediction.  
│   ├── multiple_step_mode/  
│   │   ├── CNN_multiple_step.py: CNN model adapted for multiple-step prediction.  
│   │   ├── LSTM_multiple_step.py: LSTM model for multi-step forecasting.    
│   │   ├── multiple_steps_comparison.ipynb: Notebook for comparing different multi-step models.  
├── results/  
│   ├── results_show.ipynb: Notebook displaying the results of the predictive models.  
│   ├── Train_Test_Prediction_Graphics/  
│   │   └── Various .png files: Visualization of training and testing predictions for different models.  
│   ├── Test_Set_Prediction_data/    
│   │   ├── CNN_Test_Set_Predictions.csv: Test set predictions from the CNN model.  
│   │   └── LSTM_Test_Set_Predictions.csv: Test set predictions from the LSTM model.  
│   └── Graphics/  
│       └── Sub-folders and files for additional graphical data representations, including scatter plots and daily predictions.  
└── GUI/   
    ├── GUI for maps and data showing/  
    │   ├── Various .html and .png files for displaying maps and data animations.  
    │   ├── Map with Markers and Clusters/  
    │   │   └── map_with_markers_and_clusters_darmstadt.html: HTML map showing data points with markers and clusters.  
    │   │   └── K_MEAN_c.py: Python script for K-means clustering visualization.  
    │   ├── Map with Markers Darmstadt/  
    │   │   └── HTML files and Python scripts for generating maps with specific markers.  
    │   └── Darmstadt City Map/  
    │       ├── Darmstadt_City_Map.py: Python script for visualizing Darmstadt city map.  
    │       └── Darmstadt_City_Map.html: HTML for visualizing Darmstadt city map.  
    ├── Statistical Characteristics of Cross/  
    │   └── Scripts and images for analyzing traffic flow at specific cross-sections.  
    ├── Density Map/  
    │   └── Produce_Density_Map.py: Script to generate density maps.  
    └── GUI for Model comparison/  
        └── HTML and image files for comparing different predictive models visually.  
## 2. Introduction to the code execution process
### 2.1 Overall Process
![Flowchart - Page 1 (2)](https://github.com/liyixuan12/TrafficPredictionFlow/assets/98014565/bb5401cc-fe7a-4402-b8a3-d04361e8c01b)

### 2.2 Data Cleaning
<img width="1258" alt="image" src="https://github.com/liyixuan12/TrafficPredictionFlow/assets/98014565/0cb79f16-4f9e-4d50-a0cc-454e64d0973d">  

### 2.3 Model Training & Evaluation

![Flowchart - Page 3 (2)](https://github.com/liyixuan12/TrafficPredictionFlow/assets/98014565/61715ac0-21bb-4ebf-8675-648dbbdf5b6f)

### 2.4 GUI Development

![image](https://github.com/liyixuan12/TrafficPredictionFlow/assets/98014565/07a8f1dc-4145-4807-8445-91e0b9e028d7)

## 2.5 Experiment Results
### Prediction Result
<img width="1468" alt="image" src="https://github.com/liyixuan12/TrafficPredictionFlow/assets/98014565/c7c5af88-30e9-47ce-a498-5f4b880323e2">

The comparison graph of the two models for the prediction results is shown above.   

CNN: Train RMSE: 105.66 Test RMSE: 107.11  
LSTM: Train RMSE: 107.36 Test RMSE: 108.07 
### Traffic Density Map of Darmstadt
<img width="1560" alt="image" src="https://github.com/liyixuan12/Predictive-Analysis-and-Visualization-of-Traffic-Flow-in-Darmstadt-PAaV/assets/98014565/9b4e6c32-06a5-424b-b116-73d43059f8f0">


## Data Science Canvas
<img width="1224" alt="image" src="https://github.com/liyixuan12/Predictive-Analysis-and-Visualization-of-Traffic-Flow-in-Darmstadt-PAaV/assets/98014565/134a2ea1-3d22-400b-bb23-54abd4f6a3e6">



