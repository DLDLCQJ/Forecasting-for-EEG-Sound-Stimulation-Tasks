# Forecasting-for-EEG-Sound-Stimulation-Tasks

<img width="653" alt="Screenshot 2024-03-13 at 1 22 17 PM" src="https://github.com/DLDLCQJ/Forecasting-for-EEG-Sound-Stimulation-Tasks/assets/145650040/d72ac615-76c0-4e13-83a3-131d299d2f8f">

Our ultimate goal is to employ deep learning techniques for long-term time series prediction.

The EEG dataset, which includes data from three distinct stimulation tasks, is subjected to careful preprocessing to ensure its suitability for training purposes.

## GRU model for EEG data forecasting.
We propose a classical RNN model that is based on GRU model, which can capture the dynamic time dependences of time-sequence features across time. 

#### Note:
The pipeline contains train-val-test or train-val-reference processes. The users can take one way depend on your task.

## Priliminary Test or predictive results

<img width="695" alt="Screenshot 2024-03-13 at 7 00 46 PM" src="https://github.com/DLDLCQJ/Forecasting-for-EEG-Sound-Stimulation-Tasks/assets/145650040/79348441-e017-4c90-bebb-e466855535ea">


## Usage Guidelines:

- **Preprocessing Requirement**: Prioritize preprocessing the EEG data before leveraging the model. This primarily includes filtering out noise to ensure the data quality.

- **Model Suitability**: This model is specifically designed for event-related tasks. For optimal results, consider grouping similar stimuli together in cases of repeated data.
P.S.
<img width="843" alt="Screenshot 2024-03-13 at 7 18 21 PM" src="https://github.com/DLDLCQJ/Forecasting-for-EEG-Sound-Stimulation-Tasks/assets/145650040/73259183-3eb8-4ac9-9ef6-293a9473a7b1">

- **Normalization Caution**: Pay careful attention to the normalization step within our model. It's crucial to prevent "information leakage" about the target range through scaled values.

- **Stride Experimentation**: Experiment with different strides in the slicing window. Adjusting the stride can significantly impact the prediction outcomes, and finding the right stride can optimize your results.

- **Model Generalization Capacity**: Enhancing the model's generalization ability is a pivotal challenge in long-term time series forecasting. While developing a specific model for each sample is an effective approach to address generalization issues, it necessitates retraining for new samples, which can be time-consuming. In the future, our goal is to develop a model with inherent generalization capabilities across all samples. This would allow for minimal retraining—only fine-tuning a few parameters—when predicting new samples, streamlining the process significantly.


 


## Contact

If you have any questions or suggestions, feel free to contact:

- Simon: simonwang@cuhk.edu.hk

## Acknowledgement

Thanks to all contributors !!!
