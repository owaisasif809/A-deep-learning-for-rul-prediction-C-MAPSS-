# A-Deep-Learning-Model-for-Remaining-Useful-Life-Prediction-of-Aircraft-Turbofan-Engine-on-C-MAPSS-Da
In the era of industry 4.0, safety, efficiency and reliability of industrial machinery is an elementary concern in trade sectors. The accurate remaining useful life (RUL) prediction of an equipment in due time allows us to effectively plan the maintenance operation and mitigate the downtime to raise the revenue of business. In the past decade, data driven based RUL prognostic methods had gained a lot of interest among the researchers. There exist various deep learning-based techniques which have been used for accurate RUL estimation. One of the widely used technique in this regard is the long short-term memory (LSTM) networks. To further improve the prediction accuracy of LSTM networks, this paper proposes a model in which effective pre-processing steps are combined with LSTM network. C-MAPSS turbofan engine degradation dataset released by NASA is used to validate the performance of the proposed model. One important factor in RUL predictions is to determine the starting point of the engine degradation. This work proposes an improved piecewise linear degradation model to determine the starting point of deterioration and assign the RUL target labels. The sensors data is pre-processed using the correlation analysis to choose only those sensors measurement which have a monotonous behavior with RUL, which is then filtered through a moving median filter. The updated RUL labels from the degradation model together with the pre-processed data are used to train a deep LSTM network. The deep neural network when combined with dimensionality reduction and piece-wise linear RUL function algorithms achieves improved performance on aircraft turbofan engine sensor dataset. We have tested our proposed model on all four sub-datasets in C-MAPSS and the results are then compared with the existing methods which utilizes the same dataset in their experimental work. It is concluded that our model yields improvement in RUL prediction and attains minimum root mean squared error and score function values.

Asif, Owais, et al. "A Deep Learning Model for Remaining Useful Life Prediction of Aircraft Turbofan Engine on C-MAPSS Dataset." IEEE Access 10 (2022): 95425-95440.
