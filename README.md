# Recurrent_Neural_Network_With_LSTM_Trend_Predictor


Recurrent Neural Network


Sometimes we required to remember the data or result from previous or past actions on which current decisions might get impacted. When we are under the realm of Machine Learning, vanilla neural networks learn the things during the training. Recurrent Neural Network learn the exactly same as the vanilla networks learn, but in addition they remember the things learned from the previous input while learning from current input.


Lont-Short Term Memory(LSTM)


LSTM's are special kind of RNN which is capable of long-term learning dependencies. LSTM's main aim is to remembering the information for long time and hence it will input in a learning process. LSTM have chain like structure which have different structure than standard neural networks. LSTM has four layered structure which they can interact in a special way. There are three gates namely Forget Gate, Input Gate and Output Gate through which information is regulated. Cell state is act like conveyer belt which carry information with very minor changes in LTSM.

Result Interpretation 

Here key point is to understand that the goal is to check whether the predicted stock price pattern follows the real stock price pattern or not. I train the model with historical data near about last 10 years of TCS stock prices(OPEN price, CLOSE price) and try to predict the price and hence pattern for the month of september-2020.


1. TCS Open Price Trend SEP-20

![Alt text](/Results/TCS_OPEN_PRICE_TREND.JPG?raw=true "TCS Open Price Trend")


2. TCS Close Price Trend SEP-20

![Alt text](/Results/TCS_CLOSE_PRICE_TREND.JPG?raw=true "TCS Open Price Trend")
