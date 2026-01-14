##Linear Regression Setup

This Program is a setup for Linear Regression experimentation using Fortune 500 data. to use it you can select the features, target and split. i am using 80/20 split and combination of different features to see whether the theoritical strongly corelated features are also statically corelated.

This program gives us R2 (1 - SS [predicted - average] / SS [actual - average] ) and MSE (mean squared errors).


##Analysis
Removing High and Low features had minimal impact on R2 but increased MSE, indicating that the opening price alone explains most of the variance in the closing price, while High and Low refine prediction accuracy. Volume alone shows weak explanatory power for price level prediction.

when looking at coefficients i found that open is negative. this is due to multilinear features. this doesnt mean that open is inverdsely related to close but just the model adjusting itself. this can be proved by eliminating high and low from the feature list.


##try ridge regression
introduces alpha*(SS Coefficients) -> coefficients movetowards zero and therefore doesnot force sign flippage.

##compare linear regression and ridge regression
Ridge regression selected a very high alpha, indicating that the problem does not suffer from harmful overfitting when using same-day OHLC features. Linear regression achieved lower test error, showing that regularization is unnecessary for this formulation.


##Todays close = tomorrows open
this shows that the best alpha actually shrinks to 0.003 therefore ridge is effectively linear. the data is stable and no overfitting but the features are strongly correlated. predictions lag behind so the market is hard to predict using only past OHLC data.

