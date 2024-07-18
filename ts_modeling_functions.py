import statsmodels.tsa.api as tsa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.api as tsa
import numpy as np

print("- Switch to using lp_functions instead of ts_modeling_functions to match lesson.")
    

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error)


def get_adfuller_results(ts, alpha=.05,
                         label='adfuller',
                         **kwargs):#adfuller_kws = {}):
    """Uses statsmodels' adfuller function to test a univariate time series for stationarity.
        Null hypothesis: 
            The time series is NOT stationary. (It "has a unit root".) 
        Interpretation:
            a p-value less than alpha (.05) means the ts IS stationary. 
            (We reject the null hypothesis that it is not stationary.)


    Returns
    -------
    results (DataFrame): DataFrame with the following columns/results:
    - "Test Statistic" : the adfuller test statistic.
    - "# of Lags Used": The number of lags used in the calculation.
    - "# of Observations" : The number of observations used.
    - "p-value" : p-value for hypothesis test.
    - "alpha": the significance level used for interpretin p-value
    - "sig/stationary?": simplified interpretation of p-value
    
    ADFULLER DOCUMENTATION:
    For the full adfuller documentation, see:
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

    """
    # Saving each output
    (test_stat, pval, nlags, nobs, crit_vals_d, 
    icbest )= tsa.adfuller(ts, **kwargs)
    adfuller_results = {'Test Statistic': test_stat,
                        "# of Lags Used":nlags, 
                       '# of Observations':nobs,
                        'p-value': round(pval,6),
                        'alpha': alpha,
                       'sig/stationary?': pval<alpha}
    return pd.DataFrame(adfuller_results, index=[label])




def regression_metrics(y_true, y_pred, label="", verbose=True, output_dict=False,):
    # Get metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r_squared = r2_score(y_true, y_pred)

    if verbose == True:
        # Print Result with label
        header = "---" * 20
        print(header, f"Regression Metrics: {label}", header, sep="\n")
        print(f"- MAE = {mae:,.3f}")
        print(f"- MSE = {mse:,.3f}")
        print(f"- RMSE = {rmse:,.3f}")
        print(f"- R^2 = {r_squared:,.3f}")

    if output_dict == True:
        metrics = {
            "Label": label,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R^2": r_squared,
        }
        return metrics




def evaluate_regression(reg, X_train, y_train, X_test, y_test, 
                        output_frame=False):
  # Get predictions and  results for training data
  y_train_pred = reg.predict(X_train)
  results_train = regression_metrics(y_train, y_train_pred, 
                                     output_dict=output_frame,
                                     label='Training Data')
  print()
  # Get results for test data
  y_test_pred = reg.predict(X_test)
  results_test = regression_metrics(y_test, y_test_pred, 
                                  output_dict=output_frame,
                                    label='Test Data' )
  
  if output_frame:
    results_df = pd.DataFrame([results_train,results_test])
    # Set the label as the index and get rid of the name 
    results_df = results_df.set_index('Label')
    results_df.index.name=None
    return results_df.round(3)


def get_forecast(*args,**kwargs):
    msg = "custom get_forecast function has been replaced by the built-in forecast.summary_frame() method.\n"
    msg += ">> forecast = model.get_forecast(steps)\n>> forecast_df = forecast.summary_frame()\n"
    msg += "- the new forecast column names are: 'mean','mean_ci_lower','mean_ci_upper'"
    raise Exception(msg)
    
    

def plot_forecast(ts_train, ts_test, forecast_df, n_train_lags=None, 
                  figsize=(10,4), title='Comparing Forecast vs. True Data'):
    ### PLot training data, and forecast (with upper/,lower ci)
    fig, ax = plt.subplots(figsize=figsize)

    # setting the number of train lags to plot if not specified
    if n_train_lags==None:
        n_train_lags = len(ts_train)
            
    # Plotting Training  and test data
    ts_train.iloc[-n_train_lags:].plot(ax=ax, label="train")
    ts_test.plot(label="test", ax=ax)

    # Plot forecast
    forecast_df['mean'].plot(ax=ax, color='green', label="forecast")

    # Add the shaded confidence interval
    ax.fill_between(forecast_df.index, 
                    forecast_df['mean_ci_lower'],
                   forecast_df['mean_ci_upper'],
                   color='green', alpha=0.3,  lw=2)

    # set the title and add legend
    ax.set_title(title)
    ax.legend();
    
    return fig, ax


    

def regression_metrics_ts(ts_true, ts_pred, label="", verbose=True, output_dict=False,):
    # Get metrics
    mae = mean_absolute_error(ts_true, ts_pred)
    mse = mean_squared_error(ts_true, ts_pred)
    rmse = mean_squared_error(ts_true, ts_pred, squared=False)
    r_squared = r2_score(ts_true, ts_pred)
    try:
        U = thiels_U(ts_true, ts_pred)
    except:
        U = np.nan
    mae_perc = mean_absolute_percentage_error(ts_true, ts_pred) * 100

    if verbose == True:
        # Print Result with label
        header = "---" * 20
        print(header, f"Regression Metrics: {label}", header, sep="\n")
        print(f"- MAE = {mae:,.3f}")
        print(f"- MSE = {mse:,.3f}")
        print(f"- RMSE = {rmse:,.3f}")
        print(f"- R^2 = {r_squared:,.3f}")
        print(f"- MAPE = {mae_perc:,.2f}%")
        print(f"- Thiel's U = {U:,.3f}")

    if output_dict == True:
        metrics = {
            "Label": label,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE(%)": mae_perc,
            "R^2": r_squared,
            "Thiel's U": U
        }
        return metrics

    
def thiels_U(ts_true, ts_pred):
    """Calculate's Thiel's U metric for forecasting accuracy.
    Accepts true values and predicted values.
    Original Formula Source: https://docs.oracle.com/cd/E57185_01/CBREG/ch06s02s03s04.html
    Adapted Function from Source: https://github.com/jirvingphd/predicting-the-SP500-using-trumps-tweets_capstone-project/blob/cf11f6ed88721433d2c00cb1f8486206ab179cc0/bsds/my_keras_functions.py#L735
    Returns: 
        U (float)
        
    Thiel's U Value Interpretation:
    - <1  = Forecasting is better than guessing 
    - 1   = Forecasting is about as good as guessing
    - >1  = Forecasting is worse than guessing 
    """
    import numpy as np
    # sum_list = []
    num_list=[]
    denom_list=[]
    
    for t in range(len(ts_true)-1):
        
        num_exp = (ts_pred[t+1] - ts_true[t+1])/ts_true[t]
        num_list.append([num_exp**2])
        
        denom_exp = (ts_true[t+1] - ts_true[t])/ts_true[t]
        denom_list.append([denom_exp**2])
        
    U = np.sqrt( np.sum(num_list) / np.sum(denom_list))
    return U





# def evaluate_ts_model(model, ts_train, ts_test, 
#                       return_scores=False, show_summary=True,
#                       n_train_lags=None, figsize=(9,3),
#                       title='Comparing Forecast vs. True Data'):
                    
#     # Check if auto-arima, if so, extract sarima model
#     if hasattr(model, "arima_res_"):
#         model = model.arima_res_
        
#     # Get forecast         
#     forecast = model.get_forecast(steps=len(ts_test))
#     forecast_df = forecast.summary_frame()

#     # Visualize 
#     plot_forecast(ts_train, ts_test, forecast_df, 
#                   n_train_lags=n_train_lags, figsize=figsize,
#                  title=title)
#     plt.show()
                    
#     # Get and display the regression metrics BEFORE showing plot
#     reg_res = regression_metrics_ts(ts_test, forecast_df['mean'], output_dict=True)
    
#     if show_summary==True:
#         display(model.summary())
#         model.plot_diagnostics(figsize=(8,3))
#         plt.tight_layout()

  
#     if return_scores:
#         return reg_res

def make_best_arima(auto_model, ts_train, exog=None, fit_kws={}):
    ## Fit a final model and evaluate
    best_model = tsa.SARIMAX(
        ts_train,
        exog=exog,
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        sarimax_kwargs=auto_model.sarimax_kwargs,
    ).fit(disp=False, **fit_kws)
    return best_model

    
    
    
def evaluate_ts_model(model, ts_train, ts_test, exog_train=None, exog_test=None,
                      return_scores=False, show_summary=True,
                      n_train_lags=None, figsize=(9,3),
                      title='Comparing Forecast vs. True Data'):
    """Updated version that accepts a pmdarima auto_arima
    - Also accepts exogenous variables"""
    ## GET FORECAST             
    # Check if auto-arima, if so, extract sarima model
    if hasattr(model, "arima_res_"):
        print(f"- Fitting a new ARIMA using the params from the auto_arima...")
        model = make_best_arima(model, ts_train, exog=exog_train)
        

    # Get forecast         
    forecast = model.get_forecast(exog=exog_test, steps=len(ts_test))
    forecast_df = forecast.summary_frame()

    # Visualize forecast
    plot_forecast(ts_train, ts_test, forecast_df, 
                  n_train_lags=n_train_lags, figsize=figsize,
                 title=title)

    plt.show()
                    
    # Get and display the regression metrics BEFORE showing plot
    reg_res = regression_metrics_ts(ts_test, forecast_df['mean'], 
                                        output_dict=True)
    
    if show_summary==True:
        display(model.summary())
        model.plot_diagnostics(figsize=(8,4))
        plt.tight_layout()
        plt.show()
      
    if return_scores:
        return model, reg_res
    else:
        return model

    
def plot_acf_pacf(ts, nlags=40, figsize=(12, 5),
                 acf_kws={}, pacf_kws={'method':"ywm"}):
    fig, axes = plt.subplots(nrows=2,figsize=figsize)
    tsa.graphics.plot_acf(ts, ax=axes[0], lags=nlags, **acf_kws)
    tsa.graphics.plot_pacf(ts,ax=axes[1], lags=nlags, **pacf_kws)
    fig.tight_layout()
    return fig
    
    
    
def seasonal_decomposition(ts,model='additive', period=None, figsize=(9,5)):
    
    decomp = tsa.seasonal_decompose(ts, period=period , model=model)

    fig = decomp.plot()
    fig.set_size_inches(9, 5)
    fig.tight_layout()
    
    return decomp



def get_sig_lags(ts, nlags=None,alpha=0.5):
    
    # Running the function used by plot_acf
    acf_values, conf_int = tsa.stattools.acf(ts, alpha=alpha, nlags=nlags)

    # Determine lags
    lags =range(len(acf_values))
    
    # Create a centered version of the acf_df [centered on..0??]
    acf_df = pd.DataFrame({'ACF':acf_values,
                            'Lags':lags,
                            'lower ci': conf_int[:,0]-acf_values, # subtract acf from lower ci to center
                            'upper ci': conf_int[:,1]-acf_values, # subtact acf to upper ci to center
                                 })
    acf_df = acf_df.set_index("Lags")
    
    # Getting filter for sig lags
    filter_sig_lags = (acf_df['ACF'] < acf_df['lower ci']) | (acf_df['ACF'] > acf_df['upper ci'])

    # Get lag #'s 
    sig_lags= acf_df.index[filter_sig_lags]
    sig_lags = sig_lags[sig_lags!=0]

    return sig_lags



def plot_acf_sig_lags(ts,nlags=None,alpha=0.5,ax=None ):

    # Plot acf
    fig = tsa.graphics.plot_acf(ts,ax=ax,lags=nlags,alpha=alpha)
    if ax is None: 
        ax = fig.get_axes()[0]

    ## Get significant lags
    sig_lags = get_sig_lags(ts,nlags=nlags, alpha=alpha)
    
    ## ADDING ANNOTATING SIG LAGS
    for lag in sig_lags:
        ax.axvline(lag, ls='--', lw=1, zorder=0, color='red')

    return sig_lags