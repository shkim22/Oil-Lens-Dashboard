import pmdarima as pm
import pandas as pd


df_actual = df_metadata[df_metadata["lifecycle_stage"] == "actual"]
pred_df_list = []



def get_data_and_fit_arima(row:pd.Series):

    y_pred = []
    y = []
    time_point = []

    df = get_data(dataset_ids=[row["dataset_id"]], date_from=row["date_from_min"])

    df = df.set_index("Date")

        

    median_data = df.median()[0]
        
    min_allowed = median_data/100
    df.iloc[(df.iloc[:,0]<=min_allowed),0] = np.nan

    if df.iloc[:, 0].isnull().all():
        return
        
    df = df.fillna(method='bfill').fillna(method='ffill')

    df_lens = df.shape[0]

    try:
        

        train_end = int(df_lens*0.7)

        test_lens = df_lens - train_end
        
        model = pm.auto_arima(df[:train_end], 
                        m=12,               # frequency of series                      
                        seasonal=True,  # TRUE if seasonal series
                        seasonal_test = 'ch',
                        d=None,             # let model determine 'd'
                        test='adf',         # use adftest to find optimal 'd'
                        start_p=0, start_q=0, # minimum p and q
                        max_p=12, max_q=12, # maximum p and q
                        D=None,             # let model determine 'D'
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)
        c, confint = model.predict(n_periods=test_lens, return_conf_int=True)

        y.extend(df.iloc[train_end:,0])
        y_pred.extend(c)
        time_point.extend(list(range(train_end,df_lens)))

        res_df = pd.DataFrame({'y':y,'y_pred':y_pred,'time_point':time_point})
        res_df["description"] = row["description"]
        res_df["category"] = row["category"]

        pred_df_list.append(res_df)
    except Exception as error:
        print("Error occurs: ", error)
        print(df)

df_actual.apply(get_data_and_fit_arima, axis=1)

pred_df = pd.concat(pred_df_list)