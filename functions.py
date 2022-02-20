
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def comp_auc(target, predictions):
    """
    Computes area of curve
    """
    return roc_auc_score(target, predictions)


def _one_model(x_train, x_valid, y_train, y_valid, params):
    """
    Trains one lgbm model
    """

    dtrain=lgb.Dataset(x_train, label=y_train)
    dvalid=lgb.Dataset(x_valid, label=y_valid)

    watchlist=dvalid

    booster=lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=watchlist,
        verbose_eval=200
    )

    return booster

def fit_model(x_train,y_train, x_valid, y_valid, params, preds):
    """
    Performs LGBM training
    """

    models=[]
    performance = []

    model = _one_model(x_train[preds], x_valid[preds], y_train, y_valid,params)
    models.append(model)

    x=[]
    x.append(model)
    predictions = predict(x, x_valid, preds)

    performance.append(comp_auc(y_valid, predictions))


    print('Performance on validation sets:')
    print(performance)
    print('Mean:')
    print(np.mean(performance))

    return models

def predict(models, set_to_predict, preds):
    """
    Predicts model to new set
    """
    
    predictions=np.zeros(set_to_predict.shape[0])

    for model in models:
        predictions = predictions + model.predict(set_to_predict[preds])/len(models)

    return predictions

def _comp_var_imp(models, preds):
    """
    Computes variable importance of model
    """

    importance_df=pd.DataFrame()
    importance_df['Feature']=preds
    importance_df['Importance_gain']=0
    importance_df['Importance_weight']=0

    for model in models:
        importance_df['Importance_gain'] = importance_df['Importance_gain'] + model.feature_importance(importance_type = 'gain') / len(models)
        importance_df['Importance_weight'] = importance_df['Importance_weight'] + model.feature_importance(importance_type = 'split') / len(models)

    return importance_df

def plot_importance(models, imp_type, preds ,ret=False, show=True, n_predictors = 100):
    """
    Plots variable importance of the model
    """
    if ((imp_type!= 'Importance_gain' ) & (imp_type != 'Importance_weight')):
        raise ValueError('Only importance_gain or importance_gain is accepted')

    dataframe = _comp_var_imp(models, preds)

    if (show == True):
        plt.figure(figsize = (20, len(preds)/2))
        sns.barplot(x=imp_type, y='Feature', data=dataframe.sort_values(by=imp_type, ascending= False).head(len(preds)))

    if (ret == True):
        return dataframe.sort_values(by=imp_type, ascending= False).head(len(preds))[['Feature', imp_type]]


def print_shap_values(preds, cols_cat, x_train, y_train, x_valid, y_valid, params):
    """
    Computes SHAP values of the model for the x_valid set
    """
    x_train = x_train[preds]
    x_valid = x_valid[preds]
    
    for col in cols_cat:
        if x_train[col].isnull().sum()>0:
            x_train[col] = x_train[col].cat.add_categories('NA').fillna('NA')
            x_valid[col] = x_valid[col].cat.add_categories('NA').fillna('NA')
        _ , indexer = pd.factorize(x_train[col])
        x_train[col] = indexer.get_indexer(x_train[col])
        x_valid[col] = indexer.get_indexer(x_valid[col])
    

    model= _one_model(x_train, x_valid, y_train, y_valid, params)
         
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_valid)

    if isinstance(shap_values, list):
               
        shap_values = shap_values[1]

    else:
        shap_values = shap_values


    shap.summary_plot(shap_values, x_valid)
    shap.summary_plot(shap_values, x_valid, plot_type='bar')
 
    
    return x_valid, shap_values ,explainer
        



def shap_dependence_plot(set, shap_values, x, y=None):
    """
    Plots SHAP dependence plot
    """

    if  y is None:
        shap.dependence_plot(x, shap_values, set)

    else:            
        shap.dependence_plot(x, shap_values, set, interaction_index = y)


def prepare_roc():
    """
    Prepares figure for auc plotting
    """
    plt.figure(figsize=(10,10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def _convert_times(data_it, data_seg):
    """
    Converts daytime columns to pandas Datetime type
    """
    data_it['timestamp_of_search'] = pd.to_datetime(data_it['timestamp_of_search'])
    data_seg['departure_time_utc'] = pd.to_datetime(data_seg['departure_time_utc'])
    data_seg['arrival_time_utc'] = pd.to_datetime(data_seg['arrival_time_utc'])
    
    return data_it, data_seg

def _agg_function(data, agg_by, agg_function):
    """
    performs pandas groupby 
    """
    grouped = data.groupby(agg_by).aggregate(agg_function)
    grouped.columns=['_'.join (col) for col in list(grouped)]
    grouped.reset_index(inplace=True)
    return grouped

def _compute_daytime_diff(column1, column2):
    """
    Returns datetime difference of 2 pandas columns in days
    """
    return (pd.to_datetime(column1) - pd.to_datetime(column2)).dt.total_seconds()/(3600*24)

def _time_between_search_first_flight(data_it, data_seg):
    """
    computes time distance between time of search and time of first flight in days
    """
    
    min_flights = _agg_function(data_seg, ['bid'], {'departure_time_utc':['min']})
    merged_ds = pd.merge(data_it[['bid', 'timestamp_of_search']], min_flights, how = 'left',
                        left_on = 'bid', right_on = 'bid')
    
    merged_ds['search_flight_diff_first'] = _compute_daytime_diff(merged_ds['departure_time_utc_min'],
                                                            merged_ds['timestamp_of_search'])
    
    return merged_ds[['bid','search_flight_diff_first']]


def _time_between_search_last_flight(data_it, data_seg):
    """
    computes time distance between time of search and time of last flight in days
    """
    
    min_flights = _agg_function(data_seg, ['bid'], {'arrival_time_utc':['max']})
    merged_ds = pd.merge(data_it[['bid', 'timestamp_of_search']], min_flights, how = 'left',
                        left_on = 'bid', right_on = 'bid')
    
    merged_ds['search_flight_diff_last'] = _compute_daytime_diff(merged_ds['arrival_time_utc_max'],
                                                            merged_ds['timestamp_of_search'])
    
    return merged_ds[['bid','search_flight_diff_last']]

def _time_between_first_last_flight(data_seg):
    """
    computes time distance between first departure_time_utc and last arrival_time_utc
    """
    
    min_flights = _agg_function(data_seg, ['bid'], {'departure_time_utc':['min']})
    max_flights = _agg_function(data_seg, ['bid'], {'arrival_time_utc':['max']})
    
    merged_ds = pd.merge(max_flights, min_flights, how = 'inner', left_on = 'bid', right_on = 'bid')
    
    merged_ds['first_last_flight_diff'] = _compute_daytime_diff(merged_ds['arrival_time_utc_max'],
                                                                merged_ds['departure_time_utc_min'])
    
    return merged_ds[['bid','first_last_flight_diff']]


def _count_of_different_airlines(data_seg):
    """
    computes number of unique airlines that are in the itinerary
    """
    
    n_airlines_ds = _agg_function(data_seg, ['bid'], {'airline':['nunique']})
    
    return n_airlines_ds


def _compute_month_day_of_first_flight(data_seg):
    """
    computes month and weekday of first flight in itinerary
    """
    
    min_flights = _agg_function(data_seg, ['bid'], {'departure_time_utc':['min']})
    min_flights['dayweek_of_first_flight'] = min_flights['departure_time_utc_min'].dt.weekday
    min_flights['month_of_first_flight'] = min_flights['departure_time_utc_min'].dt.month
    
    return min_flights[['bid', 'dayweek_of_first_flight', 'month_of_first_flight']]
    

def _airlines_used(data_seg):
    """
    Compute which airlines are in the itinerary
    """

    dummies = pd.get_dummies(data_seg['airline'], prefix='airline')
    dummies['bid'] = data_seg['bid']

    names = []
    for airline in data_seg['airline'].value_counts().index[0:10]:
        names.append(f'airline_{airline}')

    agg_func = {}
    for name in names:
        agg_func[name] = ['sum']

    return _agg_function(dummies, ['bid'], agg_func)


def _count_of_sectors(data_seg):
    """
    computes number of sectors
    """
    
    n_sectors_ds = _agg_function(data_seg, ['bid'], {'sector':['max']})
    
    return n_sectors_ds

def _count_of_vehicles(data_seg):
    """
    computes number of different vehicle types in itinerary
    """
    
    n_vehicles_ds = _agg_function(data_seg, ['bid'], {'vehicle_type':['nunique']})
    
    return n_vehicles_ds


def _count_of_fare_types(data_seg):
    """
    computes number of fare types vehicle types in itinerary
    """
    
    n_fare_types_ds = _agg_function(data_seg, ['bid'], {'fare_category':['nunique']})
    
    return n_fare_types_ds


def _statistics_of_airtime(data_seg):
    """
    Computes min, max, average, sum of time spent in the air
    """
    
    data_seg['time_in_air'] = _compute_daytime_diff(data_seg['arrival_time_utc'], data_seg['departure_time_utc'])

    in_air_ds = _agg_function(data_seg, ['bid'], {'time_in_air':['min','max','mean','sum']})
    
    return in_air_ds

def _number_of_flights_within_sector(data_seg):
    """
    Computes number of flights within a sector 
    This predictor -1 would be number transfers within a sector
    """
    
    data_sector_flights = _agg_function(data_seg[['bid', 'departure_time_utc', 'arrival_time_utc',
          'sector']].drop_duplicates(keep = 'first'), ['bid', 'sector'], {'departure_time_utc':['nunique']})
    
    data_sector_flights = _agg_function(data_sector_flights, ['bid'], {'departure_time_utc_nunique':['min', 'max']})
    data_sector_flights.rename(columns = {"departure_time_utc_nunique_min": "min_number_of_flights_within_sector",
                                          "departure_time_utc_nunique_max": "max_number_of_flights_within_sector"
                                         }, inplace = True)
    
    return data_sector_flights
    
    
def _compute_month_day_of_search(data_it):
    """
    computes month and weekday and hour of flight search
    """
    
    data_it['dayweek_of_search'] = data_it['timestamp_of_search'].dt.weekday
    data_it['month_of_search'] = data_it['timestamp_of_search'].dt.month
    data_it['hour_of_search'] = data_it['timestamp_of_search'].dt.hour
    
    return data_it  

def _detect_types(data):
    """
    Detects numerical and categorical features
    """
    numerical_preds=[]
    categorical_preds=[]
     
    
    for i in list(data):
        if(data[i].dtype=='object'):
            categorical_preds.append(i)
        else:
            numerical_preds.append(i)
    
    return numerical_preds, categorical_preds


def _replace_categories(train_set, test_set, categorical_preds, num_categories):
    """
    For categorical columns with many categories, selects only top n most frequent
    and concats others to one single category
    """
    for i in categorical_preds:
        if train_set[i].nunique()>num_categories:
            top_n_cat=train_set[i].value_counts()[:10].index.tolist()
            train_set[i]=np.where(train_set[i].isin(top_n_cat),train_set[i],'other')   
            test_set[i]=np.where(test_set[i].isin(top_n_cat),test_set[i],'other')
            
    return train_set, test_set


def preprocess_data(data_it, data_seg):
    """
    Compute all features
    """
    
    data_it, data_seg = _convert_times(data_it, data_seg)
    
    ds_features = []
    
    ds_features.append(_time_between_search_first_flight(data_it, data_seg))
    ds_features.append(_time_between_search_last_flight(data_it, data_seg))
    ds_features.append(_time_between_first_last_flight(data_seg))
    ds_features.append(_count_of_different_airlines(data_seg))
    ds_features.append(_compute_month_day_of_first_flight(data_seg))
    ds_features.append(_count_of_sectors(data_seg))
    ds_features.append(_count_of_vehicles(data_seg))
    ds_features.append(_count_of_fare_types(data_seg))
    ds_features.append(_statistics_of_airtime(data_seg))
    ds_features.append(_number_of_flights_within_sector(data_seg))
    ds_features.append(_airlines_used(data_seg))
    
    # merge all dataframes with features
    for ds in ds_features:
        data_it = pd.merge(data_it, ds, how = 'left', left_on = 'bid', right_on = 'bid')
    
    data_it = _compute_month_day_of_search(data_it)
    data_it['time_in_air_sum'] = data_it['time_in_air_sum']/data_it['passengers']

    for col in list(data_it):
        # in case when name of feature is not separable 
        try:
            if ((col.split('_')[0] == 'airline') & (col.split('_')[2] == 'sum')):
                data_it[col] = data_it[col]/data_it['passengers']
        except:
            pass
    
    # creation of binary target
    data_it['fare_type'] = np.where(data_it['fare_type'] == 'saver', 0, 1)
    target = data_it['fare_type']
    
    # baggage
    data_it['is_hold_bags_disabled'] = np.where(data_it['is_hold_bags_disabled'] == True, 1, 0)
    
    
    # remove columns
    cols_to_remove = ['bid', 'timestamp_of_search', 'src', 'dst', 'fare_type']
    
    data_it.drop(cols_to_remove, axis=1, inplace = True)
    
    # detects categorical and numerical columns 
    numerical_preds, categorical_preds = _detect_types(data_it)
    
    for col in categorical_preds:
        data_it[col] = data_it[col].astype('category')
        
    
    # train, test, valid split
    train, test, y_train, y_test = train_test_split( data_it, target,
                                                    test_size=0.25, random_state=134)
    train, valid, y_train, y_valid = train_test_split( train, y_train, test_size=0.3333, random_state=134)
    
    
    # replace low frequented categories within 'partner' and 'market' features
    train, valid = _replace_categories(train, valid, ['partner', 'market'], 10)
    train, test = _replace_categories(train, test, ['partner', 'market'], 10)
    
    return train, valid, test, y_train, y_valid, y_test, numerical_preds, categorical_preds


def graph_exploration(feature_binned,target):
    """
    EDA of categorical variable
    """
  
    if(sum(feature_binned.isnull())>0):
        feature_binned=feature_binned.cat.add_categories('NA').fillna('NA')
    
    result = pd.concat([feature_binned, target], axis=1)
    
    gb=result.groupby(feature_binned)
    counts = gb.size().to_frame(name='counts')
    final=counts.join(gb.agg({result.columns[1]: 'mean'}).rename(columns={result.columns[1]: 'target_mean'})).reset_index()
    final['odds_ratio']=np.log2((final['counts']*final['target_mean']+100*np.mean(target))/((100+final['counts'])*np.mean(target)))
        
    sns.set(rc={'figure.figsize':(15,10)})
    fig, ax =plt.subplots(2,1)
    sns.countplot(x=feature_binned, hue=target, data=result,ax=ax[0])
    sns.barplot(x=final.columns[0],y='odds_ratio',data=final,color="green",ax=ax[1])
    plt.show()
    
    
def graph_exploration_continuous(feature_binned,target):
    """
    EDA of numerical variable
    """
  
    if(sum(feature_binned.isnull())>0):
        feature_binned=feature_binned.cat.add_categories('NA').fillna('NA')
    
    plt.figure(figsize=(12,5))
    sns.boxplot(x=feature_binned,y=target,showfliers=False)
    plt.xticks(rotation='vertical')
    #plt.xlabel(feature_binned, fontsize=12)
    #plt.ylabel(target, fontsize=12)
    plt.show()
