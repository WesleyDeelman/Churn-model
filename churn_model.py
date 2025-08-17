import category_encoders as ce
import pandas as pd
import lightgbm as lgb
import optuna
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path
import numpy as np

class DataLoader:
    
    def __init__(self,path,customer_id,transaction_id,transaction_date,amount):
        """
        Initialize the DataLoader.

        Parameters:
            path (str): Path to the CSV file.
            customer_id (str): Column name for customer IDs.
            transaction_id (str): Column name for transaction IDs.
            transaction_date (str): Column name for transaction dates.
            amount (str): Column name for transaction amounts.
        
        """
        self.path = path
        self.customer_id = customer_id
        self.transaction_id = transaction_id
        self.transaction_date = transaction_date 
        self.amount = amount 
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch raw CSV data from the specified path.

        Parameters:
            path (str): Path to the CSV file.
        
        Returns:
            pd.DataFrame: Loaded data.
        """
        data_path = Path(self.path)
        if not data_path.exists():
            raise FileNotFoundError(f"CSV file not found at: {data_path}")
        data = pd.read_csv(data_path)
        data.rename(columns={self.customer_id:'customer_id',
                             self.transaction_id: 'transaction_id',
                             self.transaction_date: 'date',
                             self.amount: 'amount'}, inplace=True)
        data['date'] = pd.to_datetime(data['date'],format='%Y/%m/%d')
        return data
    
    def calculate_target(self,date: str, window_size: pd.Timedelta, \
                        repurchase_threshold: float, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculating the target for the model
        Parameters:
            date (str): date to calculate recency from (YYYY-MM-DD).
            window_size (pd.Timedelta): Number of days in which to consider repurchase
            repurchase_threshold: Minimum spend to be considered a repurchase
        Returns:
            pd.DataFrame: Target per CustomerID.
        """
        #caculate whether a customer made total purchases after date exceeding repurchase threshold
        data = df.copy()
        data['date '] = pd.to_datetime(data['date'])
        date = pd.to_datetime(date)
        future_purchases = data[(data['date'] > date) & \
                                (data['date'] <= date + window_size)]

        target_customers = future_purchases.groupby('customer_id')[['amount']].sum()
        target_customers = target_customers[target_customers >= repurchase_threshold].index

        total_future_purchases = future_purchases.groupby('customer_id')[['amount']].sum()
        data = data.merge(total_future_purchases.rename(columns={'amount': 'subsequent_purchases'}),
                            on='customer_id', how='left')
        data['subsequent_purchases'] = data['subsequent_purchases'].fillna(0)

        data['target'] = data['customer_id'].isin(target_customers).astype(int)
        data['date'] = date
        
        return data[['customer_id','date','target','subsequent_purchases']].drop_duplicates()
    
    def generate_churn_features(self,df, cutoff_date=None, recent_days=[30, 60, 90]):
            """
            Generate churn features from transaction data.
            
            Parameters:
            -----------
            df : pd.DataFrame
                Must have columns: 'transaction_id', 'customer_id', 'date', 'amount'
            cutoff_date : str or pd.Timestamp, optional
                If provided, features are calculated up to this date (avoiding leakage)
            recent_days : list of int
                Windows (in days) for recency/frequency/monetary calculations
            
            Returns:
            --------
            features_df : pd.DataFrame
                One row per customer with churn features
            """
            
            # Ensure correct types
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            if cutoff_date:
                cutoff_date = pd.to_datetime(cutoff_date)
                df = df[df['date'] <= cutoff_date]
            else:
                cutoff_date = df['date'].max()

            # Sort for easier calculations
            df = df.sort_values(['customer_id', 'date'])
            
            features = []

            for cust_id, group in df.groupby('customer_id'):
                group = group.sort_values('date')
                dates = group['date'].values
                amounts = group['amount'].values
                
                first_date = group['date'].min()
                last_date = group['date'].max()
                tenure_days = (last_date - first_date).days
                recency_days = (cutoff_date - last_date).days
                
                # Gaps
                if len(group) > 1:
                    gaps = np.diff(group['date']).astype('timedelta64[D]').astype(int)
                    avg_gap = np.mean(gaps)
                    max_gap = np.max(gaps)
                    std_gap = np.std(gaps)
                else:
                    avg_gap = max_gap = std_gap = np.nan

                # Monetary stats
                total_spend = amounts.sum()
                mean_spend = amounts.mean()
                median_spend = np.median(amounts)
                max_spend = amounts.max()
                min_spend = amounts.min()
                std_spend = np.std(amounts)
                cv_spend = std_spend / mean_spend if mean_spend != 0 else np.nan
                skew_spend = pd.Series(amounts).skew()

                # Frequency stats
                total_txn = len(group)
                txn_per_month = total_txn / max((tenure_days / 30.0), 1)

                # Day-of-week & weekend ratio
                dow_mode = group['date'].dt.dayofweek.mode()[0]  # 0=Monday
                weekend_ratio = np.mean(group['date'].dt.dayofweek >= 5)

                # Recent windows
                recent_feats = {}
                for nd in recent_days:
                    mask_recent = group['date'] >= (cutoff_date - pd.Timedelta(days=nd))
                    spend_recent = amounts[mask_recent].sum()
                    freq_recent = mask_recent.sum()
                    recent_feats[f'spend_last_{nd}d'] = spend_recent
                    recent_feats[f'freq_last_{nd}d'] = freq_recent

                # Ratio features (recent vs older)
                spend_last30 = recent_feats.get('spend_last_30d', 0)
                spend_prev30 = total_spend - spend_last30
                freq_last30 = recent_feats.get('freq_last_30d', 0)
                freq_prev30 = total_txn - freq_last30

                spend_change_30 = (spend_last30 - spend_prev30) / spend_prev30 if spend_prev30 != 0 else np.nan
                freq_change_30 = (freq_last30 - freq_prev30) / freq_prev30 if freq_prev30 != 0 else np.nan

                # Lifecycle flags
                is_new = int((cutoff_date - first_date).days <= 30)
                is_loyal = int(tenure_days > 180 and total_txn >= np.percentile(df.groupby('customer_id').size(), 50))
                at_risk = int((recency_days > 30) and (freq_last30 == 0))

                features.append({
                    'customer_id': cust_id,
                    'tenure_days': tenure_days,
                    'recency_days': recency_days,
                    'avg_gap_days': avg_gap,
                    'max_gap_days': max_gap,
                    'std_gap_days': std_gap,
                    'total_spend': total_spend,
                    'mean_spend': mean_spend,
                    'median_spend': median_spend,
                    'max_spend': max_spend,
                    'min_spend': min_spend,
                    'std_spend': std_spend,
                    'cv_spend': cv_spend,
                    'skew_spend': skew_spend,
                    'total_txn': total_txn,
                    'txn_per_month': txn_per_month,
                    'dow_mode': dow_mode,
                    'weekend_ratio': weekend_ratio,
                    **recent_feats,
                    'spend_change_30': spend_change_30,
                    'freq_change_30': freq_change_30,
                    'is_new': is_new,
                    'is_loyal': is_loyal,
                    'at_risk': at_risk
                })

            features_df = pd.DataFrame(features)
            return features_df
    

#create a clean and simple class for optimising lgbm in python
class LGBMOptimizer:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def objective(self, trial):

        params = {
            'objective': 'binary',
            'verbose':-1,
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
            'random_state': 42,
            'n_jobs': -1,
        }

        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        # Create datasets for training and validation
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        callbacks = [lgb.early_stopping(100)]
        # Initialize the model with native API
        model = lgb.train(params,
                        train_set=train_data,
                        valid_sets=[val_data],
                        callbacks=callbacks)
        
        y_pred_proba = model.predict(X_val)
        best_auc = 0.5
        auc = roc_auc_score(y_val, y_pred_proba)
        if best_auc < auc:
            best_auc = auc
        print(f'The best AUC up to now is: {best_auc:.2f}')

        return auc

    def optimize(self, n_trials=50):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        return trial.params

CONFIG = {'snapshot_date':'2023/01/01'}  

loader = DataLoader(r'data\customer_transaction_data_reduced.csv','CustomerID','TransactionID','PurchaseDate','TotalAmount')
data  = loader.fetch_data()
target = loader.calculate_target(CONFIG['snapshot_date'],pd.Timedelta(weeks=56),0,data)
additional_features = loader.generate_churn_features(data,CONFIG['snapshot_date'])
data= data.merge(additional_features,on=['customer_id'],how='left').drop('date',axis=1)
data = data.merge(target.drop(['date','subsequent_purchases'],axis=1),on=['customer_id'],how='left')

# Prepare data for modeling
# Drop duplicates based on customer_id to ensure one row per customer for features
model_data = data.drop_duplicates(subset=['customer_id']).copy()
X = model_data.drop(['target'],axis=1).set_index('customer_id')
print(col for col in X.columns)
y = model_data[['target']]
# Handle potential NaN values introduced by feature engineering (e.g., std_spend if only one transaction)
X = X.fillna(X.mean())

# Initialize and run the optimizer
optimizer = LGBMOptimizer(X, y)
best_params = optimizer.optimize(n_trials=50)

# Train the final model with the best parameters
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = lgb.Dataset(X_train, label=y_train)
X_test = lgb.Dataset(X_test, label=y_test)
 

#Build the final model
final_model=lgb.train(best_params, train_set=lgb.Dataset(X), num_boost_round=1000)

y_pred = np.int16(final_model.predict(X)>=0.5)
y_pred_proba = final_model.predict(X)


