import pandas as pd
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
    
    def calculate_rfm(self, snapshot_date: str, window: pd.Timedelta,df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics for customer segmentation.

        Parameters:
            snapshot_date (str): date to calculate recency from (YYYY-MM-DD).
        
        Returns:
            pd.DataFrame: RFM metrics per CustomerID.
        """
        data = df.copy()
        snapshot = pd.to_datetime(snapshot_date)
        data['date'] = pd.to_datetime(data.date)
        data  = data[(data.date<snapshot) & (data.date>(snapshot-window))]
        data['recency'] = (snapshot - pd.to_datetime(data['date'])).dt.days
        rfm = data.groupby('CustomerID').agg({
            'recency': 'min',
            'TransactionID': 'count',               # frequency: number of transactions
            'Amount': 'sum'             # monetary: total spend
        }).reset_index()

        rfm.rename(columns={'TransactionID': 'frequency', 'Amount': 'monetary'}, inplace=True)
        rfm['date'] = snapshot_date

        return rfm[['CustomerID', 'date', 'recency', 'frequency', 'monetary']]
    
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
    
    def rfm_segments(self,date: str, window: pd.Timedelta, df: pd.DataFrame) -> pd.DataFrame:
        """
        Segments customers based on their RFM score.

        Parameters:
            date (str): date to calculate recency from (YYYY-MM-DD).
            df (pd.DataFrame): Raw transactional data.
        
        Returns:
            pd.DataFrame: RFM segments per CustomerID.
        """
        data = df.copy()
        data = self.calculate_rfm(date,window,data)
        # Create RFM scores
        data['r_score'] = pd.qcut(data['recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
        data['f_score'] = pd.qcut(data['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        data['m_score'] = pd.qcut(data['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        data['rfm_score'] = data['r_score'].astype(str) + \
                            data['f_score'].astype(str) + \
                            data['m_score'].astype(str)
        data['rfm_score_int'] = data['r_score'].astype(int) + \
                                data['f_score'].astype(int)+ \
                                data['m_score'].astype(int)           
        #create a dicitionary with the rfm_score that maps into segment, the key must be a regex
        segment_map = {
                        r'^55[1-5]$': 'Champions',
                        r'^54[1-5]$': 'Loyal Customers',
                        r'^45[1-5]$': 'Potential Loyalists',
                        r'^53[1-5]$': 'New or Returning Customers',
                        r'^33[1-5]$': 'Promising',
                        r'^22[1-5]$': 'Needs Attention',
                        r'^\d{3}$': 'Others'  # Matches any other 3-digit score
                    }
        data['segment'] = data['rfm_score'].replace(segment_map, regex=True)
        data['date'] = date
        return data[['CustomerID', 'date', 'rfm_score','rfm_score_int', 'segment']]
    
    def dedup_demographic_variables(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate demographic variables for each customer, keeping the latest entry.
        
        Parameters:
            df (pd.DataFrame): Raw transactional data with demographic information.
        
        Returns:
            pd.DataFrame: Deduplicated demographic data per CustomerID.
        """
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'],format='%m-%d-%Y %H:%M:%S')
        
        # Sort by CustomerID and date to get the most recent demographic info
        data = data.sort_values(by=['customer_id', 'date'], ascending=True)
        
        # Drop duplicates, keeping the last (most recent) entry for each CustomerID
        # Assuming demographic variables are 'Gender', 'Age', 'Age Group', extend to the actual list
        demographic_cols = ['customer_id', 'Gender', 'Age','Province']
        deduplicated_demographics = data[demographic_cols].drop_duplicates(subset=['CustomerID'], keep='last')
        
        return deduplicated_demographics
    
    def transaction_descriptor_variables(self,date) -> pd.DataFrame:
        """
        Caclulates mode transaction dimensions purchased from each customer
        Parameters:
            date (str): date to calculate recency from (YYYY-MM-DD).
        Returns:
            pd.DataFrame: Most frequent transaction descriptors per CustomerID.
        """

        data = self.fetch_data()
        data['date'] = pd.to_datetime(data['date'])
        purchases = data[data.date < date]
        summary = purchases.groupby('customer_id')[['ProductCategory','PurchaseChannel',
                                        'PaymentMethod','Store']].agg(pd.Series.mode).reset_index()
        summary['date'] = date
        # Rename columns for clarity
        summary.rename(columns={'ProductCategory': 'Most_frequented_Category',
                                  'PurchaseChannel': 'Most_frequented_Channel',
                                  'PaymentMethod': 'Most_used_payment_method',
                                  'Store': 'Most_frequented_Store'}, inplace=True)
        return summary[['customer_id','date','Most_frequented_Channel','Most_frequented_Category','Most_used_payment_method','Most_frequented_Store']]
    
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

if __name__=='__main__':
    loader = DataLoader('data/customer_transaction_data.csv','CustomerID','TransactionID','PurchaseDate','TotalAmount')
    orig_data = loader.fetch_data()
    rfm = loader.calculate_rfm('2023-07-31',pd.Timedelta(days=99999),orig_data)
    print(rfm.head())
    demographics = loader.dedup_demographic_variables(orig_data)
    print(demographics.head())
    additional_features = loader.additional_features(orig_data,'2023-07-31')
    print(additional_features.head())
    
