import pandas as pd
from faker import Faker

fake = Faker()

# Number of transactions you want to create
num_transactions = 100000

# List of South African provinces for simplicity
provinces = [
    'Western Cape', 'Eastern Cape', 'Northern Cape', 'Gauteng', 
    'KwaZulu-Natal', 'Limpopo', 'Mpumalanga', 'North West', 
    'Free State'
]

# Create synthetic customer transaction data
data = {
    'TransactionID': range(1, num_transactions + 1),
    'CustomerID': [fake.user_name() for _ in range(num_transactions)],
    'ProductID': [fake.ean8() for _ in range(num_transactions)],
    'PurchaseDate': [fake.date_between(start_date='-5y', end_date='today') for _ in range(num_transactions)],
    'Quantity': [fake.random_int(min=1, max=5) for _ in range(num_transactions)],
    'PricePerUnit': [fake.pyfloat(left_digits=2, right_digits=2, positive=True) for _ in range(num_transactions)],
    'Age': [fake.random_int(min=18, max=60) for _ in range(num_transactions)],
    'Gender': ['Male' if fake.boolean() else 'Female' for _ in range(num_transactions)],
    'Province': [provinces[fake.random_int(min=0, max=len(provinces)-1)] for _ in range(num_transactions)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Simulate repeat customer purchases up to 5 times for 40% of customers
repeat_customers = df['CustomerID'].sample(int(0.5 * num_transactions))
for i, row in df.iterrows():
    if row['CustomerID'] in repeat_customers:
        # Add between 1 and 5 new rows with new dates for each repeat customer
        num_new_records = fake.random_int(min=1, max=10)
        for _ in range(num_new_records):
            new_date = fake.date_between(start_date='-5y', end_date='today')
            
            df.loc[len(df)] = {
                'TransactionID': len(df) + 1,
                'CustomerID': row['CustomerID'],
                'ProductID': fake.ean8(),
                'PurchaseDate': new_date,
                'Quantity': fake.random_int(min=1, max=5),
                'PricePerUnit': fake.pyfloat(left_digits=2, right_digits=2, positive=True),
                'Age': row['Age'],
                'Gender': row['Gender'],
                'Province': row['Province']
            }

# Simulate purchase frequency and adjust TotalAmount accordingly
customer_frequency = {}
for i, row in df.iterrows():
    customer_id = row['CustomerID']
    if customer_id not in customer_frequency:
        customer_frequency[customer_id] = 0
    customer_frequency[customer_id] += 1

df['TotalAmount'] = 0
for i, row in df.iterrows():
    quantity = row['Quantity']
    price_per_unit = row['PricePerUnit']
    frequency = customer_frequency[row['CustomerID']]
    
    # Adjust TotalAmount based on purchase frequency
    if frequency > 5:
        multiplier = fake.pyfloat(left_digits=1, right_digits=2, positive=True, max_value=3)
    elif frequency > 2:
        multiplier = fake.pyfloat(left_digits=1, right_digits=2, positive=True, max_value=2)
    else:
        multiplier = fake.pyfloat(left_digits=1, right_digits=2, positive=True, min_value=0.5, max_value=1.5)
    
    df.at[i, 'TotalAmount'] = quantity * price_per_unit * multiplier

# Save to CSV
df.to_csv(r'data\customer_transaction_data.csv', index=False)

print("Customer transaction data file created successfully.")
