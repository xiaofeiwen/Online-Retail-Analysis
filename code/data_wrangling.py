import pandas as pd
import numpy as np
np.random.seed(42)

# read data
df_original = pd.read_excel("../data/Online Retail.xlsx")

# remove rows without a customer id
df_clean = df_original[df_original.CustomerID.notnull()].copy()

# convert columns to appropriate data types
df_clean.CustomerID = df_clean.CustomerID.astype(int)
df_clean.StockCode = df_clean.StockCode.apply(str)

# remove canceled items
canceled = df_clean[df_clean.Quantity < 0]
canceled = canceled[['CustomerID','StockCode','Quantity']]
df_clean = df_clean.merge(canceled,how='left',on=['CustomerID','StockCode'],suffixes=('', '_c'))
df_clean.Quantity_c = df_clean.Quantity_c.fillna(0)
df_clean.Quantity = df_clean.Quantity + df_clean.Quantity_c
df_clean.drop('Quantity_c',axis=1,inplace=True)

# remove non-product rows
df_clean = df_clean[(df_clean.Quantity > 0) & (~df_clean.StockCode.isin(['BANK CHARGES', 'C2', 'DOT', 'PADS', 'POST','M']))]

# map stock code to new stock code
unique_StockCode = df_clean.StockCode.apply(str).sort_values().unique()
new_StockCode_df = pd.DataFrame({'StockCode':unique_StockCode,'NewStockCode':np.arange(len(unique_StockCode))})
new_StockCode_df.NewStockCode = new_StockCode_df.NewStockCode.astype(int)
df_clean = df_clean.merge(new_StockCode_df,how = 'left',on='StockCode')

# remove customers with less then 4 orders
num_invoices = df_clean.groupby('CustomerID')['InvoiceNo'].unique().apply(len)
customers_to_use = num_invoices[num_invoices > 3].index
df_clean = df_clean[df_clean.CustomerID.isin(customers_to_use)]

# combine duplicated products in each invoice
df_clean.Quantity = df_clean.Quantity.astype(int)
sum_Quantity = pd.DataFrame(df_clean.groupby(['InvoiceNo','NewStockCode'])['Quantity'].sum())
df_clean = df_clean.set_index(['InvoiceNo','NewStockCode']).\
        merge(sum_Quantity,left_index=True,right_index=True,suffixes=('_', '')).reset_index()
df_clean.drop_duplicates(['InvoiceNo','NewStockCode'],inplace=True)
df_clean.drop('Quantity_',axis=1,inplace=True)

# extract products and customers to seperate tables
products = df_clean[['NewStockCode','Description']].sort_values(['NewStockCode','Description']).\
            drop_duplicates('NewStockCode').reset_index(drop=True)
customers = df_clean.sort_values('CustomerID')[['CustomerID','Country']].drop_duplicates('CustomerID').reset_index(drop=True)

# extract orders to a seperate table
orders = df_clean[['CustomerID','InvoiceNo','InvoiceDate']].sort_values(['CustomerID','InvoiceDate']).\
            drop_duplicates('InvoiceNo')

# create new features based on the InvoiceDate feature
orders['order_number'] = orders.groupby('CustomerID').cumcount() + 1
orders['order_dow'] = orders.InvoiceDate.dt.dayofweek
orders['order_hour_of_day'] = orders.InvoiceDate.dt.hour
orders['days_since_prior_order'] = (orders.InvoiceDate - orders.groupby('CustomerID')['InvoiceDate'].shift(1)).dt.days

# split all orders into 'prior', 'train' and 'test'
last_orders = orders.groupby('CustomerID')['order_number'].max().reset_index()
n_customers = last_orders.shape[0]
split_point = int(np.ceil(n_customers * 0.8))
random_index = np.random.choice(n_customers,n_customers,replace=False)
train_index = random_index[:split_point]
test_index = random_index[split_point:]
last_orders['eval_set'] = 'placeholder'
last_orders['eval_set'].iloc[train_index] = 'train'
last_orders['eval_set'].iloc[test_index] = 'test'
orders = orders.merge(last_orders,how='left',on=['CustomerID','order_number'])
orders.eval_set = orders.eval_set.fillna('prior')

# create a new table to show what products are in each order
# and calculate add to cart order
order_products = df_clean[['InvoiceNo','NewStockCode','Quantity','UnitPrice']].copy()
order_products['add_to_cart_order'] = order_products.groupby('InvoiceNo').cumcount() + 1
order_products = order_products.merge(orders[['InvoiceNo','eval_set']],how='left',on='InvoiceNo')

d = {}
for row in order_products.itertuples():
    InvoiceNo = row.InvoiceNo
    try:
        d[InvoiceNo] += ' ' + str(row.NewStockCode)
    except:
        d[InvoiceNo] = str(row.NewStockCode)

order_products_compact = pd.DataFrame.from_dict(d, orient='index')

order_products_compact.reset_index(inplace=True)
order_products_compact.columns = ['InvoiceNo', 'NewStockCode']
order_products_compact.NewStockCode = order_products_compact.NewStockCode.str.split()
order_products_compact.sort_values('InvoiceNo',inplace=True)
order_products_compact = order_products_compact.merge(orders[['InvoiceNo','eval_set']],how='left',on='InvoiceNo')

# calculate whether a product has been ordered before
# takes a long time
def previous_orders(InvoiceNo):
    row = orders[orders.InvoiceNo == InvoiceNo]
    CustomerID = int(row.CustomerID)
    order_number = int(row.order_number)
    prev_ord_nums = range(1,order_number)
    df = orders[(orders.CustomerID == CustomerID)&(orders.order_number.isin(prev_ord_nums))]
    return df.InvoiceNo.values

def previous_items(InvoiceNo):
    prev_ord = previous_orders(InvoiceNo)
    df = order_products[order_products.InvoiceNo.isin(prev_ord)]
    return df.NewStockCode.unique()

prev_item_dict = {}
count = 0
for inv_no in orders.InvoiceNo.values:
    prev_item_dict[inv_no] = previous_items(inv_no)

order_products['reordered'] = -1
for i in range(order_products.shape[0]):
    row = order_products.iloc[i]
    prev_items = prev_item_dict[row.InvoiceNo]
    order_products.iloc[i,-1] = int(row.NewStockCode in prev_items)

# rename columns to make them cleaner and easier to understand
df_clean.columns = ['order_id', 'product_id', 'product_id_original', 'description', 'order_date',
                   'unit_price', 'user_id', 'country', 'quantity']
orders.columns = ['user_id', 'order_id', 'order_date', 'order_number', 'order_dow',
                   'order_hour_of_day', 'days_since_prior_order', 'eval_set']
products.columns = ['product_id', 'description']
customers.columns = ['user_id', 'country']
order_products.columns = ['order_id', 'product_id', 'quantity', 'unit_price', 'add_to_cart_order', 'eval_set', 'reordered']
order_products_compact.columns = ['order_id', 'product_id', 'eval_set']

# save data to hdf
df_clean.to_hdf('../data/online_retail.h5','clean')
orders.to_hdf('../data/online_retail.h5','orders')
products.to_hdf('../data/online_retail.h5','products')
customers.to_hdf('../data/online_retail.h5','customers')
order_products.to_hdf('../data/online_retail.h5','order_products')
order_products_compact.to_hdf('../data/online_retail.h5','order_products_compact')






