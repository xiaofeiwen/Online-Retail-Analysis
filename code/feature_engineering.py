import numpy as np
import pandas as pd
np.random.seed(42)

# read data
all_data = pd.read_hdf('../data/online_retail.h5','clean')
orders = pd.read_hdf('../data/online_retail.h5','orders')
products = pd.read_hdf('../data/online_retail.h5','products')
customers = pd.read_hdf('../data/online_retail.h5','customers')
order_products = pd.read_hdf('../data/online_retail.h5','order_products')
order_products_compact = pd.read_hdf('../data/online_retail.h5','order_products_compact')
priors = order_products[order_products.eval_set == 'prior'].copy()
train = order_products[order_products.eval_set == 'train'].copy()
test = order_products[order_products.eval_set == 'test'].copy()

# product features
priors['revenue'] = priors['unit_price'] * priors['quantity']
priors_orders = orders.merge(priors, on='order_id')
priors_orders.loc[:,'_user_buy_product_times'] = priors_orders.groupby(['user_id', 'product_id']).cumcount() + 1
prods = pd.DataFrame()
prods['orders'] = priors_orders.groupby('product_id').size()
prods['reorders'] = priors_orders.groupby('product_id')['reordered'].sum()
prods['reorder_rate'] = (prods.reorders / prods.orders)
prods['total_quantity'] = priors_orders.groupby('product_id')['quantity'].sum()
prods['total_revenue'] = priors_orders.groupby('product_id')['revenue'].sum()
prods['avg_price'] = prods['total_revenue'] / prods['total_quantity']
prods['prod_first_buy'] = priors_orders.groupby('product_id')['_user_buy_product_times'].agg(lambda x: sum(x==1))
prods['prod_second_buy'] = priors_orders.groupby('product_id')['_user_buy_product_times'].agg(lambda x: sum(x==2))
prods['prod_1reorder_ratio'] = prods.prod_second_buy / prods.prod_first_buy
prods['prod_nreorder_ratio'] = prods.reorders / prods.prod_first_buy
products = products.join(prods, on='product_id')
products.set_index('product_id', drop=False, inplace=True)
products.fillna(0,inplace=True)

# user features
prior_order_size = priors.groupby('order_id').size().reset_index()
prior_order_size.columns = ['order_id','order_size']
prior_order_size = orders.merge(prior_order_size,on='order_id')
usr = pd.DataFrame()
usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean()
usr['sum_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].sum()
usr['nb_orders'] = orders.groupby('user_id')['order_number'].max()
usr['order_size_mean'] = prior_order_size.groupby('user_id')['order_size'].mean()
usr['order_size_std'] = prior_order_size.groupby('user_id')['order_size'].std()
usr['order_size_max'] = prior_order_size.groupby('user_id')['order_size'].max()

users = pd.DataFrame()
users['total_items'] = priors_orders.groupby('user_id').size()
users['all_products'] = priors_orders.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.all_products.map(len))
users['total_item_quantity'] = priors_orders.groupby('user_id')['quantity'].sum()
users['total_money_spent'] = priors_orders.groupby('user_id')['revenue'].sum()
users['avg_money_spent_per_item'] = users['total_money_spent'] / users['total_item_quantity']
users['user_reorder_ratio'] = priors_orders.groupby('user_id')['reordered'].sum().\
                              divide(priors_orders.groupby('user_id')['order_number'].agg(lambda x: sum(x>1)))
users = users.join(usr)

# user-prodcut interaction features
priors_orders['user_product'] = priors_orders.product_id + priors_orders.user_id * 100000
d= dict()
for row in priors_orders.itertuples():
    z = row.user_product
    if z not in d:
        d[z] = (1,
                (row.order_number, row.order_id),
                row.add_to_cart_order)
    else:
        d[z] = (d[z][0] + 1,
                max(d[z][1], (row.order_number, row.order_id)),
                d[z][2] + row.add_to_cart_order)

userXproduct = pd.DataFrame.from_dict(d, orient='index')
userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
userXproduct.nb_orders = userXproduct.nb_orders.astype(np.int16)
userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1])
# userXproduct.sum_pos_in_cart = userXproduct.sum_pos_in_cart
up_temp = pd.DataFrame()
up_temp['up_total_quantity'] = priors_orders.groupby('user_product')['quantity'].sum()
up_temp['up_total_spent'] = priors_orders.groupby('user_product')['revenue'].sum()
up_temp['up_first_order_number'] = priors_orders.groupby('user_product')['order_number'].min()
up_temp['up_last_order_number'] = priors_orders.groupby('user_product')['order_number'].max()
userXproduct = userXproduct.join(up_temp)

# train test split
test_orders = orders[orders.eval_set == 'test']
train_orders = orders[orders.eval_set == 'train']
train.set_index(['order_id', 'product_id'], inplace=True, drop=False)

# data transformation
order_list = []
product_list = []
labels = []
last_orders = set(zip(train.order_id.values,train.product_id.values))
for row in train_orders.itertuples():
    order_id = row.order_id
    user_id = row.user_id
    user_products = list(products.product_id.values)
    product_list += user_products
    order_list += [order_id] * len(user_products)
    labels += [(order_id, product) in last_orders for product in user_products]
df = pd.DataFrame({'order_id':order_list, 'product_id':product_list, 'labels':labels}, dtype=np.int32)

orders.set_index('order_id',inplace=True)
print('user related features')
df['user_id'] = df.order_id.map(orders.user_id)
df['user_total_orders'] = df.user_id.map(users.nb_orders)
df['user_total_items'] = df.user_id.map(users.total_items)
df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
df['user_order_size_mean'] =  df.user_id.map(users.order_size_mean)
df['user_order_size_std'] =  df.user_id.map(users.order_size_std)
df['user_order_size_max'] =  df.user_id.map(users.order_size_max)
df['user_total_item_quantity'] = df.user_id.map(users.total_item_quantity)
df['user_total_spent'] = df.user_id.map(users.total_money_spent)
df['user_sum_days_between_orders'] = df.user_id.map(users.sum_days_between_orders)
df['user_reorder_ratio'] = df.user_id.map(users.user_reorder_ratio)


print('order related features')
# df['dow'] = df.order_id.map(orders.order_dow)
df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
df['order_dow'] = df.order_id.map(orders.order_dow)
df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
df['days_since_ratio'] = df.days_since_prior_order / (df.user_average_days_between_orders+.01)

print('product related features')
df['product_orders'] = df.product_id.map(products.orders)
df['product_reorders'] = df.product_id.map(products.reorders)
df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)
df['product_total_quantity_sold'] = df.product_id.map(products.total_quantity)
df['product_avg_price'] = df.product_id.map(products.avg_price)
df['prod_first_buy'] = df.product_id.map(products.prod_first_buy)
df['prod_second_buy'] = df.product_id.map(products.prod_second_buy)
df['prod_1reorder_ratio'] = df.product_id.map(products.prod_1reorder_ratio)
df['prod_nreorder_ratio'] = df.product_id.map(products.prod_nreorder_ratio)

print('user_X_product related features')
df['z'] = df.user_id * 100000 + df.product_id
# df.drop(['user_id'], axis=1, inplace=True)
df['UP_orders'] = df.z.map(userXproduct.nb_orders)
df['UP_orders_ratio'] = df.UP_orders / df.user_total_orders
# df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
# df['UP_average_pos_in_cart'] = df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders
df['UP_reorder_rate'] = df.UP_orders / df.user_total_orders
# df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
df['UP_total_quantity'] = df.z.map(userXproduct.up_total_quantity)
df['UP_first_order_number'] = df.z.map(userXproduct.up_first_order_number)
df['UP_order_rate_since_first_order'] = df.UP_orders / (df.user_total_orders - df.UP_first_order_number + 1)

df.drop(['z','UP_first_order_number'], axis=1, inplace=True)
df.fillna(0,inplace=True)

# save datadf.to_pickle('../data/train_transformed.p')
df.to_pickle('../data/train_transformed.p')
products.to_pickle('../data/product_features.p')
users.to_pickle('../data/user_features.p')
















