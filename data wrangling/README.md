# Data Wrangling
The dataset contains missing values on the product description column and the user id column. Since we are trying to predict each users next order based on his or her historical orders, we have to remove rows without user information. Each product appears multiple times on the table. After further review, each unique product has a description.

After removing missing data, I took the following steps to further clean up the dataset:
* The original table contains canceled products whose quantities are negative. For each product in each order, I subtracted the number of canceled items from the number of original orders.
* I removed rows recording non-product transactions such as bank charges, postages etc.
* Some product ids contain strings. I mapped all product ids to integers.
* I combined duplicated products in each order.
* Some users have limited order history and it will be difficult to make a prediction for them. I removed all users with less than 5 orders.

After cleaning up the dataset, I divided it into several relational data frames to make modeling easier. They are:
* The orders data frame which contains user ids, order ids and order date and time. I created several other features based on the order date and time including the order number for each user, order day of week, order hour of day, and days since prior order.
* The products data frame which contains product id and description.
* The customers data frame which contains user id and country.
* The order_products data frame which contains order id, product id, quantity and unit price. I created new features including the add to cart order and whether a product has been ordered before
* The order_products data frame which shows what products each order contains.

Finally I divided all orders into 3 evaluation sets:
* The prior set which contains all orders prior to the last orders for each user
* The train set which contains 80% of the last orders for each user
* The test set which contains 20% of the last orders for each user
