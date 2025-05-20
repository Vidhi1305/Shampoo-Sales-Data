# Databricks notebook source
# MAGIC %pip install tqdm
# MAGIC %pip install pandas

# COMMAND ----------

import pandas as pd

# COMMAND ----------

path ="dbfs:/FileStore/tables/"
file ="sales_data.csv"
data = spark.read.csv(path+file,header=True, inferSchema=True)
data.display()

# COMMAND ----------

a = data.toPandas()
a

# COMMAND ----------

print(a.columns)

# COMMAND ----------

a = a.rename(columns={'Year': 'year'})
a = a.rename(columns={'week_number': 'weeknumber'})
a = a.rename(columns={'Product': 'product'})
a = a.rename(columns={'Price': 'price'})
a = a.rename(columns={'On_Flyer': 'onflyer'})
a = a.rename(columns={' Units ': 'units'})
a = a.rename(columns={'Sales ': 'sales'})
a = a.rename(columns={'Gross Margin ': 'grossmargin'})
a = a.rename(columns={' # Transactions that contained the product ': 'transactionproduct'})
a

# COMMAND ----------

#1.	What price point is most effective at maximizing sales?
import numpy as np
max_sales=a.groupby(['price','product','year']).agg({'sales':np.sum,'weeknumber':np.count_nonzero}).reset_index()
max_sales['sale_per_week']=max_sales['sales']/max_sales['weeknumber']
max_sales=max_sales.sort_values(by=['product','year','price',],ascending=True)
spark_max_sales = spark.createDataFrame(max_sales)
spark_max_sales.display()



# COMMAND ----------

# 2. What price point is most effective at maximizing gross margin?
import numpy as np
max_grossmargin=a.groupby(['price','product','year']).agg({'grossmargin':np.sum,'weeknumber':np.count_nonzero}).reset_index()
max_grossmargin['grossmargin_per_week']=max_grossmargin['grossmargin']/max_grossmargin['weeknumber']
max_grossmargin=max_grossmargin.sort_values(by=['product','year','price'],ascending=False)
spark_max_grossmargin = spark.createDataFrame(max_grossmargin)
spark_max_grossmargin.display()


# COMMAND ----------

# 3. Is Shampoo seasonal (influenced by time of year)? Explain why or why not.
import numpy as np
max_sales=a.groupby(['product','year','weeknumber']).agg({'units':np.sum}).reset_index()
max_sales=max_sales.sort_values(by=['product','year','weeknumber'],ascending=True)
spark_max_sales = spark.createDataFrame(max_sales)
spark_max_sales.display()


# COMMAND ----------

# 4. What is the cost per unit of each product?
import numpy as np
a['cost_per_unit']=(a['sales']-a['grossmargin'])/a['units']
cost_unit = a.groupby(['product','year']).agg({'cost_per_unit':np.mean}).reset_index()
spark_cost_unit = spark.createDataFrame(cost_unit)
spark_cost_unit.display()


# COMMAND ----------

# 5. How would Pantene perform for units, sales and margin with a 25% discount?
import numpy as np
filterdata = a[a['product'] == "Pantene"]
filterdata['discounted_price']=filterdata['price']*0.75
filterdata['sales_at_discountedprice']=filterdata['discounted_price']*filterdata['units']
filterdata['new_grossmargin_at_discountedprice']=(filterdata['discounted_price']-filterdata['cost_per_unit'])*filterdata['units']
grouped = filterdata.groupby('price').agg({'discounted_price':'first','units':np.sum,'sales_at_discountedprice':np.sum,'new_grossmargin_at_discountedprice':np.sum,}).reset_index()
spark_grouped = spark.createDataFrame(grouped)
spark_grouped.display()


# COMMAND ----------

# 6. How would Pantene perform for units, sales and margin with a 60% discount?
import numpy as np
filterdata = a[a['product'] == "Pantene"]
filterdata['discounted_price']=filterdata['price']*0.40
filterdata['sales_at_discountedprice']=filterdata['discounted_price']*filterdata['units']
filterdata['new_grossmargin_at_discountedprice']=(filterdata['discounted_price']-filterdata['cost_per_unit'])*filterdata['units']
grouped = filterdata.groupby('price').agg({'discounted_price': 'first','units':np.sum,'sales_at_discountedprice': np.sum,    'new_grossmargin_at_discountedprice':np.sum}).reset_index()
spark_grouped = spark.createDataFrame(grouped)
spark_grouped.display()

# COMMAND ----------

# 7. What impact does being “On Flyer” have on performance?
import numpy as np
performance=a.groupby(['onflyer','product']).agg({'units':np.sum,'sales':np.sum,'weeknumber':np.count_nonzero}).reset_index()
performance['sale_per_week_onflyer']=performance['sales']/performance['weeknumber']
performance_onflyer=performance.sort_values(by=['product','sale_per_week_onflyer'],ascending=False)
spark_performance = spark.createDataFrame(performance_onflyer)
spark_performance.display()

# COMMAND ----------

# 8. Your director wants to change the price on an upcoming Aussie Shampoo flyer promotion. Her goal is to maximize sales, but she does not want to sacrifice too much margin. 
# a.  How would you present the data to help her make the decision?
# b.  What price would you recommend?
import numpy as np
filterdata= a[a['product']=='Aussie'] 
performance=filterdata.groupby(['price','product']).agg({'units':np.sum,'sales':np.sum,'grossmargin':np.sum,'weeknumber':np.count_nonzero}).reset_index()
performance['sale_per_week']=performance['sales']/performance['weeknumber']
performance['grossmargin_per_week']=performance['grossmargin']/performance['weeknumber']
performance_price=performance.sort_values(by='sale_per_week',ascending=False)
spark_performance = spark.createDataFrame(performance_price)
spark_performance.display()

# COMMAND ----------

# 9. Aussie Shampoo sold at $2.49 is a “loss leader” promotion. We lose money selling it at this price, but hope that people who came to buy it will purchase other items.
# i.  Is Aussie @ $2.49 an effective loss leader? Explain why or why not.
import numpy as np
filterdata= a[a['product']=='Aussie']
performance=filterdata.groupby(['price','product']).agg({'units':np.sum,'sales':np.sum,'grossmargin':np.sum,'weeknumber':np.count_nonzero,'transactionproduct':np.sum}).reset_index()
performance['per_week_grossmargin']=performance['grossmargin']/performance['weeknumber']
performance['per_week_transactionproduct']=performance['transactionproduct']/performance['weeknumber']
performance_price=performance.sort_values(by='price',ascending=True)
spark_performance = spark.createDataFrame(performance_price)
spark_performance.display()

# COMMAND ----------

# ii.  Your director proposes to change the promotion to 2 for $5 or pay 2.99 each, hoping that this will improve margin. Will this work? Explain why or why not.
import numpy as np
filterdata= a[(a['product']=='Aussie') & (a['price'].isin([2.49,2.99]))]
performance=filterdata.groupby(['price','product']).agg({'units':np.sum,'sales':np.sum,'grossmargin':np.sum,'weeknumber':np.count_nonzero,'transactionproduct':np.sum}).reset_index()
performance['per_week_grossmargin']=performance['grossmargin']/performance['weeknumber']
performance['per_week_sales']=performance['sales']/performance['weeknumber']
performance['per_week_transactionproduct']=performance['transactionproduct']/performance['weeknumber']
performance_price=performance.sort_values(by='price',ascending=True)
spark_performance = spark.createDataFrame(performance_price)
spark_performance.display()