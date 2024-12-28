import pandas as pd
import os
import matplotlib.pyplot as plt

files = os.listdir('metrics')

data = pd.concat([pd.read_csv(f'metrics/{fil}',index_col=0) for fil in files])
data.sort_values('TestRMSE',inplace=True,ascending=False)
print(data)

ax = data.plot.bar(rot=45)
plt.savefig('images/Model_RMSE_Comparison.png',bbox_inches='tight')
plt.close()
