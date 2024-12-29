import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_excel('交通数据.xlsx',sheet_name='Sheet6',index_col='日期')

df.plot(y='平均车速',ylim=(0, 100),xlim=(0,360),figsize=(16,9))
plt.savefig('平均车速.png',dpi=300)
plt.cla()
df.plot(y='日交通量',ylim=(0,30000),xlim=(0,360),figsize=(16,9))
plt.savefig('日交通量.png', dpi=300)

