# memuat numpy sebagai np
import numpy as np
 
# memuat pandas sebagai pd
import pandas as pd
# memuat data bernama 'dataset_statistics.csv' dan memasukkan hasilnya ke dalam 'raw_data'
raw_data = pd.read_csv("https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/dataset_statistic.csv", sep=';')

print (raw_data)
# melihat 10 data pada baris pertama
print (raw_data.head(10))

# melihat 5 data pada baris terakhir
print (raw_data.tail())

# melihat dimensi dari raw_data
print (raw_data.shape)

# mengambil jumlah data
print (raw_data.shape[0])

print(raw_data)
print(raw_data.columns)

print (raw_data.isna())
print (raw_data.isna().sum())

print (raw_data.describe())

# Mencari nilai maksimum dari tiap kolom
raw_data.max()
 
# Mencari nilai maksimum dari kolom 'Harga'
raw_data['Harga'].max()
 
# Mencari nilai minimum dari kolom 'Harga'
raw_data['Harga'].min()

# menghitung jumlah dari semua kolom
print (raw_data.sum())
 
# menghitung jumlah dari semua kolom bertipe data numerik saja
raw_data.sum(numeric_only=True)
 
# menghitung jumlah dari kolom 'Harga' dan 'Pendapatan'
raw_data[['Harga', 'Pendapatan']].sum()

# Memilih kolom 'Pendapatan' saja
print (raw_data['Pendapatan'])
 
# Memilih kolom 'Jenis Kelamin' dan 'Pendapatan'
print (raw_data[['Jenis Kelamin', 'Pendapatan']])

# mengambil data dari baris ke-0 sampai baris ke-(10-1) atau baris ke-9
print(raw_data[:10])
 
# mengambil data dari baris ke-3 sampai baris ke-(5-1) atau baris ke-4
print(raw_data[3:5])
 
# mengambil data pada baris ke-1, ke-3 dan ke-10
print(raw_data.loc[[1,3,10]])

# Mengambil kolom 'Jenis Kelamin' dan 'Pendapatan' dan ambil baris ke-1 sampai ke-9
print(raw_data[['Jenis Kelamin', 'Pendapatan']][1:10])
 
# Mengambil kolom 'Harga' dan 'Tingkat Kepuasan' dan ambil baris ke-1, ke-10 dan ke-15
print(raw_data[['Harga', 'Tingkat Kepuasan']].loc[[1,10,15]])

# mengambil hanya data untuk produk 'A'
produk_A = raw_data[raw_data['Produk']=='A']
 
# menghitung rerata pendapatan menggunakan method .mean pada objek pandas DataFrame
print (produk_A['Pendapatan'].mean())
 
# menghitung rerata pendapatan menggunakan method .mean pada objek pandas DataFrame dengan numpy
print (np.mean(produk_A['Pendapatan']))

print(raw_data)
# Hitung median dari pendapatan menggunakan pandas
print(produk_A['Pendapatan'].median())
 
# Hitung median dari pendapatan menggunakan numpy
print(np.median(produk_A['Pendapatan']))

# Melihat jumlah dari masing-masing produk
print(raw_data['Produk'].value_counts())

# mencari median atau 50% dari data menggunakan pandas
print(raw_data['Pendapatan'].quantile(q=0.5))
 
# mencari median atau 50% dari data menggunakan pandas
print(np.quantile(raw_data['Pendapatan'], q=0.5))

# menghitung rerata dan median 'Pendapatan' dan 'Harga'
print(raw_data[['Pendapatan','Harga']].agg([np.mean,np.median]))
 
# menghitung rerata dan median Pendapatan dan Harga dari tiap produk
print(raw_data[['Pendapatan','Harga','Produk']].groupby('Produk').agg([np.mean, np.median]))

# cari proporsi tiap Produk
print(raw_data['Produk'].value_counts()/raw_data.shape[0])

# Cari nilai rentang dari kolom 'Pendapatan'
print (raw_data['Pendapatan'].max() - raw_data['Pendapatan'].min())

# menghitung variansi umur menggunakan method .var() dari pandas
print (raw_data['Pendapatan'].var())
 
# menghitung variansi umur menggunakan method .var() dari numpy
print (np.var(raw_data['Pendapatan']))

# mengatur variansi populasi dengan method `.var()` dari pandas
print (raw_data['Pendapatan'].var(ddof=0))

# menghitung deviasi baku sampel pendapatan menggunakan method std() dari pandas
print (raw_data['Pendapatan'].std())
 
# menghitung deviasi baku sampel pendapatan menggunakan method std() dari numpy
print (np.std(raw_data['Pendapatan'], ddof = 1))

# menghitung korelasi dari setiap pasang variabel pada raw_data
print (raw_data.corr())

# mencari korelasi 'kendall' untuk tiap pasang variabel
print (raw_data.corr(method='kendall'))
 
# mencari korelasi 'spearman' untuk tiap pasang variabel
print (raw_data.corr(method='spearman'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.clf()

## mengambil data contoh
raw_data = pd.read_csv("https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/dataset_statistic.csv", sep=';')

## melihat isi dari data
print(raw_data)

plt.figure()
# visualisasi diagram pencar untuk variabel 'Pendapatan' dan 'Total' menggunakan 'plot.scatter' dari pandas
raw_data.plot.scatter(x='Pendapatan', y='Total')
plt.title('plot.scatter dari pandas', size=14)
plt.tight_layout()
plt.show()

# visualisasi diagram pencar untuk variabel 'Pendapatan' dan 'Total' menggunakan 'plt.scatter' dari matplotlib
plt.scatter(x='Pendapatan', y='Total', data=raw_data)
plt.title('plt.scatter dari matplotlib', size=14)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
# visualisasi diagram pencar untuk variabel 'Pendapatan' dan 'Total' menggunakan 'plt.scatter' dari matplotlib.pyplot
plt.scatter(x='Pendapatan', y='Total', data=raw_data)
plt.xlabel('Pendapatan')
plt.ylabel('Total')
plt.show()

import matplotlib.pyplot as plt
plt.clf()

plt.figure()
# melihat distribusi data kolom 'Pendapatan' menggunakan 'hist' dari pandas
raw_data.hist(column='Pendapatan')
plt.title('.hist dari pandas', size=14)
plt.tight_layout()
plt.show()

plt.figure()
# melihat distribusi data kolom 'Pendapatan' menggunakan 'pyplot.hist' dari matplotlib.pyplot
plt.hist(x='Pendapatan', data=raw_data)
plt.xlabel('Pendapatan')
plt.title('pyplot.hist dari matplotlib.pyplot', size=14)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
plt.clf()

# melihat box plot dari kolom 'Pendapatan' menggunakan method '.boxplot' dari pandas
plt.figure()
raw_data.boxplot(column='Pendapatan')
plt.title('.boxplot dari pandas', size=14)
plt.tight_layout()
plt.show()

# melihat box plot dari kolom 'Pendapatan' menggunakan method '.boxplot' dari matplotlib
plt.figure()
plt.boxplot(x = 'Pendapatan', data=raw_data)
plt.xlabel('Pendapatan')
plt.title('pyplot.boxplot dari matplotlib.pyplot', size=14)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
plt.clf()

# hitung frekuensi dari masing-masing nilai pada kolom 'Produk'
class_freq = raw_data['Produk'].value_counts()

# lihat nilai dari class_freq
print(class_freq)

plt.figure()
# membuat bar plot dengan method `plot.bar()` dari pandas
class_freq.plot.bar()
plt.title('.bar dari pandas', size=14)
plt.tight_layout()
plt.show()

plt.figure()
# membuat bar plot dengan method `plt.bar()` dari matplotlib
plt.bar(x=class_freq.index, height=class_freq.values)
plt.title('plt.bar dari matplotlib.pyplot', size=14)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
plt.clf()

plt.figure()
# membuat pie chart menggunakan method 'pyplot.pie()' dari matplotlib
plt.pie(class_freq.values, labels=class_freq.index)
plt.title('plt.pie dari matplotlib.pyplot', size=14)
plt.tight_layout()
plt.show()

plt.figure()
# membuat pie chart menggunakan method 'plot.pie' dari pandas
class_freq.plot.pie()
plt.title('plot.pie dari pandas', size=14)
plt.tight_layout()
plt.show()

from scipy import stats
import matplotlib.pyplot as plt
plt.clf()

plt.figure()
raw_data.hist()
plt.title('Histogram seluruh kolom', size=14)
plt.tight_layout()
plt.show()

plt.figure()
raw_data['Pendapatan'].hist()
plt.title('Histogram pendapatan', size=14)
plt.tight_layout()
plt.show()

plt.figure()
# transformasi menggunakan akar lima
np.power(raw_data['Pendapatan'], 1/5).hist()
plt.title('Histogram pendapatan - ransformasi menggunakan akar lima', size=14)
plt.tight_layout()
plt.show()

# simpan hasil transformasi
pendapatan_akar_lima = np.power(raw_data['Pendapatan'], 1/5)

plt.figure()
# membuat qqplot pendapatan - transformasi menggunakan akar lima
stats.probplot(pendapatan_akar_lima, plot=plt)
plt.title('qqplot pendapatan - transformasi menggunakan akar lima', size=14)
plt.tight_layout()
plt.show()

plt.figure()
# membuat qqplot pendapatan
stats.probplot(raw_data['Pendapatan'], plot=plt)
plt.title('qqplot pendapatan', size=14)
plt.tight_layout()
plt.show()


print(raw_data['Produk'])

data_dummy_produk = pd.get_dummies(raw_data['Produk'])

print(data_dummy_produk)

print(raw_data.skew())
#the closest variable to normal distribution is 'Total' and i believe it is because it is the smallest number that approaching 0

hasil_1 =np.power(raw_data['Pendapatan'], 1/5)

stats.skew(hasil_1)

hasil_2, _=stats.boxcox(raw_data['Pendapatan'])

stats.skew(hasil_2)

import seaborn as sns
import matplotlib.pyplot as plt
plt.clf()

# mengatur ukuran gambar/plot
plt.rcParams['figure.dpi'] = 100

plt.figure()
plt.matshow(raw_data.corr())
plt.title('Plot correlation matriks dengan .matshow', size=14)
plt.tight_layout()
plt.show()

plt.figure()
sns.heatmap(raw_data.corr(), annot=True)
plt.title('Plot correlation matriks dengan sns.heatmap', size=14)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
plt.clf()

plt.figure()
# boxplot biasa tanpa pengelompokkan
raw_data.boxplot(rot=90)
plt.title('Boxplot tanpa pengelompokkan', size=14)
plt.tight_layout()
plt.show()

plt.figure()
# box plot dengan pengelompokkan dilakukan oleh kolom 'Produk'
raw_data.boxplot(by='Produk')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
plt.clf()

plt.figure()
raw_data[raw_data['Produk'] == 'A'].hist()
plt.tight_layout()
plt.show()

plt.figure()
raw_data[raw_data['Produk'] == 'B'].hist()
plt.tight_layout()
plt.show()

plt.figure()
raw_data[raw_data['Produk'] == 'C'].hist()
plt.tight_layout()
plt.show()

plt.figure()
raw_data[raw_data['Produk'] == 'D'].hist()
plt.tight_layout()
plt.show()

plt.figure()
raw_data[raw_data['Produk'] == 'E'].hist()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
plt.clf()

plt.figure()
raw_data.plot.hexbin(x='Pendapatan', y='Total', gridsize=25, rot=90)
plt.tight_layout()
plt.show()


#THERE IS TROUBLE HERE
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
plt.clf()

_, ax = plt.subplots(1, 1, figsize=(10,10))
scatter_matrix(raw_data, ax=ax)
plt.show()


from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
plt.clf()

_, ax = plt.subplots(1, 1, figsize=(10,10))
scatter_matrix(raw_data, diagonal='kde', ax=ax)
plt.show()

import pandas as pd
import statsmodels.api as sm

print(raw_data)
list(raw_data)

#variabel tak bebas
nilai_Y=raw_data[['Total']]
#Variabel bebas
nilai_X=sm.add_constant(raw_data['Pendapatan'])

#membuat model regresi linier
model_regresi=sm.OLS(endog=nilai_Y, exog=nilai_X).fit()

model_regresi.summary()

import matplotlib.pyplot as plt
import plotnine as p9
import pandas as pd

import pandas as pd
df_penduduk = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/datakependudukandki-dqlab.csv')
df_inflasi = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/inflasi.csv')

print (df_penduduk.head())

import matplotlib.pyplot as plt
from plotnine import *
ggplot(data=df_penduduk).draw()
plt.show()

list(df_penduduk)

import matplotlib.pyplot as plt
from plotnine import *
(ggplot(data=df_penduduk)
+ aes(x='NAMA KABUPATEN/KOTA', y='JUMLAH')
).draw()
plt.show()

import matplotlib.pyplot as plt
from plotnine import *
(ggplot(data=df_penduduk)
+ aes(x='NAMA KABUPATEN/KOTA', y='JUMLAH')
+ geom_col()
).draw()
plt.show()

import matplotlib.pyplot as plt
from plotnine import *
(ggplot(data=df_penduduk)
+ aes(x='NAMA KABUPATEN/KOTA', y='JUMLAH')
+ geom_col()
+ coord_flip()
).draw()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from plotnine import *
import plotnine
plotnine.options.figure_size=(12, 4.8)
(ggplot(data=df_penduduk)
+ aes(x='NAMA KABUPATEN/KOTA', y='JUMLAH')
+ geom_col()
+ coord_flip()
+ labs(title='Jumlah penduduk per kabupaten/kota di DKI Jakarta (2013)',
x='Kabupaten/Kota',
y='Jumlah Penduduk')
).draw()
plt.show()

import matplotlib.pyplot as plt
from plotnine import *
import plotnine
plotnine.options.figure_size=(10, 3.6)
(ggplot(data=df_penduduk)
+ aes(x='NAMA KABUPATEN/KOTA', y='JUMLAH', fill= 'JENIS KELAMIN')
+ geom_col()
+ coord_flip()
+ labs(title='Jumlah penduduk per kabupaten/kota di DKI Jakarta (2013)',
x='Kabupaten/Kota',
y='Jumlah Penduduk')
).draw()
plt.show()

import matplotlib.pyplot as plt
from plotnine import *
import plotnine
plotnine.options.figure_size=(10, 3.6)
(ggplot(data=df_penduduk[df_penduduk['NAMA KECAMATAN'] == 'CENGKARENG'])
+ aes(x='NAMA KELURAHAN', y='JUMLAH', fill='JENIS KELAMIN')
+ geom_col()
+ coord_flip()
+ labs(title='Jumlah penduduk per kelurahan di DKI Jakarta (2013)',
x='Kelurahan', y='Jumlah Penduduk')
).draw()
plt.show()

import matplotlib.pyplot as plt
from plotnine import *
import plotnine
plotnine.options.figure_size=(10, 3.6)
(ggplot(data=df_penduduk[df_penduduk['NAMA KECAMATAN'] == 'CENGKARENG'])
+ aes(x='NAMA KELURAHAN', y='JUMLAH', fill='JENIS KELAMIN')
+ geom_col(position='position_dodge')
+ coord_flip()
+ labs(title='Jumlah penduduk per kelurahan di DKI Jakarta (2013)',
x='Kelurahan',
y='Jumlah Penduduk')
).draw()
plt.show()

import matplotlib.pyplot as plt
from plotnine import *

df_penduduk_luas_jumlah = df_penduduk.groupby(['NAMA KELURAHAN', 'LUAS WILAYAH (KM2)'])[['JUMLAH']].agg('sum').reset_index()

(ggplot(data=df_penduduk_luas_jumlah)
+ aes(y='LUAS WILAYAH (KM2)', x='JUMLAH')
+ geom_point()
).draw()
plt.show()

import matplotlib.pyplot as plt
from plotnine import *
import plotnine
plotnine.options.figure_size=(10, 3.6)
(ggplot(data=df_penduduk_luas_jumlah)
+ aes(y='LUAS WILAYAH (KM2)', x='JUMLAH', color='JUMLAH')
+ geom_point()
).draw()
plt.show()

list(df_penduduk)

import matplotlib.pyplot as plt
from plotnine import *
(ggplot(data=df_penduduk)
+ aes(x='LUAS WILAYAH (KM2)')
+ geom_histogram()
).draw()
plt.show()

import matplotlib.pyplot as plt
from plotnine import *
(ggplot(data=df_penduduk)
+ aes(x='LUAS WILAYAH (KM2)', y= 'stat(count/max(count))')
+ geom_histogram()
).draw()
plt.show()

import matplotlib.pyplot as plt
from plotnine import *
(ggplot(data=df_penduduk)
+ aes(x='NAMA KABUPATEN/KOTA', y='JUMLAH')
+ geom_boxplot()
+ coord_flip()
).draw()
plt.tight_layout()
plt.show()

import pandas as pd
df_inflasi = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/inflasi.csv')

print(df_inflasi)
import matplotlib.pyplot as plt
from plotnine import *
df_inflasi['Bulan'] = pd.to_datetime(df_inflasi['Bulan'])
(ggplot(data=df_inflasi[df_inflasi['Negara']=='Indonesia'])
+ aes(x='Bulan', y='Inflasi')
+ geom_line()
).draw()
plt.show()

import matplotlib.pyplot as plt
from plotnine import *
import plotnine
plotnine.options.figure_size=(10, 3.6)
(ggplot(data=df_inflasi)
+ aes(x='Bulan', y='Inflasi', color='Negara')
+ geom_line()
).draw()
plt.show()

import matplotlib.pyplot as plt
from plotnine import *
(ggplot(data=df_inflasi)
+ aes(x='Bulan', y='Inflasi', color='Negara')
+ geom_line()
+ theme(figure_size=(10, 5))
).draw()
plt.show()
