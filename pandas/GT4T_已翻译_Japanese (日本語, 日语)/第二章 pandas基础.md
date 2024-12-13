<center> <h1>第 2 章パンダスの 基礎 </h1></center>



```python
import numpy as np
import pandas as pd
```

学習を始める前に、 `pandas` のバージョン番号が以下のバージョン以下でないことを確認してください。そうでない場合は、必ずアップグレードしてください！ `xlrd, xlwt, openpyxl` 3つのパッケージがインストールされていることを確認してください。 `xlrd` バージョンは `2.0.0` より高くないでください。



```python
pd.__version__
```




    '1.1.5'



## 1. ファイルの読み込みと書き込み
### 1. ファイル読み取り

 `pandas` 読み取ることができるファイル形式はたくさんありますが、ここでは主に `csv, excel, txt` ファイルの読み取りについて説明します。



```python
df_csv = pd.read_csv('../data/my_csv.csv')
df_csv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple</td>
      <td>2020/1/1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana</td>
      <td>2020/1/2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>c</td>
      <td>2.5</td>
      <td>orange</td>
      <td>2020/1/5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>d</td>
      <td>3.2</td>
      <td>lemon</td>
      <td>2020/1/7</td>
    </tr>
  </tbody>
</table>
</div>





```python
df_txt = pd.read_table('../data/my_table.txt')
df_txt
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple 2020/1/1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana 2020/1/2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>c</td>
      <td>2.5</td>
      <td>orange 2020/1/5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>d</td>
      <td>3.2</td>
      <td>lemon 2020/1/7</td>
    </tr>
  </tbody>
</table>
</div>





```python
df_excel = pd.read_excel('../data/my_excel.xlsx')
df_excel
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple</td>
      <td>2020/1/1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana</td>
      <td>2020/1/2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>c</td>
      <td>2.5</td>
      <td>orange</td>
      <td>2020/1/5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>d</td>
      <td>3.2</td>
      <td>lemon</td>
      <td>2020/1/7</td>
    </tr>
  </tbody>
</table>
</div>



ここにはいくつかの一般的なパラメータがあります。 `header=None` は最初の行が列名にならないことを意味します。 `index_col` はある列またはいくつかの列をインデックスにすることを意味します。インデックスの内容は第3章で詳しく説明します。 `usecols` は列の集合を読み取ります。デフォルトではすべての列を読み取ります。 `parse_dates` は時間に変換する必要があります。時系列については第10章で説明します。 `nrows` 読み取ったデータの行数を示します。これらの引数は、上記の3つの関数で使用できます。



```python
pd.read_table('../data/my_table.txt', header=None)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>col1</td>
      <td>col2</td>
      <td>col3</td>
      <td>col4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple 2020/1/1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana 2020/1/2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>c</td>
      <td>2.5</td>
      <td>orange 2020/1/5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>d</td>
      <td>3.2</td>
      <td>lemon 2020/1/7</td>
    </tr>
  </tbody>
</table>
</div>





```python
pd.read_csv('../data/my_csv.csv', index_col=['col1', 'col2'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
    <tr>
      <th>col1</th>
      <th>col2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <th>a</th>
      <td>1.4</td>
      <td>apple</td>
      <td>2020/1/1</td>
    </tr>
    <tr>
      <th>3</th>
      <th>b</th>
      <td>3.4</td>
      <td>banana</td>
      <td>2020/1/2</td>
    </tr>
    <tr>
      <th>6</th>
      <th>c</th>
      <td>2.5</td>
      <td>orange</td>
      <td>2020/1/5</td>
    </tr>
    <tr>
      <th>5</th>
      <th>d</th>
      <td>3.2</td>
      <td>lemon</td>
      <td>2020/1/7</td>
    </tr>
  </tbody>
</table>
</div>





```python
pd.read_table('../data/my_table.txt', usecols=['col1', 'col2'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>d</td>
    </tr>
  </tbody>
</table>
</div>





```python
pd.read_csv('../data/my_csv.csv', parse_dates=['col5'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple</td>
      <td>2020-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana</td>
      <td>2020-01-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>c</td>
      <td>2.5</td>
      <td>orange</td>
      <td>2020-01-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>d</td>
      <td>3.2</td>
      <td>lemon</td>
      <td>2020-01-07</td>
    </tr>
  </tbody>
</table>
</div>





```python
pd.read_excel('../data/my_excel.xlsx', nrows=2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple</td>
      <td>2020/1/1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana</td>
      <td>2020/1/2</td>
    </tr>
  </tbody>
</table>
</div>



 `txt` ファイルを読み込むとき、区切り文字がスペースではないことがよくあります。 `read_table` には分割パラメータ `sep` があり、ユーザーは分割記号をカスタマイズし、 `txt` データを読み込むことができます。たとえば、次の読み込みテーブルは `||||` で分割されています：



```python
pd.read_table('../data/my_table_special_sep.txt')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1 |||| col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TS |||| This is an apple.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GQ |||| My name is Bob.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WT |||| Well done!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PT |||| May I help you?</td>
    </tr>
  </tbody>
</table>
</div>



上記の結果は明らかに理想的ではありません。この場合、 `sep` を使用し、エンジンを `python` に指定する必要があります。



```python
pd.read_table('../data/my_table_special_sep.txt', sep=' \|\|\|\| ', engine='python')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TS</td>
      <td>This is an apple.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GQ</td>
      <td>My name is Bob.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WT</td>
      <td>Well done!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PT</td>
      <td>May I help you?</td>
    </tr>
  </tbody>
</table>
</div>



#### [WARNING] `sep` は正規引数です

 `read_table` を使用するときは、パラメータ `sep` に正規表現が使用されているため、 `|` を `\|` にエスケープする必要があります。そうしないと正しい結果を読み取ることができません。正規表現の基本的な内容については、第8章またはその他の関連資料を参照することができる。

#### [END]

### 2. データ書き込み

一般的にデータ書き込みで最も一般的な操作は `index` を `False` に設定することです。特にインデックスに特別な意味がない場合、この行動は保存時にインデックスを除去することができます。



```python
df_csv.to_csv('../data/my_csv_saved.csv', index=False)
df_excel.to_excel('../data/my_excel_saved.xlsx', index=False)
```

 `to_table` 関数は `pandas` で定義されていませんが、 `to_csv` は `txt` ファイルとして保存でき、カスタム区切り文字、一般的なタブ文字 `\t` 分割が可能です：



```python
df_txt.to_csv('../data/my_txt_saved.txt', sep='\t', index=False)
```

テーブルを `markdown` と `latex` 言語にすばやく変換したい場合は、 `to_markdown` と `to_latex` 関数を使用してください。ここには `tabulate` パッケージが必要です。



```python
print(df_csv.to_markdown())
```

    |    |   col1 | col2   |   col3 | col4   | col5     |
    |---:|-------:|:-------|-------:|:-------|:---------|
    |  0 |      2 | a      |    1.4 | apple  | 2020/1/1 |
    |  1 |      3 | b      |    3.4 | banana | 2020/1/2 |
    |  2 |      6 | c      |    2.5 | orange | 2020/1/5 |
    |  3 |      5 | d      |    3.2 | lemon  | 2020/1/7 |

    


```python
print(df_csv.to_latex())
```

    \begin{tabular}{lrlrll}\toprule{} &  col1 & col2 &  col3 &    col4 &      col5 \\\midrule0 &     2 &    a &   1.4 &   apple &  2020/1/1 \\1 &     3 &    b &   3.4 &  banana &  2020/1/2 \\2 &     6 &    c &   2.5 &  orange &  2020/1/5 \\3 &     5 &    d &   3.2 &   lemon &  2020/1/7 \\\bottomrule\end{tabular}
    
    

## 二、基本的なデータ構造
 `pandas` には、1次元 `values` を格納する `Series` と2次元 `values` を格納する `DataFrame` の2つの基本的なデータ格納構造があり、この2つの構造には多くの属性とメソッドが定義されています。

### 1. Series
 `Series` は一般的に、シーケンスの値 `data`、インデックス `index`、ストレージタイプ `dtype`、シーケンスの名前 `name` の4つの部分から構成されています。その中で、インデックスはその名前を指定することもでき、デフォルトは空です。



```python
s = pd.Series(data = [100, 'a', {'dic1':5}],
              index = pd.Index(['id1', 20, 'third'], name='my_idx'),
              dtype = 'object',
              name = 'my_name')
s
```




    my_idxid1              10020                 athird    {'dic1': 5}Name: my_name, dtype: object



#### [NOTE] `object` タイプ

 `object` は、上記の例で整数、文字列、および `Python` の辞書データ構造が格納されているように、混合型を表します。また、現在 `pandas` は純粋な文字列シーケンスもデフォルトで `object` 型のシーケンスとみなされていますが、それは `string` 型で保存することもできます。テキストシーケンスの内容は第8章で說明します。

#### [END]

これらのプロパティについては、.以下の方法で取得します：



```python
s.values
```




    array([100, 'a', {'dic1': 5}], dtype=object)





```python
s.index
```




    Index(['id1', 20, 'third'], dtype='object', name='my_idx')





```python
s.dtype
```




    dtype('O')





```python
s.name
```




    'my_name'



 `.shape` を使用してシーケンスの長さを取得できます：



```python
s.shape
```




    (3,)



インデックスは `pandas` における最も重要な概念の1つであり、第3章で詳細に說明されます。単一のインデックスに対応する値を取り出したい場合は、 `[index_item]` で取り出すことができます。

### 2. DataFrame
 `DataFrame` `Series` にカラムインデックスが追加され、データフレームは2次元 `data` と行列インデックスで構成できます。



```python
data = [[1, 'a', 1.2], [2, 'b', 2.2], [3, 'c', 3.2]]
df = pd.DataFrame(data = data,
                  index = ['row_%d'%i for i in range(3)],
                  columns=['col_0', 'col_1', 'col_2'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_0</th>
      <th>col_1</th>
      <th>col_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row_0</th>
      <td>1</td>
      <td>a</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>row_1</th>
      <td>2</td>
      <td>b</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>row_2</th>
      <td>3</td>
      <td>c</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>



しかし、一般的には、列のインデックス名からデータへのマッピングを使用してデータフレームを構築し、行のインデックスを追加します：



```python
df = pd.DataFrame(data = {'col_0': [1,2,3],
                          'col_1':list('abc'),
                          'col_2': [1.2, 2.2, 3.2]},
                  index = ['row_%d'%i for i in range(3)])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_0</th>
      <th>col_1</th>
      <th>col_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row_0</th>
      <td>1</td>
      <td>a</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>row_1</th>
      <td>2</td>
      <td>b</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>row_2</th>
      <td>3</td>
      <td>c</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>



このマッピングのため、 `DataFrame` では、 `[col_name]` と `[col_list]` で対応する列と複数の列からなるテーブルを取り出すことができ、結果はそれぞれ `Series` と `DataFrame` になります：



```python
df['col_0']
```




    row_0    1row_1    2row_2    3Name: col_0, dtype: int64





```python
df[['col_0', 'col_1']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_0</th>
      <th>col_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row_0</th>
      <td>1</td>
      <td>a</td>
    </tr>
    <tr>
      <th>row_1</th>
      <td>2</td>
      <td>b</td>
    </tr>
    <tr>
      <th>row_2</th>
      <td>3</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>



 `Series` と同様に、データフレームに対応する属性を取り出すことができます：



```python
df.values
```




    array([[1, 'a', 1.2],
           [2, 'b', 2.2],[3, 'c', 3.2]], dtype=object)





```python
df.index
```




    Index(['row_0', 'row_1', 'row_2'], dtype='object')





```python
df.columns
```




    Index(['col_0', 'col_1', 'col_2'], dtype='object')





```python
df.dtypes # 返回的是值为相应列数据类型的Series
```




    col_0      int64col_1     objectcol_2    float64dtype: object





```python
df.shape
```




    (3, 3)



 `.T` は `DataFrame` を転置することができます：



```python
df.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row_0</th>
      <th>row_1</th>
      <th>row_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>col_0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>col_1</th>
      <td>a</td>
      <td>b</td>
      <td>c</td>
    </tr>
    <tr>
      <th>col_2</th>
      <td>1.2</td>
      <td>2.2</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>



## 三、よく使われる基本関数
例を示すために、次のセクションおよび残りの章では、4校の生徒の身体検査個人情報を記録した `learn_pandas.csv` の仮想データセットを使用します。



```python
df = pd.read_csv('../data/learn_pandas.csv')
df.columns
```




    Index(['School', 'Grade', 'Name', 'Gender', 'Height', 'Weight', 'Transfer',
           'Test_Number', 'Test_Date', 'Time_Record'],
          dtype='object')



上記の列名は順に学校、学年、氏名、性別、身長、体重、転系生の有無、体験回数、テスト時間、1000メートルの成績を表しており、本章ではその中の最初の7列のみを使用すればよい。



```python
df = df[df.columns[:7]]
```

### 1. 要約関数
 `head, tail` 関数は、テーブルまたはシーケンスの前の `n` 行と後の `n` 行をそれぞれ返します。ここで、 `n` はデフォルトで5になります：



```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>School</th>
      <th>Grade</th>
      <th>Name</th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Transfer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Freshman</td>
      <td>Gaopeng Yang</td>
      <td>Female</td>
      <td>158.9</td>
      <td>46.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Peking University</td>
      <td>Freshman</td>
      <td>Changqiang You</td>
      <td>Male</td>
      <td>166.5</td>
      <td>70.0</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>





```python
df.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>School</th>
      <th>Grade</th>
      <th>Name</th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Transfer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>197</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Senior</td>
      <td>Chengqiang Chu</td>
      <td>Female</td>
      <td>153.9</td>
      <td>45.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>198</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Senior</td>
      <td>Chengmei Shen</td>
      <td>Male</td>
      <td>175.3</td>
      <td>71.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>199</th>
      <td>Tsinghua University</td>
      <td>Sophomore</td>
      <td>Chunpeng Lv</td>
      <td>Male</td>
      <td>155.7</td>
      <td>51.0</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>



 `info, describe` テーブルの情報概要とテーブルの数値列に対応する主な統計をそれぞれ返します：



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200 entries, 0 to 199Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   School    200 non-null    object 
     1   Grade     200 non-null    object 
     2   Name      200 non-null    object 
     3   Gender    200 non-null    object 
     4   Height    183 non-null    float64
     5   Weight    189 non-null    float64
     6   Transfer  188 non-null    object 
    dtypes: float64(2), object(5)
    memory usage: 11.1+ KB

    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>183.000000</td>
      <td>189.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>163.218033</td>
      <td>55.015873</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.608879</td>
      <td>12.824294</td>
    </tr>
    <tr>
      <th>min</th>
      <td>145.400000</td>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>157.150000</td>
      <td>46.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>161.900000</td>
      <td>51.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>167.500000</td>
      <td>65.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>193.900000</td>
      <td>89.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### [NOTE] より包括的なデータまとめ

 `info, describe` は少ない情報を表示することしかできませんが、データセットの包括的かつ効果的な観察を行いたい場合、特に列が多い場合は、[pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/)パッケージを使用することをお勧めします。これは第11章で再び取り上げられます。

#### [END]

### 2. 特徴統計関数
多くの統計関数は `Series` と `DataFrame` で定義されており、最も一般的なのは `sum, mean, median, var, std, max, min` です。たとえば、デモンストレーションのために「身長」と「体重」列を選択します：



```python
df_demo = df[['Height', 'Weight']]
df_demo.mean()
```




    Height    163.218033Weight     55.015873dtype: float64





```python
df_demo.max()
```




    Height    193.9Weight     89.0dtype: float64



さらに、 `quantile, count, idxmax` という3つの関数を紹介します。これらはそれぞれ分位数、欠落していない値の数、最大値に対応するインデックスを返します：



```python
df_demo.quantile(0.75)
```




    Height    167.5Weight     65.0Name: 0.75, dtype: float64





```python
df_demo.count()
```




    Height    183Weight    189dtype: int64





```python
df_demo.idxmax() # idxmin是对应的函数
```




    Height    193Weight      2dtype: int64



これらのすべての関数は、操作後にスカラーを返すため、集約関数とも呼ばれます。これらの関数は共通引数 `axis` を持っています。デフォルトではカラムごとに集約するために0に設定されています。1に設定されている場合は、行ごとに集約することになります：



```python
df_demo.mean(axis=1).head() # 在这个数据集上体重和身高的均值并没有意义
```




    0    102.451    118.252    138.953     41.004    124.00dtype: float64



### 3. ユニーク値関数
シーケンスに `unique` と `nunique` を使用すると、それぞれ一意の値のリストと一意の値の数が得られます：



```python
df['School'].unique()
```




    array(['Shanghai Jiao Tong University', 'Peking University',
           'Fudan University', 'Tsinghua University'], dtype=object)





```python
df['School'].nunique()
```




    4



 `value_counts` 一意の値とそれに対応する頻度を得ることができます：



```python
df['School'].value_counts()
```




    Tsinghua University              69Shanghai Jiao Tong University    57Fudan University                 40Peking University                34Name: School, dtype: int64



複数の列の組み合わせの一意の値を観察したい場合は、 `drop_duplicates` を使用します。キーパラメータは `keep` で、デフォルト値 `first` は各組み合わせの最初の出現行を保持すること、 `last` は最後の出現行を保持すること、 `False` はすべての繰り返し組み合わせの行を除去することを意味します。



```python
df_demo = df[['Gender','Transfer','Name']]
df_demo.drop_duplicates(['Gender', 'Transfer'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Transfer</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>N</td>
      <td>Gaopeng Yang</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>N</td>
      <td>Changqiang You</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Female</td>
      <td>NaN</td>
      <td>Peng You</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Male</td>
      <td>NaN</td>
      <td>Xiaopeng Shen</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Male</td>
      <td>Y</td>
      <td>Xiaojuan Qin</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Female</td>
      <td>Y</td>
      <td>Gaoli Feng</td>
    </tr>
  </tbody>
</table>
</div>





```python
df_demo.drop_duplicates(['Gender', 'Transfer'], keep='last')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Transfer</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>147</th>
      <td>Male</td>
      <td>NaN</td>
      <td>Juan You</td>
    </tr>
    <tr>
      <th>150</th>
      <td>Male</td>
      <td>Y</td>
      <td>Chengpeng You</td>
    </tr>
    <tr>
      <th>169</th>
      <td>Female</td>
      <td>Y</td>
      <td>Chengquan Qin</td>
    </tr>
    <tr>
      <th>194</th>
      <td>Female</td>
      <td>NaN</td>
      <td>Yanmei Qian</td>
    </tr>
    <tr>
      <th>197</th>
      <td>Female</td>
      <td>N</td>
      <td>Chengqiang Chu</td>
    </tr>
    <tr>
      <th>199</th>
      <td>Male</td>
      <td>N</td>
      <td>Chunpeng Lv</td>
    </tr>
  </tbody>
</table>
</div>





```python
df_demo.drop_duplicates(['Name', 'Gender'], keep=False).head() # 保留只出现过一次的性别和姓名组合
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Transfer</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>N</td>
      <td>Gaopeng Yang</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>N</td>
      <td>Changqiang You</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>N</td>
      <td>Mei Sun</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>N</td>
      <td>Gaojuan You</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Female</td>
      <td>N</td>
      <td>Xiaoli Qian</td>
    </tr>
  </tbody>
</table>
</div>





```python
df['School'].drop_duplicates() # 在Series上也可以使用
```




    0    Shanghai Jiao Tong University1                Peking University3                 Fudan University5              Tsinghua UniversityName: School, dtype: object



さらに、 `duplicated` と `drop_duplicates` は同様の机能を持っていますが、前者は一意の値であるかどうかのブールリストを返し、 `keep` 引数は後者と同じです。返されるシーケンスは、繰り返し要素を `True`、そうでなければ `False` に設定します。 `drop_duplicates` は、 `duplicated` が `True` の対応する行を除去することに等しい。



```python
df_demo.duplicated(['Gender', 'Transfer']).head()
```




    0    False1    False2     True3     True4     Truedtype: bool





```python
df['School'].duplicated().head() # 在Series上也可以使用
```




    0    False1    False2     True3    False4     TrueName: School, dtype: bool



### 4. 置換関数
一般的に、置換はカラムに対して行われるので、以下の例は `Series` で例を挙げます。 `pandas` の置換関数は、マッピング置換、論理置換、数値置換の3種類にまとめることができます。マッピング置換には、 `replace` メソッド、第8章の `str.replace` メソッド、および第9章の `cat.codes` メソッドが含まれています。ここでは、 `replace` の使用方法について説明します。

 `replace` では、辞書構造を使用するか、2つのリストを渡すことによって置換できます。



```python
df['Gender'].replace({'Female':0, 'Male':1}).head()
```




    0    01    12    13    04    1Name: Gender, dtype: int64





```python
df['Gender'].replace(['Female', 'Male'], [0, 1]).head()
```




    0    01    12    13    04    1Name: Gender, dtype: int64



さらに、 `replace` 特別な方向置換があります。 `method` パラメータを `ffill` と指定すると、直前の置換されていない値を置き換え、 `bfill` を指定すると、直前の置換されていない値を置き換えることができます。次の例からわかるように、それらの結果は異なります。



```python
s = pd.Series(['a', 1, 'b', 2, 1, 1, 'a'])
s.replace([1, 2], method='ffill')
```




    0    a1    a2    b3    b4    b5    b6    adtype: object





```python
s.replace([1, 2], method='bfill')
```




    0    a1    b2    b3    a4    a5    a6    adtype: object



#### [WARNING] 正規置換は `str.replace` を使用してください。

正規置換は `replace` で使用できますが、現在のバージョンでは `string` タイプの正規置換には `bug` が存在していますので、必要な場合は、 `str.replace` を選択して置換してください。具体的な方法は第8章で説明します。

#### [END]

論理置換には、完全に対称な `where` と `mask` 関数が含まれます。 `where` 関数は、受信条件 `False` の対応する行で置換し、 `mask` の対応する行で置換し、置換値が指定されていない場合は、欠落した値で置換します。



```python
s = pd.Series([-1, 1.2345, 100, -50])
s.where(s<0)
```




    0    -1.01     NaN2     NaN3   -50.0dtype: float64





```python
s.where(s<0, 100)
```




    0     -1.01    100.02    100.03    -50.0dtype: float64





```python
s.mask(s<0)
```




    0         NaN1      1.23452    100.00003         NaNdtype: float64





```python
s.mask(s<0, -50)
```




    0    -50.00001      1.23452    100.00003    -50.0000dtype: float64



送信される条件は、呼び出された `Series` インデックスと一致するブールシーケンスであればよいことに注意してください。



```python
s_condition= pd.Series([True,False,False,True],index=s.index)
s.mask(s_condition, -50)
```




    0    -50.00001      1.23452    100.00003    -50.0000dtype: float64



数値置換には、所定の精度で切り捨て、絶対値取り、切り捨てを表す `round, abs, clip` メソッドが含まれています：



```python
s = pd.Series([-1, 1.2345, 100, -50])
s.round(2)
```




    0     -1.001      1.232    100.003    -50.00dtype: float64





```python
s.abs()
```




    0      1.00001      1.23452    100.00003     50.0000dtype: float64





```python
s.clip(0, 2) # 前两个数分别表示上下截断边界
```




    0    0.00001    1.23452    2.00003    0.0000dtype: float64



#### [練習して練習する]

clipでは境界を超えたものは境界値にしか切り捨てることができませんが、境界を超えたものをカスタム値に置き換える場合はどうすればいいですか？

#### [END]

### 5. ソート関数
ソートには2つの方法があります。1つは値ソートであり、もう1つはインデックスソートであり、対応する関数は `sort_values` と `sort_index` です。

ソート関数を実証するために、以下ではまず `set_index` メソッドを用いて学年と名前の2列をインデックスとし、マルチレベルインデックスの内容とインデックス設定の方法を第3章で詳しく説明する。



```python
df_demo = df[['Grade', 'Name', 'Height', 'Weight']].set_index(['Grade','Name'])
df_demo.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Grade</th>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Freshman</th>
      <th>Gaopeng Yang</th>
      <td>158.9</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>Changqiang You</th>
      <td>166.5</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>Senior</th>
      <th>Mei Sun</th>
      <td>188.9</td>
      <td>89.0</td>
    </tr>
  </tbody>
</table>
</div>



身長をソートします。デフォルトのパラメータ `ascending=True` は昇順です：



```python
df_demo.sort_values('Height').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Grade</th>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Junior</th>
      <th>Xiaoli Chu</th>
      <td>145.4</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Senior</th>
      <th>Gaomei Lv</th>
      <td>147.3</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Sophomore</th>
      <th>Peng Han</th>
      <td>147.8</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Senior</th>
      <th>Changli Lv</th>
      <td>148.7</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>Sophomore</th>
      <th>Changjuan You</th>
      <td>150.5</td>
      <td>40.0</td>
    </tr>
  </tbody>
</table>
</div>





```python
df_demo.sort_values('Height', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Grade</th>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">Senior</th>
      <th>Xiaoqiang Qin</th>
      <td>193.9</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>Mei Sun</th>
      <td>188.9</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>Gaoli Zhao</th>
      <td>186.5</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>Freshman</th>
      <th>Qiang Han</th>
      <td>185.3</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>Senior</th>
      <th>Qiang Zheng</th>
      <td>183.9</td>
      <td>87.0</td>
    </tr>
  </tbody>
</table>
</div>



ソートでは、同じ体重の場合、身長をソートし、身長の降順、体重の昇順を維持するなど、複数列のソートの問題がよく発生します：



```python
df_demo.sort_values(['Weight','Height'],ascending=[True,False]).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Grade</th>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sophomore</th>
      <th>Peng Han</th>
      <td>147.8</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Senior</th>
      <th>Gaomei Lv</th>
      <td>147.3</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Junior</th>
      <th>Xiaoli Chu</th>
      <td>145.4</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Sophomore</th>
      <th>Qiang Zhou</th>
      <td>150.5</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>Freshman</th>
      <th>Yanqiang Xu</th>
      <td>152.4</td>
      <td>38.0</td>
    </tr>
  </tbody>
</table>
</div>



インデックスソートの使用法は値ソートと完全に同じですが、要素の値はインデックスにあり、この場合はインデックスレイヤーの名前またはレイヤー番号を指定し、パラメータ `level` で表される必要があります。なお、文字列の配列順序はアルファベット順に決まることに注意してください。



```python
df_demo.sort_index(level=['Grade','Name'],ascending=[True,False]).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Grade</th>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Freshman</th>
      <th>Yanquan Wang</th>
      <td>163.5</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>Yanqiang Xu</th>
      <td>152.4</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>Yanqiang Feng</th>
      <td>162.3</td>
      <td>51.0</td>
    </tr>
    <tr>
      <th>Yanpeng Lv</th>
      <td>NaN</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>Yanli Zhang</th>
      <td>165.1</td>
      <td>52.0</td>
    </tr>
  </tbody>
</table>
</div>



### 6. applyメソッド
 `apply` メソッドは `DataFrame` の行反復または列反復によく使用され、その `axis` の意味は小節2の統計集約関数と一致し、 `apply` の引数は往々にしてシーケンスを入力とする関数です。たとえば、 `.mean()` の場合、 `apply` を使用して次のように書くことができます：



```python
df_demo = df[['Height', 'Weight']]
def my_mean(x):
     res = x.mean()
     return res
df_demo.apply(my_mean)
```




    Height    163.218033Weight     55.015873dtype: float64



同様に、 `lambda` 式を使って簡潔に書くことができます。ここで `x` は呼び出された `df_demo` テーブルに入力されたシーケンスを指します。



```python
df_demo.apply(lambda x:x.mean())
```




    Height    163.218033Weight     55.015873dtype: float64



 `axis=1` を指定すると、行要素の `Series` が関数に渡されるたびに、結果は以前の行単位の平均値と一致します。



```python
df_demo.apply(lambda x:x.mean(), axis=1).head()
```




    0    102.451    118.252    138.953     41.004    124.00dtype: float64



ここでもう1つの例を挙げます。 `mad` 関数は、シーケンスの平均から逸脱した絶対値の大きさの平均を返します。例えば、シーケンス1,3,7,10では、平均値は5.25であり、各要素の逸脱の絶対値は4.25,2.25,1.75,4.75であり、この逸脱シーケンスの平均値は3.25です。次に、増加と体重の `apply` メトリックを `mad` 計算します：



```python
df_demo.apply(lambda x:(x-x.mean()).abs().mean())
```




    Height     6.707229Weight    10.391870dtype: float64



これは、組み込みの `mad` 関数を使用して計算された結果と一致します。



```python
df_demo.mad()
```




    Height     6.707229Weight    10.391870dtype: float64



#### [WARNING] 慎重に使う `apply`

カスタム関数に渡される処理のおかげで、 `apply` の自由度は高いが、これはパフォーマンスのコストを伴う。一般的に、 `pandas` の組み込み関数処理と `apply` を使用して同じタスクを処理するのは、速度が大きく異なりますので、カスタマイズの必要性がある場合にのみ `apply` を検討してください。

#### [END]

## 4. ウィンドウオブジェクト
 `pandas` には、スライドウィンドウ `rolling`、拡張ウィンドウ `expanding`、指数重みウィンドウ `ewm` の3種類があります。なお、日付オフセットをウィンドウサイズとするスライディングウィンドウについては第10章で説明し、指数重みウィンドウについては本章の練習を参照してください。

### 1. スライドウィンドウオブジェクト
スライドウィンドウ関数を使用するには、まずシーケンスに `.rolling` を使用してスライドウィンドウオブジェクトを取得する必要があります。その最も重要な引数はウィンドウサイズ `window` です。



```python
s = pd.Series([1,2,3,4,5])
roller = s.rolling(window = 3)
roller
```




    Rolling [window=3,center=False,axis=0]



スライドウィンドウオブジェクトを取得したら、対応する集約関数を使用して計算できます。ウィンドウには現在の行がある要素が含まれていることに注意してください。たとえば、4番目の位置で平均値を計算する場合は、 (1+2+3)/3ではなく (2+3+4)/3を計算する必要があります：



```python
roller.mean()
```




    0    NaN1    NaN2    2.03    3.04    4.0dtype: float64





```python
roller.sum()
```




    0     NaN1     NaN2     6.03     9.04    12.0dtype: float64



スライド相関係数またはスライド共分散の計算については、次のように書くことができます。



```python
s2 = pd.Series([1,2,6,16,30])
roller.cov(s2)
```




    0     NaN1     NaN2     2.53     7.04    12.0dtype: float64





```python
roller.corr(s2)
```




    0         NaN1         NaN2    0.9449113    0.9707254    0.995402dtype: float64



また、 `apply` を使用してカスタム関数を受信することもサポートされています。受信値は対応するウィンドウの `Series` です。たとえば、上記の平均関数と同等に表すことができます：



```python
roller.apply(lambda x:x.mean())
```




    0    NaN1    NaN2    2.03    3.04    4.0dtype: float64



 `shift, diff, pct_change` は、共通引数 `periods=n` で、デフォルトは1であり、方向前の `n` 番目の要素の値、前方の `n` 番目の要素との差分（ `n` 番目の差分とは異なり）、前方の `n` 番目の要素と比較して成長率を計算します。ここで `n` は負であり、逆方向の同様の操作を示すことができます。



```python
s = pd.Series([1,3,6,10,15])
s.shift(2)
```




    0    NaN1    NaN2    1.03    3.04    6.0dtype: float64





```python
s.diff(3)
```




    0     NaN1     NaN2     NaN3     9.04    12.0dtype: float64





```python
s.pct_change()
```




    0         NaN1    2.0000002    1.0000003    0.6666674    0.500000dtype: float64





```python
s.shift(-1)
```




    0     3.01     6.02    10.03    15.04     NaNdtype: float64





```python
s.diff(-2)
```




    0   -5.01   -7.02   -9.03    NaN4    NaNdtype: float64



これらをスライドウィンドウ関数とみなす理由は、これらの机能を `n+1` ウィンドウサイズの `rolling` メソッドに置き換えることができるからです。



```python
s.rolling(3).apply(lambda x:list(x)[0]) # s.shift(2)
```




    0    NaN1    NaN2    1.03    3.04    6.0dtype: float64





```python
 s.rolling(4).apply(lambda x:list(x)[-1]-list(x)[0]) # s.diff(3)
```




    0     NaN1     NaN2     NaN3     9.04    12.0dtype: float64





```python
def my_pct(x):
     L = list(x)
     return L[-1]/L[0]-1
s.rolling(2).apply(my_pct) # s.pct_change()
```




    0         NaN1    2.0000002    1.0000003    0.6666674    0.500000dtype: float64



#### [練習して練習する]

 `rolling` オブジェクトのデフォルトのウィンドウ方向はすべて前方です。場合によっては、ユーザーが後方のウィンドウを必要とする場合があります。例えば、1,2,3に対して後方のウィンドウを2に設定した `sum` 操作は、結果は3,5,NaNです。この場合、後方のスライドウィンドウ操作をどのように実現すればよいでしょうか？

#### [END]

### 2. 拡張ウィンドウ
拡張ウィンドウは累積ウィンドウとも呼ばれ、動的長さのウィンドウと理解することができ、そのウィンドウのサイズはシーケンスの開始から具体的な操作の対応する位置までであり、それが使用する集約関数はこれらの段階的に拡張されるウィンドウに作用する。具体的には、系列をa1，a2，a3，a4とすると、その各位置に対応するウィンドウは\ [a1\]、\ [a1，a2\]、\ [a1，a2，a3\]、\ [a1，a2，a3，a4\] である。



```python
s = pd.Series([1, 3, 6, 10])
s.expanding().mean()
```




    0    1.0000001    2.0000002    3.3333333    5.000000dtype: float64



#### [練習して練習する]

 `cummax, cumsum, cumprod` 関数は典型的なクラス拡張ウィンドウ関数です。 `expanding` オブジェクトを使用して順次実装してください。

#### [END]

## 五、練習
### Ex1：ポケモンデータセット
既存のポケモンのデータセットがあります。以下にいくつかの背景を説明します：

*  `#` 全国図鑑番号を表し、行ごとに同じ数字があれば、その妖怪の異なる状態を表す

* 妖怪には単一属性と二重属性の2種類があり、単一属性の妖怪の場合、 `Type 2` は欠落値です。
*  `Total, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed` それぞれ種族値、体力、物体攻撃、防御、特攻、特防、速度を表し、種族値は後の6項目の合計である



```python
df = pd.read_csv('../data/pokemon.csv')
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>



1.  `HP, Attack, Defense, Sp. Atk, Sp. Def, Speed` を合計し、 `Total` 値であるかどうかを確認します。

2.  `#` 重複するモンスターの場合、最初のレコードのみを保持します。次の問題を解決します：

* 第1属性の種類数と上位3多数に対応する種類を求める
* 第1属性と第2属性の組み合わせ種類を求める
* まだ表示されていない属性の組み合わせを探す

3. 次のように `Series` を構築します：

* 対象攻撃を取り出し、120以上の場合は `high`、50未満の場合は `low`、そうでない場合は `mid` に置き換えます。
* 最初の属性を取り出し、すべての文字をそれぞれ `replace` と `apply` で大文字に置き換えます
* 各妖怪の6つの能力の偏差、すなわち、すべての能力の中で中央値から最も大きくずれた値を求め、 `df` に加算し、大きいから小さい順に並べ替えます。

### Ex2指数重み付けウィンドウ
1. 拡張ウィンドウとしての `ewm` ウィンドウ

拡張ウィンドウでは、ユーザーはさまざまな関数を使用して履歴の累積指標統計を行うことができますが、これらの組み込み統計関数は、ウィンドウ内のすべての要素に同じ重みを付与することが多いです。実際、ウィンドウ内の要素に異なる重みを与えることができ、指数重みウィンドウはそのような特殊な拡張ウィンドウである。

このうち、最も重要な引数は `alpha` で、デフォルトのウィンドウ重みを$w_i= (1−\alpha) ^i,i\in\ {0,1,...,t\} $に決定します。ここで、$i=t$は現在の要素を表し、$i=0$はシーケンスの最初の要素を表します。

重み式から分かるように、現在の値から離れるほど重みは小さくなり、元の系列を$x$、更新された現在の要素を$y_t$とすると、重み式で正規化すると、次のことがわかる：

$$
\begin{split}y_t &=\frac{\sum_{i=0}^{t} w_i x_{t-i}}{\sum_{i=0}^{t} w_i} \\
&=\frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ...
+ (1 - \alpha)^{t} x_{0}}{1 + (1 - \alpha) + (1 - \alpha)^2 + ...
+ (1 - \alpha)^{t}}\\\end{split}
$$

 `Series` の場合、指数平滑化されたシーケンスは、 `ewm` オブジェクトを使用して次のように計算できます：



```python
np.random.seed(0)
s = pd.Series(np.random.randint(-1,2,30).cumsum())
s.head()
```




    0   -11   -12   -23   -24   -2dtype: int32





```python
s.ewm(alpha=0.2).mean().head()
```




    0   -1.0000001   -1.0000002   -1.4098363   -1.6097564   -1.725845dtype: float64



 `expanding` ウィンドウで実装してください。

2. スライドウィンドウとしての `ewm` ウィンドウ

質問1からわかるように、 `ewm` は、拡張ウィンドウの特別なケースとして、シーケンスの最初の要素からのみ重み付けることができます。制限されたウィンドウ `n` を与えて、ウィンドウとして自分自身を含む最も近い `n` 要素のみをスライディング重み付けスムージングしたいと思います。スライドウィンドウ関数に基づいて、新しい `wi` と `yt` の更新式を与え、これを `rolling` ウィンドウで実装します。
