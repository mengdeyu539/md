<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"> <ul class="toc-item"><li><span><a href="#基本指标" data-toc-modified-id="基本指标-1"><span class="toc-item-num">1 &nbsp;&nbsp;</span>基本指標 </a></span><li><span></span>2 </li><li><a href="#柱状图" data-toc-modified-id="柱状图-3"><a href="#柱状图" data-toc-modified-id="柱状图-3"><span class="toc-item-num"><a href="#柱状图" data-toc-modified-id="柱状图-3"><span class="toc-item-num">&nbsp;&nbsp;ランダムジェネレータ </a></span><ul class="toc-item"><span><a href="#Generator" data-toc-modified-id="Generator-5.1"><span class="toc-item-num">5.1 &nbsp;&nbsp;</a><li><span><a href="#并行" data-toc-modified-id="并行-5.2">&nbsp;&nbsp;</li></li>&nbsp;&nbsp;<gt r=整数ランダム系列</a></span></li><li><span><a href="#均匀随机序列" data-toc-modified-id="均匀随机序列-5.6"><span class="toc-item-num">5.6 &nbsp;&nbsp;</span>均一ランダム系列 </a></span><li><a href="#随机采样" data-toc-modified-id="随机采样-5.7"><span class="toc-item-num">5.7 &nbsp;&nbsp;</span>ランダムサンプリング </a><span><span class="toc-item-num"></a></li><gt r="348"対数相関を指す</a></span></li><li><span><span class="toc-item-num">7.4 &nbsp;&nbsp;</span>検査関連 </a></ul></li><span class="toc-item-num">8 </span><span class="toc-item-num"></div>



```python
import numpy as np
np.__version__
```




    '1.22.3'



ドキュメントの読み取り手順：

- 🐧はTipを示します
- ⚠️注意事項を示す

一般的な基礎APIの一部は『小白から入門まで』で紹介されているので、ここでは説明しません。

## 基本指標

主に平均、中央、分散、標准偏差 - 非値（NaN）をサポートする場合。



```python
a = np.array([
    [1, 2, 3, 4],
    [5, np.nan, np.nan, 6],
    [7, 8, np.nan, 9]
])
a
```




    array([[ 1.,  2.,  3.,  4.],
           [ 5., nan, nan,  6.],[ 7.,  8., nan,  9.]])



平均値：



```python
np.average(a)
```




    nan





```python
np.nanmean(a)
```




    5.0





```python
np.mean(a, axis=0)
```




    array([4.33333333,        nan,        nan, 6.33333333])





```python
np.nanmean(a, axis=0), 13/3, 10/2, 3/1, 19/3
```




    (array([4.33333333, 5.        , 3.        , 6.33333333]),
     4.333333333333333,5.0,3.0,6.333333333333333)



中央値：



```python
np.median(a)
```




    nan





```python
np.nanmedian(a)
```




    5.0



標準偏差：



```python
np.nanstd(a)
```




    2.581988897471611





```python
np.nanvar(a, axis=1)
```




    array([1.25      , 0.25      , 0.66666667])



分位数：



```python
a
```




    array([[ 1.,  2.,  3.,  4.],
           [ 5., nan, nan,  6.],[ 7.,  8., nan,  9.]])





```python
np.percentile(a, 25)
```




    nan





```python
# 百分位
np.nanpercentile(a, 25)
```




    3.0





```python
# 分位数
np.nanquantile(a, 0.25)
```




    3.0





```python
a = np.arange(12).reshape(3, 4)
a
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],[ 8,  9, 10, 11]])





```python
# 极值
np.ptp(a)
```




    11





```python
np.ptp(a, axis=0)
```




    array([8, 8, 8, 8])





```python
np.ptp(a, axis=1)
```




    array([3, 3, 3])



## 相関性

 `correlate` 2つの1次元配列の相互相関を計算します。引数：

- 配列：a,v
- mode：valid、同じ、またはfull、デフォルトのvalidで、ch04の「畳み込み」のmodeと同じ意味です。



```python
a = np.array([1,2,3])
b = np.array([4,5,6])
```



```python
np.correlate(a, b), np.sum(a * np.conj(b))
```




    (array([32]), 32)





```python
np.correlate(a, b, "same")
```




    array([17, 32, 23])





```python
(
    np.sum(np.array([0,1,2])*np.array([4,5,6])),
    np.sum(np.array([1,2,3])*np.array([4,5,6])),
    np.sum(np.array([2,3,0])*np.array([4,5,6])),
)
```




    (17, 32, 23)





```python
np.correlate(a, b, "full")
```




    array([ 6, 17, 32, 23, 12])





```python
(
    np.sum(np.array([0,0,1])*np.array([4,5,6])),
    np.sum(np.array([0,1,2])*np.array([4,5,6])),
    np.sum(np.array([1,2,3])*np.array([4,5,6])),
    np.sum(np.array([2,3,0])*np.array([4,5,6])),
    np.sum(np.array([3,0,0])*np.array([4,5,6])),
)
```




    (6, 17, 32, 23, 12)



 `corrcoef` はピアソン相関係数、パラメータ：

- 配列
- rowvar：デフォルトはTrueです。Trueの場合、各行は変数を表し、Falseの場合、各列は変数を表します。

式：

$$
R_{ij} = \frac{ C_{ij} } { \sqrt{ C_{ii} * C_{jj} } }
$$



```python
a * b /np.sqrt((a*a) * (b*b))
```




    array([1., 1., 1.])





```python
np.corrcoef(a,b)
```




    array([[1., 1.],
           [1., 1.]])



 `cov` は、次のパラメータを含む共分散行列です。

- 配列
- rowvar： `corrcoef` と同じブール値。
- bias：ブール値、デフォルトはFalse、正規化値はN-1、Nは観測数、Trueの場合、正規化値はNです。ddofで上書きされます。
- ddof： `int`、1または0を取ることができます。1の場合はN-1を意味します（次の2つのweightsの設定にかかわらず）、0の場合はNを意味します。
- fweights：各観測値の繰り返し回数を表す周波数重み。
- aweights：観測ベクトルの重み、重要な相対重みは大きく、重要でない相対重みは小さい。ddof=0の場合は、観測ベクトルに確率を割り当てるために使用できます。


式：
$$
\operatorname{cov}_{x, y}=\frac{\sum\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{N-1}
$$

ここで、uおよびvはそれぞれXおよびYの期待値である。

返される行列の `c[i][j]` は `x[i]` と `x[j]` の共分散であり、 `c[i][i]` は `x{i]` の分散である。



```python
np.cov(a,b)
```




    array([[1., 1.],
           [1., 1.]])





```python
np.sum((a-np.average(a)) * (b-np.average(b)))/2
```




    1.0





```python
x = np.array([[0, 2], [1, 1], [2, 0]]).T
x
```




    array([[0, 1, 2],
           [2, 1, 0]])





```python
# 默认每一行是一个变量
# 对角线是方差，反对角线是协方差
np.cov(x)
```




    array([[ 1., -1.],
           [-1.,  1.]])





```python
# [0-1, 1-1, 2-1] * [2-1, 1-1, 0-1] = -1 + 0 + -1 = -2 
# -2 / (3-1) = -1
```



```python
# 再来一个例子
x = np.array([-2.1, -1,  4.3])
y = np.array([3,  1.1,  0.12])
X = np.stack((x, y), axis=0)
X
```




    array([[-2.1 , -1.  ,  4.3 ],
           [ 3.  ,  1.1 ,  0.12]])





```python
np.cov(X), np.cov(x, y)
```




    (array([[11.71      , -4.286     ],
            [-4.286     ,  2.14413333]]),
     array([[11.71      , -4.286     ],
            [-4.286     ,  2.14413333]]))





```python
np.cov(x), np.cov(y)
```




    (array(11.71), array(2.14413333))





```python
np.sum((x - np.average(x)) * (y - np.average(y))) / (3-1)
```




    -4.2860000000000005



## 棒グラフ

棒グラフは実際には異なるデータ分布を表したものです。



```python
from collections import Counter
```



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 5, (3, 4))
a
```




    array([[0, 3, 3, 2],
           [2, 4, 0, 3],[1, 0, 2, 4]])



 `histogram` いくつかの引数を受け入れます：

- 配列が引き分けられます
- bins：デフォルト10、1つ `int` または1つ `int` または1つ `str`、1つは等間隔で、もう1つは右端を含む単調増加するエッジの配列で、不均一なbin幅を許容します。 `str` の場合は、以下の `histogram_bin_edges` インタフェースを参照してください。
- range：Tupleの浮動小数点数。binの上下境界。提供されない場合はデフォルトで `(a.min(), a.max())` に設定されます。超過は無視されます。最初の要素は2番目より小さくなければなりません。
- weights：aと同じshapeの数のセットで、 `density=True` の場合は正規化されます。
- density：ブール値、Falseの場合はbinごとのサンプル数を含み、Trueの場合は確率密度関数となります。

histとbinの境界を返します。最後の境界セットを除いて、他の境界はすべて左閉右開です。



```python
sa = sorted(Counter(a.flatten()).items())
sa
```




    [(0, 3), (1, 1), (2, 3), (3, 3), (4, 2)]





```python
# 注意：3个数其实是2个区间
np.histogram(a, bins=[0, 3, 5]), np.sum([v[1] for v in sa if v[0] < 3]), np.sum([v[1] for v in sa if v[0] >= 3])
```




    ((array([7, 5]), array([0, 3, 5])), 7, 5)





```python
total = a.shape[0] * a.shape[1]
sa = [(v[0], v[1]/total) for v in sorted(Counter(a.flatten()).items())]
sa
```




    [(0, 0.25),
     (1, 0.08333333333333333),(2, 0.25),(3, 0.25),(4, 0.16666666666666666)]





```python
(
    np.histogram(a, bins=[0, 3, 5], density=True), 
    np.sum([v[1] for v in sa if v[0] < 3])/3, np.sum([v[1] for v in sa if v[0] >= 3])/2
)
```




    ((array([0.19444444, 0.20833333]), array([0, 3, 5])),
     0.19444444444444442,0.20833333333333331)





```python
# 带权重的
np.histogram(a, bins=[0, 3, 5], weights=a,  density=True)
```




    (array([0.09722222, 0.35416667]), array([0, 3, 5]))





```python
a.flatten()
```




    array([0, 3, 3, 2, 2, 4, 0, 3, 1, 0, 2, 4])





```python
# 0个0，3个3 …… 4个4
b = [3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 3, 3, 3, 1, 2, 2, 4, 4, 4, 4]
```



```python
np.histogram(b, bins=[0,3,5], density=True)
```




    (array([0.09722222, 0.35416667]), array([0, 3, 5]))



 `histogram2d` は前のインタフェースの2次元バージョンで、2つの配列 (1次元) を処理します。

- x,y：xとyの一次元配列
- bins：意味は上記と同じですが、より多くの場合（ただしstringはサポートされていません）、サポートされています：
  - xとyは、 `int`、 `array` を共有します。
  - xとyはそれぞれ `[int,int]`、 `[array, array]`
  -  `int` はbinの数を示し、 `array` はグループ化境界を示します。 `[int, array]`、 `[array, int]`
- レンジ：上記と同じ
- density：前述と同じ、デフォルトはFalseで、binあたりのサンプル数を返し、Trueではbinの確率密度を返します。
- weights：上記と同じ

xとyがまったく交点がなければ、実はそれぞれと同じですが、両者に交点があるのは二次元平面上で統計されます。



```python
a
```




    array([[0, 3, 3, 2],
           [2, 4, 0, 3],[1, 0, 2, 4]])





```python
rng = np.random.default_rng(42)
c = rng.integers(5, 10, (3, 4))
c
```




    array([[5, 8, 8, 7],
           [7, 9, 5, 8],[6, 5, 7, 9]])





```python
rng = np.random.default_rng(42)
d = rng.integers(0, 5, (3, 4))
d
```




    array([[0, 3, 3, 2],
           [2, 4, 0, 3],[1, 0, 2, 4]])





```python
Hac,xe,ye =np.histogram2d(a.flatten(), c.flatten(), bins=[[0,3,5],[5,9,10]])
Hac
```




    array([[7., 0.],
           [3., 2.]])





```python
(a.flatten() < 3).sum(), (a.flatten() >=3).sum(), 7+0, 3+2
```




    (7, 5, 7, 5)





```python
(c.flatten()<9).sum(), (c.flatten()>=9).sum(), 7+3, 0+2
```




    (10, 2, 10, 2)





```python
Had, xe,ye = np.histogram2d(a.flatten(), d.flatten(), bins=[[0,3,4], [0,2,4]])
Had
```




    array([[4., 3.],
           [0., 5.]])





```python
(a.flatten() < 3).sum(), (a.flatten() >=3).sum(), 4+3, 0+5
```




    (7, 5, 7, 5)





```python
(d.flatten()<2).sum(), (d.flatten()>=2).sum(), 4+0,3+5
```




    (4, 8, 4, 8)





```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(131, title='imshow: square bins')
plt.imshow(Had, interpolation='nearest', origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]])
```




    <matplotlib.image.AxesImage at 0x1137c5af0>




    
![png](ch05-probability_statistics_files/ch05-probability_statistics_70_1.png)
    


 `histogramdd` は多次元バージョンであり、入力された配列以外のパラメータはほぼ前のパラメータと同じです。



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (100, 3))
```



```python
H, edges = np.histogramdd(a, bins=[5, 3, 4])
```



```python
H.shape
```




    (5, 3, 4)





```python
edges
```




    [array([0. , 1.8, 3.6, 5.4, 7.2, 9. ]),
     array([0., 3., 6., 9.]),array([0.  , 2.25, 4.5 , 6.75, 9.  ])]



 `histogram_bin_edges` binの境界を計算するために使用され、境界のみを計算します。

ただし、ここのbinパラメータはstringをサポートしており、ここでは多くのメソッドが定義されています：

- auto：折衷的な方法で良い効果が得られ、小さなデータセットでは一般的に採用されます
- fd：Freedam Diaconis Estimator、大データセットで一般的に使用されます。

$$
2 \frac {IQR} {\sqrt[3] {n}}
$$

- scott：標準偏差に比例し、データセットのサイズの立方根に反比例し、小さいデータセットには保守的すぎるが、大きいデータセットには比例している。標準偏差は異常値に対してあまりロバストではありません。異常値がない場合、その値はFDとよく似ています。

$$
h = \sigma \sqrt[3]{\frac{24 * \sqrt{\pi}}{n}}
$$

- rice：binの数はデータセットのサイズの立方根に反比例し、バーの数を過大評価する傾向があり、データの可変性を考慮しない。

$$
n_h = 2 n^{1/3}
$$

- sturges：データセットsizeの2はベース対数であり、この推定器はデータが正規であると仮定し、大きな非正規データセットに対しては保守的すぎる。R言語の `hist` のデフォルトメソッド。

$$
n_h = \log_{2}{n} +1
$$

- doane：非正常なデータセットに対してより良い推定値を生成するSturgesの改良されたバージョン。この推定器は、データのスキューを解釈しようとします。

$$
n_h = 1 + \log_{2}(n) + \log_{2} (1 + \frac {|g_1|} {\sigma_{g_1}}) \\
g_1 = mean[(\frac{x - \mu}{\sigma})^3]\\
\sigma_{g_1} = \sqrt{\frac{6(n - 2)}{(n + 1)(n + 3)}}
$$

- sqrt：データセットのサイズのみを考慮する最も簡単かつ迅速な推定器。

$$
n_h = \sqrt n
$$



```python
# 随机数本来就是0-9
np.histogram_bin_edges(a, bins=3)
```




    array([0., 3., 6., 9.])





```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, 500)
```



```python
np.histogram_bin_edges(a, bins="auto")
```




    array([0. , 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9. ])



 `fd`：



```python
IQR = np.percentile(a, 75) - np.percentile(a, 25)
```



```python
2 * IQR / (a.size ** (1/3))
```




    1.2599210498948732





```python
np.histogram_bin_edges(a, bins="fd")
```




    array([0.   , 1.125, 2.25 , 3.375, 4.5  , 5.625, 6.75 , 7.875, 9.   ])



`scott`:



```python
np.histogram_bin_edges(a, bins="scott")
```




    array([0.   , 1.125, 2.25 , 3.375, 4.5  , 5.625, 6.75 , 7.875, 9.   ])





```python
(24*np.sqrt(np.pi)/a.size)**(1/3)* np.std(a)
```




    1.2524886807479167



`rice`:



```python
# 16个区间（柱子）
np.histogram_bin_edges(a, bins="rice").shape
```




    (17,)





```python
2 * (a.size**(1/3))
```




    15.874010519681994



`sturges`:



```python
np.histogram_bin_edges(a, bins="sturges")
```




    array([0. , 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9. ])





```python
np.histogram_bin_edges(a, bins="sturges").shape
```




    (11,)





```python
np.log2(a.size) + 1
```




    9.965784284662087



 `doane`：



```python
np.histogram_bin_edges(a, bins="doane").shape
```




    (11,)





```python
(1+
 np.log2(a.size)+
 np.log2(1+abs(np.mean(((a - np.mean(a))/np.std(a))**3)) / np.sqrt(6*(a.size-2)/((a.size+1)*(a.size+3)))))
```




    9.970039609658638



`sqrt`:



```python
np.histogram_bin_edges(a, bins="sqrt").size
```




    24





```python
np.sqrt(a.size)
```




    22.360679774997898



 `auto`： `fd` と `sturges` の大きい方が選択されます。



```python
np.histogram_bin_edges(a, bins="fd").size, np.histogram_bin_edges(a, bins="sturges").size
```




    (9, 11)





```python
np.histogram_bin_edges(a, "auto").size
```




    11



 `digitize` は、入力された配列がどのbinに属しているかを示します。



```python
x = np.array([0.2, 6.4, 3.0, 1.6])
bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
inds = np.digitize(x, bins)
```



```python
# 0.0 <= 0.2 < 1.0
# 4.0 <= 6.4 < 10.0
# 2.5 <= 3.0 < 4.0
# 1.0 <= 1.6 < 2.5
inds
```




    array([1, 4, 3, 2])



## カウント

 `bincount` 非負の配列内の各値の出現回数を数えるために使用されます。それは棒グラフにも関係しています。



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, 100)
```



```python
sorted(Counter(a).items())
```




    [(0, 9),
     (1, 9),(2, 6),(3, 9),(4, 15),(5, 7),(6, 11),(7, 15),(8, 12),(9, 7)]





```python
np.bincount(a)
```




    array([ 9,  9,  6,  9, 15,  7, 11, 15, 12,  7])





```python
b = np.array([1, 1, 1, -1, -1, -1])
```



```python
# 必须非负
np.bincount(b)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-765-54605025760c> in <module>
          1 # 必须非负
    ----> 2 np.bincount(b)
    

    /usr/local/lib/python3.8/site-packages/numpy/core/overrides.py in bincount(*args, **kwargs)
    

    ValueError: 'list' argument must have no negative elements




```python
# 必须一维
np.bincount(np.array([[1],[2]]))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-758-f01c373d0299> in <module>
          1 # 必须一维
    ----> 2 np.bincount(np.array([[1],[2]]))
    

    /usr/local/lib/python3.8/site-packages/numpy/core/overrides.py in bincount(*args, **kwargs)
    

    ValueError: object too deep for desired array


さらに、ウェイトを指定できます：

`out[n] = out[n] + weight[i]`

デフォルトは次のとおりです：

`out[n] = out[n] + 1`

注：与えられた配列は位置であり、重みはこれらの位置に対応します。



```python
c = np.array([2,3,4,2,3,4])
```



```python
# 在第2 3 4 个位置上+1
# 在第2 3 4 个位置上+1
np.bincount(c)
```




    array([0, 0, 2, 2, 2])





```python
# 在第2 3 4 个位置上+2 3 4
# 在第2 3 4 个位置上+2 3 4
np.bincount(c, weights=[2,3,4,2,3,4])
```




    array([0., 0., 4., 6., 8.])





```python
# loc 0  0.3
# loc 1  0.5+0.2
# loc 2  0.7+1.0-0.6
w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6])
x = np.array([0, 1, 1, 2, 2, 2])
np.bincount(x,  weights=w)
```




    array([0.3, 0.7, 1.1])



返されるbinの数を指定することもできます：



```python
np.bincount(a)
```




    array([ 9,  9,  6,  9, 15,  7, 11, 15, 12,  7])





```python
# 多出2个数字
np.bincount(a, minlength=12)
```




    array([ 9,  9,  6,  9, 15,  7, 11, 15, 12,  7,  0,  0])



 `count_nonzero` ゼロ以外の数量を数えるために使用されます。



```python
# 统计非0
np.count_nonzero(np.eye(4))
```




    4





```python
np.count_nonzero(a)
```




    453





```python
(a>0).sum()
```




    453



## ランダムジェネレータ

コンピュータ上の乱数はすべて擬似乱数です。擬似ランダムとは、確定的なアルゴリズムで生成されたランダムに見えるが、実際にはランダムではないプロセスである。スタート値が変わらなければ乱数の順序も変わらない。計算が簡単であり、少ない数値でそのアルゴリズムを推計するのが難しいというメリットがある。一般的には、擬似乱数計算の開始値として、コンピュータ上の時間などの偽乱数が使用されます。時には、真のランダムさが必要な場合があります（例えば、暗号化された財布を作成するとき）。この場合、通常、マウスを勝手に動かすように要求され、この場合、マウスが停止したときの位置はランダムの開始値として扱われます。

まず順序が変わらないとは何かを見てみましょう：



```python
rng = np.random.default_rng(42)
rng.integers(1, 10, 2)
```




    array([1, 7])





```python
rng = np.random.default_rng(42)
rng.integers(1, 10, 5)
```




    array([1, 7, 6, 4, 4])





```python
rng = np.random.default_rng(42)
rng.integers(1, 10, 8)
```




    array([1, 7, 6, 4, 4, 8, 1, 7])



ランダムシードを指定すると、このシードによって生成されるランダム数列は変わりません。

よく使われる擬似乱数生成アルゴリズムには、線形同数法、二乗中央取り法、M-系列、キャリー乗数法、メーソン回転アルゴリズム、擬似乱数2進数列などが含まれる。C/C++に組み込まれた擬似乱数生成アルゴリズムである線形同一法を簡単に紹介します。

$$
X_{n+1} = (aX_n + c)(\mod m)\\
m > 0, 0<a<m, 0<=x_0<=m
$$

mはモジュール、aは乗算係数、cは増分、x0は初期値（シード）である。

m a cは一般的に自分で考える必要はなく、すでに多くの研究が様々な最適値を見つけている。詳細を参照：[维基百科](https://en.wikipedia.org/wiki/Linear_congruential_generator)。たとえば、m=2^16+1 a=75 c=74のグループの1つを使用します。



```python
def lcg(n=1, seed=42):
    for i in range(n):
        seed =  (75*seed + 74) % (2**16 + 1)
        yield seed
```



```python
list(lcg(5))
```




    [3224, 45263, 52412, 64291, 37698]



小数を取得するには、取得した整数を最大値で割って乱数を0 - 1区間にマッピングすることができます。

NumPyの最新バージョンは、PCG-64 (Permutation Congruence Generator）擬似乱数生成アルゴリズムを採用しています。 `RandomState` を使用するのではなく、 `Generator` は乱数生成器の推奨コンストラクタです。


注：新しいバージョンでは、サンプリングに基づくMr.Cheng `Generator` を使用することをお勧めします。これも私たちがこれまで使ってきた方法です。

ただし、古いコードが古い方法を使用している場合があります。古いインタフェースについては、以下を参照してください：

-  [Legacy Random Generation-NumPy v1.24.de v0マニュアル](https://numpy.org/devdocs/reference/random/legacy.html)
-  [What’s New or Different-NumPy v1.24.de v0マニュアル](https://numpy.org/devdocs/reference/random/new-or-different.html)

### Generator

NumPyの乱数生成は、 `BitGenerator` を使用してシーケンスを生成し、次に、 `Generator` を使用してランダムシーケンスを特定の確率分布に変換します。

 `Generator` 生成されたランダム値は `BitGenerator` から生成されます。ただし、 `BitGenerators` は乱数を直接提供するものではなく、種子のメソッド、ステータスの取得または設定、ステータスのジャンプまたは進行、および提供された機能に効率的にアクセスできるコードのための低レベルのラッパーへのアクセスのみを含んでいます。


サポートされているBitGenerator：

- PCG64
- PCG64DXSM
- MT19937
- Philox
- SFC64

以下、PCG64を例に挙げて説明する。



```python
# 指定种子
bg = np.random.PCG64(seed=42)
```



```python
# 使用操作系统的熵作为种子
ss = np.random.SeedSequence()
ss
```




    SeedSequence(
        entropy=125958451863476122535671492956036760397,
    )





```python
bgr = np.random.PCG64(ss.entropy)
```

オペレーティングシステムによって収集されるエントロピーは128ビット整数を使用し、デフォルトのシードです。32ビット以下の小さなシードは一般的な用途には推奨されません。小さなシードが大きな状態空間をインスタンス化することは、いくつかの初期状態が到達できないことを意味します。

Generatorを使用するには：



```python
pcg = np.random.PCG64()
pcg
```




    <numpy.random._pcg64.PCG64 at 0x117298720>





```python
rng = np.random.Generator(pcg)
rng
```




    Generator(PCG64) at 0x1172B33C0





```python
rng.standard_normal()
```




    -1.5260230207039922





```python
rng.bit_generator is pcg
```




    True



### 並列

NumPyは3つの戦略を実装しており、復数のプロセス（ローカルまたは分散）にわたって繰り返し可能な擬似乱数を生成するために使用できます。

最初はSeedSequence spawningです。SeedSequenceは、ユーザーから提供されたシードを、通常は何らかのサイズの整数として処理し、それをBitGeneratorの初期状態に変換するアルゴリズムを実装します。低品質のシードが高品質の初期状態に変わることを保証するために、ハッシング技術を使用します（少なくとも、非常に高い確率で）。



```python
ss = np.random.SeedSequence(42)
```



```python
child_seeds = ss.spawn(10)
```



```python
bgs = [np.random.PCG64(s) for s in child_seeds]
bgs
```




    [<numpy.random._pcg64.PCG64 at 0x1199777d0>,
     <numpy.random._pcg64.PCG64 at 0x1173d9bf0>,<numpy.random._pcg64.PCG64 at 0x1173d9f60>,<numpy.random._pcg64.PCG64 at 0x1173d9040>,<numpy.random._pcg64.PCG64 at 0x119b7cf60>,<numpy.random._pcg64.PCG64 at 0x119b7c930>,<numpy.random._pcg64.PCG64 at 0x119b7cb40>,<numpy.random._pcg64.PCG64 at 0x119b7cbf0>,<numpy.random._pcg64.PCG64 at 0x119b7ce00>,<numpy.random._pcg64.PCG64 at 0x119b7ceb0>]





```python
streams = [np.random.Generator(bg) for bg in bgs]
```



```python
streams
```




    [Generator(PCG64) at 0x119BC69E0,
     Generator(PCG64) at 0x119BC6AC0,Generator(PCG64) at 0x119BC6BA0,Generator(PCG64) at 0x119BC6C80,Generator(PCG64) at 0x119BC6D60,Generator(PCG64) at 0x119BC6E40,Generator(PCG64) at 0x119BC6F20,Generator(PCG64) at 0x119B4E040,Generator(PCG64) at 0x119B4E120,Generator(PCG64) at 0x119B4E200]



子SeedSequenceは孫オブジェクトを生成し続けることもできます：



```python
# default_rng = Generator(BitGenerator)
grandchildren_seeds = child_seeds[0].spawn(4)
grand_streams = [
    np.random.default_rng(s) for s in grandchildren_seeds]
```



```python
grand_streams
```




    [Generator(PCG64) at 0x117376F20,
     Generator(PCG64) at 0x1173BAC80,Generator(PCG64) at 0x1173BA580,Generator(PCG64) at 0x1173BA900]



それから `Philox`、カウンタベースのRNG（ランダムナンバージェネレータ）は、弱い暗号化プリミティブを用いてカウンタを暗号化して増加させることで値を生成します。

シードは暗号化に使用されるキーを決定し、ユニークなキーはユニークで独立したストリームを作成します。 `Philox` シードをバイパスして128ビットキーを直接設定することができます。類似しているが異なる鍵は、独立したストリームを作成します。

 `Philox` のkeyとseedは異なるものであることに注意してください。



```python
import secrets
```



```python
root_seed = secrets.randbits(128)
```



```python
bgs = [
    np.random.Philox(key=root_seed + stream_id) 
    for stream_id in range(10)
]
```



```python
bgs
```




    [<numpy.random._philox.Philox at 0x1173e4040>,
     <numpy.random._philox.Philox at 0x1173e4130>,<numpy.random._philox.Philox at 0x1173e4220>,<numpy.random._philox.Philox at 0x1173e4310>,<numpy.random._philox.Philox at 0x1173e4400>,<numpy.random._philox.Philox at 0x1173e44f0>,<numpy.random._philox.Philox at 0x1173e45e0>,<numpy.random._philox.Philox at 0x1173e46d0>,<numpy.random._philox.Philox at 0x1173e47c0>,<numpy.random._philox.Philox at 0x1173e48b0>]



最后の1つはjumpを使用して、jumpはたくさんの乱数を抽出したかのように `BitGenerator` の状態を推進し、その状態を持つ新しいインスタンスを返します。


|BitGenerator|周期|jump大小|位数/每次抽取|
|-------------|----|------|-----------|
|PCG64|2^128|2^127|64|
|PCG64DXSM|2^128|2^127|64|
|MT19937|2^19937-1|2^128|32|
|Philox|2^256|2^128|64|

PCG64およびPCG64DXSMのジャンプサイズは実際に次のとおりです： `(黄金比例-1)*2^128`

次に具体的な例を見てみましょう：



```python
bg = np.random.PCG64(42)
```



```python
bgs = [bg.jumped(i) for i in range(10)]
```



```python
bgs
```




    [<numpy.random._pcg64.PCG64 at 0x1199f2250>,
     <numpy.random._pcg64.PCG64 at 0x117209e00>,<numpy.random._pcg64.PCG64 at 0x1172db720>,<numpy.random._pcg64.PCG64 at 0x1173e2eb0>,<numpy.random._pcg64.PCG64 at 0x1173e2930>,<numpy.random._pcg64.PCG64 at 0x1173e21a0>,<numpy.random._pcg64.PCG64 at 0x1173e2ca0>,<numpy.random._pcg64.PCG64 at 0x1173e2040>,<numpy.random._pcg64.PCG64 at 0x1173e2a90>,<numpy.random._pcg64.PCG64 at 0x1173e23b0>]





```python
bg = np.random.PCG64(42)
bg.state
```




    {'bit_generator': 'PCG64',
     'state': {'state': 274674114334540486603088602300644985544,
      'inc': 332724090758049132448979897138935081983},
     'has_uint32': 0,'uinteger': 0}





```python
# 一个周期
bg.advance(2**128).state
```




    {'bit_generator': 'PCG64',
     'state': {'state': 274674114334540486603088602300644985544,
      'inc': 332724090758049132448979897138935081983},
     'has_uint32': 0,'uinteger': 0}





```python
# 前进一步
bg.jumped(1).state
```




    {'bit_generator': 'PCG64',
     'state': {'state': 246721301968239085263295379140720340427,
      'inc': 332724090758049132448979897138935081983},
     'has_uint32': 0,'uinteger': 0}





```python
# 等于前进jump size的步数
bg.advance(210306068529402873165736369884012333109).state
```




    {'bit_generator': 'PCG64',
     'state': {'state': 246721301968239085263295379140720340427,
      'inc': 332724090758049132448979897138935081983},
     'has_uint32': 0,'uinteger': 0}



また、大規模并列環境での `PCG64 BitGenerator` の使用は統計的に弱いことが証明されており、NumPyは新しい `PCG64DXSM BitGenerator` を導入しました。これは最終的に将来のリリースで使用される `default_rng` の新しいデフォルトの `BitGenerator` 実装となります。 `PCG64DXSM` 統計上の弱点を解決し、 `PCG64` のパフォーマンスと特性を維持します。

あなたが以下の場合：

- 1つの `Generator` インスタンスのみを使用する
-  `RandomState` または `numpy.random` の関数のみを使用する
-  `PCG64.jumped` メソッドのみを使用して並列ストリームを生成する
-  `PCG64` 以外の `BitGenerator` を明示的に使用する

この弱点は影響を与えません。

### マルチスレッド



```python
from numpy.random import default_rng, SeedSequence
import multiprocessing
import concurrent.futures

"""
代码来自：https://numpy.org/devdocs/reference/random/multithreading.html
"""

class MultithreadedRNG:
    def __init__(self, n, seed=None, threads=None):
        if threads is None:
            threads = multiprocessing.cpu_count()
        # 线程数量
        self.threads = threads

        seq = SeedSequence(seed)
        # 使用spawn生成threads个Generator
        self._random_generators = [default_rng(s)
                                   for s in seq.spawn(threads)]

        self.n = n
        # 生成对应的线程executor
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)
        # 存储value
        self.values = np.zeros(n)
        
        self.step = np.ceil(n / threads).astype(np.int_)

    def fill(self):
        # 填充随机值
        # random, standard_normal, standard_exponential, standard_gamma支持
        def _fill(random_state, out, first, last):
            # 每次生成threads个随机数
#             print(f"first: {first}, last: {last}\n")
            random_state.standard_normal(out=out[first:last])

        futures = {}
        for i in range(self.threads):
            args = (_fill,
                    self._random_generators[i],
                    self.values,
                    i * self.step,
                    (i + 1) * self.step)
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)

    def __del__(self):
        self.executor.shutdown(False)
```



```python
N = 10000000
seed = 42
mrng = MultithreadedRNG(N, seed=42)
mrng.values
```

    array([0., 0., 0., ..., 0., 0., 0.])





```python
mrng.fill()
```



```python
mrng.values[:10]
```




    array([ 0.41832997,  0.60557617,  0.02878786, -1.084246  ,  1.46422098,
            0.29072736, -1.33075642, -0.03472346,  0.28041847,  0.10749307])





```python
%timeit mrng.fill()
```

    61.5 ms ± 15.3 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

    


```python
values = np.zeros(N)
rg = np.random.default_rng()
%timeit rg.standard_normal(out=values)
```

    117 ms ± 8.35 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

配列作成にもオーバーヘッドがあります：



```python
rg = default_rng()
%timeit rg.standard_normal(N)
```

    126 ms ± 7.04 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

### パフォーマンス

さまざまな `Generator` について、公式には以下のような参考提案があります。

- パラレル性の高いユースケースには、 `PCG64` またはそのアップグレードされたバリエーション `PCG64DXSM` が推奨されます。ほとんどのプラットフォームで統計的に高品質、フル機能、高速ですが、32ビットプロセス用にコンパイルすると少し遅くなります。
-  `Philox` は遅いですが、統計属性の質は非常に高く、一意のキーを使って信頼性の高い独立したストリームを簡単に得ることができます。このように並列にしたい場合は選択できます。
-  `SFC64` 統計的に高品質で非常に高速です。しかし、ジャンプすることはできません。机能が必要でなく、高速化を望む場合は (32ビットプロセスも同様) 選択することができます。
-  `MT19937` いくつかの統計テストに失敗し、現代のPRNGと比較すると特に速いものではありません。ほとんどの場合単独で使用することは推奨されておらず、現在は主に古い `RandomState` バージョンで使用されています。

以下の比較は、ns単位で特定の分布から乱数が生成される時間である。


|  |RandomState|MT19937|PCG64|PCG64DXSM|Philox|SFC64|
|---|----------|-------|-----|---------|------|-----|
|32-bit Unsigned Ints|3.1|3.3|1.9|2.0|3.3|**1.8**|
|64-bit Unsigned Ints|5.5|5.6|3.2|2.9|4.9|**2.5**|
|Uniforms            |6.0|5.9|3.1|2.9|5.0|**2.6**|
|Normals             |56.8|13.9|10.8|10.5|12.0|**8.3**|
|Exponenitials       |63.9|9.1|6.0|5.8|8.1|**5.4**|


詳細はこちらを参照してください：[パフォーマンス-NumPy v1.24.de v0マニュアル](https://numpy.org/devdocs/reference/random/performance.html)

### 整数ランダム系列



```python
rng = np.random.default_rng(42)

rng.integers(1, 10, (2,3))
```




    array([[1, 7, 6],
           [4, 4, 8]])



 `integers` のAPIは直感的です：

- low：下界
- high：上限
- サイズ：shape
- dtype：データ型
- endpoint：デフォルトのFalse、上限は含まれていない



```python
rng.integers(0, 2, 3), rng.integers(0, 2, 3, endpoint=True)
```




    (array([1, 0, 1]), array([1, 0, 2]))



もちろん、lowとhighの両方は、複数の上下境界を表す配列であることができます。



```python
rng.integers(0, [2,5,10], size=(2,3))
```




    array([[1, 3, 7],
           [0, 1, 4]])





```python
rng.integers([2, 5, 10], [4, 10, 20], size=(2,3))
```




    array([[ 2,  7, 16],
           [ 3,  7, 18]])





```python
# 第一行 2 ==> 8 10 20
# 第二行 5 ==> 8 10 20
rng.integers([[2],[5]], [8, 10, 20])
```




    array([[ 5,  2, 15],
           [ 6,  8, 13]])



### 均一ランダム系列

 `random` 0 - 1（左閉右開）乱数を生成するために使用されます。



```python
rng = np.random.default_rng(42)
rng.random((2, 3))
```




    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235]])





```python
rng = np.random.default_rng(42)
rng.uniform(0, 1, (2,3))
```




    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235]])



### ランダムサンプリング

 `choice` 与えられたシーケンスからランダムにサンプリングするために使用されます。『シロから入門へ』で紹介されているので、言及しません。

## ランダム配列

最初に `shuffle` インタフェースがあり、与えられた配列をshuffleするために使用されます。配列 (任意の次元) を受け入れ、shuffleの次元 (何次元目) を指定することもできます。

 `shuffle` は `in-place` であることを📢に注意してください。



```python
rng = np.random.default_rng(42)
a = rng.integers(1, 100, (5, 4))
a
```




    array([[ 9, 77, 65, 44],
           [43, 86,  9, 70],[20, 10, 53, 97],[73, 76, 72, 78],[51, 13, 84, 45]])





```python
rng.shuffle(a)
a
```




    array([[51, 13, 84, 45],
           [ 9, 77, 65, 44],[20, 10, 53, 97],[73, 76, 72, 78],[43, 86,  9, 70]])





```python
rng = np.random.default_rng(42)
a = rng.integers(1, 100, (5, 4))
a
```




    array([[ 9, 77, 65, 44],
           [43, 86,  9, 70],[20, 10, 53, 97],[73, 76, 72, 78],[51, 13, 84, 45]])



 `shuffle` 次元を指定すると、ブロックごとに並べ替えられることに注意してください。



```python
# 每一行顺序会变，但总的元素不变，而且，每一列也不变
# 也就是以列/行（通过axis控制）为单位在重排
rng.shuffle(a, axis=1)
a
```




    array([[44,  9, 65, 77],
           [70, 43,  9, 86],[97, 20, 53, 10],[78, 73, 72, 76],[45, 51, 84, 13]])



 `permutation` ランダム配列シーケンス、パラメータ：

- 配列または整数。配列の場合はcopyを返し、整数の場合は `arange` のランダムを生成します。
- 次元



```python
rng.permutation(5)
```




    array([0, 3, 4, 2, 1])





```python
rng = np.random.default_rng(42)
a = rng.integers(1, 100, (5, 4))
a
```




    array([[ 9, 77, 65, 44],
           [43, 86,  9, 70],[20, 10, 53, 97],[73, 76, 72, 78],[51, 13, 84, 45]])





```python
b = rng.permutation(a)
b
```




    array([[43, 86,  9, 70],
           [ 9, 77, 65, 44],[20, 10, 53, 97],[51, 13, 84, 45],[73, 76, 72, 78]])



axisは `shuffle` と同じで、1行または1列単位で並べ替えられています。



```python
# 行不变
rng.permutation(a, axis=0)
```




    array([[73, 76, 72, 78],
           [20, 10, 53, 97],[43, 86,  9, 70],[ 9, 77, 65, 44],[51, 13, 84, 45]])



 `permuted` は `permutation` よりも1つのパラメータが追加されています。これは以前にも何度も言及されています。また、前の2つとは、axisが独立しており、1行や1列全体の再配置が発生しない点が異なります。



```python
rng.permuted([2,3,1])
```




    array([2, 3, 1])





```python
a
```




    array([[ 9, 77, 65, 44],
           [43, 86,  9, 70],[20, 10, 53, 97],[73, 76, 72, 78],[51, 13, 84, 45]])





```python
b = rng.permuted(a)
b
```




    array([[97, 86, 20, 77],
           [53, 45, 73,  9],[13, 44, 65,  9],[10, 72, 70, 43],[84, 78, 51, 76]])





```python
rng = np.random.default_rng(42)
a = rng.integers(1, 100, (5, 4))
a
```




    array([[ 9, 77, 65, 44],
           [43, 86,  9, 70],[20, 10, 53, 97],[73, 76, 72, 78],[51, 13, 84, 45]])





```python
# 行元素不变，但顺序都重排了，由于不是成块变，所以列也变了
# 也就是说，每一行都是独立的在重排
b = rng.permuted(a, axis=1)
b
```




    array([[44,  9, 65, 77],
           [43, 70,  9, 86],[10, 20, 53, 97],[73, 76, 78, 72],[13, 51, 84, 45]])



上記の3つのAPIをまとめると、以下のようになります。

|api|copy/in-place|axis|
|----|-------|--------|
|shuffle|in-place|视为一维|
|permutation|copy|视为一维|
|permuted|outでin-placeに変えることができます|维度独立|

もちろん、ほとんどの場合axisというパラメータは必要ありません。直接shuffleすれば完了です。

## ランダム分布

NumPyの `random` には多くの分布が組み込まれているので、簡単に見てみましょう。



```python
import matplotlib.pyplot as plt
```

 `uniform` 均一に分布し、左閉右開のlowとhighのパラメータは、最も簡単な連続分布です。

$$
p(x) = \frac{1}{b - a}
$$

しかし、最も一般的/使用されているのは正規分布であるはずです：

 `normal` 以下のパラメータを含むガウス分布を返します。

- loc：平均、デフォルト0.0
- スケール：標准偏差、デフォルト1.0
- size：デフォルトNone (このパラメータは一般的なため、以下の說明ではこのパラメータを無視します)

$$
p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }}
                 e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} }
$$



```python
# 不给入参时，返回一个标准正太的值
rng = np.random.default_rng(42)
rng.normal()
```




    0.30471707975443135





```python
# 指定size
rng = np.random.default_rng(42)
rng.normal(size=(2,3))
```




    array([[ 0.30471708, -1.03998411,  0.7504512 ],
           [ 0.94056472, -1.95103519, -1.30217951]])



画像で見てみましょう（他の分布は似ているので、繰り返さない）：



```python
# 指定均值、标准差
rng = np.random.default_rng(42)
N  = 10000
a1 = rng.normal(0.0, 1.0, N)
a2 = rng.normal(0.0, 4.0, N)
a3 = rng.normal(0.0, 8.0, N)
```



```python
fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, tight_layout=True)
n_bins=100
axs[0].hist(a1, bins=n_bins);
axs[1].hist(a2, bins=n_bins);
axs[2].hist(a3, bins=n_bins);
# 标准差越大，图像越宽
```


    
![png](ch05-probability_statistics_files/ch05-probability_statistics_227_0.png)
    


 `standard_normal` 標准正規分布（すなわち、 `normal` デフォルトの平均値と標准偏差の場合）には、sizeに加えてdtypeとoutが追加されており、これらのパラメータの意味はもう一度說明しません。

 `multivarite_normal` 多変量正規分布は、1次元正規分布を高次元に一般化します。パラメータは次のとおりです：

- mean：N次元分布の平均値であり、N次元空間内の座標であり、生成される可能性が最も高いサンプル位置を表す
- cov：2つの変数が一緒に変化するレベルを表す、対称かつ半正定である必要がある分布の共分散行列
- size：Noneの場合はN個、そうでない場合は `shape+(N,)` 個を生成します。
- check_valid：共分散行列が半正定であるかどうかをチェックします。デフォルトのwarn、オプションのwarn、raise、ignore
- tol：共分散行列の特異値をチェックするときの公差。covはチェック前に倍増します。デフォルト1e-8
- method：因子行列A ( `A@A.T = cov`) を計算します。デフォルトsvd (最も遅い)、choleskyは最も速いがロバストではありません。eigen分解を使用して、2つの間で速度が高いです。



```python
(
    rng.multivariate_normal([0,0], [[1,0],[0,100]]).shape,
    rng.multivariate_normal([0,0], [[1,0],[0,100]], 3).shape,
    rng.multivariate_normal([0,0], [[1,0],[0,100]], (3, 4)).shape
)
```




    ((2,), (3, 2), (3, 4, 2))



### ガウス相関

 `lognormal` 対数正規分布で、平均meanと標准偏差sigmaを受け入れます。

$$
p(x) = \frac{1}{\sigma x \sqrt{2\pi}}
e^{(-\frac{(ln(x)-\mu)^2}{2\sigma^2})}
$$

 `log(x)` が正規分布を満たす場合、xは対数分布を満たす。

 `laplace` 位置パラメータlocとスケールパラメータscaleを受け入れるラプラス分布は、ガウス分布と同様に、ピークではより尖っていて、尾ではより平坦です。

$$
(x; \mu, \lambda) = \frac{1}{2\lambda}
                               \exp\left(-\frac{|x - \mu|}{\lambda}\right)
                               $$

 `rayleigh` レイリー分布、パラメータscaleは標准偏差を表します。

$$
P(x;scale) = \frac{x}{scale^2}e^{\frac{-x^2}{2 \cdotp scale^2}}
$$

ランダムな2次元ベクトルの2つの成分が独立して、同じ分散を持ち、平均値0の正規分布を示す場合、ベクトルのモジュールはレイリー分布を示す。

 `gumbel` ゲンベル分布には、loc（中心パラメータμと拡張パラメータσ）を指定する必要があります。

$$
p(x) = \frac{e^{-(x - \mu)/ \beta}}{\beta} e^{ -e^{-(x - \mu)/\beta}}\\
={\displaystyle {\frac {1}{\beta }}e^{-(z+e^{-z})}}\\
{\displaystyle z={\frac {x-\mu }{\beta }}}\\
$$

 `standard_cauchy` 標准コーシー分布（物理学上のローレンツ分布）。

$$
P(x; x_0, \gamma) = \frac{1}{\pi \gamma \bigl[ 1+(\frac{x-x_0}{\gamma})^2 \bigr] }
$$

標准コーシー分布では、γ=1、x0=0である。確率密度関数は次のように簡略化できます：

$$
P(x;0,1) = \frac{1}{\pi \bigl[ 1+x^2 \bigr] }
$$

仮説の正規性の仮説検定を検討する際には、コーシー分布からのデータに対して検定がどのように実行されているかを見ることが良い指標である。コーシー分布はガウス分布のように見えますが、尾の方が重いからです。

 `vonmises` フォン・ミセス分布は円上の連続確率分布であり、パラメータmuは中心モードを表し、kappaは濃度の測定値であり、kappa=0であれば分布が均一であることを示し、非常に小さい場合は均一に近い分布であり、kappaが大きい場合は角度μで分布が非常に集中してしまう。

$$
p(x) = \frac{e^{\kappa cos(x-\mu)}}{2\pi I_0(\kappa)}
$$

I_0は0次修正されたベッセル関数である。

 `wald` 逆ガウス分布で、scale（パラメータλ）が無限大に近づくと、分布はよりガウス分布に似ています。

$$
P(x;mean,scale) = \sqrt{\frac{scale}{2\pi x^3}}e^\frac{-scale(x-mean)^2}{2\cdotp mean^2x}
$$

 `triangular` 三角分布では、下限left、上限right、モードmodeを指定する必要があります。

$$
P(x;l, m, r) = \begin{cases}
          \frac{2(x-l)}{(r-l)(m-l)}& \text{for $l \leq x \leq m$},\\
          \frac{2(r-x)}{(r-l)(r-m)}& \text{for $m \leq x \leq r$},\\
          0& \text{otherwise}.
          \end{cases}
$$

三角分布は定義が不明で、潜在的な分布が不明であるが、限界やモードに関する知識がある問題でよく用いられる。

### 離散分布

 `binomial` は二項分布であり、パラメータはnとpであり、nは試験回数であり、pは成功確率である。n=1のときはベルヌーリ分布である。

n回の実験でk回の成功確率が得られる：

$$
{\displaystyle f(k,n,p)=\Pr(X=k)={n \choose k}p^{k}(1-p)^{n-k}}
$$



```python
# 丢10次硬币，正面向上的结果
rng.binomial(10, 0.5)
```




    5



 `multinomial` 多項式分布は二項分布の一般化であり、二項を複数の状態に一般化することは多項分布である。パラメータには、試験回数nと様々な結果の確率分布が含まれます（和は1でなければなりません）。

$$
{\displaystyle {\begin{aligned}f(x_{1},\ldots ,x_{k};n,p_{1},\ldots ,p_{k})&{}=\Pr(X_{1}=x_{1}{\text{ and }}\dots {\text{ and }}X_{k}=x_{k})\\&{}={\begin{cases}{\displaystyle {n! \over x_{1}!\cdots x_{k}!}p_{1}^{x_{1}}\times \cdots \times p_{k}^{x_{k}}},\quad &{\text{when }}\sum _{i=1}^{k}x_{i}=n\\\\0&{\text{otherwise,}}\end{cases}}\end{aligned}}}
$$

詳細については、[Multinomial distribution  -  Wikipedia](https://en.wikipedia.org/wiki/Multinomial_distribution)を参照してください。



```python
# 比如丢一个骰子20次
# 结果表示每个点数出现的次数
rng.multinomial(20, [1/6] * 6)
```




    array([4, 6, 2, 1, 3, 4])



 `negative_binomial` 負の二項分布（またはパスカル分布）で、パラメータnは成功回数を表し、pは成功確率を表す。

$$
f(k;n,p) = \frac{\Gamma(n+k)}{k!\Gamma(n)}p^{n}(1-p)^{k}\\
\frac{\Gamma(n+k)}{N!\Gamma(n)} = \binom{k+n-1}{k}
$$

二項分布との違いは、二項分布は総回数nを固定した独立実験における成功回数kの分布であることである。一方、負二項分布は、n回まで成功したときに終了するすべての独立試験における失敗回数kの分布である。すなわち、成功回数n、失敗回数kである。



詳細については、[Negative binomial distribution  -  Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution#:~:text=In%20probability%20theory%20and%20statistics,failures%20(denoted%20r)%20occur.）を参照してください。

 `poisson` ポアソン分布も比較的一般的であり、離散分布であり、二項分布の限界である。パラメータlamは、パラメータλを表す。

$$
f(k; \lambda)=\frac{\lambda^k e^{-\lambda}}{k!}
$$

ポアソン分布の長さは、ある時間帯のウェブサイトのクリック数など、カウントプロセスと関連している。

ポアソン分布は二項分布の近似（ポアソン定理）としても使用でき、nが無限大に近づくとき、二項分布の限界はポアソン分布である。

詳細については、[Poisson distribution  -  Wikipedia](https://en.wikipedia.org/wiki/Poisson_distribution#:~:text=In%20probability%20theory%20and%20statistics,time%20since%20the%20last%20event.)を参照してください。

 `geometric` 幾何分布は成功確率pを指定する必要があります。

もし各試験の成功確率がpであれば、k回目の試験のうち、k回目に成功する確率は次のとおりです：

$$
f(k) = (1 - p)^{k - 1} p
$$

 `hypergeometric` ハイパージオメトリック分布は、正しく選択されたメソッド数ngood、誤って選択されたメソッド数nbad、サンプル数、nsample <=ngood+nbadを指定する必要があります。

N個のサンプル、g個が条件を満たし、b個が満たさない、N個の中からn個を抽出し、xが条件を満たす確率：

$$
P(x) = \frac{\binom{g}{x}\binom{b}{n-x}}{\binom{N}{n}}\\
g=\text{good}, b=\text{bad}
$$

詳細については、[Hypergeometric distribution  -  Wikipedia](https://en.wikipedia.org/wiki/Hypergeometric_distribution)を参照してください。

 `multivariate_hypergeometric` 多変数超幾何学的分布は、N個の異なるタイプの集合からランダムに置換せずにnsampleサンプルを選択します。パラメータは次のとおりです：

- colors：コレクション内の各タイプの数
- nsample：サンプル数
- method：アルゴリズム、marginals (デフォルト) またはcountを生成します。いくつかの場合（colorsが比較的小さい数字を含むなど）、countメソッドはmarginalsよりも速いです。


$$
f(x)=\frac{\left(\begin{array}{c}
D \\
x
\end{array}\right)\left(\begin{array}{c}
N-D \\
n-x
\end{array}\right)}{\left(\begin{array}{c}
N \\
n
\end{array}\right)}
$$


[Multivariate Hypergeometric distribution | Vose Software](https://www.vosesoftware.com/riskwiki/MultivariateHypergeometricdistribution.php#:~:text=The%20Multivariate%20Hypergeometric%20distribution%20is%20an%20array%20distribution%2C%20in%20this,%2C%20French%2C%20and%20Canadian)



```python
colors = [3,2,1,4]
nsample = 4
```



```python
# 从一组分别有 3 2 1 4 个不同类别的集合中选择4个
rng.multivariate_hypergeometric(colors, nsample)
```




    array([0, 1, 1, 2])





```python
rng.multivariate_hypergeometric(colors, nsample, method="count")
```




    array([2, 1, 0, 1])



countメソッドは以下のメソッドと同等です：



```python
choices = np.repeat(np.arange(len(colors)), colors)
choices
```




    array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])





```python
selection = rng.choice(choices, nsample, replace=False)
selection
```




    array([2, 1, 1, 0])





```python
# 0 1 2 3 分别出现的「次数」
variate = np.bincount(selection, minlength=len(colors))
variate
```




    array([1, 2, 1, 0])



marginalsは実際に繰り返し要求する単変数ハイパージオメトリック分布サンプラーです：



```python
colors, nsample
```




    ([3, 2, 1, 4], 4)





```python
variate = np.zeros(len(colors), dtype=np.int64)
# 每次采样后剩下的数
remaining = np.cumsum(colors[::-1])[::-1]
remaining
```




    array([10,  7,  5,  4])





```python
for i in range(len(colors) - 1):
    if nsample < 1:
        break
    variate[i] = rng.hypergeometric(colors[i], remaining[i+1], nsample)
    nsample -= variate[i]
variate[-1] = nsample
variate
```




    array([2, 0, 0, 2])



### 対数相関を指す

 `power` 入力パラメータaを受け入れるべき力律分布。

$$
P(x; a) = ax^{a-1}, 0 \le x \le 1, a>0
$$

べき乗律分布はちょうどパレートの逆数であり、ベータ分布の特例と見なすこともできる。生活の中で多くの現象はすべて幂律分布を呈して、例えば都市の規模と収入、種の餌探しパターン、大部分の言語の用語頻度など。

 `pareto` パレート分布は、ロマックス分布から得られる。パラメータaは形状パラメータであり、確率分布パラメータファミリーの数値パラメータであり、位置パラメータでもスケールパラメータでもない。

$$
p(x) = \frac{am^a}{x^{a+1}}
$$


パレート分布はべき乗律確率分布であり、現実世界に多数存在する（経済学外ではブラッドフォード分布と呼ばれる）。


詳細については、[Pareto distribution  -  Wikipedia](https://en.wikipedia.org/wiki/Pareto_distribution)を参照してください。

 `zipf` ジフの法則（ゼータ分布とも呼ばれる）：単語の頻度は頻度表でのランキングと反比例します。パラメータaは表す

$$
p(k) = \frac{k^{-a}}{\zeta(a)}
$$

ジフの法則は、べき乗法則の確率分布に関連するあらゆるものの参考として用いられている。

 `beta` ベータ（またはB）分布は、ディリクレ分布の特別な形態であり、ガンマ分布に関連している。パラメータには、aとbが含まれ、それぞれalphaとbetaを表します。

$$
f(x; a,b) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1}
(1 - x)^{\beta - 1},\\
B(\alpha, \beta) = \int_0^1 t^{\alpha - 1}
                             (1 - t)^{\beta - 1} dt.
$$

 `dirichlet` はディリクレット分布であり、分布の引数として長さkの順序浮動小数点数を受け入れる。

$$
{\displaystyle f(x_{1},\dots ,x_{K};\alpha _{1},\dots ,\alpha _{K})={\frac {1}{\mathrm {B} (\alpha )}}\prod _{i=1}^{K}x_{i}^{\alpha _{i}-1}}
$$

 `gamma` 分布には次のパラメータが含まれます：

- shape：ガンマ分布のk値
- scale：ガンマ分布のテータ値、デフォルト1.0

$$
p(x) = x^{k-1}\frac{e^{-x/\theta}}{\theta^k\Gamma(k)}
$$

 `exponential` は指数分布で、次のパラメータを持っています：

- スケール：1/beta、デフォルト1.0

$$
f(x; \frac{1}{\beta}) = \frac{1}{\beta} \exp(-\frac{x}{\beta})
$$


 `weibull` ウェーバー分布、パラメータaはshapeを表し、信頼性分析と寿命検証の理論的基礎となっている。

$$
p(x) = \frac{a}{\lambda}(\frac{x}{\lambda})^{a-1}e^{-(x/\lambda)^a}
$$

a=1のときは指数分布に劣化する。

 `logseries` 対数分布は対数級数分布とも呼ばれ、パラメータpは確率を表す。

$$
P(k) = \frac{-p^k}{k \ln(1-p)}
$$

詳細を参照：[Logarithmic distribution  -  Wikipedia](https://en.wikipedia.org/wiki/Logarithmic_distribution)

 `logistic` 論理分布は成長分布とも呼ばれ、位置パラメータlocとスケールパラメータscaleを受け入れる。

$$
P(x) = \frac{e^{-(x-\mu)/s}}{s(1+e^{-(x-\mu)/s})^2}
$$

### 検査関連

 `chisquare` カイ二乗分布は、自由度の数を表す浮動小数点数または浮動小数点数のセットを指定します。

$$
p(x) = \frac{(1/2)^{k/2}}{\Gamma(k/2)}
                 x^{k/2 - 1} e^{-x/2}
$$

 `noncentral_chisquare` 非中心カイ二乗分布、パラメータカイ二乗分布の自由度に加えて、非中心パラメータλを表すnoncパラメータがあります。

$$
P(x;df,nonc) = \sum^{\infty}_{i=0} \frac{e^{-nonc/2}(nonc/2)^{i}}{i!} P_{Y_{df+2i}}(x)
$$

Y_qは、自由度qのカイ二乗分布である。

 `f` F分布には2つの引数が含まれます。dfnumは分子の自由度、dfdenは分母の自由度です。

$$
{\begin{aligned}f(x;d_{1},d_{2})&={\frac  {{\sqrt  {{\frac  {(d_{1}\,x)^{{d_{1}}}\,\,d_{2}^{{d_{2}}}}{(d_{1}\,x+d_{2})^{{d_{1}+d_{2}}}}}}}}{x\,{\mathrm  {B}}\!\left({\frac  {d_{1}}{2}},{\frac  {d_{2}}{2}}\right)}}\\&={\frac  {1}{{\mathrm  {B}}\!\left({\frac  {d_{1}}{2}},{\frac  {d_{2}}{2}}\right)}}\left({\frac  {d_{1}}{d_{2}}}\right)^{{{\frac  {d_{1}}{2}}}}x^{{{\frac  {d_{1}}{2}}-1}}\left(1+{\frac  {d_{1}}{d_{2}}}\,x\right)^{{-{\frac  {d_{1}+d_{2}}{2}}}}\end{aligned}}
$$

詳細については、[F-distribution  -  Wikipedia](https://en.wikipedia.org/wiki/F-distribution)を参照してください。

 `noncentral_f` 非中心F分布は、非中心パラメータλを表すF分布よりも1つのnoncパラメータが増えている。

$$
p(f)=\sum \limits _{{k=0}}^{\infty }{\frac  {e^{{-\lambda /2}}(\lambda /2)^{k}}{B\left({\frac  {\nu _{2}}{2}},{\frac  {\nu _{1}}{2}}+k\right)k!}}\left({\frac  {\nu _{1}}{\nu _{2}}}\right)^{{{\frac  {\nu _{1}}{2}}+k}}\left({\frac  {\nu _{2}}{\nu _{2}+\nu _{1}f}}\right)^{{{\frac  {\nu _{1}+\nu _{2}}{2}}+k}}f^{{\nu _{1}/2-1+k}}\\
B(x,y)={\frac  {\Gamma (x)\Gamma (y)}{\Gamma (x+y)}}.
$$

詳細については、[Noncentral F-distribution  -  Wikipedia](https://en.wikipedia.org/wiki/Noncentral_F-distribution)を参照してください。

 `standard_t` は学生のt分布であり、パラメータdfは自由度である。


$$
P(x, df) = \frac{\Gamma(\frac{df+1}{2})}{\sqrt{\pi df} \Gamma(\frac{df}{2})}\Bigl( 1+\frac{x^2}{df} \Bigr)^{-(df+1)/2}
$$

主に正規分布に属するかどうかを検証するために使用されます。ドキュメントの女性のカロリー摂取を例に挙げてください：



```python
# 11 个女性日卡路里摄入
intake = np.array([5260., 5470, 5640, 6180, 6390, 
                   6515, 6805, 7515, 7515, 8230, 8770])
```

上記の結果は推奨の7725KJから系統的に逸脱しているのでしょうか？まずゼロ仮説を持つ必要があります。つまり、仮定に偏りがなく、代替仮説にはプラスまたはマイナスの影響が存在し、二尾分布であるという仮説です。

11の値は10の自由度に対応し、有意度を95%に設定し、標準偏差を計算します：



```python
np.std(intake, ddof=1)
```




    1142.1232221373727





```python
# T统计量
t = (np.mean(intake) - 7725) / \
(np.std(intake, ddof=1) / np.sqrt(len(intake)))
t 
```




    -2.8207540608310198





```python
# 生成学生t分布（自由度10）
s = rng.standard_t(10, 10000)
```



```python
# 计算p值，远小于0.05，所以拒绝原假设，认为它们是有偏差的
np.sum(np.abs(t) < np.abs(s)) / len(s)
```




    0.0185



## まとめ

## 参考

-  [ランダムジェネレータ-NumPy v1.24.de v0マニュアル](https://numpy.org/devdocs/reference/random/generator.html)



```python

```
