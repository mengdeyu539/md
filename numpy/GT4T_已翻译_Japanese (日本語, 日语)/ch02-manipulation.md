<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"> <ul class="toc-item"><li><a href="#广播" data-toc-modified-id="广播-1"><span class="toc-item-num">1 &nbsp;&nbsp;</span></span><span><span class="toc-item-num">2 </a></li><span><li><span><span class="toc-item-num"></span></a></span>6"/> <span><a href="#重排元素" data-toc-modified-id="重排元素-3.6"><span class="toc-item-num">3.6 &nbsp;&nbsp;</span>要素の再配置 </a></li><a href="#排序搜索" data-toc-modified-id="排序搜索-4"><a href="#排序搜索" data-toc-modified-id="排序搜索-4"></span>4 </a><li><span>"412"/> </li><li><span><a href="#并集" data-toc-modified-id="并集-5.3"><span class="toc-item-num">5.3 &nbsp;&nbsp;</span>ユニオン </a></li><a href="#差集" data-toc-modified-id="差集-5.4"><a href="#差集" data-toc-modified-id="差集-5.4"><span class="toc-item-num">5.4 </span><li><a href="#异或集" data-toc-modified-id="异或集-5.5"><li><a href="#异或集" data-toc-modified-id="异或集-5.5">> <a href="#异或集" data-toc-modified-id="异或集-5.5"><a href="#异或集" data-toc-modified-id="异或集-5.5"><a href="#异或集" data-toc-modified-id="异或集-5.5"><a href="#异或集" data-toc-modified-id="异或集-5.5">> <a href="#异或集" data-toc-modified-id="异或集-5.5">> <a href="#异或集" data-toc-modified-id="异或集-5.5">> <a href="#异或集" data-toc-modified-id="异或集-5.5"><a href="#异或集" data-toc-modified-id="异或集-5.5">> > <a href="#异或集" data-toc-modified-id="异或集-5.5">> >関数型プログラミング </a></span><li><a href="#测试" data-toc-modified-id="测试-7"><span class="toc-item-num">7 &nbsp;&nbsp;</span>テスト </a><ul class="toc-item"><a href="#相等" data-toc-modified-id="相等-7.1"><span class="toc-item-num">7.1 &nbsp;&nbsp;</a></span></span></li><span></div>



```python
import numpy as np
np.__version__
```




    '1.22.3'



ドキュメントの読み取り手順：

- 🐧はTipを示します
- ⚠️注意事項を示す

## 放送

この放送では、NumPyが数値計算で異なる形状の配列をどのように処理するかについて説明しています。特定の制限の下では、小さい配列は大きい配列にブロードキャストされ、その形状に合わせます。



```python
# 最简单的例子
a = np.array([1., 2., 3.])
a
```




    array([1., 2., 3.])





```python
a * 2
```




    array([2., 4., 6.])





```python
# 上面的例子等价于
b = np.array([2, 2, 2])
a * b
```




    array([2., 4., 6.])



放送のルールは以下の通りです。

- 右から左への比較
- 互換性がある場合は等しいか、次元が1
- 配列には異なる次元があります



```python
a = np.ones((8, 1, 6, 1))
b = np.ones((7, 1, 5))
```



```python
# b 等价于变成了 (1, 7, 1, 5) 的shape
c = a + b
c.shape
```




    (8, 7, 6, 5)





```python
# 不能广播的例子
a = np.ones((2, 3, 4))
b = np.ones((2, 3))
a + b
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-475-796f45d71953> in <module>
          2 a = np.ones((2, 3, 4))3 b = np.ones((2, 3))
    ----> 4 a + b
    

    ValueError: operands could not be broadcast together with shapes (2,3,4) (2,3) 




```python
# 当然一维向量数字的数量就是维度
# 这样是不行的
a = np.ones((2, 3))
b = np.ones((2, ))
b.shape
```




    (2,)





```python
a + b
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-472-bd58363a63fc> in <module>
    ----> 1 a + b
    

    ValueError: operands could not be broadcast together with shapes (2,3) (2,) 




```python
# 这样就可以
b = np.ones((3, ))
a + b
```




    array([[2., 2., 2.],
           [2., 2., 2.]])



ほとんどの場合、ダメージなく使用できますが、データ量が多い場合、ブロードキャストは配列をコピーし、メモリオーバーフローを引き起こす可能性があります。



```python
rng = np.random.default_rng(42)
a = rng.random((102400, 256, 64))
b = rng.random((102400, 256))
c = rng.random((256, 64))
```



```python
# a=12.5G
102400*256*64*8/1024/1024/1024
```




    12.5





```python
# 如果内存小于25G，这里会超出内存，因为差的结果还是会存在临时空间
a[:] = b[:, :, np.newaxis] - c
```



```python
# 可以用之前提到的 out 参数，不会额外增加内存
d = np.subtract(b[:, :, np.newaxis], c, out=a)
```

ただし、一般的には仮想メモリが使われているので、あまり心配しないでください。

## 通関数

通関数（ufunc）は、ndarrayを要素ごとに操作する関数であり、配列放送や型変換などをサポートする。NumPyでは、汎用関数はすべて `np.ufunc` のインスタンスです。



```python
isinstance(np.add, np.ufunc)
```




    True



 `+` は `np.add` へのショートカットです。他の関数も同様です。



```python
x1 = np.arange(9).reshape((3, 3))
x2 = np.arange(3)
```



```python
x1 + x2
```




    array([[ 0,  2,  4],
           [ 3,  5,  7],[ 6,  8, 10]])





```python
np.add(x1, x2)
```




    array([[ 0,  2,  4],
           [ 3,  5,  7],[ 6,  8, 10]])





```python
x1 * x2
```




    array([[ 0,  1,  4],
           [ 0,  4, 10],[ 0,  7, 16]])





```python
np.multiply(x1, x2)
```




    array([[ 0,  1,  4],
           [ 0,  4, 10],[ 0,  7, 16]])



outパラメータは主に計算結果を格納するために使用されます：



```python
x3 = np.zeros((3, 3), dtype=np.int8)
x3
```




    array([[0, 0, 0],
           [0, 0, 0],[0, 0, 0]], dtype=int8)





```python
np.add(x1, x2, out=x3)
```




    array([[ 0,  2,  4],
           [ 3,  5,  7],[ 6,  8, 10]], dtype=int8)





```python
x3
```




    array([[ 0,  2,  4],
           [ 3,  5,  7],[ 6,  8, 10]], dtype=int8)



whereパラメータは、どれを保存できるかを決定します：



```python
x4 = np.zeros((3, 3), dtype=np.int8)
x4
```




    array([[0, 0, 0],
           [0, 0, 0],[0, 0, 0]], dtype=int8)





```python
np.add(x1, x2, out=x4, where=[True, False, False])
```




    array([[0, 0, 0],
           [3, 0, 0],[6, 0, 0]], dtype=int8)





```python
x4
```




    array([[0, 0, 0],
           [3, 0, 0],[6, 0, 0]], dtype=int8)



NumPyには数学演算、三角関数、ビット操作、論理関数、浮動小数点関数など多くのufuncがあります。詳細は次のとおりです：

- https://numpy.org/devdocs/reference/ufuncs.html#available-ufuncs

ufuncは `__array_ufunc__` メソッドで上書きすることができます。詳細は第1章「コアコンセプト：カスタム配列コンテナ」を参照してください。

ufuncは次のメソッドをサポートしています：

- reduce：次元に沿って蓄積する
- accumulate：すべての要素が蓄積される
- reduceat：特定の次元に沿って指定されたスライスの累積
- outer：AとBのすべての要素に対して演算する
- at：指定されたインデックスの要素に対してバッファリングなしのインプレイス演算を実行する



```python
a = np.arange(12).reshape(4, 3)
a
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])



 `reduce` のパラメータは、初級コースで紹介されている多くのインタフェースと同じです：

- array：配列
- axis：次元
- dtype：データ型
- out：上記と同じ
- where：上と同じ
- keepdims：次元を維持するかどうか、『基本チュートリアル』に紹介されています
- initial：初期値



```python
np.add.reduce(a, axis=0, initial=10)
```




    array([28, 32, 36])





```python
np.add.reduce(a, axis=1, initial=10, keepdims=True)
```




    array([[13],
           [22],[31],[40]])



 `accumulate` の引数は非常に少ない：

- array：配列
- axis：次元
- dtype：データ型
- out：上記と同じ



```python
a
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])





```python
# 沿着行
np.multiply.accumulate(a, axis=1)
```




    array([[  0,   0,   0],
           [  3,  12,  60],[  6,  42, 336],[  9,  90, 990]])





```python
np.multiply.accumulate(a)
```




    array([[  0,   1,   2],
           [  0,   4,  10],[  0,  28,  80],[  0, 280, 880]])



 `reduceat` `accumulate` よりもインデックス位置が増加しました：

- index：indexの復数
- その他のパラメータは `accumulate` と同じです


 `array[indices[i]:indices[i+1]]` を計算します。iはi番目の行/列を表します。計算ルールは次のとおりです：

-  `i = len(indices) - 1` (最后のインデックス)： `indices[i+1] = array.shape[axis]`
-  `indices[i] >= indices[i + 1]` の場合、i番目は `array[indices[i]]` です。
-  `indices[i] >= len(array)` または `indices[i] < 0` の場合、エラー



```python
a
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])





```python
np.add.reduceat(a, [1,2])
```




    array([[ 3,  4,  5],
           [15, 17, 19]])





```python
# 四列
# 第0列：indices[0]: indices[1]，即第0:2累积（0+1列）
# 第1列：indices[1] > indices[2]，等于第2列
# 第2列：indices[2] > indices[3]，等于第1列
# 第3列：最后一个，indices[3]: indices[a.shape[1]=3]，等于0:3累积（0,1,2列）
np.add.reduceat(a, [0, 2, 1, 0], axis=1)
```




    array([[ 1,  2,  1,  3],
           [ 7,  5,  4, 12],[13,  8,  7, 21],[19, 11, 10, 30]])



 `outer` 2つの配列を受け入れます。これは次の結果に相当します：


```python
r = empty(len(A),len(B))
for i in range(len(A)):
    for j in range(len(B)):
        r[i,j] = op(A[i], B[j])
```



```python
np.add.outer(a, a).shape
```




    (4, 3, 4, 3)





```python
np.add.outer([1,2,3], [4,5,6])
```




    array([[5, 6, 7],
           [6, 7, 8],[7, 8, 9]])



 `at` パラメータは次のとおりです：

- A：配列
- インデックス：インデックス
- b：2つのオペレーターの場合、もう1つのオペレーター



```python
a = np.arange(12).reshape(4, 3)
a
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])





```python
# 对第0和1行加1
np.add.at(a, [0, 1], [1])
```



```python
a
```




    array([[ 1,  2,  3],
           [ 4,  5,  6],[ 6,  7,  8],[ 9, 10, 11]])





```python
# 4x3 和 1x3 可以通过广播运算
a = np.arange(12).reshape(4, 3)
a
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])





```python
np.add.at(a, [0, 1], np.array([[1, 2, 3]]))
a
```




    array([[ 1,  3,  5],
           [ 4,  6,  8],[ 6,  7,  8],[ 9, 10, 11]])



 `np.frompyfunc` を使用して、パス関数のインスタンスを生成できます。本章の後でさらに紹介しますので、繰り返しは省略します。

## 基本的な操作

その中でよく使われる操作の多くは、例えば『入門から小白へ』で紹介しています。 `shape`, `reshape`, `squeeze`, `expand_dims`, `stack`, `concatenate`, `split`, `repeat` など、残りの使用頻度は少し低いが重要な操作を紹介します。

### Shape



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (3, 4))
a
```




    array([[0, 7, 6, 4],
           [4, 8, 0, 6],[2, 0, 5, 9]])





```python
# flat
list(a.flat)
```




    [0, 7, 6, 4, 4, 8, 0, 6, 2, 0, 5, 9]





```python
# 返回copy
a.flatten()
```




    array([0, 7, 6, 4, 4, 8, 0, 6, 2, 0, 5, 9])





```python
# 不同的Style
a.flatten("F")
```




    array([0, 4, 2, 7, 8, 0, 6, 0, 5, 4, 6, 9])





```python
# 返回view
np.ravel(a)
```




    array([0, 7, 6, 4, 4, 8, 0, 6, 2, 0, 5, 9])





```python
np.ravel(a, "F")
```




    array([0, 4, 2, 7, 8, 0, 6, 0, 5, 4, 6, 9])



### 軸変換



```python
a = np.ones((3, 4, 5))
```



```python
np.moveaxis(a, 0, 1).shape
```




    (4, 3, 5)





```python
# 坐标轴的值可以是数组
np.moveaxis(a, [0, 2], [1, 0]).shape
```




    (5, 3, 4)





```python
# 坐标轴的值必须是整数
np.swapaxes(a, 0, 1).shape
```




    (4, 3, 5)



### 次元



```python
np.atleast_1d(1)
```




    array([1])





```python
np.atleast_1d(1, 2)
```




    [array([1]), array([2])]





```python
np.atleast_2d(1, 2)
```




    [array([[1]]), array([[2]])]





```python
a = np.ones((2, 3))
```



```python
np.atleast_3d(a).shape
```




    (2, 3, 1)



### 構造

 `block` はブロック行列を構築するためによく使用されます。



```python
a = np.eye(2)*2
b = np.eye(3)*3
```



```python
np.block([
    [a, np.zeros((2, 3))],
    [np.ones((3, 2)), b]
])
```




    array([[2., 0., 0., 0., 0.],
           [0., 2., 0., 0., 0.],[1., 1., 3., 0., 0.],[1., 1., 0., 3., 0.],[1., 1., 0., 0., 3.]])



さらに、blockはいくつかの他のAPIと同等の場合もあります：

- depth=1の場合、 `hstack`
- depth=2の場合、 `vstack`
-  `atleast_1d` と `atleast_2d` を置き換えることができます。

「小白から入門まで」で述べたように、 `concatenate` と `hstack`、 `vstack`、 `dstack` は汎用的であり、前者に異なる次元を加えることで後者の効果を達成することができる。

しかし、後のいくつかは0次元（つまり整数）を処理することができ、前者はできません。ベクトルの扱いにもいくつかの違いがあります。2次元以上の場合は、以前のやり方に従うことをお勧めします。1次元またはゼロ次元の場合は、先に処理してから以前の方法を実行することもできます。



```python
a = np.array([0, 1, 2])
```



```python
np.hstack((a, 3))
```




    array([0, 1, 2, 3])





```python
# 这样方可以
np.concatenate((a, [3]))
```




    array([0, 1, 2, 3])





```python
np.vstack((a, (3,4,5)))
```




    array([[0, 1, 2],
           [3, 4, 5]])





```python
np.concatenate((np.atleast_2d(a), [[3,4,5]]))
```




    array([[0, 1, 2],
           [3, 4, 5]])



 `tile` 入力配列値を所定の回数繰り返す配列を構築するために使用されます。



```python
a = np.array([0, 1, 2])
```



```python
np.tile(a, 2)
```




    array([0, 1, 2, 0, 1, 2])



 `repeat` との違いに注意してください。



```python
np.repeat(a, 2)
```




    array([0, 0, 1, 1, 2, 2])



複数の次元を繰り返すように指定できます：



```python
np.tile(a, (2, 2, 2)).shape
```




    (2, 2, 6)



多次元配列の場合も同様です：



```python
b = np.array([[1,2], [3,4]])
```



```python
np.tile(b, 2)
```




    array([[1, 2, 1, 2],
           [3, 4, 3, 4]])





```python
np.repeat(b, 2, axis=1)
```




    array([[1, 1, 2, 2],
           [3, 3, 4, 4]])





```python
np.tile(b, (4, 1)).shape
```




    (8, 2)



### 要素の追加/削除

 `delte` 要素を削除するために使用されます。axisを指定しないと引き分けられます。そうしないと、indexに対応するすべての要素が削除されます。



```python
a = np.arange(6).reshape(3, 2)
a
```




    array([[0, 1],
           [2, 3],[4, 5]])





```python
np.delete(a, 5)
```




    array([0, 1, 2, 3, 4])





```python
np.delete(a, [3, 5])
```




    array([0, 1, 2, 4])





```python
# 指定axis=0，删除行
np.delete(a, 1, 0)
```




    array([[0, 1],
           [4, 5]])



 `insert` 削除と同様に要素を挿入するために使用されます。



```python
a
```




    array([[0, 1],
           [2, 3],[4, 5]])





```python
# 在index=1的位置插入-1，不指定axis
np.insert(a, 1, -1)
```




    array([ 0, -1,  1,  2,  3,  4,  5])





```python
# 在index=[1,2]的位置插入-1，不指定axis
np.insert(a, [1,2], -1)
```




    array([ 0, -1,  1, -1,  2,  3,  4,  5])





```python
# 在index=1的位置插入[-1, -2]，不指定axis
np.insert(a, 1, [-1, -2])
```




    array([ 0, -1, -2,  1,  2,  3,  4,  5])





```python
# 在index=[1,2]的位置插入[-1, -2]，不指定axis
np.insert(a, [1,2], [-1, -2])
```




    array([ 0, -1,  1, -2,  2,  3,  4,  5])





```python
# 指定axis，index=1的位置插入
np.insert(a, 1, -1, axis=0)
```




    array([[ 0,  1],
           [-1, -1],[ 2,  3],[ 4,  5]])





```python
# index=1的位置插入不同值
np.insert(a, 1, [-1, -2], axis=0)
```




    array([[ 0,  1],
           [-1, -2],[ 2,  3],[ 4,  5]])





```python
np.insert(a, [1,2], -1, axis=0)
```




    array([[ 0,  1],
           [-1, -1],[ 2,  3],[-1, -1],[ 4,  5]])





```python
np.insert(a, [1,2], [-1, -2], axis=0)
```




    array([[ 0,  1],
           [-1, -2],[ 2,  3],[-1, -2],[ 4,  5]])





```python
np.insert(a, [1,2], [[-1, -2]], axis=0)
```




    array([[ 0,  1],
           [-1, -2],[ 2,  3],[-1, -2],[ 4,  5]])



 `append` Pythonのappendに似ています。



```python
np.append(a, 1)
```




    array([0, 1, 2, 3, 4, 5, 1])





```python
np.append(a, [1,2])
```




    array([0, 1, 2, 3, 4, 5, 1, 2])





```python
a
```




    array([[0, 1],
           [2, 3],[4, 5]])





```python
np.append(a, [[6,7]], axis=0)
```




    array([[0, 1],
           [2, 3],[4, 5],[6, 7]])





```python
np.append(a, [[6],[7],[8]], axis=1)
```




    array([[0, 1, 6],
           [2, 3, 7],[4, 5, 8]])



 `trim_zeros` 1次元配列またはシーケンスのゼロ値をクリーンアップするために使用されます。



```python
a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
np.trim_zeros(a)
```




    array([1, 2, 3, 0, 2, 1])





```python
np.trim_zeros(a, "b")
```




    array([0, 0, 0, 1, 2, 3, 0, 2, 1])





```python
np.trim_zeros(a, "f")
```




    array([1, 2, 3, 0, 2, 1, 0])



### エレメントの再配置

 `flip` 主にフリップ要素：



```python
a = np.arange(8).reshape((2,2,2))
```



```python
np.flip(a)
```




    array([[[7, 6],
            [5, 4]],
    
           [[3, 2],
            [1, 0]]])





```python
np.flip(a, [0,1,2])
```




    array([[[7, 6],
            [5, 4]],
    
           [[3, 2],
            [1, 0]]])





```python
np.flip(a, 0)
```




    array([[[4, 5],
            [6, 7]],
    
           [[0, 1],
            [2, 3]]])





```python
np.flip(a, 1)
```




    array([[[2, 3],
            [0, 1]],
    
           [[6, 7],
            [4, 5]]])





```python
np.flip(a, 2)
```




    array([[[1, 0],
            [3, 2]],
    
           [[5, 4],
            [7, 6]]])





```python
np.flip(a, [0,1])
```




    array([[[6, 7],
            [4, 5]],
    
           [[2, 3],
            [0, 1]]])



 `roll` と `rot90` は要素を回転させるために使用されますが、同じではありません。

 `roll` 指定された軸を回転させる要素。次のパラメータを含む：

- 配列
- shift：整数またはタプル整数、回転数、tupleの場合は対応するaxisと同じ長さ
- axis：座標軸、整数またはタプル整数。デフォルトでは、フラットになってからshiftし、それからreshapeします。



```python
a = np.arange(10)
```



```python
np.roll(a, 3)
```




    array([7, 8, 9, 0, 1, 2, 3, 4, 5, 6])





```python
np.roll(a, -2)
```




    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])





```python
np.roll(a, (1, 2))
```




    array([7, 8, 9, 0, 1, 2, 3, 4, 5, 6])





```python
a = np.arange(10).reshape(2, 5)
a
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])





```python
np.roll(a, 1, axis=0)
```




    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])





```python
np.roll(a, 1, axis=1)
```




    array([[4, 0, 1, 2, 3],
           [9, 5, 6, 7, 8]])





```python
np.roll(a, 1)
```




    array([[9, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])





```python
# 等价于
np.roll(a.flatten(), 1).reshape((2, 5))
```




    array([[9, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])





```python
np.roll(a, -1)
```




    array([[1, 2, 3, 4, 5],
           [6, 7, 8, 9, 0]])





```python
# 两个tuple等长
np.roll(a, (1, 1), axis=(1, 0))
```




    array([[9, 5, 6, 7, 8],
           [4, 0, 1, 2, 3]])





```python
# 上式等价于
np.roll(a, 1, axis=(1, 0))
```




    array([[9, 5, 6, 7, 8],
           [4, 0, 1, 2, 3]])





```python
np.roll(a, (2, 1), axis=(1, 0))
```




    array([[8, 9, 5, 6, 7],
           [3, 4, 0, 1, 2]])



 `rot90` は、指定された軸で90°回転し、方向は第1軸から第2軸へと回転することを意味します。パラメータは次のとおりです：

- 配列：2次元以上
- k：整数、回転数、デフォルト1
- axes：2つ以上の要素のtupleで、要素が異なる必要があります。デフォルト (0,1)



```python
a
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])





```python
np.rot90(a)
```




    array([[4, 9],
           [3, 8],[2, 7],[1, 6],[0, 5]])





```python
# 180°
np.rot90(a, 2)
```




    array([[9, 8, 7, 6, 5],
           [4, 3, 2, 1, 0]])





```python
# 转回来
np.rot90(a, 4)
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])





```python
# 顺时针
np.rot90(a, 1, (1, 0))
```




    array([[5, 0],
           [6, 1],[7, 2],[8, 3],[9, 4]])



## ソート検索

### 極端値

最大最小値に関する方法はたくさんありますが、見てみましょう。

 `np.maximum` と `np.minimum` は通常関数であり、maxを例にとると、他には `np.max`（同等の `np.amax`）、 `np.fmax`、 `np.nanmax` が関連しています。以下のように区別されます。

-  `minimum`：2つの配列を要素単位で比較する
-  `fmax`：上と同じですが、欠落した値は無視されます
-  `amax`：与えられた次元に沿って
-  `nanmax`：上と同じですが、欠落した値は無視します

最大値を例に挙げましょう。



```python
# 等价于 np.where(x1 >= x2, x1, x2)
np.fmax([np.nan, 2, 3], [1, 5, np.nan])
```




    array([1., 5., 3.])





```python
np.nanmax([[np.nan, 3, 5], [2, 1, np.nan]], axis=0)
```




    array([2., 3., 5.])



### 検索

 `argmax/argmin` 私たちはすでに『小白から入門まで』で紹介しましたが、ここでは主に非数値 (NaN) を持つバージョンを紹介します。最小値を例にします。



```python
a = np.array([
    [np.nan, 2, 3],
    [1, np.nan, 4]
])
```



```python
np.argmin(a, axis=0)
```




    array([0, 1, 0])





```python
np.nanargmin(a, axis=0)
```




    array([1, 0, 0])



 `argwhere` は、 `where` の弱体化されたバージョンである `where` とは異なり、0以外のすべての要素のindexを返します。



```python
a = np.arange(6).reshape(2, 3)
a
```




    array([[0, 1, 2],
           [3, 4, 5]])





```python
# 默认返回非0的元素index
np.argwhere(a)
```




    array([[0, 1],
           [0, 2],[1, 0],[1, 1],[1, 2]])





```python
# 这个不是指定条件，而是a>3本身是个数组
np.argwhere(a>3)
```




    array([[1, 1],
           [1, 2]])



0以外の位置のインデックスには、さらに2つのAPIがあります：



```python
# 返回非0元素的索引
np.nonzero(a>2)
```




    (array([1, 1, 1]), array([0, 1, 2]))





```python
# 返回打平后的索引
np.flatnonzero(a>2)
```




    array([3, 4, 5])



最後に `searchsorted` があり、指定された値を挿入すべき位置のインデックスを返します。指定された配列は1次元でなければなりませんが、挿入された値は配列であることができます。パラメータは以下のとおりです

- 挿入された1次元配列。
- 挿入する値、配列。
- サイド：デフォルトleft、最初の適切な位置。rightは最后の適切な位置です。
- sorter：1次元配列、任意のインデックス、挿入する配列を昇順にソートします。



```python
a = np.arange(1, 6)
b = np.array([3, 2, 1, 5, 4])
a
```




    array([1, 2, 3, 4, 5])





```python
np.searchsorted(a, 2)
```




    1





```python
np.searchsorted(b, 2)
```




    3





```python
# 要插入的值可以是多维数组
np.searchsorted(a, [[2,2],[2,2]])
```




    array([[1, 1],
           [1, 1]])



sideは、すべての適切な位置に挿入される位置のindexを制御します：



```python
np.searchsorted([1,1,1,1,1], 1, "right")
```




    5





```python
np.searchsorted([1,1,1,1,1], 1, "left")
```




    0



sorterは配列に挿入された要素のインデックスを表し、配列を昇順に並べます。



```python
b
```




    array([3, 2, 1, 5, 4])





```python
np.searchsorted(b, 2)
```




    3





```python
# 1 2 3 4 5
np.searchsorted(b, 2, sorter=[2,1,0,4,3])
```




    1





```python
np.searchsorted([1,2,3,4,5],2)
```




    1





```python
# 5 4 3 2 1
np.searchsorted(b, 2, sorter=[3,4,0,1,2])
```




    0





```python
np.searchsorted([5,4,3,2,1],2)
```




    0



### 並べ替え

 `argsort` 『小白から入門まで』で紹介しているので、ここでは詳しくは言いません。次のいくつかのソート関連のAPIを紹介します：

-  `sort/lexsort/ndarray.sort`：配列ソート
-  `msort`：第1軸をソートする
-  `sort_complex`：複素ソート
-  `prtition/argpartition/ndarray.partition`：部分的なソート



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (3, 4))
a
```




    array([[0, 7, 6, 4],
           [4, 8, 0, 6],[2, 0, 5, 9]])



 `sort` 次のパラメータを含みます：

- 配列
- axis：デフォルト最后の軸 (- 1)
- kind：kindはソートアルゴリズムで、サポート：quicksort、mergesort、heapsort、stable、デフォルトのquicksort
- order：ソートするためのフィールド（配列にフィールドがある場合）

最後の次元 (デフォルト) に沿ってソートするときに一時的なコピーは作成されないので、最も高速で余分なスペースを占有しません。

stableは、ソートされるデータ型に基づいて、最も安定したソートアルゴリズムを自動的に選択します。



```python
# 默认axis=-1
np.sort(a)
```




    array([[0, 4, 6, 7],
           [0, 4, 6, 8],[0, 2, 5, 9]])





```python
# 指定轴
np.sort(a, axis=0)
```




    array([[0, 0, 0, 4],
           [2, 7, 5, 6],[4, 8, 6, 9]])





```python
# 指定排序算法
np.sort(a, kind="stable")
```




    array([[0, 4, 6, 7],
           [0, 4, 6, 8],[0, 2, 5, 9]])





```python
# 指定order
# String, float16, int32
dtype = [("name", "U10"), ("height", "f2"), ("age", "i4")]
values = [
    ("Arthur", 1.8, 41), 
    ("Lancelot", 1.9, 38), 
    ("Galahad", 1.7, 38)
]
b = np.array(values, dtype=dtype)
np.sort(b, order=["height"])
```




    array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41), ('Lancelot', 1.9, 38)],
          dtype=[('name', '<U10'), ('height', '<f2'), ('age', '<i4')])



 `lexsort`、后のキーが優先されます。



```python
surnames =    ('Zertz',    'Halilei', 'Halilei')
first_names = ('Heinrich', 'Gzlileo', 'Gustav')
# 先surname，在按firstname
ind = np.lexsort((first_names, surnames))
ind
```




    array([2, 1, 0])





```python
[surnames[i] + ", " + first_names[i] for i in ind]
```




    ['Halilei, Gustav', 'Halilei, Gzlileo', 'Zertz, Heinrich']



 `ndarray.sort` は、in-placeソートです。残りの引数は `np.sort` と同じです。



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (3, 4))
a
```




    array([[0, 7, 6, 4],
           [4, 8, 0, 6],[2, 0, 5, 9]])





```python
a.sort()
```



```python
a
```




    array([[0, 4, 6, 7],
           [0, 4, 6, 8],[0, 2, 5, 9]])



 `msort` は `np.sort(a, axis=0)` と同等です。



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (3, 4))
a
```




    array([[0, 7, 6, 4],
           [4, 8, 0, 6],[2, 0, 5, 9]])





```python
np.msort(a)
```




    array([[0, 0, 0, 4],
           [2, 7, 5, 6],[4, 8, 6, 9]])



 `sort_complex` 実部を使用し、次に虚部を使用します。



```python
np.sort_complex([1 + 2j, 1+1j, 2 - 1j, 3 - 2j, 3 - 3j, 3 + 5j])
```




    array([1.+1.j, 1.+2.j, 2.-1.j, 3.-3.j, 3.-2.j, 3.+5.j])



 `partition` と `argpartition` の関係と `sort` は `argsort` と同様です。パラメータは次のとおりです：

- 配列
- kth：分割の位置、整数または整数の系列
- axis：軸、デフォルト - 1
- kind：選択アルゴリズム、デフォルト `introselect`
- order：strまたはList [str] は、先に說明した `sort` の引数と同じです。

返された結果のk番目の要素の位置は、順序付け時の位置であることに注意してください（元の配列のindexではありません）。



```python
a = [100, 99, 87, 101, 88, 78, 98]
# 88 排好时在index=2的数字
# 比88小的在左边，比88大或相等的在右边
np.partition(a, 2)
```




    array([ 78,  87,  88, 101,  99, 100,  98])





```python
np.argpartition(a, 2)
```




    array([5, 2, 4, 3, 1, 0, 6])





```python
a = [3, 4, 2, 1]
# 4 是排好序时index=3的数字
np.partition(a, 3)
```




    array([2, 1, 3, 4])





```python
np.argpartition(a, 3)
```




    array([2, 3, 0, 1])





```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (3, 8))
a
```




    array([[0, 7, 6, 4, 4, 8, 0, 6],
           [2, 0, 5, 9, 7, 7, 7, 7],[5, 1, 8, 4, 5, 3, 1, 9]])





```python
# 注意第一行，比4大或相等的在右边 
np.partition(a, 2)
```




    array([[0, 0, 4, 6, 4, 8, 7, 6],
           [0, 2, 5, 9, 7, 7, 7, 7],[1, 1, 3, 4, 5, 8, 5, 9]])





```python
np.argpartition(a, 2)
```




    array([[0, 6, 3, 2, 4, 5, 1, 7],
           [1, 0, 2, 3, 4, 5, 6, 7],[1, 6, 5, 3, 4, 2, 0, 7]])



## コレクション操作

### 含む

最初の配列の要素が2番目の配列にあるかどうか：



```python
a = np.array([[1, 2], [2, 3]])
np.unique(a)
```




    array([1, 2, 3])





```python
np.in1d([1,2,3,4], a)
```




    array([ True,  True,  True, False])





```python
# 翻转
np.in1d([1,2,3,4], a, invert=True)
```




    array([False, False, False,  True])





```python
# 两个数组里的值均unique
# 可以加速计算
np.in1d([1,2,3,4], a, assume_unique=True)
```




    array([ True,  True,  True, False])





```python
# 打平
np.in1d([[1,2],[3,4]], a)
```




    array([ True,  True,  True, False])





```python
# 不打平
np.isin([1,2,3,4], a)
```




    array([ True,  True,  True, False])





```python
np.isin([[1,2],[3,4]], a)
```




    array([[ True,  True],
           [ True, False]])



残りの引数は `in1d` と同じです。

### 交差する

交差は、入力された配列が引き分けられます：



```python
a
```




    array([[1, 2],
           [2, 3]])





```python
b = np.array([[2, 4, 1]])
```



```python
np.intersect1d(a, b)
```




    array([1, 2])





```python
# 如果假设为unique，其实不是，结果会有误
np.intersect1d(a, b, assume_unique=True)
```




    array([1, 2, 2])





```python
# 返回索引，如果有多个，返回第一个
np.intersect1d(a, b, return_indices=True)
```




    (array([1, 2]), array([0, 1]), array([2, 0]))



### ユニオン

ユニオンは、入力された配列が引き分けられます：



```python
a
```




    array([[1, 2],
           [2, 3]])





```python
b
```




    array([[2, 4, 1]])





```python
np.union1d(a, b)
```




    array([1, 2, 3, 4])



### 差分集合

差分セットは、配列1で配列2ではない値を返します：



```python
np.setdiff1d(a, b)
```




    array([3])





```python
np.setdiff1d(b, a)
```




    array([4])



### XOR集合

XORセット：



```python
np.setxor1d(a, b)
```




    array([3, 4])



複数の配列の場合は、 `reduce` メソッドを使用できます：



```python
from functools import reduce
```



```python
reduce(
    np.union1d, 
    ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2])
)
```




    array([1, 2, 3, 4, 6])



## 関数型プログラミング

NumPyのベクトル化計算もカスタマイズして使用することができます。NumPyでカスタムメソッドを使う方法を見ていきましょう。

 `apply_along_axis` は、次元に沿って関数を適用することで、元の配列値を次元に沿ってマッピングすることと同じです。



```python
def func(a):
    return (a[0] + a[-1]) / 2
```



```python
b = np.arange(1, 10).reshape(3, 3)
b
```




    array([[1, 2, 3],
           [4, 5, 6],[7, 8, 9]])





```python
np.apply_along_axis(func, 0, b)
```




    array([4., 5., 6.])





```python
np.apply_along_axis(func, 1, b)
```




    array([2., 5., 8.])





```python
b = np.array([[8,1,7], [4,3,9], [5,2,6]])
b
```




    array([[8, 1, 7],
           [4, 3, 9],[5, 2, 6]])





```python
np.apply_along_axis(sorted, 1, b)
```




    array([[1, 7, 8],
           [3, 4, 9],[2, 5, 6]])





```python
np.apply_along_axis(sorted, 0, b)
```




    array([[4, 1, 6],
           [5, 2, 7],[8, 3, 9]])





```python
np.apply_along_axis(np.diag, 1, b)
```




    array([[[8, 0, 0],
            [0, 1, 0],[0, 0, 7]],
    
           [[4, 0, 0],
            [0, 3, 0],[0, 0, 9]],
    
           [[5, 0, 0],
            [0, 2, 0],[0, 0, 6]]])





```python
np.apply_along_axis(np.diag, 0, b).T
```




    array([[[8, 0, 0],
            [0, 4, 0],[0, 0, 5]],
    
           [[1, 0, 0],
            [0, 3, 0],[0, 0, 2]],
    
           [[7, 0, 0],
            [0, 9, 0],[0, 0, 6]]])



 `apply_over_axes` 関数が複数の軸に複数回繰り返し適用されます。

注：関数は配列と次元の2つの引数 ( `apply_along_axis` の1つの引数とは異なります) を受け取ります。



```python
b
```




    array([[8, 1, 7],
           [4, 3, 9],[5, 2, 6]])





```python
np.apply_over_axes(np.sum, b, 1)
```




    array([[16],
           [16],[13]])





```python
np.apply_over_axes(np.sum, b, 0)
```




    array([[17,  6, 22]])





```python
np.apply_over_axes(np.sum, b, [0, 1])
```




    array([[45]])



 `vectorize` は、パフォーマンス向上のためではなく、利便性のためにカスタム関数をベクトル化する汎用クラスであり、内部は実際にはforループです。



```python
def func(a, b):
    if a > b:
        return a - b
    else:
        return a + b
```



```python
vfunc = np.vectorize(func)
```



```python
vfunc(np.arange(10), 3)
```




    array([3, 4, 5, 6, 1, 2, 3, 4, 5, 6])



出力のタイプを指定できます：



```python
vfunc = np.vectorize(func, otypes=[np.float16])
```



```python
vfunc(np.arange(10), 3)
```




    array([3., 4., 5., 6., 1., 2., 3., 4., 5., 6.], dtype=float16)



または、ベクトル化しないパラメータを指定します：



```python
def func(a, b):
    res = np.zeros_like(b)
    for i in a:
        res += i * b
    return res
```



```python
func([1,2,3], np.array([2,4]))
```




    array([12, 24])





```python
vfunc = np.vectorize(func, excluded=["a"])
```



```python
vfunc(a=[1,2,3], b=[2,4])
```




    array([12, 24])



または後で指定します：



```python
vfunc = np.vectorize(func)
# 第一个参数排除
vfunc.excluded.add(0)
```



```python
vfunc([1,2,3], [2,4])
```




    array([12, 24])



入力されたパラメータが直接実行できない場合は、署名が必要です：



```python
conv = np.vectorize(np.convolve, signature="(n),(m)->(k)")
```



```python
conv([[1,2,3],[4,5,6],[7,8,9]], [1, 0.5])
```




    array([[ 1. ,  2.5,  4. ,  1.5],
           [ 4. ,  7. ,  8.5,  3. ],[ 7. , 11.5, 13. ,  4.5]])





```python
np.convolve([1,2,3,4,5,6,7,8,9], [1, 0.5])
```




    array([ 1. ,  2.5,  4. ,  5.5,  7. ,  8.5, 10. , 11.5, 13. ,  4.5])



 `vectorize` よりも、 `frompyfunc` は、任意のPython関数を通信関数に変換することができます。



```python
def func(x):
    return str(x)
```



```python
ufunc = np.frompyfunc(func, 1, 1)
```



```python
ufunc(np.arange(5))
```




    array(['0', '1', '2', '3', '4'], dtype=object)



 `piecewise` 条件と対応する関数のセットが実行され、条件が真のときに実行されます。



```python
a = np.arange(10)
```



```python
np.piecewise(a, [a<5, a==5, a>5], [-1, 0, 1])
```




    array([-1, -1, -1, -1, -1,  0,  1,  1,  1,  1])





```python
def func(a, b):
    return a + b
```



```python
# 注意后面的参数是所有函数通用的
np.piecewise(a, [a<5, a==5, a>5], [func, 0, lambda x, b: x**2], b=5)
```




    array([ 5,  6,  7,  8,  9,  0, 36, 49, 64, 81])



## テスト

主にAssertsに関するAPIを紹介し、第3節の『論理演算』と一定の関連がある。

### 等しい

Equal Asserts：

- `assert_equal`
- `assert_array_equal`
- `assert_string_equal`

3つのAPIは少し似ています。最初の2つは実際の値と期待値に加えて、さらに多くのAPIがあります：

-  `err_msg`：失敗時に印刷されるメッセージ
-  `verbose`：Trueの場合、競合する値はメッセージの后ろに追加されます。

 `assert_equal` 2つのオブジェクト（スカラー、配列、タプル、辞書、NumPy配列など）を比較します。一方がスカラーで、もう一方が配列である場合、スカラーは配列の各要素と比較されます。同じ位置がすべて非数値（NaN）であれば、等しいとみなされる。



```python
np.testing.assert_equal([1,2], [1,4], "msg: not equal")
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-750-bfa692ba2952> in <module>
    ----> 1 np.testing.assert_equal([1,2], [1,4], "msg: not equal")
    

        [... skipping hidden 1 frame]
    

    /usr/local/lib/python3.8/site-packages/numpy/testing/_private/utils.py in assert_equal(actual, desired, err_msg, verbose)
        423         # Explicitly use __eq__ for comparison, gh-2552424         if not (desired == actual):
    --> 425             raise AssertionError(msg)
        426 427     except (DeprecationWarning, FutureWarning) as e:
    

    AssertionError: Items are not equal:item=1msg: not equal
     ACTUAL: 2DESIRED: 4




```python
# 标量与NumPy array_like 数组
np.testing.assert_equal(3, np.array([3, 3, 3]))
```



```python
# 含有非数值（NaN）的情况
np.testing.assert_equal([1, np.nan], [1, np.nan])
```

 `assert_array_equal` の範囲は `assert_equal` よりも小さく、後者は前者を内部的に呼び出します。入力がarray_likeであれば、両者に違いはありません。後者は複数、時間、非数値（NaN）などで異なる。



```python
np.testing.assert_array_equal(np.nan, [np.nan])
```



```python
np.testing.assert_equal(np.nan, [np.nan])
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-9-ff23e2edb3c8> in <module>
    ----> 1 np.testing.assert_equal(np.nan, [np.nan])
    

    /usr/local/lib/python3.8/site-packages/numpy/testing/_private/utils.py in assert_equal(actual, desired, err_msg, verbose)
        375     # isscalar test to check cases such as [np.nan] != np.nan376     if isscalar(desired) != isscalar(actual):
    --> 377         raise AssertionError(msg)
        378 379     try:
    

    AssertionError: Items are not equal:
     ACTUAL: nanDESIRED: [nan]


 `assert_string_equal` 文字列を比較するには、2つの文字列を入力します。



```python
np.testing.assert_string_equal("abc", "abc")
```



```python
np.testing.assert_string_equal("abc", "abC")
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-20-c25a53711d83> in <module>
    ----> 1 np.testing.assert_string_equal("abc", "abC")
    

    /usr/local/lib/python3.8/site-packages/numpy/testing/_private/utils.py in assert_string_equal(actual, desired)
       1204     msg = f"Differences in strings:\n{''.join(diff_list).rstrip()}"1205     if actual != desired:
    -> 1206         raise AssertionError(msg)
       1207 1208 
    

    AssertionError: Differences in strings:
    - abc+ abC


### 近い

近いAsserts：

-  `assert_allclose`：最も多く使用されています
- `assert_array_almost_equal_nulp`
- `assert_array_max_ulp`

 `assert_allclose` は、次のパラメータを含みます：

- 実際値
- 期待値
- rtol=1e-07
- atol=0
- equal_nan=True
- err_msg=""
- verbose=True

後の2つのパラメータは前と同じで、これ以上説明しません。 `equal_nan` 非数値（NaN）をTrueと同じとみなされます。rtolとatolは精度を評価するために使用され、両者の差が `atol + rtol*|desired|` 未満であれば十分に近いと考えられます。



```python
np.testing.assert_allclose(1+1e-7, 1+1e-8)
```



```python
1e-7-1e-8 < (1+1e-8)*1e-7
```




    True





```python
np.testing.assert_allclose(5, 4, atol=1)
```



```python
np.testing.assert_allclose([5, np.nan], [4, np.nan], 
                           atol=1, equal_nan=True)
```

 `assert_array_almost_equal_nulp` 2つの振幅可変キューの相対的な堅牢性を比較するために使用できます。

- x
- y
- nulp=1、最後の許容差の最大単位数、次式を満たす場合：

`|x-y| <= nulps * spacing(max(|x|, |y|))`

計算はほぼ等しい。



```python
np.spacing(np.float64(1)) == np.finfo(np.float64).eps
```




    True





```python
x = np.array([1, 0.1, 1e-10, 1e-20])
x.dtype
```




    dtype('float64')





```python
eps = np.finfo(x.dtype).eps
eps
```




    2.220446049250313e-16





```python
x * eps <= 1 * np.spacing(x + x*eps)
```




    array([ True, False, False, False])





```python
np.testing.assert_array_almost_equal_nulp(
    x, x + x*eps, nulp=1
)
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-91-4807c4e32d47> in <module>
    ----> 1 np.testing.assert_array_almost_equal_nulp(
          2     x, x + x*eps, nulp=13 )
    

    /usr/local/lib/python3.8/site-packages/numpy/testing/_private/utils.py in assert_array_almost_equal_nulp(x, y, nulp)
       1592             max_nulp = np.max(nulp_diff(x, y))1593             msg = "X and Y are not equal to %d ULP (max is %g)" % (nulp, max_nulp)
    -> 1594         raise AssertionError(msg)
       1595 1596 
    

    AssertionError: X and Y are not equal to 1 ULP (max is 2)




```python
np.testing.assert_array_almost_equal_nulp(
    .1, .1+eps, nulp=1
)
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-93-7d1275786d24> in <module>
    ----> 1 np.testing.assert_array_almost_equal_nulp(
          2     .1, .1+eps, nulp=13 )
    

    /usr/local/lib/python3.8/site-packages/numpy/testing/_private/utils.py in assert_array_almost_equal_nulp(x, y, nulp)
       1592             max_nulp = np.max(nulp_diff(x, y))1593             msg = "X and Y are not equal to %d ULP (max is %g)" % (nulp, max_nulp)
    -> 1594         raise AssertionError(msg)
       1595 1596 
    

    AssertionError: X and Y are not equal to 1 ULP (max is 16)




```python
np.testing.assert_array_almost_equal_nulp(
    1, 1+eps, nulp=1
)
```



```python
np.testing.assert_array_almost_equal_nulp(
    4, 8, nulp=1e16
)
```



```python
1e16 * np.spacing(8)
```




    17.763568394002505



 `assert_array_max_ulp` 配列内のすべての要素が最大N単位で異なることを確認するために使用されます。



```python
np.testing.assert_array_max_ulp(1, 1)
```




    array([0.])





```python
a = np.linspace(0., 1., 10)
np.testing.assert_array_max_ulp(a, np.arcsin(np.sin(a)))
```




    array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]])





```python
np.testing.assert_array_max_ulp(1, 2, maxulp=1e16)
```




    array([4.50359963e+15])





```python
# NaN不区分
np.testing.assert_array_max_ulp(np.nan, np.nan)
```




    array([0.])



### より小さい

以下：

-  `assert_array_less`：パラメータは `assert_array_equal` と同じで、より小さいかどうかを比較するために使用されます。NaNも比較されます。同じ位置がNaNである場合はスキップします（例外はスローされません）。



```python
np.testing.assert_array_less(
    [1, 1, np.nan],
    [2, 2, np.nan]
)
```



```python
np.testing.assert_array_less(
    3, [1, 4]
)
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-148-4b4915dff584> in <module>
    ----> 1 np.testing.assert_array_less(
          2     3, [1, 4]3 )
    

        [... skipping hidden 1 frame]
    

    /usr/local/lib/python3.8/site-packages/numpy/testing/_private/utils.py in assert_array_compare(comparison, x, y, err_msg, verbose, header, precision, equal_nan, equal_inf)
        842                                 verbose=verbose, header=header,843                                 names=('x', 'y'), precision=precision)
    --> 844             raise AssertionError(msg)
        845     except ValueError:846         import traceback
    

    AssertionError: Arrays are not less-ordered
    
    Mismatched elements: 1 / 2 (50%)Max absolute difference: 2Max relative difference: 2.
     x: array(3)y: array([1, 4])


### 異常

例外：

- `assert_raises`
- `assert_raises_regex`
- `assert_warns`

これら3つのAPIはすべてコンテキストマネージャとして使用できます。

 `assert_raises` 2つの方法があります：

- `assert_raises(exception_class, callable, *args, **kwargs)`
- `assert_raises(exception_class)`



```python
1/0
```


    ---------------------------------------------------------------------------

    ZeroDivisionError                         Traceback (most recent call last)

    <ipython-input-158-9e1622b385b6> in <module>
    ----> 1 1/0
    

    ZeroDivisionError: division by zero




```python
with np.testing.assert_raises(ZeroDivisionError):
    1/0
```



```python
def div(a,b):a/b
np.testing.assert_raises(ZeroDivisionError, div, 1, 0)
```



```python
with np.testing.assert_raises(ZeroDivisionError):
    1/1
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-164-3bd112b16131> in <module>
          1 with np.testing.assert_raises(ZeroDivisionError):
    ----> 2     1/1
    

    /usr/local/Cellar/python@3.8/3.8.10/Frameworks/Python.framework/Versions/3.8/lib/python3.8/unittest/case.py in __exit__(self, exc_type, exc_value, tb)
        225                                                                 self.obj_name))226             else:
    --> 227                 self._raiseFailure("{} not raised".format(exc_name))
        228         else:229             traceback.clear_frames(tb)
    

    /usr/local/Cellar/python@3.8/3.8.10/Frameworks/Python.framework/Versions/3.8/lib/python3.8/unittest/case.py in _raiseFailure(self, standardMsg)
        162     def _raiseFailure(self, standardMsg):163         msg = self.test_case._formatMessage(self.msg, standardMsg)
    --> 164         raise self.test_case.failureException(msg)
        165 166 class _AssertRaisesBaseContext(_BaseTestCaseContext):
    

    AssertionError: ZeroDivisionError not raised


 `assert_raises_regex` 前者よりも正規を指定できます。



```python
with np.testing.assert_raises_regex(
    ZeroDivisionError, "division by zero"
):
    1/0
```



```python
with np.testing.assert_raises_regex(
    ZeroDivisionError, "[a-z]+"
):
    1/0
```

 `assert_warns` は `assert_raises` と同じように使用されますが、例外を投げるのではなく警告が表示されます。



```python
import warnings
```



```python
def func(n):
    warnings.warn("msg", UserWarning)
    return n * n
```



```python
np.testing.assert_warns(UserWarning, func, 2)
```




    4





```python
func(2)
```

    <ipython-input-187-23e97a00966d>:2: UserWarning: msg
      warnings.warn("msg", UserWarning)
    




    4



警告の種類については、次のドキュメントを参照してください：

-  [warnings—Warning control—Python 3.10.4 documentation](https://docs.python.org/3/library/warnings.html)
## まとめ

## 参考

-  [NumPyドキュメント—NumPy v1.23.de v0マニュアル](https://numpy.org/devdocs/index.html)



```python

```
