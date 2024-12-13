<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"> <ul class="toc-item"><li><span><a href="#数组乘法" data-toc-modified-id="数组乘法-1"><span class="toc-item-num">1 &nbsp;&nbsp;</span>配列乗算 </a></span><li><a href="#点积/内积/数量积/标量积" data-toc-modified-id="点积/内积/数量积/标量积-1.1"><span class="toc-item-num">1.1 &nbsp;&nbsp;</span>ドット積/内積/数量積/スカラー積 </a></span></li><span><a href="#叉积/外积/向量积" data-toc-modified-id="叉积/外积/向量积-1.2"><span class="toc-item-num">1.2 &nbsp;&nbsp;</span>クロス積/外積/ベクトル積 </a></span></li><li><span><a href="#张量积/外积" data-toc-modified-id="张量积/外积-1.3"><span class="toc-item-num">1.3 &nbsp;&nbsp;</span>テンソル積/外積 </a></span></li><li><a href="#矩阵乘法" data-toc-modified-id="矩阵乘法-1.4"><span class="toc-item-num">1.4 &nbsp;&nbsp;</span>行列乗算 </a></span><li><span><span class="toc-item-num">1.5 &nbsp;&nbsp;</span>クロネッカー積 </a></span></li><li><span><a href="#多矩阵乘法" data-toc-modified-id="多矩阵乘法-1.6"><span class="toc-item-num">1.6 &nbsp;&nbsp;</span>マルチマトリクス乗算 </a></span></li></li><li><a href="#基础概念" data-toc-modified-id="基础概念-2"><span class="toc-item-num">2 </a><ul class="toc-item"><a href="#范数" data-toc-modified-id="范数-2.1"><span class="toc-item-num">2.1 </a><li><span class="toc-item-num">3.3 &nbsp;&nbsp;</span>行列分解 </a></span></ul><span><a href="#Einsum" data-toc-modified-id="Einsum-4"><span class="toc-item-num">4 &nbsp;&nbsp;</span>Einsum </a><li><span class="toc-item-num">5 &nbsp;&nbsp;</span>r="505"/> 7.4 &nbsp;&nbsp;</span>修正 </a></span><li><span><a href="#索引切片" data-toc-modified-id="索引切片-7.5"><span class="toc-item-num">7.5 &nbsp;&nbsp;</span><li><a href="#代数运算" data-toc-modified-id="代数运算-7.6"><span class="toc-item-num">7.6 </a></span>代数演算 </a></span></span></li></div>



```python
import numpy as np
np.__version__
```




    '1.22.3'



ドキュメントの読み取り手順：

- 🐧はTipを示します
- ⚠️注意事項を示す

## 配列乗算

注意：彼らが何と呼ばれているかにあまり注目しないで、彼らが何をしているかを見てください。

### ドット積/内積/数量積/スカラー積

 **ドット積**

`np.dot`:

- aとbが一次元であれば、内積 `np.inner`
- aとbが二次元である場合は行列乗算 `np.matmul or a @ b`
- aまたはbのいずれかが定数である場合 `np.multiply or a * b`
- aがN次元であり、bが1次元である場合 `sum product`
- aがN次元で、bがM次元である場合 `dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])`

 `np.vdot`：多次元入力はフラッテンされてドット積が計算されます。また、複素数の計算は `np.dot` とは異なります。



 **内積**

 `np.inner`：1次元配列ベクトルの一般的な内積（複素共役なし）は、より高い次元では、最后の軸上のsum productです。

- 1次元配列の場合、要素の積の和`sum (a*b)
- 1つがスカラーであれば、それは直接乗算されます
- 多次元配列の場合は `np.tensordot(a, b, axes=(-1, -1))` に等しく、特定のインデックスの場合は乗算して最後の次元で合計します `inner(a, b)[i0,...,ir-2,j0,...,js-2] = sum(a[i0,...,ir-2,:]*b[j0,...,js-2,:])`

 [数積 - ウィキペディア、自由百科事典](https://zh.wikipedia.org/wiki/%E7%82%B9%E7%A7%AF)

aとbはどちらも一次元である：



```python
a = np.array([1, 2, 4])
b = np.array([4, 5, 6])
```



```python
np.dot(a, b), 1*4 + 2*5 + 4*6, np.inner(a, b), sum(a * b)
```




    (38, 38, 38, 38)





```python
np.vdot(a,b)
```




    38



aとbは両方とも2次元です。



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (3, 4))
b = rng.integers(0, 10, (3, 4))
```



```python
np.inner(a, b)
```




    array([[119,  71,  63],
           [126,  52,  98],[112,  86,  96]])





```python
np.tensordot(a, b, axes=(-1, -1))
```




    array([[119,  71,  63],
           [126,  52,  98],[112,  86,  96]])





```python
a @ b.T
```




    array([[119,  71,  63],
           [126,  52,  98],[112,  86,  96]])





```python
np.matmul(a, b.T)
```




    array([[119,  71,  63],
           [126,  52,  98],[112,  86,  96]])





```python
np.vdot(a, b), np.dot(a.flatten(), b.flatten())
```




    (267, 267)



aまたはbは定数です：



```python
a * 2
```




    array([[ 0, 14, 12,  8],
           [ 8, 16,  0, 12],[ 4,  0, 10, 18]])





```python
np.dot(a, 2)
```




    array([[ 0, 14, 12,  8],
           [ 8, 16,  0, 12],[ 4,  0, 10, 18]])





```python
np.multiply(a, 2)
```




    array([[ 0, 14, 12,  8],
           [ 8, 16,  0, 12],[ 4,  0, 10, 18]])





```python
np.inner(a, 2)
```




    array([[ 0, 14, 12,  8],
           [ 8, 16,  0, 12],[ 4,  0, 10, 18]])



aは多次元でありbは一次元である：



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (2, 3, 4))
b = np.array([1, 2, 3, 4])
```



```python
np.dot(a, b)
```




    array([[48, 44, 53],
           [70, 47, 50]])





```python
# 最后一个维度乘积和
np.inner(a, b)
```




    array([[48, 44, 53],
           [70, 47, 50]])





```python
np.tensordot(a, b, axes=(-1, -1))
```




    array([[48, 44, 53],
           [70, 47, 50]])





```python
np.sum(a*b, axis=-1)
```




    array([[48, 44, 53],
           [70, 47, 50]])





```python
a @ b
```




    array([[48, 44, 53],
           [70, 47, 50]])



aはm次元でありbはn次元である：



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (2, 4, 3))
b = rng.integers(0, 10, (2, 3, 3))
```



```python
# a 和 b 最后一个维度可以不一样，也就是最后一个维度是自由的
dab = np.dot(a, b)
dab.shape
```




    (2, 4, 2, 3)





```python
# dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
dab[1,2,1,0], sum(a[1,2,:] * b[1,:,0])
```




    (102, 102)





```python
# a 和 b 最后一个维度必须一样
iab = np.inner(a, b)
iab.shape
```




    (2, 4, 2, 3)





```python
# inner(a, b)[i0,...,ir-2,j0,...,js-2] = sum(a[i0,...,ir-2,:]*b[j0,...,js-2,:])
iab[1,2,1,0], sum(a[1,2,:] * b[1,0,:])
```




    (72, 72)





```python
np.alltrue(iab == np.tensordot(a, b, axes=(-1, -1)))
```




    True





```python
np.alltrue(iab == dab), np.any(iab == dab)
```




    (False, False)



### クロス積/外積/ベクトル積


\begin{aligned}\mathbf {u\times v} &={\begin{vmatrix}u_{2}&u_{3}\\v_{2}&v_{3}\end{vmatrix}}\mathbf {i} -{\begin{vmatrix}u_{1}&u_{3}\\v_{1}&v_{3}\end{vmatrix}}\mathbf {j} +{\begin{vmatrix}u_{1}&u_{2}\\v_{1}&v_{2}\end{vmatrix}}\mathbf {k} \\&=(u_{2}v_{3}-u_{3}v_{2})\mathbf {i} -(u_{1}v_{3}-u_{3}v_{1})\mathbf {j} +(u_{1}v_{2}-u_{2}v_{1})\mathbf {k} \end{aligned}

2次元または3次元のみがサポートされ、uとvの両方に垂直なベクトルを表します。

 [交差積 - ウィキペディア、自由百科事典](https://zh.wikipedia.org/wiki/%E5%8F%89%E7%A7%AF)



```python
# 二维
a = [1, 1]
b = [-1, 1]
```



```python
# 向量积在二维中不起作用，因为返回的向量在二维之外
# 长度就是面积（根号2×根号2）
np.cross(a, b)
```




    array(2)





```python
# 行变多并不等于维度变多
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (2, 2))
b = rng.integers(0, 10, (2, 2))
a, b
```




    (array([[0, 7],
            [6, 4]]),
     array([[4, 8],
            [0, 6]]))





```python
np.cross(a, b)
```




    array([-28,  36])





```python
np.cross([0,7], [4,8]), np.cross([6, 4], [0, 6])
```




    (array(-28), array(36))





```python
# 三维
a = [1, 2, 4]
b = [4, 5, 6]
```



```python
2*6-4*5, -(1*6-4*4), 1*5-2*4
```




    (-8, 10, -3)





```python
# 与a和b都垂直的向量
np.cross(a, b)
```




    array([-8, 10, -3])



また、配列の定義を変更するための次元に関するいくつかの3つの引数があります（CとFスタイルに似ています）。



```python
# 行变多并不等于维度变多
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (2, 3))
b = rng.integers(0, 10, (2, 3))
a, b
```




    (array([[0, 7, 6],
            [4, 4, 8]]),
     array([[0, 6, 2],
            [0, 5, 9]]))





```python
np.cross(a, b)
```




    array([[-22,   0,   0],
           [ -4, -36,  20]])





```python
np.cross([0,7,6], [0,6,2]), np.cross([4,4,8], [0,5,9])
```




    (array([-22,   0,   0]), array([ -4, -36,  20]))





```python
np.cross(a, b, axisc=0)
```




    array([[-22,  -4],
           [  0, -36],[  0,  20]])





```python
a.T,b.T
```




    (array([[0, 4],
            [7, 4],[6, 8]]),
     array([[0, 0],
            [6, 5],[2, 9]]))





```python
np.cross(a.T, b.T)
```




    array([ 0, 11, 38])





```python
np.cross(a, b, axisa=0, axisb=0)
```




    array([ 0, 11, 38])





```python
np.cross([0,4], [0,0]), np.cross([7,4],[6,5]), np.cross([6,8], [2,9])
```




    (array(0), array(11), array(38))





```python
# 这个是实际计算时沿着的维度，和上面的不一样
np.cross(a, b, axis=0)
```




    array([ 0, 11, 38])



### テンソル積/外積

-  [テンソル積 - ウィキペディア、自由百科事典](https://zh.wikipedia.org/wiki/%E5%BC%A0%E9%87%8F%E7%A7%AF)
-  [外積-Wikipedia、自由百科事典](https://zh.wikipedia.org/wiki/%E5%A4%96%E7%A7%AF)



```python
a = [1, 2, 4]
b = [4, 5, 6]
```



```python
np.outer(a, b)
```




    array([[ 4,  5,  6],
           [ 8, 10, 12],[16, 20, 24]])





```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (2, 3, 4))
b = rng.integers(0, 10, (2, 3, 4))
```



```python
np.alltrue(
    np.outer(a,b) == a.ravel().reshape(24, 1) @ b.ravel().reshape(1, 24)
)
```




    True





```python
np.alltrue(np.outer(a, b) == np.tensordot(a.ravel(), b.ravel(), axes=((), ())))
```




    True



### 行列乗算

 `np.dot` 上記のように、それに似た `np.matmul` がありますが、いくつかの違いがあります。

-  `dot` は通信関数ではなく、 `matmul` は通信関数であり、つまり通信関数の一般的な引数があることを意味します。
-  `matmul` ベクトルと数値の乗算はサポートされていません
-  `matmul` マトリクス（要素のように）がスタックされて放送されます

に関する `np.matmul`：

- すべてが2次元であれば、通常の行列乗算です
- いずれかが多次元 (> 2) であれば、それは最后の2つのインデックスに存在するマトリクススタックとして扱い、それに応じてブロードキャストされます。
- 1つ目が1次元であれば、その次元の前に1を付けて行列に昇格し、行列の乗算後に前に付加した1を削除します。
- 2番目が1次元であれば、次元上でappend 1、行列乗算後に追加された1を削除します。

2次元の場合：



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (2, 3))
b = rng.integers(0, 10, (3, 2))
a, b
```




    (array([[0, 7, 6],
            [4, 4, 8]]),
     array([[0, 6],
            [2, 0],[5, 9]]))





```python
np.matmul(a, b)
```




    array([[44, 54],
           [48, 96]])



そのうちの1つは1次元のケースです：



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (2, 3))
b = rng.integers(0, 10, (3,))
a, b
```




    (array([[0, 7, 6],
            [4, 4, 8]]),
     array([0, 6, 2]))





```python
np.matmul(a, b), (a @ np.stack((b, [1, 1, 1]), axis=1))[:,0]
```




    (array([54, 40]), array([54, 40]))





```python
np.matmul(b, a.T), (np.stack(([1, 1, 1], b), axis=0) @ a.T)[1,:]
```




    (array([54, 40]), array([54, 40]))



いずれかが多次元である：



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (2, 3, 4, 5))
b = rng.integers(0, 10, (2, 3, 5, 9))
```



```python
np.matmul(a, b).shape
```




    (2, 3, 4, 9)





```python
(a @ b).shape
```




    (2, 3, 4, 9)





```python
np.dot(a, b).shape
```




    (2, 3, 4, 2, 3, 9)





```python
np.alltrue(np.dot(a, b) == np.matmul(a, b))
```




    True



簡単な例を見てみましょう：



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (2, 3, 2))
b = rng.integers(0, 10, (2, 2, 2))
```



```python
# 2x3x2
np.matmul(a, b)
```




    array([[[49, 49],
            [70, 70],[84, 84]],
    
           [[48, 24],
            [10,  2],[97, 41]]])





```python
a[0,:,:] @ b[0,:,:]
```




    array([[49, 49],
           [70, 70],[84, 84]])





```python
a[1,:,:] @ b[1,:,:]
```




    array([[48, 24],
           [10,  2],[97, 41]])





```python
# 2x3x2
np.matmul(a[0,:,:], b)
```




    array([[[49, 49],
            [70, 70],[84, 84]],
    
           [[56, 28],
            [62, 22],[84, 36]]])





```python
# 2x3x2
np.matmul(a[1,:,:], b)
```




    array([[[42, 42],
            [14, 14],[98, 98]],
    
           [[48, 24],
            [10,  2],[97, 41]]])



dotがどのように動作しているかを見てみましょう。



```python
# 2x3x2x2
np.dot(a,b)
```




    array([[[[49, 49],
             [56, 28]],
    
            [[70, 70],
             [62, 22]],
    
            [[84, 84],
             [84, 36]]],
    
    
           [[[42, 42],
             [48, 24]],
    
            [[14, 14],
             [10,  2]],
    
            [[98, 98],
             [97, 41]]]])





```python
# 3x2x2
np.dot(a[0,:,:], b[:,:,:])
```




    array([[[49, 49],
            [56, 28]],
    
           [[70, 70],
            [62, 22]],
    
           [[84, 84],
            [84, 36]]])





```python
# 3x2x2
np.dot(a[1,:,:], b[:,:,:])
```




    array([[[42, 42],
            [48, 24]],
    
           [[14, 14],
            [10,  2]],
    
           [[98, 98],
            [97, 41]]])



### クロネッカー積

> Wikipedia：は任意の大きさの2つの行列間の演算で、⊗と表されています。クロネッカー積は、ベクトルから行列への外積の一般化であり、標准基礎でのテンソル積の行列表現でもある。

 [クロネッカーの積 - ウィキペディア、自由百科事典](https://zh.m.wikipedia.org/zh-hans/%E5%85%8B%E7%BD%97%E5%86%85%E5%85%8B%E7%A7%AF)

$$
A\otimes B={\begin{bmatrix}a_{{11}}B&\cdots &a_{{1n}}B\\\vdots &\ddots &\vdots \\a_{{m1}}B&\cdots &a_{{mn}}B\end{bmatrix}}.
$$

$$
{\begin{bmatrix}1&2\\3&1\\\end{bmatrix}}\otimes {\begin{bmatrix}0&3\\2&1\\\end{bmatrix}}={\begin{bmatrix}1\cdot 0&1\cdot 3&2\cdot 0&2\cdot 3\\1\cdot 2&1\cdot 1&2\cdot 2&2\cdot 1\\3\cdot 0&3\cdot 3&1\cdot 0&1\cdot 3\\3\cdot 2&3\cdot 1&1\cdot 2&1\cdot 1\\\end{bmatrix}}={\begin{bmatrix}0&3&0&6\\2&1&4&2\\0&9&0&3\\6&3&2&1\end{bmatrix}}.
$$



```python
np.kron([1,2,3], [1, 10, 100])
```




    array([  1,  10, 100,   2,  20, 200,   3,  30, 300])





```python
np.kron([1, 10, 100], [1,2,3])
```




    array([  1,   2,   3,  10,  20,  30, 100, 200, 300])





```python
a = np.array([[1,2],[3,1]])
b = np.array([[0,3],[2,1]])
```



```python
np.kron(a,b)
```




    array([[0, 3, 0, 6],
           [2, 1, 4, 2],[0, 9, 0, 3],[6, 3, 2, 1]])





```python
a = np.ones((2,5,2,5))
b = np.ones((2,3,4))
np.kron(a,b).shape
```




    (2, 10, 6, 20)



### マルチマトリクス乗算

 `linalg.multi_dot` チェーン呼び出し `np.dot` は、自動的に最も速い順序を選択します。

- 最初の配列が1次元であれば、行ベクトルとして扱われます。
- 最后の配列が1次元の場合は、列ベクトルとして扱われます。
- 2つ以上のベクトルが入力された場合、他のベクトルは2次元でなければなりません。



```python
a = np.ones((2, 4))
b = np.ones((4, 3))
c = np.ones((3, 5))
```



```python
np.linalg.multi_dot((a,b,c)).shape
```




    (2, 5)



次のような順序によってパフォーマンスが異なります：

`A_{10x100}, B_{100x5}, C_{5x50}`


`cost((AB)C) = 10*100*5 + 10*5*50   = 5000 + 2500   = 7500``cost(A(BC)) = 10*100*50 + 100*5*50 = 50000 + 25000 = 75000`



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (10, 100))
b = rng.integers(0, 10, (100, 5))
c = rng.integers(0, 10, (5, 50))
```



```python
%timeit a.dot(b).dot(c).shape
```

    8.74 µs ± 745 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    


```python
%timeit a.dot(b.dot(c)).shape
```

    66 µs ± 4.96 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    


```python
%timeit np.linalg.multi_dot((a, b, c)).shape
```

    13.6 µs ± 335 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    

先頭と末尾が1次元である場合：



```python
a = np.ones((3))
b = np.ones((3, 5))
c = np.ones((5, 8))
d = np.ones(8)
```



```python
a.dot(b).dot(c).shape
```




    (8,)





```python
# a=1x3
np.linalg.multi_dot((a, b, c)).shape
```




    (8,)





```python
# d=8x1
np.linalg.multi_dot((b, c, d)).shape
```




    (3,)





```python
b.dot(c).dot(d).shape
```




    (3,)



## 基本概念

線形代数のいくつかのよく使われるAPIを紹介し、数学知識は含まない。



```python
from numpy import linalg as LA
```

### ノルム

含まれています：


```
=====  ============================  ==========================
ord    norm for matrices             norm for vectors
=====  ============================  ==========================
None   Frobenius norm                2-norm
'fro'  Frobenius norm                --
'nuc'  nuclear norm                  --
inf    max(sum(abs(x), axis=1))      max(abs(x))
-inf   min(sum(abs(x), axis=1))      min(abs(x))
0      --                            sum(x != 0)
1      max(sum(abs(x), axis=0))      as below
-1     min(sum(abs(x), axis=0))      as below
2      2-norm (largest sing. value)  as below
-2     smallest singular value       as below
other  --                            sum(abs(x)**ord)**(1./ord)
=====  ============================  ==========================
```

- [Matrix norm - Wikipedia](https://en.wikipedia.org/wiki/Matrix_norm)



```python
a = np.arange(6).reshape(2, 3)
a
```




    array([[0, 1, 2],
           [3, 4, 5]])





```python
LA.norm(a)
```




    7.416198487095663





```python
# F范数
LA.norm(a, "fro"), np.sqrt(np.sum(a**2))
```




    (7.416198487095663, 7.416198487095663)





```python
# 核范数
LA.norm(a, "nuc"), np.sum(LA.svd(a)[1])
```




    (8.348469228349535, 8.348469228349535)





```python
# inf
LA.norm(a, np.inf), np.max(np.sum(abs(a), axis=1))
```




    (12.0, 12)





```python
# -inf
LA.norm(a, -np.inf), np.min(np.sum(abs(a), axis=1))
```




    (3.0, 3)





```python
# 1
LA.norm(a, 1), np.max(np.sum(abs(a), axis=0))
```




    (7.0, 7)





```python
# -1
LA.norm(a, -1), np.min(np.sum(abs(a), axis=0))
```




    (3.0, 3)





```python
# 2
LA.norm(a, 2), np.max(LA.svd(a)[1])
```




    (7.3484692283495345, 7.3484692283495345)





```python
# -2
LA.norm(a, -2), np.min(LA.svd(a)[1])
```




    (0.9999999999999998, 0.9999999999999998)



### 行列式、跡



```python
a = np.array(([1,5],[3,4]))
a
```




    array([[1, 5],
           [3, 4]])





```python
LA.det(a), 1*4-5*3
```




    (-11.000000000000002, -11)





```python
a[0,0] + a[1,1]
```




    5





```python
# 二维
np.trace(a)
```




    5





```python
a = np.arange(8).reshape((2,2,2))
a
```




    array([[[0, 1],
            [2, 3]],
    
           [[4, 5],
            [6, 7]]])





```python
a[0,0] + a[1,1]
```




    array([6, 8])





```python
np.trace(a)
```




    array([6, 8])



### 固有値

 `eig` 正方形マトリックスの固有値と右の特徴ベクトルを計算します。 `eigvals` は特徴ベクトルを返さないことと違います。



```python
a = np.diag([1,2,3])
w, v = LA.eig(a)
```



```python
w
```




    array([1., 2., 3.])





```python
v
```




    array([[1., 0., 0.],
           [0., 1., 0.],[0., 0., 1.]])





```python
LA.eigvals(a)
```




    array([1., 2., 3.])





```python
a @ v == w * v
```




    array([[ True,  True,  True],
           [ True,  True,  True],[ True,  True,  True]])





```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (3, 3))
a
```




    array([[0, 7, 6],
           [4, 4, 8],[0, 6, 2]])





```python
w, v = LA.eig(a)
```



```python
np.allclose(a @ v, w * v)
```




    True





```python
LA.eigvals(a) == w
```




    array([ True,  True,  True])



 `eigh` エルミート行列または実対称行列の固有値と特徴ベクトルを計算します。 `eigvalsh` は、後者が特徴ベクトルを返さないことと異なります。


-  [エルミート行列 - ウィキペディア](https://zh.wikipedia.org/wiki/%E5%9F%83%E5%B0%94%E7%B1%B3%E7%89%B9%E7%9F%A9%E9%98%B5)
-  [対称マトリックス - ウィキペディア](https://zh.wikipedia.org/zh/%E5%B0%8D%E7%A8%B1%E7%9F%A9%E9%99%A3)


実対称行列を例にとる。



```python
a = np.array([
    [1, 2, 3],
    [2, 4, -5],
    [3, -5, 6]
])
```



```python
LA.eigh(a)
```




    (array([-3.07730361,  3.84139016, 10.23591345]),
     array([[-0.65271955,  0.74655097,  0.12891408],
            [ 0.55145509,  0.58486128, -0.59483995],[ 0.5194752 ,  0.31717334,  0.79343972]]))





```python
LA.eigvalsh(a)
```




    array([-3.07730361,  3.84139016, 10.23591345])





```python
LA.eig(a)
```




    (array([-3.07730361,  3.84139016, 10.23591345]),
     array([[-0.65271955,  0.74655097,  0.12891408],
            [ 0.55145509,  0.58486128, -0.59483995],[ 0.5194752 ,  0.31717334,  0.79343972]]))



## 行列演算

### 行列解決

 `solve` 直接解決できます：



```python
x = np.array([[1, 2], [3, 5]])
y = np.array([1, 2])
w = LA.solve(x, y)
```



```python
np.allclose(x.dot(w), y)
```




    True



 `tensorsolve` もっと一般的に：



```python
LA.tensorsolve(x, y)
```




    array([-1.,  1.])





```python
rng = np.random.default_rng(42)
x = rng.integers(0, 10, (6, 4, 2, 3, 4))
y = rng.integers(0, 10, (6, 4))
```



```python
LA.tensorsolve(x, y).shape
```




    (2, 3, 4)





```python
LA.solve(x,y)
```


    ---------------------------------------------------------------------------

    LinAlgError                               Traceback (most recent call last)

    <ipython-input-1256-9be96e929103> in <module>
    ----> 1 LA.solve(x,y)
    

    /usr/local/lib/python3.8/site-packages/numpy/core/overrides.py in solve(*args, **kwargs)
    

    /usr/local/lib/python3.8/site-packages/numpy/linalg/linalg.py in solve(a, b)
        378     a, _ = _makearray(a)379     _assert_stacked_2d(a)
    --> 380     _assert_stacked_square(a)
        381     b, wrap = _makearray(b)382     t, result_t = _commonType(a, b)
    

    /usr/local/lib/python3.8/site-packages/numpy/linalg/linalg.py in _assert_stacked_square(*arrays)
        201         m, n = a.shape[-2:]202         if m != n:
    --> 203             raise LinAlgError('Last 2 dimensions of the array must be square')
        204 205 def _assert_finite(*arrays):
    

    LinAlgError: Last 2 dimensions of the array must be square


最小二乗近似を使用して解くことができます：



```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
```



```python
data = load_iris()
```



```python
x = data["data"]
y = data["target"]
```



```python
x_train, x_test, y_train, y_test = \
train_test_split(x, y, test_size=0.2)
```



```python
w = LA.lstsq(x_train, y_train, rcond=None)[0]
```



```python
# 精准率
np.sum(
    np.abs(x_test.dot(w)).round()==y_test
)/len(y_test)
```




    1.0



### 逆行列

 `inv` 行列の逆を求めるために使用できます：



```python
a = np.arange(1, 5).reshape(2, 2)
a
```




    array([[1, 2],
           [3, 4]])





```python
inva = LA.inv(a)
```



```python
np.allclose(inva.dot(a), np.eye(2))
```




    True



特異値分解を用いて行列擬似逆を計算する。



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (2, 3))
a
```




    array([[0, 7, 6],
           [4, 4, 8]])





```python
inva = LA.pinv(a)
inva
```




    array([[-0.12751678,  0.14261745],
           [ 0.15436242, -0.08053691],[-0.01342282,  0.09395973]])





```python
np.allclose(a.dot(inva), np.eye(2))
```




    True





```python
np.allclose(a.dot(inva).dot(a), a)
```




    True





```python
np.allclose(a.dot(inva.dot(a)), a)
```




    True





```python
np.allclose(inva.dot(a.dot(inva)), inva)
```




    True



 `tensorinv` 高次元配列の逆値に適しています：



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (4, 6, 8, 3))
```



```python
ainv = np.linalg.tensorinv(a, ind=2)
```



```python
np.tensordot(a, ainv).shape
```




    (4, 6, 4, 6)





```python
eye = np.eye(4*6)
eye.shape = (4, 6, 4, 6)
```



```python
np.allclose(np.tensordot(a, ainv), eye)
```




    True



### 行列分解

 `cholesky` 分解とは、対称正定行列を下三角行列Lとその転置の積として表現する分解である。すべての固有値がゼロより大きい必要があります。


 [コレスキー分解 - ウィキペディア、自由百科事典](https://zh.wikipedia.org/zh-sg/%E7%A7%91%E5%88%97%E6%96%AF%E5%9F%BA%E5%88%86%E8%A7%A3)



```python
a = np.array([
    [4, 12, -16],
    [12, 37, -43],
    [-16, -43, 98]
])
```



```python
ca = LA.cholesky(a)
ca
```




    array([[ 2.,  0.,  0.],
           [ 6.,  1.,  0.],[-8.,  5.,  3.]])





```python
np.array_equal(ca.dot(ca.T), a)
```




    True



 `qr` 分解は行列を直交行列と上三角行列の積に分解します。


 [QR分解 - ウィキペディア、自由百科事典](https://zh.wikipedia.org/zh-hans/QR%E5%88%86%E8%A7%A3)



```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (4, 5))
a
```




    array([[0, 7, 6, 4, 4],
           [8, 0, 6, 2, 0],[5, 9, 7, 7, 7],[7, 5, 1, 8, 4]])





```python
# q是正交矩阵，r是上三角矩阵
q, r = LA.qr(a)
```



```python
q.shape, r.shape
```




    ((4, 4), (4, 5))





```python
np.allclose(q.dot(r), a)
```




    True





```python
# 转置=逆
np.allclose(LA.inv(q), q.T)
```




    True



 `svd` 分解は行列をユニタリー行列 `U`、非負実対角行列 `Σ`、共役転置行列 `V*` の積に分解する。Σの対角線上の要素は特異値である。

 `M = UΣV*`


 [特異値分解 - ウィキペディア](https://zh.wikipedia.org/zh-hans/%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3)



```python
a
```




    array([[0, 7, 6, 4, 4],
           [8, 0, 6, 2, 0],[5, 9, 7, 7, 7],[7, 5, 1, 8, 4]])





```python
u, s, vh = LA.svd(a)
```



```python
u.shape, s.shape, vh.shape
```




    ((4, 4), (4,), (5, 5))





```python
xgm = np.insert(np.diag(s), s.shape[0], 0, axis=1)
xgm.shape
```




    (4, 5)





```python
np.allclose(u.dot(xgm).dot(vh), a)
```




    True



## Einsum

 `einsum` を使用すると、多くの一般的な配列演算を簡潔に表現できます。

サブスクリプト文字列は、カンマ区切りのサブスクリプトラベルのリストであり、各ラベルは対応する操作の次元を参照しています。
- ラベルが繰り返された場合は、 `np.einsum("i,i", a, b)` と合計され、 `np.inner(a,b)` に相当する。
-  `np.einsum("i", a)` が一度だけ表示された場合は、自分のviewを返します。
- 繰り返し下書きラベルは対角線 `np.einsum("ii", a)` を取り、 `np.trace(a)` に相当する


暗黙のモードでは、下書きが重要で、出力の軸はアルファベット順に並べ替えられます。例えば：
-  `np.einsum("ij",a)` は2次元配列には影響しませんが、 `np.einsum("ji",a)` はトランスポートを返します。
-  `np.einsum("ij,jk", a, b)` は行列乗算を返し、 `np.einsum("ij,jh", a, b)` は乗算の転置を返します。


明示モードでは、出力サブスクリプトラベルを指定することで、出力を直接制御できます。この場合、 `->` 識別子が必要です。
-  `np.einsum("i->", a)` は `np.sum(a, axis=-1)` と似ています。
-  `np.einsum("ii->i", a)` は `np.diag(a)` と似ています。
-  `np.einsum("ij,jh->ih", a, b)` 結果のトランスポートではなく乗算結果を返します。

 `einsum` ブロードキャストはデフォルトではサポートされていません。有効にするには（左に追加した）省略記号が必要です。
- `np.einsum("..ii->...i", a)`
- 最初の次元と最後の次元を追跡します： `np.einsum("i...i", a)`
- 左端の軸行列を乗算します： `np.einsum("ij...,jk...->ik...", a, b)`



```python
a = np.arange(3)
b = np.arange(9).reshape(3, 3)
c = np.arange(6).reshape(2,3)
d = np.arange(6).reshape(3,2)
e = np.arange(60).reshape(3,4,5)
f = np.arange(24).reshape(4,3,2)
g = np.arange(30).reshape(3,5,2)
```

1つのタグ：



```python
# 返回自己的view
np.einsum("i", a)
```




    array([0, 1, 2])





```python
# 广播
np.einsum("...i", b)
```




    array([[0, 1, 2],
           [3, 4, 5],[6, 7, 8]])





```python
# 内积
np.einsum("i,i", a,a), np.inner(a, a)
```




    (5, 5)





```python
# 迹
np.einsum("ii", b), np.trace(b)
```




    (12, 12)



暗黙のモード：



```python
# 不影响结果
np.einsum("ij", c)
```




    array([[0, 1, 2],
           [3, 4, 5]])





```python
# 返回转置
np.einsum("ji", c)
```




    array([[0, 3],
           [1, 4],[2, 5]])





```python
# 外积
np.einsum("i,j", a,a), np.outer(a, a)
```




    (array([[0, 0, 0],
            [0, 1, 2],[0, 2, 4]]),
     array([[0, 0, 0],
            [0, 1, 2],[0, 2, 4]]))





```python
# 矩阵乘法
np.einsum("ij,jk", c, d)
```




    array([[10, 13],
           [28, 40]])





```python
# h在i之前，返回转置
np.einsum("ij,jh", c, d)
```




    array([[10, 28],
           [13, 40]])





```python
np.einsum("ij,j", c,a), np.dot(c,a)
```




    (array([ 5, 14]), array([ 5, 14]))



明示モード：



```python
# 求和
np.einsum("i->", a)
```




    10





```python
# np.sum(b, axis=1)
np.einsum("ij->i", b), np.sum(b, axis=1)
```




    (array([ 3, 12, 21]), array([ 3, 12, 21]))





```python
# 求和
np.einsum("ij->i", c), np.sum(c, axis=1)
```




    (array([ 3, 12]), array([ 3, 12]))





```python
# np.sum(d, axis=0)
np.einsum("ij->j", d), np.sum(d, axis=0)
```




    (array([6, 9]), array([6, 9]))





```python
# 元素相乘
np.einsum("i,i->i", a, a), a*a
```




    (array([0, 1, 4]), array([0, 1, 4]))





```python
np.einsum("i,j->", a, a), np.outer(a,a).sum()
```




    (9, 9)





```python
# 显式，转置
np.einsum("ij->ji", c)
```




    array([[0, 3],
           [1, 4],[2, 5]])





```python
# 返回diag
np.einsum("ii->i", b)
```




    array([0, 4, 8])





```python
# 元素相乘
np.einsum("ij,ij->ij", c, c)
```




    array([[ 0,  1,  4],
           [ 9, 16, 25]])





```python
# 矩阵乘法，因为显式指定，不会转置
np.einsum("ij,jh->ih", c,d)
```




    array([[10, 13],
           [28, 40]])





```python
# 求和
np.einsum("ij,jh->i", c, d)
```




    array([23, 68])





```python
np.einsum("ij,jh->h", c, d)
```




    array([38, 53])





```python
# 多维
# e=(3,4,5), f=(4,3,2)
np.einsum("ijk,jil->kl", e, f).shape
```




    (5, 2)





```python
np.einsum("ijk,ikh->ijh", e, g).shape
```




    (3, 4, 2)





```python
np.array_equal(
    np.einsum("ijk,ikh->ijh", e, g), 
    np.matmul(e,g)
)
```




    True





```python
np.einsum("ij,kl -> ijkl", c, d).shape
```




    (2, 3, 3, 2)





```python
np.array_equal(
    np.einsum("ij,kl->ijkl", c, d),
    c[:,:,None,None] * d
)
```




    True





```python
np.einsum("ij,jk->ijk", c, d).shape
```




    (2, 3, 2)





```python
np.array_equal(np.einsum("ij,jk->ijk", c, d), c[:,:,None]*d)
```




    True



放送：



```python
# 矩阵乘法
# c=(2,3), a=(3,)
np.einsum("...j,j", c, a)
```




    array([ 5, 14])





```python
# a=(3,), c=(2,3)
np.einsum("j,...j", a, c)
```




    array([ 5, 14])





```python
# a=(3,), d=(3, 2)
np.einsum("j,j...", a, d)
```




    array([10, 13])





```python
# 延续上面的乘法，显式
np.einsum("ij,j...->i...", c, d)
```




    array([[10, 13],
           [28, 40]])





```python
# 隐式
np.einsum("...j,ji", c, d)
```




    array([[10, 13],
           [28, 40]])



## Padding

次のパラメータを指定したPadding操作：

- 配列
- pad_width：シーケンス、整数、または配列で、各軸のエッジの拡張数。
- モード：デフォルトのconstant。また、edge,linear_ramp,maximum,mean,median,minimum,reflect,symmetric,wrap,emptyを含んでいます。
- stat_length：シーケンス、整数、または配列。パターンは `maximum`, `minimum`, `mean`, `median` \の場合、各軸のエッジの値を計算します。デフォルトはNoneです。
- constant_values：シーケンスまたはスカラー、パッドの値、デフォルトは0です。
- end_values：仮想列またはスカラー。パターンが `linear_ramp` のときに使用されます。終了値に使用されます。配列のエッジを埋めるようになります。デフォルトは0です。
- reflect_type：モードが `reflect` と `symmetric` のときに使用されます。デフォルトは `even` スタイルで、エッジ値の周りの反射は変更されません。 `odd` モードで、配列の拡張部分は、エッジ値の2倍から反射値を減算することで作成されます。

まずpad_widthパラメータを参照してください。



```python
# tuple
np.pad(
    [1,2,3,4,5], 
    (2,3),  # 等于((2,3),), ((2,3))
    "constant", 
    constant_values=(4, 6)
)
```




    array([4, 4, 1, 2, 3, 4, 5, 6, 6, 6])





```python
np.pad(
    [1,2,3,4,5], 
    2, # 等于(2), (2, )
    "constant", 
    constant_values=(4, 6)
)
```




    array([4, 4, 1, 2, 3, 4, 5, 6, 6])





```python
# 分别左上，右下
np.pad(
    [[1,2,3],[4,5,6]],
    (1, 2)
)
```




    array([[0, 0, 0, 0, 0, 0],
           [0, 1, 2, 3, 0, 0],[0, 4, 5, 6, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]])





```python
# 行（1,2）
# 列（2,1）
np.pad(
    [[1,2,3],[4,5,6]],
    ((1, 2), (2, 1))
)
```




    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 1, 2, 3, 0],[0, 0, 4, 5, 6, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]])





```python
np.pad(
    [[1,2,3],[4,5,6]],
    1
)
```




    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 0],[0, 4, 5, 6, 0],[0, 0, 0, 0, 0]])



次に、異なるモードを見て、それぞれのモードに対応する追加パラメータについて説明します。単純化のために、 `pad_width` 整数を統一的に使用します。



```python
a = np.arange(1, 7).reshape(3, 2)
a
```




    array([[1, 2],
           [3, 4],[5, 6]])





```python
# edge
np.pad(a, 1, "edge")
```




    array([[1, 1, 2, 2],
           [1, 1, 2, 2],[3, 3, 4, 4],[5, 5, 6, 6],[5, 5, 6, 6]])





```python
# linear_ramp
# 需要额外参数：end_values，默认0
np.pad(a, 1, "linear_ramp")
```




    array([[0, 0, 0, 0],
           [0, 1, 2, 0],[0, 3, 4, 0],[0, 5, 6, 0],[0, 0, 0, 0]])





```python
np.pad(a, 1, "linear_ramp", end_values=(1, ))
```




    array([[1, 1, 1, 1],
           [1, 1, 2, 1],[1, 3, 4, 1],[1, 5, 6, 1],[1, 1, 1, 1]])





```python
np.pad(a, 1, "linear_ramp", end_values=(1, 2), )
```




    array([[1, 1, 1, 2],
           [1, 1, 2, 2],[1, 3, 4, 2],[1, 5, 6, 2],[1, 2, 2, 2]])





```python
# 行（1,2）
# 列（3,4）
np.pad(a, 1, "linear_ramp", end_values=((1, 2), (3, 4)))
```




    array([[3, 1, 1, 4],
           [3, 1, 2, 4],[3, 3, 4, 4],[3, 5, 6, 4],[3, 2, 2, 4]])





```python
# 和这个等价
np.pad(
    a,
    1,
    "constant", 
    constant_values=((1, 2), (3,4))
)
```




    array([[3, 1, 1, 4],
           [3, 1, 2, 4],[3, 3, 4, 4],[3, 5, 6, 4],[3, 2, 2, 4]])





```python
# maximum, minmum, mean, median
# 需要额外参数stat_length，默认None，使用该轴所有值
np.pad(a, 1, "maximum")
```




    array([[6, 5, 6, 6],
           [2, 1, 2, 2],[4, 3, 4, 4],[6, 5, 6, 6],[6, 5, 6, 6]])





```python
# 只取2个
np.pad(a, 1, "maximum", stat_length=2)
```




    array([[4, 3, 4, 4],
           [2, 1, 2, 2],[4, 3, 4, 4],[6, 5, 6, 6],[6, 5, 6, 6]])





```python
# 分别取，左上2，右下1
np.pad(a, 1, "maximum", stat_length=((2, 1), ))
```




    array([[4, 3, 4, 4],
           [2, 1, 2, 2],[4, 3, 4, 4],[6, 5, 6, 6],[6, 5, 6, 6]])





```python
# 各自分别指定
# 行（2,1）
# 列（1,2）
np.pad(a, 1, "maximum", stat_length=((2, 1), (1, 2)))
```




    array([[3, 3, 4, 4],
           [1, 1, 2, 2],[3, 3, 4, 4],[5, 5, 6, 6],[5, 5, 6, 6]])





```python
b = a.astype(np.float16)
```



```python
# mean和median类似
# 行（2,1）
# 列（1,2）
np.pad(b, 1, "mean", stat_length=((2, 1), (1, 2)))
```




    array([[2. , 2. , 3. , 2.5],
           [1. , 1. , 2. , 1.5],[3. , 3. , 4. , 3.5],[5. , 5. , 6. , 5.5],[5. , 5. , 6. , 5.5]], dtype=float16)





```python
a = [1,2,3,4,5]
```



```python
# reflect, symmetric
# 需要额外参数reflect_type，默认even
# 首末是对称轴
np.pad(a, 3, "reflect")
```




    array([4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2])





```python
# 首末是对称轴
np.pad(a, 3, "reflect", reflect_type="odd")
```




    array([-2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8])





```python
# 边缘是对称轴
np.pad(a, 3, "symmetric")
```




    array([3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3])





```python
# 边缘是对称轴
np.pad(a, 3, "symmetric", reflect_type="odd")
```




    array([-1,  0,  1,  1,  2,  3,  4,  5,  5,  6,  7])





```python
# wrap
# 首尾互换
np.pad(a, 2, "wrap")
```




    array([4, 5, 1, 2, 3, 4, 5, 1, 2])





```python
# empty
# 未定义值扩展
np.pad(a, 1, "empty")
```




    array([              0,               1,               2,               3,
                         4,               5, 123145302310976])



## 畳み込み

畳み込み関数（1次元）は、次のルールに従います。

$$
(a * v)[n] = \sum_{m = -\infty}^{\infty} a[m] v[n - m]
$$



```python
np.convolve([1, 2, 3], [0, 1, 0.5], "valid"), 0.5*1+1*2+0*3
```




    (array([2.5]), 2.5)





```python
np.convolve([1, 2, 3], [0, 1, 0.5], "same")
```




    array([1. , 2.5, 4. ])





```python
np.convolve([0, 1, 2, 3, 0], [0, 1, 0.5], "valid") 
```




    array([1. , 2.5, 4. ])





```python
# 默认 full
np.convolve([1, 2, 3], [0, 1, 0.5])
```




    array([0. , 1. , 2.5, 4. , 1.5])





```python
np.convolve([0, 0, 1, 2, 3, 0, 0], [0, 1, 0.5], "valid")
```




    array([0. , 1. , 2.5, 4. , 1.5])



## マスク演算

NumPyの `ma` モジュールを使用して、NumPyと同じ多くの机能を持っています。私たちは主にMASKに関連する机能を紹介します。

このモジュールは、不完全なデータや無効なデータが含まれている場合、またはデータの一部を人為的に上書きする必要がある場合に主に使用されます。

mask配列はndarrayとmaskの組み合わせです。maskは `nomask` であってもよく、関連付けられた配列に対応する値が無効であることを示します。ブール配列のいずれかで、Falseの場合は相関配列に対応する位置値が有効（マスクされていない）、Trueの場合は無効（マスクされている）です。


 [Masked arrays—NumPy v1.23.de v0マニュアル](https://numpy.org/devdocs/reference/maskedarray.html)



```python
import numpy.ma as ma
```

### プロフィール



```python
x = np.array([1, 2, 3, -1, 5])
```



```python
mx = ma.masked_array(x, mask=[0,0,0,1,0])
mx
```




    masked_array(data=[1, 2, 3, --, 5],
                 mask=[False, False, False,  True, False],
           fill_value=999999)





```python
mx.mean(), (1+2+3+5)/4
```




    (2.75, 2.75)



マスクの値を指定することもできます。ここでは、2つの閾値rtolとatolです。デフォルト値は `np.allclose` と同じです。この範囲内の値がマスクされていることを意味します。



```python
# mask掉1.1
ma.masked_values([1, 1.1, 1.1+1e-8, 2, 3, 4], 1.1)
```




    masked_array(data=[1.0, --, --, 2.0, 3.0, 4.0],
                 mask=[False,  True,  True, False, False, False],
           fill_value=1.1)





```python
# 整数时完全相等才算
ma.masked_values([1, 2, 3, 4], 2, rtol=1, atol=2)
```




    masked_array(data=[1, --, 3, 4],
                 mask=[False,  True, False, False],
           fill_value=2)





```python
# 小数时按np.isclose`
# abs(a-b) < atol + rtol * abs(b)
# 以4为例，4-2 < 1.5+1*2
ma.masked_values([1., 2., 3., 4.], 2, rtol=1, atol=1.5)
```




    masked_array(data=[--, --, --, --],
                 mask=[ True,  True,  True,  True],
           fill_value=2.0,
                dtype=float64)





```python
(np.isclose(4, 2, rtol=1, atol=1.5), 
 np.allclose(4,2,rtol=1, atol=1.5))
```




    (True, True)



### 作成

mask配列を構築するには、いくつかの方法があります。

-  `MaskedArray` クラスを直接呼び出す：データ配列とMask配列を指定する必要があります。
- コンストラクタを使用します： `array` と `MaskedArray` のaliasである `masked_array`、前者は引数がわずかに異なります。
- 既存の配列を `view` でmask配列に変換します。
- 上記の `masked_values` のような他の組み込み関数や、与えられた値と完全に等しいものがマスクされる `masked_object`、条件に基づいてマスクされる `masked_where` などがあります。詳細については、以下のドキュメントを参照してください。

 [numpy.maモジュール-NumPy v1.23.de v0マニュアル](https://numpy.org/devdocs/reference/maskedarray.generic.html#constructing-masked-arrays)



```python
a = np.arange(6).reshape(2, 3)
mask = [[False, True, False],[False, False, True]]
```



```python
# 直接调用类
ma.MaskedArray(a, mask=mask)
```




    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)





```python
# 使用构造器
ma.array(a, mask=mask)
```




    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)





```python
# 类的alias
ma.masked_array(a, mask)
```




    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)





```python
# view
a.view(ma.MaskedArray)
```




    masked_array(
      data=[[0, 1, 2],
            [3, 4, 5]],
      mask=False,fill_value=999999)





```python
# masked_object一般用于对象
ma.masked_object(a, 3)
```




    masked_array(
      data=[[0, 1, 2],
            [--, 4, 5]],
      mask=[[False, False, False],
            [ True, False, False]],
      fill_value=3)





```python
ma.masked_object(
    np.array(["a", "b", "c"], dtype=object), 
    "b"
)
```




    masked_array(data=['a', --, 'c'],
                 mask=[False,  True, False],
           fill_value='b',
                dtype=object)





```python
# 有条件的
ma.masked_where(a>3, a)
```




    masked_array(
      data=[[0, 1, 2],
            [3, --, --]],
      mask=[[False, False, False],
            [False,  True,  True]],
      fill_value=999999)



### 取得



```python
a
```




    array([[0, 1, 2],
           [3, 4, 5]])





```python
m = ma.masked_where(a%2==0, a)
m
```




    masked_array(
      data=[[--, 1, --],
            [3, --, 5]],
      mask=[[ True, False,  True],
            [False,  True, False]],
      fill_value=999999)





```python
m.data
```




    array([[0, 1, 2],
           [3, 4, 5]])





```python
m.mask
```




    array([[ True, False,  True],
           [False,  True, False]])





```python
a[m.mask]
```




    array([0, 2, 4])





```python
a[~m.mask]
```




    array([1, 3, 5])





```python
cm = m.compressed()
cm
```




    array([1, 3, 5])





```python
cm.data.obj
```




    array([1, 3, 5])



### 修正

mask 1つ以上の値を直接指定できます。

まずはmask操作です：



```python
a = np.arange(6).reshape(2, 3)
# 第0行第2列，第1行第1列
a[(0,1),(2,1)] = ma.masked
a
```




    array([[0, 1, 0],
           [3, 0, 5]])





```python
a = np.arange(6).reshape(2, 3)
# 第1列
a[:,1] = ma.masked
a
```




    array([[0, 0, 2],
           [3, 0, 5]])





```python
a = np.arange(6).reshape(2, 3)
# 第1列以后的
a[:,1:] = ma.masked
a
```




    array([[0, 0, 0],
           [3, 0, 0]])





```python
a = ma.arange(6).reshape(2, 3)
a.mask = [1,0,1]
a
```




    masked_array(
      data=[[--, 1, --],
            [--, 4, --]],
      mask=[[ True, False,  True],
            [ True, False,  True]],
      fill_value=999999)





```python
ma.arange(6).reshape(2,3)
```




    masked_array(
      data=[[0, 1, 2],
            [3, 4, 5]],
      mask=False,fill_value=999999)



Maskをキャンセルするには、対応する位置に有効な値を与えるだけでよい。



```python
a = np.arange(6).reshape(2, 3)
mask = [[False, True, False],[False, False, True]]
```



```python
x = ma.array(a, mask=mask)
x
```




    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)





```python
x[0,1] = -1
x
```




    masked_array(
      data=[[0, -1, 2],
            [3, 4, --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value=999999)



hardmaskの場合（maskの値はunmaskではありません）は、まずsoftが必要です。



```python
x = ma.array(a, mask=mask, hard_mask=True)
x
```




    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)





```python
# 看，没啥用
x[0,1] = -1
x
```




    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)





```python
# 成功
x.soften_mask()
x[0,1] = -1
x
```




    masked_array(
      data=[[0, -1, 2],
            [3, 4, --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value=999999)





```python
# 再转成hard
x.harden_mask()
```




    masked_array(
      data=[[0, -1, 2],
            [3, 4, --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value=999999)





```python
x
```




    masked_array(
      data=[[0, -1, 2],
            [3, 4, --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value=999999)





```python
# hard后就不能在unmask了
x[1,2] = -2
x
```




    masked_array(
      data=[[0, -1, 2],
            [3, 4, --]],
      mask=[[False, False, False],
            [False, False,  True]],
      fill_value=999999)



すべてをアンマスクしたい場合は、 `nomask` を使用します：



```python
a = np.arange(6).reshape(2, 3)
mask = [[False, True, False],[False, False, True]]
x = ma.array(a, mask=mask)
x
```




    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)





```python
# 注意，hard后是不可以的
x.mask = ma.nomask
x
```




    masked_array(
      data=[[0, 1, 2],
            [3, 4, 5]],
      mask=[[False, False, False],
            [False, False, False]],
      fill_value=999999)



### インデックススライス

 `ndarray` のサブクラスなので、 `array` と同様であり、現在、他の多くの点でも共通しています。



```python
a = np.arange(6).reshape(2, 3)
mask = [[False, True, False],[False, False, True]]
x = ma.array(a, mask=mask)
x
```




    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)





```python
x[0]
```




    masked_array(data=[0, --, 2],
                 mask=[False,  True, False],
           fill_value=999999)





```python
x[0,0]
```




    0





```python
x[0,1]
```



    masked





```python
x[:,-1]
```




    masked_array(data=[2, --],
                 mask=[False,  True],
           fill_value=999999)





```python
x[:1]
```




    masked_array(data=[[0, --, 2]],
                 mask=[[False,  True, False]],
           fill_value=999999)



### 代数演算

 `ma` モジュールにはほとんどの通常関数の特定のタイミングがあり、位置がマスクまたは値の計算によって無効になった場合、直接マスクになります。

 `ma` 標准的な通関数もサポートされており、入力はmask配列で、出力は対応する位置も依然としてmaskです。



```python
ma.log([-1, 0, 1, 2])
```




    masked_array(data=[--, --, 0.0, 0.6931471805599453],
                 mask=[ True,  True, False, False],
           fill_value=1e+20)





```python
x
```




    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)





```python
np.log(x)
```

    <ipython-input-581-de666c833898>:1: RuntimeWarning: divide by zero encountered in log
      np.log(x)
    




    masked_array(
      data=[[--, --, 0.6931471805599453],
            [1.0986122886681098, 1.3862943611198906, --]],
      mask=[[ True,  True, False],
            [False, False,  True]],
      fill_value=999999)





```python
np.exp(x)
```




    masked_array(
      data=[[1.0, --, 7.38905609893065],
            [20.085536923187668, 54.598150033144236, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)



### 使用事例

一般的に欠落値や異常値の処理に使用されます。



```python
# 假设a[0,2]这个值是缺失值
a[0,2] = -9999
x = ma.masked_values(a, -9999)
x
```




    masked_array(
      data=[[0, 1, --],
            [3, 4, 5]],
      mask=[[False, False,  True],
            [False, False, False]],
      fill_value=-9999)





```python
x.mean(), 13/5
```




    (2.6, 2.6)





```python
x - x.mean()
```




    masked_array(
      data=[[-2.6, -1.6, --],
            [0.3999999999999999, 1.4, 2.4]],
      mask=[[False, False,  True],
            [False, False, False]],
      fill_value=-9999)





```python
x.anom()
```




    masked_array(
      data=[[-2.6, -1.6, --],
            [0.3999999999999999, 1.4, 2.4]],
      mask=[[False, False,  True],
            [False, False, False]],
      fill_value=-9999)





```python
x.anom(axis=0)
```




    masked_array(
      data=[[-1.5, -1.5, --],
            [1.5, 1.5, 0.0]],
      mask=[[False, False,  True],
            [False, False, False]],
      fill_value=-9999)





```python
x - x.mean(axis=0)
```




    masked_array(
      data=[[-1.5, -1.5, --],
            [1.5, 1.5, 0.0]],
      mask=[[False, False,  True],
            [False, False, False]],
      fill_value=-9999)



欠落している値を入力します：



```python
x.filled(x.mean())
```




    array([[0, 1, 2],
           [3, 4, 5]])





```python
x
```




    masked_array(
      data=[[0, 1, --],
            [3, 4, 5]],
      mask=[[False, False,  True],
            [False, False, False]],
      fill_value=-9999)



2つのマスクが落とした配列も計算できます：



```python
a = np.arange(6).reshape(2, 3)
mask1 = [[False, True, False],[False, False, True]]
mask2 = [[False, False, True],[False, False, True]]
x1 = ma.array(a, mask=mask1)
x2 = ma.array(a, mask=mask2)
x1, x2
```




    (masked_array(
       data=[[0, --, 2],
             [3, 4, --]],
       mask=[[False,  True, False],
             [False, False,  True]],
       fill_value=999999),
     masked_array(
       data=[[0, 1, --],
             [3, 4, --]],
       mask=[[False, False,  True],
             [False, False,  True]],
       fill_value=999999))





```python
x1+x2
```




    masked_array(
      data=[[0, --, --],
            [6, 8, --]],
      mask=[[False,  True,  True],
            [False, False,  True]],
      fill_value=999999)





```python
# 0/0无效，直接变成mask
np.sqrt(x1/x2)
```




    masked_array(
      data=[[--, --, --],
            [1.0, 1.0, --]],
      mask=[[ True,  True,  True],
            [False, False,  True]],
      fill_value=999999)



条件に基づいて配列を処理できます：



```python
a = np.arange(6).reshape(2, 3)
a
```




    array([[0, 1, 2],
           [3, 4, 5]])





```python
# 在给定范围之外的就给mask掉
m = ma.masked_outside(a, 1, 4)
m
```




    masked_array(
      data=[[--, 1, 2],
            [3, 4, --]],
      mask=[[ True, False, False],
            [False, False,  True]],
      fill_value=999999)





```python
m.mean(), (1+2+3+4)/4
```




    (2.5, 2.5)





```python
# 3到5之间的都给mask掉
m = ma.masked_inside(a, 3, 5)
m
```




    masked_array(
      data=[[0, 1, 2],
            [--, --, --]],
      mask=[[False, False, False],
            [ True,  True,  True]],
      fill_value=999999)





```python
m = ma.masked_greater(a, 2)
m
```




    masked_array(
      data=[[0, 1, 2],
            [--, --, --]],
      mask=[[False, False, False],
            [ True,  True,  True]],
      fill_value=999999)



## まとめ



## 参考

- [python - Understanding NumPy's einsum - Stack Overflow](https://stackoverflow.com/questions/26089893/understanding-numpys-einsum)
-  [ティム・ロックテシェル](https://rockt.github.io/2018/04/30/einsum)



```python

```
