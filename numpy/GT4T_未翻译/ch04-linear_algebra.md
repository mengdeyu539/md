<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#数组乘法" data-toc-modified-id="数组乘法-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>数组乘法</a></span><ul class="toc-item"><li><span><a href="#点积/内积/数量积/标量积" data-toc-modified-id="点积/内积/数量积/标量积-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>点积/内积/数量积/标量积</a></span></li><li><span><a href="#叉积/外积/向量积" data-toc-modified-id="叉积/外积/向量积-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>叉积/外积/向量积</a></span></li><li><span><a href="#张量积/外积" data-toc-modified-id="张量积/外积-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>张量积/外积</a></span></li><li><span><a href="#矩阵乘法" data-toc-modified-id="矩阵乘法-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>矩阵乘法</a></span></li><li><span><a href="#克罗内克积" data-toc-modified-id="克罗内克积-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>克罗内克积</a></span></li><li><span><a href="#多矩阵乘法" data-toc-modified-id="多矩阵乘法-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>多矩阵乘法</a></span></li></ul></li><li><span><a href="#基础概念" data-toc-modified-id="基础概念-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>基础概念</a></span><ul class="toc-item"><li><span><a href="#范数" data-toc-modified-id="范数-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>范数</a></span></li><li><span><a href="#行列式、迹" data-toc-modified-id="行列式、迹-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>行列式、迹</a></span></li><li><span><a href="#特征值" data-toc-modified-id="特征值-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>特征值</a></span></li></ul></li><li><span><a href="#矩阵运算" data-toc-modified-id="矩阵运算-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>矩阵运算</a></span><ul class="toc-item"><li><span><a href="#矩阵求解" data-toc-modified-id="矩阵求解-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>矩阵求解</a></span></li><li><span><a href="#逆矩阵" data-toc-modified-id="逆矩阵-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>逆矩阵</a></span></li><li><span><a href="#矩阵分解" data-toc-modified-id="矩阵分解-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>矩阵分解</a></span></li></ul></li><li><span><a href="#Einsum" data-toc-modified-id="Einsum-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Einsum</a></span></li><li><span><a href="#Padding" data-toc-modified-id="Padding-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Padding</a></span></li><li><span><a href="#卷积" data-toc-modified-id="卷积-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>卷积</a></span></li><li><span><a href="#掩码运算" data-toc-modified-id="掩码运算-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>掩码运算</a></span><ul class="toc-item"><li><span><a href="#简介" data-toc-modified-id="简介-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>简介</a></span></li><li><span><a href="#创建" data-toc-modified-id="创建-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>创建</a></span></li><li><span><a href="#获取" data-toc-modified-id="获取-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>获取</a></span></li><li><span><a href="#修改" data-toc-modified-id="修改-7.4"><span class="toc-item-num">7.4&nbsp;&nbsp;</span>修改</a></span></li><li><span><a href="#索引切片" data-toc-modified-id="索引切片-7.5"><span class="toc-item-num">7.5&nbsp;&nbsp;</span>索引切片</a></span></li><li><span><a href="#代数运算" data-toc-modified-id="代数运算-7.6"><span class="toc-item-num">7.6&nbsp;&nbsp;</span>代数运算</a></span></li><li><span><a href="#使用案例" data-toc-modified-id="使用案例-7.7"><span class="toc-item-num">7.7&nbsp;&nbsp;</span>使用案例</a></span></li></ul></li><li><span><a href="#小结" data-toc-modified-id="小结-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>小结</a></span></li><li><span><a href="#参考" data-toc-modified-id="参考-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>参考</a></span></li></ul></div>


```python
import numpy as np
np.__version__
```




    '1.22.3'



文档阅读说明：

- 🐧 表示 Tip
- ⚠️ 表示注意事项

## 数组乘法

注意：不要太关注它们叫什么，看看它们做了什么。

### 点积/内积/数量积/标量积

**点积**

`np.dot`:

- 如果 a 和 b 是一维的，就是内积 `np.inner`
- 如果 a 和 b 是二维的，是矩阵乘法 `np.matmul or a @ b`
- 如果 a 或 b 任意一个是常量 `np.multiply or a * b`
- 如果 a 是 N 维，b 是一维 `sum product`
- 如果 a 是 N 维，b 是 M 维 `dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])`

`np.vdot`: 多维输入会被flatten后计算点积。另外在计算复数时与`np.dot`有所不同。



**内积**

`np.inner`: 一维数组向量的普通内积（无复数共轭），在更高维度上，是最后一个轴上的sum product。

- 对一维数组，就是元素乘积之和 `sum(a * b)
- 如果有一个是标量，那就是直接相乘
- 对多维数组，等于 `np.tensordot(a, b, axes=(-1, -1))`，对于某个具体的索引，就是相乘后在最后一个维度上求和 `inner(a, b)[i0,...,ir-2,j0,...,js-2] = sum(a[i0,...,ir-2,:]*b[j0,...,js-2,:])`

[数量积 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E7%82%B9%E7%A7%AF)

a和b都是一维：


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



a和b都是二维：


```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (3, 4))
b = rng.integers(0, 10, (3, 4))
```


```python
np.inner(a, b)
```




    array([[119,  71,  63],
           [126,  52,  98],
           [112,  86,  96]])




```python
np.tensordot(a, b, axes=(-1, -1))
```




    array([[119,  71,  63],
           [126,  52,  98],
           [112,  86,  96]])




```python
a @ b.T
```




    array([[119,  71,  63],
           [126,  52,  98],
           [112,  86,  96]])




```python
np.matmul(a, b.T)
```




    array([[119,  71,  63],
           [126,  52,  98],
           [112,  86,  96]])




```python
np.vdot(a, b), np.dot(a.flatten(), b.flatten())
```




    (267, 267)



a或b是常量：


```python
a * 2
```




    array([[ 0, 14, 12,  8],
           [ 8, 16,  0, 12],
           [ 4,  0, 10, 18]])




```python
np.dot(a, 2)
```




    array([[ 0, 14, 12,  8],
           [ 8, 16,  0, 12],
           [ 4,  0, 10, 18]])




```python
np.multiply(a, 2)
```




    array([[ 0, 14, 12,  8],
           [ 8, 16,  0, 12],
           [ 4,  0, 10, 18]])




```python
np.inner(a, 2)
```




    array([[ 0, 14, 12,  8],
           [ 8, 16,  0, 12],
           [ 4,  0, 10, 18]])



a是多维b是一维：


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



a是m维b是n维：


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



### 叉积/外积/向量积


\begin{aligned}\mathbf {u\times v} &={\begin{vmatrix}u_{2}&u_{3}\\v_{2}&v_{3}\end{vmatrix}}\mathbf {i} -{\begin{vmatrix}u_{1}&u_{3}\\v_{1}&v_{3}\end{vmatrix}}\mathbf {j} +{\begin{vmatrix}u_{1}&u_{2}\\v_{1}&v_{2}\end{vmatrix}}\mathbf {k} \\&=(u_{2}v_{3}-u_{3}v_{2})\mathbf {i} -(u_{1}v_{3}-u_{3}v_{1})\mathbf {j} +(u_{1}v_{2}-u_{2}v_{1})\mathbf {k} \end{aligned}

只支持二维或三维，表示与u和v都垂直的向量。

[叉积 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E5%8F%89%E7%A7%AF)


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



还有几个关于维度的三参数，用于改变数组的定义（有点类似C和F Style）。


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
           [  0, -36],
           [  0,  20]])




```python
a.T,b.T
```




    (array([[0, 4],
            [7, 4],
            [6, 8]]),
     array([[0, 0],
            [6, 5],
            [2, 9]]))




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



### 张量积/外积

- [张量积 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E5%BC%A0%E9%87%8F%E7%A7%AF)
- [外积 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E5%A4%96%E7%A7%AF)


```python
a = [1, 2, 4]
b = [4, 5, 6]
```


```python
np.outer(a, b)
```




    array([[ 4,  5,  6],
           [ 8, 10, 12],
           [16, 20, 24]])




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



### 矩阵乘法

`np.dot`上面已经提过了，其实还有个`np.matmul`和它很类似，但二者是有一些区别的。

- `dot`不是通函数，而`matmul`是通函数，也就意味这有通函数的一些通用参数
- `matmul`不支持向量和数字相乘
- `matmul`矩阵（好像元素一样）堆叠在一起广播

关于`np.matmul`：

- 如果都是二维，就是常规的矩阵乘法
- 如果任意个是多维（>2），则把它当成驻留在最后两个索引中的矩阵堆栈，并相应地广播
- 如果第一个是一维，则通过在其维度前面加上 1 来将其提升为矩阵，矩阵乘法后删除前面附加的 1
- 如果第二个是一维，则在维度上append 1，矩阵乘法后再删除后面附加的 1

二维的情况：


```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (2, 3))
b = rng.integers(0, 10, (3, 2))
a, b
```




    (array([[0, 7, 6],
            [4, 4, 8]]),
     array([[0, 6],
            [2, 0],
            [5, 9]]))




```python
np.matmul(a, b)
```




    array([[44, 54],
           [48, 96]])



其中一个是一维的情况：


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



任意一个是多维：


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



看个简单点的例子：


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
            [70, 70],
            [84, 84]],
    
           [[48, 24],
            [10,  2],
            [97, 41]]])




```python
a[0,:,:] @ b[0,:,:]
```




    array([[49, 49],
           [70, 70],
           [84, 84]])




```python
a[1,:,:] @ b[1,:,:]
```




    array([[48, 24],
           [10,  2],
           [97, 41]])




```python
# 2x3x2
np.matmul(a[0,:,:], b)
```




    array([[[49, 49],
            [70, 70],
            [84, 84]],
    
           [[56, 28],
            [62, 22],
            [84, 36]]])




```python
# 2x3x2
np.matmul(a[1,:,:], b)
```




    array([[[42, 42],
            [14, 14],
            [98, 98]],
    
           [[48, 24],
            [10,  2],
            [97, 41]]])



看看dot是咋样表现的：


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



### 克罗内克积

>维基百科：是两个任意大小的矩阵间的运算，表示为⊗。克罗内克积是外积从向量到矩阵的推广，也是张量积在标准基下的矩阵表示。

[克罗内克积 - 维基百科，自由的百科全书](https://zh.m.wikipedia.org/zh-hans/%E5%85%8B%E7%BD%97%E5%86%85%E5%85%8B%E7%A7%AF)

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
           [2, 1, 4, 2],
           [0, 9, 0, 3],
           [6, 3, 2, 1]])




```python
a = np.ones((2,5,2,5))
b = np.ones((2,3,4))
np.kron(a,b).shape
```




    (2, 10, 6, 20)



### 多矩阵乘法

`linalg.multi_dot`链式调用`np.dot`，自动选择最快的顺序。

- 如果第一个数组是一维，则被当做行向量
- 如果最后一个数组是一维，则被当做列向量
- 如果输入的向量超过两个，则其他向量必须是二维


```python
a = np.ones((2, 4))
b = np.ones((4, 3))
c = np.ones((3, 5))
```


```python
np.linalg.multi_dot((a,b,c)).shape
```




    (2, 5)



不同的顺序性能不同，比如：

`A_{10x100}, B_{100x5}, C_{5x50}`


`cost((AB)C) = 10*100*5 + 10*5*50   = 5000 + 2500   = 7500`
`cost(A(BC)) = 10*100*50 + 100*5*50 = 50000 + 25000 = 75000`


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
    

如果首尾是一维：


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



## 基础概念

介绍线性代数几个常用的API，不涉及数学知识。


```python
from numpy import linalg as LA
```

### 范数

共包括：

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



### 行列式、迹


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



### 特征值

`eig`计算一个方阵的特征值和右特征向量，`eigvals`与之的区别是不返回特征向量。


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
           [0., 1., 0.],
           [0., 0., 1.]])




```python
LA.eigvals(a)
```




    array([1., 2., 3.])




```python
a @ v == w * v
```




    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]])




```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (3, 3))
a
```




    array([[0, 7, 6],
           [4, 4, 8],
           [0, 6, 2]])




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



`eigh`计算埃尔米特矩阵或实对称矩阵的特征值和特征向量，`eigvalsh`与之的区别是后者不返回特征向量。


- [埃尔米特矩阵 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E5%9F%83%E5%B0%94%E7%B1%B3%E7%89%B9%E7%9F%A9%E9%98%B5)
- [對稱矩陣 - 维基百科，自由的百科全书](https://zh.wikipedia.org/zh/%E5%B0%8D%E7%A8%B1%E7%9F%A9%E9%99%A3)


以实对称矩阵为例。


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
            [ 0.55145509,  0.58486128, -0.59483995],
            [ 0.5194752 ,  0.31717334,  0.79343972]]))




```python
LA.eigvalsh(a)
```




    array([-3.07730361,  3.84139016, 10.23591345])




```python
LA.eig(a)
```




    (array([-3.07730361,  3.84139016, 10.23591345]),
     array([[-0.65271955,  0.74655097,  0.12891408],
            [ 0.55145509,  0.58486128, -0.59483995],
            [ 0.5194752 ,  0.31717334,  0.79343972]]))



## 矩阵运算

### 矩阵求解

`solve`可直接求解：


```python
x = np.array([[1, 2], [3, 5]])
y = np.array([1, 2])
w = LA.solve(x, y)
```


```python
np.allclose(x.dot(w), y)
```




    True



`tensorsolve`更加通用一些：


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
        378     a, _ = _makearray(a)
        379     _assert_stacked_2d(a)
    --> 380     _assert_stacked_square(a)
        381     b, wrap = _makearray(b)
        382     t, result_t = _commonType(a, b)
    

    /usr/local/lib/python3.8/site-packages/numpy/linalg/linalg.py in _assert_stacked_square(*arrays)
        201         m, n = a.shape[-2:]
        202         if m != n:
    --> 203             raise LinAlgError('Last 2 dimensions of the array must be square')
        204 
        205 def _assert_finite(*arrays):
    

    LinAlgError: Last 2 dimensions of the array must be square


可以使用最小二乘法近似求解：


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



### 逆矩阵

`inv`可用于求矩阵逆：


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



使用奇异值分解计算矩阵伪逆。


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
           [ 0.15436242, -0.08053691],
           [-0.01342282,  0.09395973]])




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



`tensorinv`适用于高维数组求逆：


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



### 矩阵分解

`cholesky`分解是把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解。要求所有的特征值必须大于零。


[科列斯基分解 - 维基百科，自由的百科全书](https://zh.wikipedia.org/zh-sg/%E7%A7%91%E5%88%97%E6%96%AF%E5%9F%BA%E5%88%86%E8%A7%A3)


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
           [ 6.,  1.,  0.],
           [-8.,  5.,  3.]])




```python
np.array_equal(ca.dot(ca.T), a)
```




    True



`qr`分解将矩阵分解成一个正交矩阵和一个上三角矩阵的积。


[QR 分解 - 维基百科，自由的百科全书](https://zh.wikipedia.org/zh-hans/QR%E5%88%86%E8%A7%A3)


```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (4, 5))
a
```




    array([[0, 7, 6, 4, 4],
           [8, 0, 6, 2, 0],
           [5, 9, 7, 7, 7],
           [7, 5, 1, 8, 4]])




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



`svd`分解将矩阵分解为一个酉矩阵`U`、一个非负实对角矩阵`Σ`和一个共轭转置矩阵`V*`的乘积。Σ对角线上的元素为奇异值。

`M = UΣV*`


[奇异值分解 - 维基百科，自由的百科全书](https://zh.wikipedia.org/zh-hans/%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3)


```python
a
```




    array([[0, 7, 6, 4, 4],
           [8, 0, 6, 2, 0],
           [5, 9, 7, 7, 7],
           [7, 5, 1, 8, 4]])




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

使用`einsum`可以让很多常见的数组运算以简洁的方式表示。

下标字符串是一个逗号分隔的下标标签列表，每个标签指的是相应操作的一个维度。
- 标签重复时会被求和`np.einsum("i,i", a, b)`，等价于`np.inner(a,b)`。
- 如果只出现一次`np.einsum("i", a)`，返回自己的view。
- 重复下标标签取对角线`np.einsum("ii", a)`，等价于`np.trace(a)`


在隐式模式下，下标很重要，输出的轴会按字母重新排序。比如：
- `np.einsum("ij",a)`不会影响二维数组，但`np.einsum("ji",a)`则返回转置。
- `np.einsum("ij,jk", a, b)`会返回矩阵乘法，而`np.einsum("ij,jh", a, b)`则返回乘法的转置。


在显式模式下，可以通过指定输出下标标签直接控制输出。此时需要`->`标识符。
- `np.einsum("i->", a)`类似于`np.sum(a, axis=-1)`。
- `np.einsum("ii->i", a)`类似于`np.diag(a)`。
- `np.einsum("ij,jh->ih", a, b)`返回乘法结果，而不是结果的转置。

`einsum`默认不支持广播，要启用需使用（在左侧添加）省略号。
- `np.einsum("..ii->...i", a)`
- 跟踪第一个和最后一个维度：`np.einsum("i...i", a)`
- 用最左边的轴矩阵乘法：`np.einsum("ij...,jk...->ik...", a, b)`


```python
a = np.arange(3)
b = np.arange(9).reshape(3, 3)
c = np.arange(6).reshape(2,3)
d = np.arange(6).reshape(3,2)
e = np.arange(60).reshape(3,4,5)
f = np.arange(24).reshape(4,3,2)
g = np.arange(30).reshape(3,5,2)
```

一个标签：


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
           [3, 4, 5],
           [6, 7, 8]])




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



隐式模式：


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
           [1, 4],
           [2, 5]])




```python
# 外积
np.einsum("i,j", a,a), np.outer(a, a)
```




    (array([[0, 0, 0],
            [0, 1, 2],
            [0, 2, 4]]),
     array([[0, 0, 0],
            [0, 1, 2],
            [0, 2, 4]]))




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



显式模式：


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
           [1, 4],
           [2, 5]])




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



广播：


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

Padding操作，参数如下：

- 数组
- pad_width：序列、整数或数组，每个轴边缘扩展的数量。
- 模式：默认constant。还包括：edge, linear_ramp, maximum, mean, median, minimum, reflect, symmetric, wrap, empty。
- stat_length：序列、整数或数组，模式是`maximum`, `minimum`, `mean`, `median`\时，用来计算每个轴边缘的值，默认None。
- constant_values：序列或标量，padding的值，默认0。
- end_values：虚列或标量，模式是`linear_ramp`时使用，用于结束值，经形成填充数组的边缘，默认0。
- reflect_type：模式是`reflect`和`symmetric`时使用，默认使用`even`风格，边缘值周围不改变反射，`odd`模式，数组的拓展部分是通过从边缘值的两倍中减去反射值来创建的。

首先看pad_width参数：


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
           [0, 1, 2, 3, 0, 0],
           [0, 4, 5, 6, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]])




```python
# 行（1,2）
# 列（2,1）
np.pad(
    [[1,2,3],[4,5,6]],
    ((1, 2), (2, 1))
)
```




    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 1, 2, 3, 0],
           [0, 0, 4, 5, 6, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]])




```python
np.pad(
    [[1,2,3],[4,5,6]],
    1
)
```




    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 0],
           [0, 4, 5, 6, 0],
           [0, 0, 0, 0, 0]])



接下来看下不同模式，顺带了解不同模式对应的额外参数。简单起见，`pad_width`我们统一使用整数。


```python
a = np.arange(1, 7).reshape(3, 2)
a
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
# edge
np.pad(a, 1, "edge")
```




    array([[1, 1, 2, 2],
           [1, 1, 2, 2],
           [3, 3, 4, 4],
           [5, 5, 6, 6],
           [5, 5, 6, 6]])




```python
# linear_ramp
# 需要额外参数：end_values，默认0
np.pad(a, 1, "linear_ramp")
```




    array([[0, 0, 0, 0],
           [0, 1, 2, 0],
           [0, 3, 4, 0],
           [0, 5, 6, 0],
           [0, 0, 0, 0]])




```python
np.pad(a, 1, "linear_ramp", end_values=(1, ))
```




    array([[1, 1, 1, 1],
           [1, 1, 2, 1],
           [1, 3, 4, 1],
           [1, 5, 6, 1],
           [1, 1, 1, 1]])




```python
np.pad(a, 1, "linear_ramp", end_values=(1, 2), )
```




    array([[1, 1, 1, 2],
           [1, 1, 2, 2],
           [1, 3, 4, 2],
           [1, 5, 6, 2],
           [1, 2, 2, 2]])




```python
# 行（1,2）
# 列（3,4）
np.pad(a, 1, "linear_ramp", end_values=((1, 2), (3, 4)))
```




    array([[3, 1, 1, 4],
           [3, 1, 2, 4],
           [3, 3, 4, 4],
           [3, 5, 6, 4],
           [3, 2, 2, 4]])




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
           [3, 1, 2, 4],
           [3, 3, 4, 4],
           [3, 5, 6, 4],
           [3, 2, 2, 4]])




```python
# maximum, minmum, mean, median
# 需要额外参数stat_length，默认None，使用该轴所有值
np.pad(a, 1, "maximum")
```




    array([[6, 5, 6, 6],
           [2, 1, 2, 2],
           [4, 3, 4, 4],
           [6, 5, 6, 6],
           [6, 5, 6, 6]])




```python
# 只取2个
np.pad(a, 1, "maximum", stat_length=2)
```




    array([[4, 3, 4, 4],
           [2, 1, 2, 2],
           [4, 3, 4, 4],
           [6, 5, 6, 6],
           [6, 5, 6, 6]])




```python
# 分别取，左上2，右下1
np.pad(a, 1, "maximum", stat_length=((2, 1), ))
```




    array([[4, 3, 4, 4],
           [2, 1, 2, 2],
           [4, 3, 4, 4],
           [6, 5, 6, 6],
           [6, 5, 6, 6]])




```python
# 各自分别指定
# 行（2,1）
# 列（1,2）
np.pad(a, 1, "maximum", stat_length=((2, 1), (1, 2)))
```




    array([[3, 3, 4, 4],
           [1, 1, 2, 2],
           [3, 3, 4, 4],
           [5, 5, 6, 6],
           [5, 5, 6, 6]])




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
           [1. , 1. , 2. , 1.5],
           [3. , 3. , 4. , 3.5],
           [5. , 5. , 6. , 5.5],
           [5. , 5. , 6. , 5.5]], dtype=float16)




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



## 卷积

卷积函数（一维）遵循以下规则：

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



## 掩码运算

使用NumPy的`ma`模块，其很多功能和NumPy一样，我们主要介绍与MASK相关的功能。

本模块主要用于处理不完整数据或包含无效数据，或需要人为覆盖掉一部分数据的情况。

一个mask数组是一个ndarray和一个mask的组合。mask既可以是`nomask`，表示相关联数组对应的值是无效的；要么是一个布尔数组，为False时，关联数组对应位置值有效（未遮掩），为True时无效（被遮掩）。


[Masked arrays — NumPy v1.23.dev0 Manual](https://numpy.org/devdocs/reference/maskedarray.html)


```python
import numpy.ma as ma
```

### 简介


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



也可以指定mask的值，这里两个阈值rtol和atol，默认值和`np.allclose`一样，表示在该范围内的值都被mask。


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



### 创建

有多种方法可以构造mask数组：

- 直接调用`MaskedArray`类：需要指定数据数组和Mask数组。
- 使用构造器：`array`和`masked_array`，后者是`MaskedArray`的alias，前者在参数上略有不同。
- 对已有数组通过`view`转为mask数组。
- 其他一些内置的函数，比如上面提到的`masked_values`，还有比如与给定value完全相等的会被mask的`masked_object`，根据条件mask的`masked_where`等。更多可参考下面的文档。

[The numpy.ma module — NumPy v1.23.dev0 Manual](https://numpy.org/devdocs/reference/maskedarray.generic.html#constructing-masked-arrays)


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
      mask=False,
      fill_value=999999)




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



### 获取


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



### 修改

mask一个或多个值可以直接指定。

首先是mask操作：


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
      mask=False,
      fill_value=999999)



取消Mask，只需要给对应位置一个有效值即可。


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



如果是hardmask（mask的值不能unmask），则需要先soft：


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



如果想要unmask掉所有的，直接用`nomask`：


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



### 索引切片

因为是`ndarray`的子类，所以和`array`是类似的，当前，很多其他方面也是通用的。


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



### 代数运算

`ma`模块有大多数通函数的特定时限，位置被mask或值计算无效时，都会直接变成mask。

`ma`也支持标准的通函数，输入mask数组，输出对应位置依然也是mask的。


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



### 使用案例

一般用于缺失值或异常值的处理。


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



填充缺失值：


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



两个mask掉的数组也可以计算：


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



可以根据条件处理数组：


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



## 小结



## 参考

- [python - Understanding NumPy's einsum - Stack Overflow](https://stackoverflow.com/questions/26089893/understanding-numpys-einsum)
- [Tim Rocktäschel](https://rockt.github.io/2018/04/30/einsum)


```python

```
