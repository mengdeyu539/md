<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#数学函数" data-toc-modified-id="数学函数-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>数学函数 </a></span><ul class="toc-item"><li><span><a href="#三角/双曲函数" data-toc-modified-id="三角/双曲函数-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>三角 / 双曲函数 </a></span></li><li><span><a href="#指数和对数" data-toc-modified-id="指数和对数-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>指数和对数 </a></span></li><li><span><a href="#算术操作" data-toc-modified-id="算术操作-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>算术操作 </a></span></li><li><span><a href="#自动域" data-toc-modified-id="自动域-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>自动域 </a></span></li></ul></li><li><span><a href="#数值计算" data-toc-modified-id="数值计算-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>数值计算 </a></span><ul class="toc-item"><li><span><a href="#舍入" data-toc-modified-id="舍入-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>舍入 </a></span></li><li><span><a href="#和积差" data-toc-modified-id="和积差-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>和积差 </a></span></li><li><span><a href="#符号函数" data-toc-modified-id="符号函数-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>符号函数 </a></span></li><li><span><a href="#截断" data-toc-modified-id="截断-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>截断 </a></span></li><li><span><a href="#插值" data-toc-modified-id="插值-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>插值 </a></span></li></ul></li><li><span><a href="#导数和微积分" data-toc-modified-id="导数和微积分-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>导数和微积分 </a></span><ul class="toc-item"><li><span><a href="#梯度" data-toc-modified-id="梯度-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>梯度 </a></span></li><li><span><a href="#梯形公式" data-toc-modified-id="梯形公式-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>梯形公式 </a></span></li></ul></li><li><span><a href="#多项式" data-toc-modified-id="多项式-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>多项式 </a></span><ul class="toc-item"><li><span><a href="#简介" data-toc-modified-id="简介-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>简介 </a></span></li><li><span><a href="#便捷类" data-toc-modified-id="便捷类-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>便捷类 </a></span></li></ul></li><li><span><a href="#关系运算" data-toc-modified-id="关系运算-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>关系运算</a></span><ul class="toc-item"><li><span><a href="#真值测试" data-toc-modified-id="真值测试-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>真值测试 </a></span></li><li><span><a href="#值和类型" data-toc-modified-id="值和类型-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>值和类型 </a></span></li><li><span><a href="#逻辑运算" data-toc-modified-id="逻辑运算-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>逻辑运算 </a></span></li><li><span><a href="#比较" data-toc-modified-id="比较-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>比较 </a></span></li></ul></li><li><span><a href="#二进制运算" data-toc-modified-id="二进制运算-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>二进制运算 </a></span><ul class="toc-item"><li><span><a href="#位运算" data-toc-modified-id="位运算-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>位运算 </a></span></li><li><span><a href="#左右移" data-toc-modified-id="左右移-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>左右移 </a></span></li><li><span><a href="#打包解包" data-toc-modified-id="打包解包-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>打包解包 </a></span></li></ul></li><li><span><a href="#字符串" data-toc-modified-id="字符串-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>字符串 </a></span><ul class="toc-item"><li><span><a href="#基本操作" data-toc-modified-id="基本操作-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>基本操作 </a></span></li><li><span><a href="#比较" data-toc-modified-id="比较-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>比较 </a></span></li><li><span><a href="#基本信息" data-toc-modified-id="基本信息-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>基本信息 </a></span></li></ul></li><li><span><a href="#小结" data-toc-modified-id="小结-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>小结 </a></span></li><li><span><a href="#参考" data-toc-modified-id="参考-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>参考 </a></span></li></ul></div>


```python
import numpy as np
np.__version__
```




    '1.22.3'



文档阅读说明：

- 🐧 表示 Tip
- ⚠️ 表示注意事项

## 数学函数

NumPy内置了很多数学函数，包括：

- 三角/双曲函数
- 四舍五入
- 和、积、差
- 导数和微积分
- 指数和对数
- 算术操作
- 综合

关于这一部分，我们主要介绍一些特殊的方法，对比较简单的就跳过了，可以参考：

- [Mathematical functions — NumPy v1.23.dev0 Manual](https://numpy.org/devdocs/reference/routines.math.html)

### 三角/双曲函数

三角函数和双曲中大部分都很容易理解，也都是通函数，我们主要介绍一个看起来不太明显的：`unwrap`，它的主要目的是对周期取大增量补码。参数如下：

- p：数组
- discont：数值间的最大中断，默认`period/2`，低于该值的被设为该值
- axis：轴，默认最后一个轴
- period：周期范围，默认`2pi`


```python
phase = np.linspace(0, np.pi, num=5)
phase[3:] += np.pi
phase
```




    array([0.        , 0.78539816, 1.57079633, 5.49778714, 6.28318531])




```python
# 超过pi的，剪掉period
np.unwrap(phase)
```




    array([ 0.        ,  0.78539816,  1.57079633, -0.78539816,  0.        ])




```python
# 要减掉 1 个周期
np.unwrap([1, 5]), 5 - 2*np.pi
```




    (array([ 1.        , -1.28318531]), -1.2831853071795862)




```python
# 要减掉3个周期
np.unwrap([1, 20]), 20 - 3*2*np.pi
```




    (array([1.        , 1.15044408]), 1.1504440784612413)




```python
# 超过 pi 的处理掉！
np.unwrap([1, 1.1+np.pi]), 1.1+np.pi-2*np.pi
```




    (array([ 1.        , -2.04159265]), -2.0415926535897935)



再看几个例子：


```python
# 超过4/2，加4
np.unwrap([0, 1, 2, -1, 0], period=4)
```




    array([0, 1, 2, 3, 4])




```python
# 为什么要加而不是减，因为只有加才能满足条件
np.unwrap([1, -2, -1, 0], period=4)
```




    array([1, 2, 3, 4])




```python
# 同上，5 后面的数字都要加 4
np.unwrap([2, 3, 4, 5, 2, 3, 4, 5], period=4)
```




    array([2, 3, 4, 5, 6, 7, 8, 9])



另外需要注意的是，`deg2rad` == `radians`，`rad2deg` == `degrees`，前面的表达更加清晰一些。

更多细节可查阅：

- https://numpy.org/devdocs/reference/routines.math.html

### 指数和对数

大部分的API都比较容易理解，比如自然指数`np.exp`，2为底指数`np.exp2`等，对应的log也有`np.log`，`np.log2`，`np.log10`等，而且所有API都是通函数。

另外，也有`np.expm1`表示exp后减1，对应的就是加1后log的`np.log1p`：


```python
np.log(np.exp(2)), np.log1p(np.expm1(2))
```




    (2.0, 2.0)



还有两个求和的`np.logaddexp`和以2为底的`np.logaddexp2`，计算公式：`log(exp(x1) + exp(x2))`


```python
np.logaddexp([1], [2]), np.log(np.exp(1) + np.exp(2))
```




    (array([2.31326169]), 2.3132616875182226)




```python
np.logaddexp2([1], [2]), np.log2(np.exp2(1) + np.exp2(2))
```




    (array([2.5849625]), 2.584962500721156)



`np.frexp`和`np.ldexp`是一对操作，后者等于`x1 * 2**x2`，而前者是将一个数组分解成mantissa和exponent，而依据就是`x = mantissa * 2**exponent`，就是对应前面的x1和x2了。


```python
np.ldexp(2, np.arange(5)), 2 * 2**np.arange(5)
```




    (array([ 2.,  4.,  8., 16., 32.], dtype=float16), array([ 2,  4,  8, 16, 32]))




```python
np.frexp(np.arange(2, 5))
```




    (array([0.5 , 0.75, 0.5 ]), array([2, 2, 3], dtype=int32))




```python
np.array([0.5, 0.75, 0.5]) * 2 ** np.array([2, 2, 3]), np.arange(2, 5)
```




    (array([2., 3., 4.]), array([2, 3, 4]))



### 算术操作

主要是中小学学过的加减乘除、乘方、开方、取余、倒数、绝对值，以及对应的一些特殊方法等，它们也都是通函数。按照惯例，我们主要介绍特殊的。

关于除法和Python的类似：


```python
# 地板除，等价于python的 //
np.floor_divide(5, 2)
```




    2




```python
np.floor_divide([7, 8], [3, 5])
```




    array([2, 1])



绝对值有相应的兼容复数的方法。


```python
np.abs(-1-1j)
```




    1.4142135623730951




```python
np.fabs(-1-1j)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-412-0b4518d62178> in <module>
    ----> 1 np.fabs(-1-1j)
    

    TypeError: ufunc 'fabs' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''


还有几个关于取余的方法：


```python
# 等价于Python的 x1%x2
np.remainder(np.arange(3), 2)
```




    array([0, 1, 0])




```python
# 和 np.reminder 一样
np.mod([12, 13], [4, 5])
```




    array([0, 3])




```python
# mod 结果的符号是x2的符号
np.mod([-3, -2, -1, 1, 2, 3], 2)
```




    array([1, 0, 1, 1, 0, 1])




```python
np.mod([2, 3], -2)
```




    array([ 0, -1])




```python
# 而fmod 结果的符号是x1的符号
np.fmod([-3, -2, -1, 1, 2, 3], 2)
```




    array([-1,  0, -1,  1,  0,  1])




```python
np.fmod([2, 3], -2)
```




    array([0, 1])



接下来这两个稍微不一样些。


```python
# 按元素返回数组的小数部分和整数部分。
np.modf([0, 3.5, 2.0])
```




    (array([0. , 0.5, 0. ]), array([0., 3., 2.]))




```python
np.modf(-1)
```




    (-0.0, -1.0)




```python
# 同时返回(x // y, x % y)
np.divmod([12, 13, 15], 2)
```




    (array([6, 6, 7]), array([0, 1, 1]))




```python
np.divmod([-3, -2, -1, 1, 2, 3], 2)
```




    (array([-2, -1, -1,  0,  1,  1]), array([1, 0, 1, 1, 0, 1]))



还有两个小学数学用过的：最大公倍数和最小公约数。


```python
# 最小公倍数
np.lcm(12, 20)
```




    60




```python
# 多个值可以用reduce
np.lcm.reduce([2, 3, 5, 8])
```




    120




```python
# 同时求多个
np.lcm([2, 3, 5, 8], 3)
```




    array([ 6,  3, 15, 24])




```python
# 最大公约数
np.gcd(12, 20)
```




    4




```python
np.gcd.reduce([15, 20, 30])
```




    5




```python
np.gcd([2, 3, 5, 8], 20)
```




    array([2, 1, 5, 4])



### 自动域

函数的输出数据类型与输入的某些域中的输入数据类型不同时，可以使用`np.emath`。

支持以下API：
- `sqrt`, `power`
- `log`, `log2`, `log10`, `logn`
- `arccos`, `arcsin`, `arctan`


```python
np.emath.sqrt(-1)
```




    1j




```python
np.sqrt(-1)
```

    <ipython-input-565-597592b72a04>:1: RuntimeWarning: invalid value encountered in sqrt
      np.sqrt(-1)
    




    nan




```python
import math
np.emath.log(-math.exp(1)) == 1+1j*math.pi
```




    True




```python
np.power([2, 4], -2)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-575-f02dcfa4fde4> in <module>
    ----> 1 np.power([2, 4], -2)
    

    ValueError: Integers to negative integer powers are not allowed.



```python
np.emath.power([2, 4], -2)
```




    array([0.25  , 0.0625])




```python
# 如果x包含负数则转为复数
np.emath.power([-2, 4], 1)
```




    array([-2.+0.j,  4.+0.j])



## 数值计算

### 舍入

然后是四舍五入，`round`和`around`一样。


```python
rng = np.random.default_rng(42)
arr = rng.random((2,3))
arr
```




    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235]])




```python
np.around(arr, 2)
```




    array([[0.77, 0.44, 0.86],
           [0.7 , 0.09, 0.98]])




```python
np.array(arr).round(2)
```




    array([[0.77, 0.44, 0.86],
           [0.7 , 0.09, 0.98]])



其他的接口都比较类似，除了`fix`的其他函数都是通函数。如下，不再赘述：


```python
lst = [2.1, -1.5, 3.2, 4.9]
```


```python
np.fix(lst)
```




    array([ 2., -1.,  3.,  4.])




```python
np.trunc(lst)
```




    array([ 2., -1.,  3.,  4.])




```python
np.rint(lst)
```




    array([ 2., -2.,  3.,  5.])




```python
np.floor(lst)
```




    array([ 2., -2.,  3.,  4.])




```python
np.ceil(lst)
```




    array([ 3., -1.,  4.,  5.])



### 和积差

这里包含了基础和积和累积和积，以及对应的有空值（`nan`）版本，API都比较简单，此处就不再赘述。主要介绍剩下的几个不太熟悉的。

首先是`diff`，它包含以下参数：

- 数组
- 计算次数，就是计算几次diff
- 维度
- prepend/append：沿着维度放在原数组前面/后面然后再计算


```python
rng = np.random.default_rng(42)
a = rng.integers(0, 10, (3, 4))
a
```




    array([[0, 7, 6, 4],
           [4, 8, 0, 6],
           [2, 0, 5, 9]])




```python
np.diff(a)
```




    array([[ 7, -1, -2],
           [ 4, -8,  6],
           [-2,  5,  4]])




```python
# 注意，连续算两次
np.diff(a, 2)
```




    array([[ -8,  -1],
           [-12,  14],
           [  7,  -1]])




```python
np.diff(a, 2, append=[[0], [0], [0]])
```




    array([[ -8,  -1,  -2],
           [-12,  14, -12],
           [  7,  -1, -13]])




```python
# 等价于
pd = np.full((3, 1), 0)
cct = np.concatenate((a, pd), axis=1)
np.diff(cct, 2)
```




    array([[ -8,  -1,  -2],
           [-12,  14, -12],
           [  7,  -1, -13]])




```python
# prepend 同理
np.diff(a, 2, prepend=[[0], [0], [0]])
```




    array([[  7,  -8,  -1],
           [  0, -12,  14],
           [ -4,   7,  -1]])



值得一提的是也可以对时间进行处理：


```python
dts = np.arange("2022-01-02", "2022-01-05", dtype=np.datetime64)
np.diff(dts, 1)
```




    array([1, 1], dtype='timedelta64[D]')



`np.ediff1d`会先flattern再diff，所以返回的是一维。

### 符号函数


```python
# 小于0为-1，等于0，为0，大于0为1
np.sign([-5, 0, 5])
```




    array([-1,  0,  1])




```python
# x1<0时为0，x1=0时为x2，x1>0时为1
np.heaviside([-5, 0, 5], 0.5)
```




    array([0. , 0.5, 1. ])



### 截断


```python
# 截断
np.clip(np.arange(10).reshape(2,5), a_min=3, a_max=7)
```




    array([[3, 3, 3, 3, 4],
           [5, 6, 7, 7, 7]])



### 插值

`np.interp`是一维线性插值方法，支持弧度。


```python
def func(x):
    return  2 * x + 3
```


```python
x = np.arange(1, 5)
y = func(x)
x, y
```




    (array([1, 2, 3, 4]), array([ 5,  7,  9, 11]))




```python
np.interp([2.5, 8], x, y), func(2.5), func(8)
```




    (array([ 8., 11.]), 8.0, 19)




```python
# 指定左右边界
np.interp([0.5, 2, 5.5], x, y)
```




    array([ 5.,  7., 11.])




```python
np.interp([0.5, 2, 5.5], x, y, left=1, right=60)
```




    array([ 1.,  7., 60.])



## 导数和微积分

### 梯度

梯度是使用内部点中的二阶精确中心差和边界处的一阶精确单侧（向前或向后）差异计算的。

主要是利用泰勒二阶展开计算导数：

$$
f(x) \approx f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right)\left(x-x_{0}\right)+\frac{f^{\prime \prime}\left(x_{0}\right)}{2 !}\left(x-x_{0}\right)^{2}+\frac{f^{\prime \prime \prime}\left(x_{0}\right)}{3 !}\left(x-x_{0}\right)^{3}+\cdots .
$$

等价于：

$$
f(x_0 + h) \approx f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right) (h)+\frac{f^{\prime \prime}\left(x_{0}\right)}{2 !} h^{2}+ O(h^3)
$$

又有：

$$
f(x_0 - h) \approx f\left(x_{0}\right)
-f^{\prime}\left(x_{0}\right) (h)+\frac{f^{\prime \prime}\left(x_{0}\right)}{2 !} h^{2}
 + O(h^3)
$$

两式相减，得到：

$$
f(x_0 + h) - f(x_0 - h) = 2 \cdot f^{\prime}(x_0)(h) 
$$

即：

$$
f^{\prime} (x_0) = \frac{f(x_0 + h) - f(x_0 - h)}{2h} + O(h^2)
$$

接下来就是利用上式来计算梯度（导数）。

参数如下：

- f，就是f(x)，数组
- varargs：f值的间隔，有多种可能的取值
- edge_order：在边界时采用顺序1或2计算
- axis：坐标轴


```python
# 表示f(0)=1, f(1)=2 ...
fx = np.array([1, 2, 4, 7, 11, 16])
```


```python
# 默认 h=1，第一个和最后一个（边界）有点特殊
# f'(0) = (f(0+1) - f(0))   / 1 = (2-1)/1   = 1
# f'(1) = (f(1+1) - f(1-1)) / 2 = (4-1)/2*1 = 1.5
# f'(2) = (f(2+1) - f(2-1)) / 2 = (7-2)/2*1 = 2.5
# ...
# f'(5) = (f(5) - f(5-1))   / 1 = (16-11)/1 = 5
np.gradient(fx)
```




    array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])




```python
# h 这个可以为其他数，比如0.5，此时表示 f(0)=1, f(0.5)=2 ...
# f'(0.0) = (f(0+0.5)   - f(0))       / 0.5   = (2-1)/0.5   = 2
# f'(0.5) = (f(0.5+0.5) - f(0.5-0.5)) / 2*0.5 = (4-1)/1.0   = 3
# f'(1.0) = (f(1.0+0.5) - f(1.0-0.5)) / 2*0.5 = (7-2)/1.0   = 5
# ...
# f'(2.5) = (f(2.5)     - f(2.5-0.5)) / 2*0.5 = (16-11)/0.5 = 10
np.gradient(fx, 0.5)
```




    array([ 2.,  3.,  5.,  7.,  9., 10.])



输入数组，会按照行列分别计算后返回，也可以指定坐标轴。


```python
rng = np.random.default_rng(42)
arr = rng.integers(1, 10, (2, 3))
arr
```




    array([[1, 7, 6],
           [4, 4, 8]])




```python
np.gradient(arr)
```




    [array([[ 3., -3.,  2.],
            [ 3., -3.,  2.]]),
     array([[ 6. ,  2.5, -1. ],
            [ 0. ,  2. ,  4. ]])]




```python
np.gradient(arr[:, 0]), np.gradient(arr[:, 1]), np.gradient(arr[:, 2])
```




    (array([3., 3.]), array([-3., -3.]), array([2., 2.]))




```python
np.gradient(arr[0,:]), np.gradient(arr[1,:])
```




    (array([ 6. ,  2.5, -1. ]), array([0., 2., 4.]))




```python
# 指定坐标轴
np.gradient(arr, axis=0)
```




    array([[ 3., -3.,  2.],
           [ 3., -3.,  2.]])



第二个参数`varargs`控制fx值之间的间隔，但它有多种方式：

- 1. 单个标量，用于指定所有尺寸的样本距离
- 2. N 个标量，用于为每个维度指定恒定的采样距离，即 “dx”， “dy”， “dz”， ...
- 3. N 个数组，用于指定值沿 F 的每个维度的坐标，数组的长度必须与相应尺寸的大小匹配
- 4. N 个标量/数组的任意组合，含义为 2 和 3

上面的例子是最简单的「单个标量」的情况，接下来看N个标量的情况。

$$
a = \frac{\frac{-dx_2}{dx_1}}{ dx_1+dx_2} \\
b = \frac{1}{dx_1} - \frac{1}{dx_2} \\
c = \frac{\frac{dx_1}{dx_2}}{ dx_1+dx_2} \\
a + b + c = 0
$$


```python
# N 个标量
x = np.array([0, 1, 1.5, 3.5, 4, 6])
fx = np.array([ 1,  2,  4,  7, 11, 16])
np.gradient(fx, x)
```




    array([1. , 3. , 3.5, 6.7, 6.9, 2.5])




```python
ax_dx = np.diff(x)

dx1 = ax_dx[0:-1]
dx2 = ax_dx[1:]
a = -dx2 / (dx1 * (dx1 + dx2))
b = (dx2 - dx1) / (dx1 * dx2)
c = dx1 / (dx2 * (dx1 + dx2))

N = fx.ndim

slice1 = [slice(None)]*N
slice2 = [slice(None)]*N
slice3 = [slice(None)]*N
slice4 = [slice(None)]*N
axis = 0
slice1[axis] = slice(1, -1)
slice2[axis] = slice(None, -2)
slice3[axis] = slice(1, -1)
slice4[axis] = slice(2, None)
```


```python
a, b, c
```




    (array([-0.33333333, -1.6       , -0.1       , -1.6       ]),
     array([-1. ,  1.5, -1.5,  1.5]),
     array([1.33333333, 0.1       , 1.6       , 0.1       ]))




```python
fx[tuple(slice2)], fx[tuple(slice3)], fx[tuple(slice4)]
```




    (array([1, 2, 4, 7]), array([ 2,  4,  7, 11]), array([ 4,  7, 11, 16]))




```python
# out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
out = np.empty_like(fx, dtype=np.float16)
out[tuple(slice1)] = a * fx[tuple(slice2)] + b * fx[tuple(slice3)] + c * fx[tuple(slice4)]
out
```




    array([0. , 3. , 3.5, 6.7, 6.9, 0. ], dtype=float16)



多个数组或数组与标量的组合，此时会分别对应到各自轴上计算：


```python
arr
```




    array([[1, 7, 6],
           [4, 4, 8]])




```python
np.gradient(arr, [3,5], [1,2,3])
```




    [array([[ 1.5, -1.5,  1. ],
            [ 1.5, -1.5,  1. ]]),
     array([[ 6. ,  2.5, -1. ],
            [ 0. ,  2. ,  4. ]])]




```python
np.gradient(arr, [3,5], axis=0)
```




    array([[ 1.5, -1.5,  1. ],
           [ 1.5, -1.5,  1. ]])




```python
np.gradient(arr, [1,2,3], axis=1)
```




    array([[ 6. ,  2.5, -1. ],
           [ 0. ,  2. ,  4. ]])



还有一个`edge_order`参数需要说明一下，它主要控制在边界位置如何计算梯度。


```python
x = np.array([1, 2, 4, 7])
y = x ** 2 + 2 * x + 1
y
```




    array([ 4,  9, 25, 64])




```python
np.gradient(y, x), 2*x + 2., np.gradient(y, x, edge_order=2)
```




    (array([ 5.,  6., 10., 13.]),
     array([ 4.,  6., 10., 16.]),
     array([ 4.,  6., 10., 16.]))




```python
(9 - 4)/(2-1), (64-25)/(7-4)
```




    (5.0, 13.0)



具体公式如下：

$$
f'({x_l}) = \frac{f(x_l + h) - f(x_l)}{h} \\
f'({x_r}) = \frac{f(x_r) - f(x_r - h)}{h} \\
$$

### 梯形公式

另一个API是梯形公式，可用来求积分，原理就是把被积函数切成很多个小的梯形。





$$
\int_{a}^{b} f(x) d x \approx \frac{\Delta x}{2}\left(f\left(x_{0}\right)+2 f\left(x_{1}\right)+2 f\left(x_{2}\right)+2 f\left(x_{3}\right)+2 f\left(x_{4}\right)+\cdots+2 f\left(x_{N-1}\right)+f\left(x_{N}\right)\right)
$$

参数如下：

- y，就是f(x)
- x，默认为None，如果指定则根据x的元素计算dx
- dx，默认1.0，如果没有x，则使用dx
- axis，坐标轴

更多参考：

- [Trapezoidal rule - Wikipedia](https://en.wikipedia.org/wiki/Trapezoidal_rule)


```python
y = np.array([1, 2, 3])
x = np.array([4, 6, 8])
```


```python
np.trapz(y), 1/2 * (1 + 2*2 + 3)
```




    (4.0, 4.0)




```python
diff = np.diff(x)
np.trapz(y, x), 1/2*((1+2)*diff[0] + (2+3)*diff[1])
```




    (8.0, 8.0)




```python
z = np.array([1, 2, 4])
diff = np.diff(z)
np.trapz(y, z), 1/2*((1+2)*diff[0] + (2+3)*diff[1])
```




    (6.5, 6.5)




```python
y = np.array([[1, 2, 3], [4, 5, 6]])
y
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
# 按行
np.trapz(y, axis=1)
```




    array([ 4., 10.])




```python
# 列
np.trapz(y, axis=0)
```




    array([2.5, 3.5, 4.5])



## 多项式

多项式的在新的版本有专门的library：`polynomial`。其中包括：

- 幂级数
- 切比雪夫多项式
- 埃尔米特多项式（物理学）
- 埃尔米特多项式（概率学）
- 拉盖尔多项式
- 勒让德多项式

具体概念可参考：

- [幂级数 - 维基百科，自由的百科全书](https://zh.m.wikipedia.org/zh-hans/%E5%B9%82%E7%BA%A7%E6%95%B0)
- [切比雪夫多项式 - 维基百科，自由的百科全书](https://zh.wikipedia.org/zh-hans/%E5%88%87%E6%AF%94%E9%9B%AA%E5%A4%AB%E5%A4%9A%E9%A1%B9%E5%BC%8F)
- [埃尔米特多项式 - 维基百科，自由的百科全书](https://zh.m.wikipedia.org/zh-hans/%E5%9F%83%E5%B0%94%E7%B1%B3%E7%89%B9%E5%A4%9A%E9%A1%B9%E5%BC%8F)
- [拉盖尔多项式 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E6%8B%89%E7%9B%96%E5%B0%94%E5%A4%9A%E9%A1%B9%E5%BC%8F)
- [勒让德多项式 - 维基百科，自由的百科全书](https://zh.m.wikipedia.org/zh/%E5%8B%92%E8%AE%A9%E5%BE%B7%E5%A4%9A%E9%A1%B9%E5%BC%8F)


```python
from numpy.polynomial import Polynomial as P
```

### 简介


```python
# 幂序列
p = np.polynomial.Polynomial([3, 2, 1])
p
```




$x \mapsto \text{3.0} + \text{2.0}\,x + \text{1.0}\,x^{2}$




```python
p(3)
```




    18.0




```python
rng = np.random.default_rng(42)
x = np.arange(10)
y = x + rng.standard_normal(10) 
```


```python
# 拟合
fitted = np.polynomial.Polynomial.fit(x, y, deg=1)
fitted
```




$x \mapsto \text{4.1644286915941295} + \text{4.216899419361024}\,\left(\text{-1.0} + \text{0.2222222222222222}x\right)$




```python
fitted.convert()
```




$x \mapsto \text{-0.05247072776689432} + \text{0.9370887598580052}\,x$




```python
# 根据根得到表达式
p = P.fromroots([1, 2])
p
```




$x \mapsto \text{2.0} - \text{3.0}\,x + \text{1.0}\,x^{2}$




```python
p.convert()
```




$x \mapsto \text{2.0} - \text{3.0}\,x + \text{1.0}\,x^{2}$




```python
p.convert(domain=[0,1])
```




$x \mapsto \text{0.75} - \text{1.0}\,\left(\text{-1.0} + \text{2.0}x\right) + \text{0.25}\,\left(\text{-1.0} + \text{2.0}x\right)^{2}$



类型之间转换也可以，不过不建议使用。当级数增加时会造成精度损失严重。


```python
from numpy.polynomial import Chebyshev as T
```


```python
T.cast(p)
```




$x \mapsto \text{2.5}\,{T}_{0}(x) - \text{3.0}\,{T}_{1}(x) + \text{0.5}\,{T}_{2}(x)$




```python
c1 = (1,2,3)
c2 = (3,2,1)
sum = P.polyadd(c1,c2)
```


```python
P.polyval(2, sum)
```




    28.0



### 便捷类

NumPy提供了不同类型多项式的便捷使用方式，提供了统一的创建、操作、拟合接口。以下的介绍以幂级数为例，更多可访问文档。

- [Power Series (numpy.polynomial.polynomial) — NumPy v1.23.dev0 Manual](https://numpy.org/devdocs/reference/routines.polynomials.polynomial.html)
- [Chebyshev Series (numpy.polynomial.chebyshev) — NumPy v1.23.dev0 Manual](https://numpy.org/devdocs/reference/routines.polynomials.chebyshev.html)
- [Hermite Series, “Physicists” (numpy.polynomial.hermite) — NumPy v1.23.dev0 Manual](https://numpy.org/devdocs/reference/routines.polynomials.hermite.html)
- [HermiteE Series, “Probabilists” (numpy.polynomial.hermite_e) — NumPy v1.23.dev0 Manual](https://numpy.org/devdocs/reference/routines.polynomials.hermite_e.html)
- [Laguerre Series (numpy.polynomial.laguerre) — NumPy v1.23.dev0 Manual](https://numpy.org/devdocs/reference/routines.polynomials.laguerre.html)
- [Legendre Series (numpy.polynomial.legendre) — NumPy v1.23.dev0 Manual](https://numpy.org/devdocs/reference/routines.polynomials.legendre.html)


```python
# 初始化一个实例
p = P([1, 2, 3])
p
```




$x \mapsto \text{1.0} + \text{2.0}\,x + \text{3.0}\,x^{2}$




```python
p.coef, p.domain, p.window
```




    (array([1., 2., 3.]), array([-1,  1]), array([-1,  1]))




```python
# x_new = -1+x
p1 = P([1, 2, 3], domain=[0, 1], window=[-1, 0])
p1
```




$x \mapsto \text{1.0} + \text{2.0}\,\left(\text{-1.0} + x\right) + \text{3.0}\,\left(\text{-1.0} + x\right)^{2}$




```python
# x_new = -1+x + x
p2 = P([1, 2, 3], domain=[0, 1], window=[-1, 1])
p2
```




$x \mapsto \text{1.0} + \text{2.0}\,\left(\text{-1.0} + \text{2.0}x\right) + \text{3.0}\,\left(\text{-1.0} + \text{2.0}x\right)^{2}$




```python
p3 = P([1, 2, 3], domain=[2, 5], window=[-1, 1])
p3
```




$x \mapsto \text{1.0} + \text{2.0}\,\left(\text{-2.3333333333333335} + \text{0.6666666666666666}x\right) + \text{3.0}\,\left(\text{-2.3333333333333335} + \text{0.6666666666666666}x\right)^{2}$




```python
from numpy.polynomial import polyutils as pu
```


```python
# 映射domain
pu.mapparms([2, 5], [-1, 1])
```




    (-2.3333333333333335, 0.6666666666666666)




```python
(5*-1 - 2*1)/(5-2), (1--1)/(5-2)
```




    (-2.3333333333333335, 0.6666666666666666)




```python
print(p)
```

    1.0 + 2.0·x¹ + 3.0·x²
    

可以选择不同的打印风格：


```python
np.polynomial.set_default_printstyle("ascii")
```


```python
print(p)
```

    1.0 + 2.0 x**1 + 3.0 x**2
    


```python
# 或
print(f"{p:unicode}")
```

    1.0 + 2.0·x¹ + 3.0·x²
    

多项式的基本运算：


```python
p + p
```




$x \mapsto \text{2.0} + \text{4.0}\,x + \text{6.0}\,x^{2}$




```python
p - p
```




$x \mapsto \color{LightGray}{\text{0.0}}$




```python
p * p
```




$x \mapsto \text{1.0} + \text{4.0}\,x + \text{10.0}\,x^{2} + \text{12.0}\,x^{3} + \text{9.0}\,x^{4}$




```python
p ** 2
```




$x \mapsto \text{1.0} + \text{4.0}\,x + \text{10.0}\,x^{2} + \text{12.0}\,x^{3} + \text{9.0}\,x^{4}$




```python
p // P([-1, 1])
```




$x \mapsto \text{5.0} + \text{3.0}\,x$




```python
p
```




$x \mapsto \text{1.0} + \text{2.0}\,x + \text{3.0}\,x^{2}$




```python
P([-1, 1]) * P([5, 3])
```




$x \mapsto \text{-5.0} + \text{2.0}\,x + \text{3.0}\,x^{2}$




```python
# 可以整除（因式分解）
P([2, 3, 1]) == P([1, 1]) * P([2, 1])
```




    True




```python
# 取余
p % P([-1, 1])
```




$x \mapsto \text{6.0}$




```python
# 分解+余
divmod(p, P([-1, 1]))
```




    (Polynomial([5., 3.], domain=[-1.,  1.], window=[-1.,  1.]),
     Polynomial([6.], domain=[-1.,  1.], window=[-1.,  1.]))




```python
# 求值
x = np.arange(5)
p(x)
```




    array([ 1.,  6., 17., 34., 57.])




```python
3*x**2 + 2*x + 1.
```




    array([ 1.,  6., 17., 34., 57.])




```python
# 嵌套
p(p)
```




$x \mapsto \text{6.0} + \text{16.0}\,x + \text{36.0}\,x^{2} + \text{36.0}\,x^{3} + \text{27.0}\,x^{4}$




```python
# 根
p.roots()
```




    array([-0.33333333-0.47140452j, -0.33333333+0.47140452j])




```python
# 有有理根
P([2, -3, 1]).roots()
```




    array([1., 2.])




```python
p
```




$x \mapsto \text{1.0} + \text{2.0}\,x + \text{3.0}\,x^{2}$




```python
p + [1, 2, 3]
```




$x \mapsto \text{2.0} + \text{4.0}\,x + \text{6.0}\,x^{2}$




```python
p + [1,2]
```




$x \mapsto \text{2.0} + \text{4.0}\,x + \text{3.0}\,x^{2}$




```python
p / 2
```




$x \mapsto \text{0.5} + \text{1.0}\,x + \text{1.5}\,x^{2}$



注意：上面的运算在不同domain、window或类型时无法使用。


```python
# 不同domain
p + P([1], domain=[0, 1])
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-1073-c9477ae81d90> in <module>
          1 # 不同domain
    ----> 2 p + P([1], domain=[0, 1])
    

    /usr/local/lib/python3.8/site-packages/numpy/polynomial/_polybase.py in __add__(self, other)
        498 
        499     def __add__(self, other):
    --> 500         othercoef = self._get_coefficients(other)
        501         try:
        502             coef = self._add(self.coef, othercoef)
    

    /usr/local/lib/python3.8/site-packages/numpy/polynomial/_polybase.py in _get_coefficients(self, other)
        282                 raise TypeError("Polynomial types differ")
        283             elif not np.all(self.domain == other.domain):
    --> 284                 raise TypeError("Domains differ")
        285             elif not np.all(self.window == other.window):
        286                 raise TypeError("Windows differ")
    

    TypeError: Domains differ


计算积分：


```python
p = P([3, 2, 1])
p
```




$x \mapsto \text{3.0} + \text{2.0}\,x + \text{1.0}\,x^{2}$




```python
# 定积分
p.integ()
```




$x \mapsto \color{LightGray}{\text{0.0}} + \text{3.0}\,x + \text{1.0}\,x^{2} + \text{0.3333333333333333}\,x^{3}$




```python
# 指定积分次数
p.integ(m=2)
```




$x \mapsto \color{LightGray}{\text{0.0}}\color{LightGray}{ + \text{0.0}\,x} + \text{1.5}\,x^{2} + \text{0.3333333333333333}\,x^{3} + \text{0.08333333333333333}\,x^{4}$




```python
# 指定下界（默认是0）为-1，常数项发生变化
p.integ(lbnd=-1)
```




$x \mapsto \text{2.333333333333333} + \text{3.0}\,x + \text{1.0}\,x^{2} + \text{0.3333333333333333}\,x^{3}$




```python
p.integ(k=[1], lbnd=-1)
```




$x \mapsto \text{3.333333333333333} + \text{3.0}\,x + \text{1.0}\,x^{2} + \text{0.3333333333333333}\,x^{3}$



计算导数：


```python
p.deriv()
```




$x \mapsto \text{2.0} + \text{2.0}\,x$




```python
p.deriv(2)
```




$x \mapsto \text{2.0}$




```python
p.deriv()(1)
```




    4.0




```python
p.deriv(2)(10)
```




    2.0



## 关系运算

NumPy中的关系运算一般用于判断数组是否满足指定条件，返回结果为布尔数组。这部分的API大多都是通函数。

### 真值测试

常用于判断元素是否全部满足或任一满足条件。


```python
a = np.array([
    [1, 0, 2],
    [2, 3, 0]
])
```


```python
np.all(a)
```




    False




```python
np.all(a, axis=0)
```




    array([ True, False, False])




```python
np.any(a)
```




    True




```python
np.any(a, axis=0)
```




    array([ True,  True,  True])




```python
np.any([0, 0, 0])
```




    False




```python
np.alltrue(a)
```




    False




```python
np.alltrue([1, 2, 3])
```




    True



### 值和类型

判断数组的值是否符合条件：

- `isfinite`：不是无穷并且不是非数字
- `isnan`：非数字
- `isnat`：非时间
- `isinf/isneginf/isposinf`：正/负无穷

类型：

- `iscomplex`：复数
- `iscomplexobj`：复数类型
- `isfortran`：F-Style
- `isreal`：实数
- `isrealobj`：不是复数类型
- `isscalar`：标量


```python
np.isfinite([np.nan, 0, np.inf, 1])
```




    array([False,  True, False,  True])




```python
np.isnan([np.nan, 2, np.inf])
```




    array([ True, False, False])




```python
np.isnat([np.datetime64("2016-01-01")])
```




    array([False])




```python
# 只支持时间格式
np.isnat([2])
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-48-b44857751f20> in <module>
          1 # 只支持时间格式
    ----> 2 np.isnat([2])
    

    TypeError: ufunc 'isnat' is only defined for datetime and timedelta.



```python
np.isinf([np.nan, np.inf])
```




    array([False,  True])




```python
np.isneginf([np.inf, -np.inf, np.NINF])
```




    array([False,  True,  True])




```python
a = np.array([[2, 3], [1+1j, 0]])
a
```




    array([[2.+0.j, 3.+0.j],
           [1.+1.j, 0.+0.j]])




```python
np.iscomplex(a)
```




    array([[False, False],
           [ True, False]])




```python
np.iscomplexobj(a)
```




    True




```python
np.isreal(a)
```




    array([[ True,  True],
           [False,  True]])




```python
np.isrealobj(a)
```




    False




```python
np.isfortran(a)
```




    False




```python
np.isscalar(np.array([2, 3]))
```




    False




```python
np.isscalar(2)
```




    True




```python
np.isscalar("fdf")
```




    True



### 逻辑运算

包括与、或、异或、非，以与为例。


```python
np.logical_and(True, False)
```




    False




```python
np.logical_and([2, 3], [4, False])
```




    array([ True, False])




```python
a = np.arange(5)
np.logical_and(a>1, a<4)
```




    array([False, False,  True,  True, False])




```python
# & 等价
np.array([1, 0]) & np.array([0, 1])
```




    array([0, 0])



### 比较

相近判断：

`allclose/isclose` 用于判断所有值是否在阈值范围内：

$$
|a - b| <= (\text{atol} + \text{rtol}  * |b|)
$$

- atol默认值为1e-08
- rtol默认值为1e-05

常用于验证精度。


```python
np.allclose(1.0000089, 1.000009)
```




    True




```python
np.allclose([1e10, 1e-9], [1.000001e10, 1e-8])
```




    True




```python
np.allclose([1, np.nan], [1, np.nan])
```




    False




```python
np.allclose([1, np.nan], [1, np.nan], equal_nan=True)
```




    True




```python
np.isclose(1.0000089, 1.000009)
```




    True




```python
np.isclose([1e10, 1e-9], [1.000001e10, 1e-8])
```




    array([ True,  True])




```python
np.isclose([1, np.nan], [1, np.nan])
```




    array([ True, False])




```python
np.isclose([1, np.nan], [1, np.nan], equal_nan=True)
```




    array([ True,  True])



相等判断：


```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 2], [3, 4]])
c = np.array([[1, np.nan], [3, np.nan]])
d = np.array([[1, np.nan], [3, np.nan]])
```


```python
np.array_equal(a, b)
```




    True




```python
np.array_equal(a, c)
```




    False




```python
np.array_equal(c, d)
```




    False




```python
np.array_equal(c, d, equal_nan=True)
```




    True




```python
a = np.array([1, 2])
b = np.array([[1, 2], [1, 2]])
c = np.array([[1, 2]])
```


```python
np.array_equal(a, b)
```




    False




```python
# shape一致值相等
np.array_equiv(a, b)
```




    True




```python
np.array_equiv(a, c)
```




    True




```python
np.array_equiv(b, c)
```




    True



注意，这里的shape一致是指一个可以广播到另一个上。

最后是比较运算，包括：>, >=, <, <=, ==, !=。


```python
a = np.array([[1, 2], [4, 2]])
b = np.array([[1, 3], [2, 2]])
```


```python
np.greater(a, b)
```




    array([[False, False],
           [ True, False]])




```python
np.greater_equal(a, b)
```




    array([[ True, False],
           [ True,  True]])




```python
np.less(a, b)
```




    array([[False,  True],
           [False, False]])




```python
np.less_equal(a, b)
```




    array([[ True,  True],
           [False,  True]])




```python
np.equal(a, b)
```




    array([[ True, False],
           [False,  True]])




```python
np.not_equal(a, b)
```




    array([[False,  True],
           [ True, False]])



## 二进制运算

主要是位运算相关API。

首先是逐位位运算，均为通函数，包括：与、或、异或、非、左移、右移等。

### 位运算


```python
int("000011", base=2), int("001100", base=2)
```




    (3, 12)




```python
# 逐位与
np.bitwise_and(3, 12), 3 & 12
```




    (0, 0)




```python
# 非
x = np.invert(np.array(13, dtype=np.uint8))
x, 2**7+2**6+2**5+2**4+2
```




    (242, 242)




```python
np.binary_repr(x, width=8), int("00001101", base=2)
```




    ('11110010', 13)



### 左右移

或、异或和 与 类似，不再赘述。接下来的按位移动，以左移为例。


```python
# 左移
```


```python
2 << 1
```




    4




```python
np.left_shift(2, 1)
```




    4




```python
np.left_shift(2, [1, 2, 3])
```




    array([ 4,  8, 16])




```python
np.left_shift([1, 2, 3], 2)
```




    array([ 4,  8, 12])



### 打包解包

最后介绍下打包和解包，分别将二进制转为 uint8 数组或反之。


```python
arr = np.array([[1, 1, 0], [1, 0, 1]])
```


```python
# 打包
np.packbits(arr)
```




    array([212], dtype=uint8)




```python
np.packbits(np.ravel(arr))
```




    array([212], dtype=uint8)




```python
# 110101
int("11010100", base=2)
```




    212




```python
# 2**7+2**6 2**7 2**6
np.packbits(arr, axis=0)
```




    array([[192, 128,  64]], dtype=uint8)




```python
# 解包，注意，dtype 须为 uint8
b = np.array([2], dtype=np.uint8)
```


```python
np.unpackbits(b)
```




    array([0, 0, 0, 0, 0, 0, 1, 0], dtype=uint8)



## 字符串

字符串在`NumPy`中也有很好的支持。所有的API在`np.char`下面，针对以下两种数据类型：


```python
np.str_, np.unicode_, np.str0
```




    (numpy.str_, numpy.str_, numpy.str_)




```python
np.bytes_, np.string_
```




    (numpy.bytes_, numpy.bytes_)



### 基本操作

首先是一些常用的字符串操作，与Python自带的类似。


```python
a = np.array(["1", "2"], dtype=np.str_)
b = np.array(["a", "b"], dtype=np.str_)
```

加法就是字符的拼接：


```python
# 加法
np.char.add(a, b)
```




    array(['1a', '2b'], dtype='<U2')




```python
np.array([1, 2], dtype=np.bytes_) + np.array([1, 2], dtype=np.bytes_)
```


    ---------------------------------------------------------------------------

    UFuncTypeError                            Traceback (most recent call last)

    <ipython-input-617-49d28b8ba097> in <module>
    ----> 1 np.array([1, 2], dtype=np.bytes_) + np.array([1, 2], dtype=np.bytes_)
    

    UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('S1'), dtype('S1')) -> None



```python
np.char.add(np.array([1, 2], dtype=np.bytes_), np.array([1, 2], dtype=np.bytes_))
```




    array([b'11', b'22'], dtype='|S2')




```python
np.char.add(np.array([1, 2], dtype=np.bytes_), np.array([1, 2], dtype=np.str_))
```




    array(['11', '22'], dtype='<U2')



乘法是字符的重复：


```python
np.char.multiply(np.array([1],dtype=np.str_), 3)
```




    array(['111'], dtype='<U3')




```python
# 次数小于0时为0
np.char.multiply(np.array([1],dtype=np.str_), -3)
```




    array([''], dtype='<U1')



其他API也与str自带的类似：
- `capitalize`：首字母大写
- `title`：按标题大写
- `center`：给定长度居中填充
- `ljust/rjust`：给定长度左右填充
- `zfill`：0左填充
- `decode/encode`：解编码
- `expandtabs`：tab会被替换成一个或多个空格
- `join`：拼接
- `lower/upper`：大小写
- `swapcase`：大小写互换
- `lstrip/rstrip/strip`：strip
- `replace`：替换
- `translate`：转换
- `partition/rpatition`：分为三元组（左/右）
- `split/splitlines`：切分


```python
# 首字母大写
np.char.capitalize("ab b c")
```




    array('Ab b c', dtype='<U6')




```python
# 标题大写
np.char.title("ab b c")
```




    array('Ab B C', dtype='<U6')




```python
# 给定长度居中
(np.char.center("ab", 5, "~"), 
 np.char.ljust("a", 5, "~"), 
 np.char.rjust("a", 5, "~"),
 np.char.zfill("a", 6)
)
```




    (array('~~ab~', dtype='<U5'),
     array('a~~~~', dtype='<U5'),
     array('~~~~a', dtype='<U5'),
     array('00000a', dtype='<U6'))




```python
# 编解码
np.char.encode("abc", encoding="utf8")
```




    array(b'abc', dtype='|S3')




```python
# 替换tab
val = np.char.expandtabs("\ta", tabsize=1)
val
```




    array(' a', dtype='<U2')




```python
val.tolist(), val.tolist()[0] == " "
```




    (' a', True)




```python
# 拼接
np.char.join("a", "12345")
```




    array('1a2a3a4a5', dtype='<U9')




```python
# 大小写
np.char.lower(np.array(["A"],dtype=np.str_)), np.char.upper("a")
```




    (array(['a'], dtype='<U1'), array('A', dtype='<U1'))




```python
# 互换
np.char.swapcase("aBc")
```




    array('AbC', dtype='<U3')




```python
# strip
np.char.strip("abc "), np.char.strip("abc", "c")
```




    (array('abc', dtype='<U4'), array('ab', dtype='<U3'))




```python
# replace
np.char.replace("aaabc", "a", "A", count=2)
```




    array('AAabc', dtype='<U5')




```python
# 转换
np.char.translate(["abc", "a"],  "1"*255, deletechars=None)
```




    array(['111', '1'], dtype='<U3')




```python
# 非unicode时才会删除
np.char.translate(
    np.array(["abc", "a"], dtype=np.bytes_), b"1"*256, deletechars=b"a")
```




    array([b'11', b''], dtype='|S3')




```python
# partition
(
    np.char.partition("abc", "b"), 
    np.char.rpartition("abc", "b"),
    np.char.partition("abca", "a"),
    np.char.rpartition("abca", "a")
)
```




    (array(['a', 'b', 'c'], dtype='<U1'),
     array(['a', 'b', 'c'], dtype='<U1'),
     array(['', 'a', 'bca'], dtype='<U3'),
     array(['abc', 'a', ''], dtype='<U3'))




```python
# split
np.char.split("a b c", " "), np.char.splitlines("a\nb\nc")
```




    (array(list(['a', 'b', 'c']), dtype=object),
     array(list(['a', 'b', 'c']), dtype=object))



### 比较

主要比较字符串大小相等。


```python
np.char.equal(["abc", "ab"], ["abd", "ab"])
```




    array([False,  True])




```python
np.char.not_equal(["abc", "ab"], ["abd", "ab"])
```




    array([ True, False])




```python
# >=
np.char.greater_equal(["abc", "ab"], ["abd", "ab"]), "abc">"abd"
```




    (array([False,  True]), False)




```python
np.char.greater(["abc", "ab"], ["abd", "ab"])
```




    array([False, False])




```python
# <=
np.char.less_equal(["abc", "ab"], ["abd", "ab"])
```




    array([ True,  True])




```python
np.char.less(["abc", "ab"], ["abd", "ab"])
```




    array([ True, False])




```python
# 比较
# cmp可以取 < <= == >= > !=
np.char.compare_chararrays(
    ["abc", "ab", "a"],
    ["ab", "ad", "ae"],
    cmp="<",
    rstrip=True
)
```




    array([False,  True,  True])



### 基本信息

包括基本的判断和统计。


```python
np.char.count("abcab", "a", start=0, end=None)
```




    array(2)




```python
np.char.str_len(["abcab", "a"])
```




    array([5, 1])




```python
(
    np.char.find("abcab", "a", start=0, end=None),
    np.char.rfind("abcab", "a", start=0, end=None)
)
```




    (array(0), array(3))




```python
# 找不到返回-1
np.char.find("abcab", "d", start=2, end=None)
```




    array(-1)




```python
(
    np.char.index("abcab", "a", start=0),
    np.char.rindex("abcab", "a", start=0)
)
```




    (array(0), array(3))




```python
# 找不到抛出异常
np.char.index("abcab", "d", start=2)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-908-6904110ea61f> in <module>
          1 # 找不到抛出异常
    ----> 2 np.char.index("abcab", "d", start=2)
    

    /usr/local/lib/python3.8/site-packages/numpy/core/overrides.py in index(*args, **kwargs)
    

    /usr/local/lib/python3.8/site-packages/numpy/core/defchararray.py in index(a, sub, start, end)
        742 
        743     """
    --> 744     return _vec_string(
        745         a, int_, 'index', [sub, start] + _clean_args(end))
        746 
    

    ValueError: substring not found



```python
# starts/ends
(
    np.char.endswith(["a", "ba"], "a", start=0, end=1),
    np.char.startswith(["a", "ba"], "a", start=1, end=3)
)
```




    (array([ True, False]), array([False,  True]))




```python
# 只有空格
np.char.isspace(["   \t\n", "a"])
```




    array([ True, False])




```python
# 所有字符小/大写，首字母大写
(
    np.char.islower(["a", "Ab"]), 
    np.char.isupper(["a", "Ab", "AB"]),
    np.char.istitle(["Aa", "aB", "AB"])
)
```




    (array([ True, False]),
     array([False, False,  True]),
     array([ True, False, False]))




```python
# 判断
lst = ["a", "1", "01", "０３", "⒊⒏", "a1", "1.1", ""]
(
    # 每个元素的所有字符都为字母，至少一个字符
    np.char.isalpha(lst),
    # 同上，字母或数字
    np.char.isalnum(lst),
    "",
    # 只有decimal（小数点不算）
    np.char.isdecimal(lst),
    # 只有digit
    np.char.isdigit(lst),
    # 只有numeric
    np.char.isnumeric(lst)
)
```




    (array([ True, False, False, False, False, False, False, False]),
     array([ True,  True,  True,  True,  True,  True, False, False]),
     '',
     array([False,  True,  True,  True, False, False, False, False]),
     array([False,  True,  True,  True,  True, False, False, False]),
     array([False,  True,  True,  True,  True, False, False, False]))



关于decimal、digit和numeric的区别可参考：

- [string - What's the difference between str.isdigit, isnumeric and isdecimal in python? - Stack Overflow](https://stackoverflow.com/questions/44891070/whats-the-difference-between-str-isdigit-isnumeric-and-isdecimal-in-python)

它们的主要区别在处理unicode的方式上。

## 小结

## 参考

- [NumPy documentation — NumPy v1.23.dev0 Manual](https://numpy.org/devdocs/index.html)
- [python - Memory growth with broadcast operations in NumPy - Stack Overflow](https://stackoverflow.com/questions/31536504/memory-growth-with-broadcast-operations-in-numpy)


```python

```
