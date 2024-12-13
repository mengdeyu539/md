<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#常量" data-toc-modified-id="常量-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>常量</a></span><ul class="toc-item"><li><span><a href="#特殊值" data-toc-modified-id="特殊值-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>特殊值</a></span></li><li><span><a href="#空值" data-toc-modified-id="空值-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>空值</a></span></li><li><span><a href="#无穷" data-toc-modified-id="无穷-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>无穷</a></span></li></ul></li><li><span><a href="#数据类型" data-toc-modified-id="数据类型-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>数据类型</a></span><ul class="toc-item"><li><span><a href="#类型" data-toc-modified-id="类型-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>类型</a></span></li><li><span><a href="#大小" data-toc-modified-id="大小-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>大小</a></span></li><li><span><a href="#顺序" data-toc-modified-id="顺序-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>顺序</a></span></li></ul></li><li><span><a href="#结构化数组" data-toc-modified-id="结构化数组-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>结构化数组</a></span></li><li><span><a href="#时间数组" data-toc-modified-id="时间数组-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>时间数组</a></span></li><li><span><a href="#数组对象" data-toc-modified-id="数组对象-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>数组对象</a></span><ul class="toc-item"><li><span><a href="#ndarray" data-toc-modified-id="ndarray-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>ndarray</a></span></li><li><span><a href="#array" data-toc-modified-id="array-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>array</a></span></li></ul></li><li><span><a href="#自定义数组容器" data-toc-modified-id="自定义数组容器-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>自定义数组容器</a></span></li><li><span><a href="#子类化与标准子类" data-toc-modified-id="子类化与标准子类-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>子类化与标准子类</a></span></li><li><span><a href="#小结" data-toc-modified-id="小结-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>小结</a></span></li><li><span><a href="#参考" data-toc-modified-id="参考-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>参考</a></span></li></ul></div>


```python
# 安装watermark
!pip install watermark
```


```python
%load_ext watermark
```


```python
%watermark
```

    Last updated: 2022-11-05T09:11:26.064679+08:00
    
    Python implementation: CPython
    Python version       : 3.8.13
    IPython version      : 7.23.1
    
    Compiler    : Clang 13.1.6 (clang-1316.0.21.2)
    OS          : Darwin
    Release     : 21.1.0
    Machine     : x86_64
    Processor   : i386
    CPU cores   : 4
    Architecture: 64bit
    
    


```python
import numpy as np
```


```python
%watermark --iversions
```

    numpy: 1.23.0
    
    

文档阅读说明：

- 🐧 表示 Tip
- ⚠️ 表示注意事项

## 常量

NumPy 中自带一部分常用的常量，方便直接使用。

### 特殊值


```python
# 自然对数
np.e
```




    2.718281828459045




```python
# PI
np.pi
```




    3.141592653589793




```python
# 0
np.PZERO
```




    0.0




```python
# -0
np.NZERO
```




    -0.0




```python
# None
np.newaxis
```

### 空值


```python
# 空值
np.nan
```




    nan




```python
type(np.nan)
```




    float



⚠️ 注意，`np.nan` 是一个值，两个 `np.nan` 不相等，虽然它们同属于一个类型。


```python
np.nan is np.nan
```




    True




```python
np.nan == np.nan
```




    False



可以使用 `np.isnan` 方法进行判断。


```python
np.isnan(1), np.isnan(2.0), np.isnan(np.nan), np.isnan(np.log(-10.))
```

    <ipython-input-34-1060db748605>:1: RuntimeWarning: invalid value encountered in log
      np.isnan(1), np.isnan(2.0), np.isnan(np.nan), np.isnan(np.log(-10.))
    




    (False, False, True, True)




```python
# 以下等价
np.nan is np.NAN is np.NaN
```




    True



### 无穷


```python
# 正无穷
np.inf
```




    inf




```python
# 负无穷
np.NINF == -np.inf
```




    True




```python
type(np.inf)
```




    float




```python
np.log(0)
```

    <ipython-input-93-f6e7c0610b57>:1: RuntimeWarning: divide by zero encountered in log
      np.log(0)
    




    -inf




```python
-np.inf < -100
```




    True




```python
np.inf < 10
```




    False



可以使用 `np.isxx` 进行判断。


```python
# 是否正或负去穷
np.isinf(-np.inf)
```




    True




```python
# 哪些元素正无穷
np.isposinf(-np.inf)
```




    False




```python
# 哪些元素负无穷
np.isneginf(np.inf)
```




    False




```python
# 哪些元素有限的（不是非数字、正无穷或负无穷）
np.isfinite(3)
```




    True




```python
np.isfinite(np.inf)
```




    False




```python
# 以下几个方法等价
np.inf == np.Inf == np.Infinity == np.infty == np.PINF 
```




    True



## 数据类型


numpy 支持丰富的数据类型，[官方文档](https://numpy.org/devdocs/user/basics.types.html)中介绍的非常全面。这里我们不要陷入太多纠结，尝试从整体的角度重新梳理一遍。其实我们更需要关注的应该是其内置的数据类型对象 `dtype`，也就是这个文档：[Data type objects](https://numpy.org/devdocs/reference/arrays.dtypes.html#arrays-dtypes)。


```python
# 数据类型 和 数据类型对象
type(np.int8), type(np.dtype(np.int8))
```




    (type, numpy.dtype[int8])



数据类型对象描述了如何解释与数组项对应的固定大小的内存块中的字节。主要包括以下几个方面（当然有很多其他信息）：

- 数据类型
- 数据大小
- 数据的顺序
- 如果是「结构化数据类型」则是其他数据类型的集合
- 如果数据类型是子数组，它的形状和数据类型


之前咱们创建 array 的时候都没有关心过数据类型，这种情况下，numpy 会自动匹配当前输入最合适的数据类型，并将其 cast 到所有元素。


总的来说可以大致分成以下几种，而我们绝大多数情况下最应该关注的其实就是 int 和 float 这两种：

- bool：`bool8`, `bool_`，不是 int
- int：`int8/byte`, `int16/short`, `int32`, `int64/longlong`, `int_`
- uint：无符号类型，表示 `unsigned`，对应 int
- float：`float16/half`, `float32/single`, `float64/double`, `float_`
- complex：复数，`complex64`, `complex128`, `complex_`
- str：`str0`, `str_`，表示 unicode 编码
- bytes: `bytes_`, `string_`
- datetime/timedelta
- structed array

后面的数字表示一个数字在内存中占几位，**一般比较推荐使用这种表示**；带下划线的表示 python 的数据类型，numpy 可以自动将 python 的类型转为它；此外，浮点数还支持不同精度以及扩展精度。

### 类型

首先看这个图：

![](https://numpy.org/devdocs/_images/dtype-hierarchy.png)

来自：[Scalars — NumPy v1.23.dev0 Manual](https://numpy.org/devdocs/reference/arrays.scalars.html)

基本涵盖了上面除 datetime 和 structed array 之外的所有类型，这两种类型我们后面单独来说。


```python
# 直观验证上图的关系
(
    isinstance(np.str_(), np.flexible),
    isinstance(np.bytes_(), np.flexible),
    isinstance(np.void(b""), np.flexible),
    
    isinstance(np.int_(), np.integer),
    isinstance(np.float_(), np.floating),
    isinstance(np.complex_(), np.complexfloating)
)
```




    (True, True, True, True, True, True)




```python
# 很多类型都有 alias，它们其实是一回事
(
    np.int_ is np.int64, np.intc is np.int32, np.short is np.int16, np.byte is np.int8,
    # 不同精度
    np.half is np.float16, np.single is np.float32, np.double is np.float64,
    # 扩展精度
    np.longfloat is np.longdouble,
    # 字符串
    np.unicode_ is np.str_,
    # bytes
    np.bytes_ is np.string_
)
```




    (True, True, True, True, True, True, True, True, True, True)




```python
# python 内置类型
(
    np.bool_ is np.bool8, 
    np.int_ is np.int64,
    np.float_ is np.float64,
    np.str_ is np.str0,
    np.complex_ is np.complex128
)
```




    (True, True, True, True, True)



接下来以整型为例来说明，其他的类似。


```python
# 创建一个「数据类型对象」
# 如果使用 python 的类型，会自动识别支持，不过建议使用 numpy 的 dtype 类型指定类型
i32 = np.dtype("int32")
i32
```




    dtype('int32')




```python
# numpy 支持的 python 类型
np.int_, np.float_, np.bool_, np.complex_, np.str_
```




    (numpy.int64, numpy.float64, numpy.bool_, numpy.complex128, numpy.str_)




```python
# 比较推荐这样创建
i32 = np.dtype(np.int32)
i32
```




    dtype('int32')



🐧 ：建议在创建 array 时指定数据类型，且使用统一的数据类型计算。


```python
%timeit np.arange(100, dtype=np.float32).reshape(10, 10) * np.arange(100, dtype=np.int32).reshape(10, 10)
```

    4.64 µs ± 151 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    


```python
%timeit np.arange(100, dtype=np.int32).reshape(10, 10) * np.arange(100, dtype=np.int32).reshape(10, 10)
```

    2.41 µs ± 70.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    

### 大小

类型暂时先看到这儿，先来看看大小：


```python
# int 默认 64 位
arr = np.array([2**63-1])
arr.dtype
```




    dtype('int64')




```python
# 每个数字 8 bytes = 64 bits
arr.nbytes
```




    8




```python
bytes(arr)
```




    b'\xff\xff\xff\xff\xff\xff\xff\x7f'




```python
list(map(hex, arr))
```




    ['0x7fffffffffffffff']




```python
15 * (sum([16**v for v in range(15)])) + 7*16**15 == 2 ** 63 - 1
```




    True




```python
# 超出64位表示的范围，自动转为uint64
arr = np.array([2**63])
arr.dtype
```




    dtype('uint64')




```python
# 一共是 8 个字节（Byte），64 位（bit）
arr.nbytes
```




    8




```python
bytes(arr)
```




    b'\x00\x00\x00\x00\x00\x00\x00\x80'




```python
list(map(hex, arr))
```




    ['0x8000000000000000']




```python
8 * 16**15  == 2 ** 63
```




    True




```python
# 可以使用 iinfo 查看
np.iinfo(np.int64)
```




    iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)




```python
np.iinfo(np.uint64)
```




    iinfo(min=0, max=18446744073709551615, dtype=uint64)




```python
# finfo 查看 float
np.finfo(np.float64)
```




    finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)



🐧 ：再次建议在创建 array 时指定数据类型，不光是为了性能，还能节约内存，而且强迫自己思考每一个数组的范围，做到心中有数。


```python
# 溢出
np.array(128, dtype=np.int8)
```




    array(-128, dtype=int8)




```python
# 如果不指定会默认int64，有时候没必要，浪费内存
np.array(128).dtype
```




    dtype('int64')




```python
# 看看不同类型的占用空间
(
    np.array(1, dtype=np.int8).nbytes,
    np.array(1, dtype=np.int16).nbytes,
    np.array(1, dtype=np.int32).nbytes,
    np.array(1, dtype=np.int64).nbytes,
)
```




    (1, 2, 4, 8)



### 顺序

这个看起来就不那么直观了。咱们先了解下背景知识。

字节顺序（Endianness）在计算机科学中指内存中字节的排列顺序。字节的排列有两个通用规则：

- 将低位放在较小的地址处，高位放在较大的地址处，称为小端序（little-endian）。
- 与上面相反的就是大端序（big-endian）。


![](https://qnimg.lovevivian.cn/cs-endian-1.jpg)

图片来自：[Endianness - Wikipedia](https://en.wikipedia.org/wiki/Endianness)


我们常用的 x86 计算机是小端位，因为内存地址一般是从低到高逐渐增加的，而我们的二进制（或者其他进制）是高位在前，低位在后，这样用小端序就会很自然，也便于编程。不过刚好人读起来正好是反着的。

拿上面的图片为例，0A0B0C0D 是自然顺序，0D 是低位，在小端序中就被放在了低地址；0A 是高位，在大端序中被放在低位。


在 `numpy` 中，dtype 的每个类型都可以用一个字符表示，而使用字符表示时，可以增加字节序。

支持的字符表示如下（大小写分别表示无符号和有符号）：


Format | C Type | Python Type | Standard Size 
-------|--------|-------------|--------------
`?`    | `_Bool`| `bool`      | 1
`b/B`  | `char` | `int`       | 1
`h/H`  | `short`| `int`       | 2
`i/I`  | `int`  | `int`       | 4
`l/L`  | `long` | `int`       | 4
`q/Q`  | `long long` | `int`  | 8
`e`    | `half`   | `float`   | 2
`f`    | `float` | `float`    | 4
`d`    | `double` | `float`   | 8


另外还有几个复杂类型：

- `c`：复数浮点
- `m/M`: timedelta / datetime
- `O`: Python 对象
- `U`: Unicode 字符串
- `V`: void
- `S/a`: 零终止字节（不推荐）

而字节序共有以下几种：

Character | Byte order | Size 
-----------|-----------|-------
`=`       | native     | standard
`<`       | little-endian     | standard
`>`       | big-endian     | standard

默认是 `=`。

上面部分参考自：https://docs.python.org/3/library/struct.html


```python
# 不同顺序
np.dtype('<i'), np.dtype('>i'), np.dtype("=i")
```




    (dtype('int32'), dtype('>i4'), dtype('int32'))




```python
# 小端序 int 类型
np.dtype("<i") == np.dtype(np.int32)
```




    True




```python
# 默认为 =
np.dtype(np.int32).byteorder, np.dtype("<i").byteorder, np.dtype(">i").byteorder
```




    ('=', '=', '>')




```python
# 也可以显式指定长度
# U 可以是任意长度
np.dtype('<i4'), np.dtype('=f8'), np.dtype('<U3')
```




    (dtype('int32'), dtype('float64'), dtype('<U3'))




```python
# 但是你不能随意指定不存在的
np.dtype("<i3")
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-66-5d7ad68fd17a> in <module>
          1 # 但是你不能随意指定不存在的
    ----> 2 np.dtype("<i3")
    

    TypeError: data type '<i3' not understood



```python
np.array([259], dtype="<i2"), np.array([259], dtype=">i2")
```




    (array([259], dtype=int16), array([259], dtype=int16))




```python
# 01 03，高位在前低位在后
3*16**0 + 0*16**1 + 1*16**2 + 0*16**3
```




    259




```python
# 存储时大小端二者顺序相反
# 小端位，从左到右地址由低到高；大端位，从左到右地址由高到低
bytes(np.array([259], dtype="<i2")), bytes(np.array([259], dtype=">i2"))
```




    (b'\x03\x01', b'\x01\x03')



字节顺序常用于**不同设备**（字节序不同）之间数据交互，也可以互相转换。

## 结构化数组

结构化数组就是数据类型是**一组**（而不是只有一个）不同的类型的数组，常常用于计算时需要将多个类型的数据放在一起的场景。


```python
arr = np.array(
    [('Rex', 9, 81.0), ('Fido', 3, 27.0)],
    dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]
)
arr
```




    array([('Rex', 9, 81.), ('Fido', 3, 27.)],
          dtype=[('name', '<U10'), ('age', '<i4'), ('weight', '<f4')])




```python
arr.shape
```




    (2,)



如果我们去掉 `dtype`，就和之前介绍的一样了（转成统一的类型）：


```python
np.array(
    [('Rex', 9, 81.0), ('Fido', 3, 27.0)])
```




    array([['Rex', '9', '81.0'],
           ['Fido', '3', '27.0']], dtype='<U32')



因为每个元素都是「一组结构化的数据」，所以也叫结构化数组。

⚠️ 注意：`dtype` 的每一个 tuple 对应元素中的一个元素。比如上面的例子中，第一个元素是 U10 类型，表示的是每一个 tuple 的第一个元素是 U10 类型。


```python
arr[0]
```




    ('Rex', 9, 81.)



另外，我们也可以创建多维的结构化数组，但这个多维和正常的多维不一样，它会把每个元素重复多遍；也就是说，每个 array 的类型其实还是一致的，然后变成了多个 array。


```python
marr = np.array([
    [1, 2, 5], 
    [4, 5, 7], 
    [7, 8 ,11], 
    [10, 11, 12]
], dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'U3'), ('u', 'i8'), ('v', 'f8')])
marr
```




    array([[( 1,  1., '1',  1,  1.), ( 2,  2., '2',  2,  2.),
            ( 5,  5., '5',  5,  5.)],
           [( 4,  4., '4',  4,  4.), ( 5,  5., '5',  5,  5.),
            ( 7,  7., '7',  7,  7.)],
           [( 7,  7., '7',  7,  7.), ( 8,  8., '8',  8,  8.),
            (11, 11., '11', 11, 11.)],
           [(10, 10., '10', 10, 10.), (11, 11., '11', 11, 11.),
            (12, 12., '12', 12, 12.)]],
          dtype=[('x', '<i4'), ('y', '<f4'), ('z', '<U3'), ('u', '<i8'), ('v', '<f8')])




```python
marr.shape == (4, 3)
```




    True




```python
marr['x']
```




    array([[ 1,  2,  5],
           [ 4,  5,  7],
           [ 7,  8, 11],
           [10, 11, 12]], dtype=int32)




```python
marr['u']
```




    array([[ 1,  2,  5],
           [ 4,  5,  7],
           [ 7,  8, 11],
           [10, 11, 12]])




```python
# 用 tuple 不行哦
np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
             dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8'), ('u', 'i8')])
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-125-1d00de367151> in <module>
          1 # 用 tuple 不行哦
    ----> 2 np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
          3              dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8'), ('u', 'i8')])
    

    ValueError: could not assign tuple of length 3 to structure with 4 fields.



```python
np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
             dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
```




    array([( 1,  2.,  5.), ( 4,  5.,  7.), ( 7,  8., 11.), (10, 11., 12.)],
          dtype=[('x', '<i4'), ('y', '<f4'), ('z', '<f8')])




```python
_["x"]
```




    array([ 1,  4,  7, 10], dtype=int32)



`zeros` 和 `ones` 和正常接口一样，可以快速创建多维（结构化）数组。


```python
# zeros 会有不同反应，注意第一个元素是空
np.zeros((2, 3), dtype=[('a', 'S1'), ('b', 'i4')])
```




    array([[(b'', 0), (b'', 0), (b'', 0)],
           [(b'', 0), (b'', 0), (b'', 0)]], dtype=[('a', 'S1'), ('b', '<i4')])




```python
np.ones((2, 3), dtype=[('a', 'S1'), ('b', 'i4')])
```




    array([[(b'1', 1), (b'1', 1), (b'1', 1)],
           [(b'1', 1), (b'1', 1), (b'1', 1)]],
          dtype=[('a', 'S1'), ('b', '<i4')])



`rec` 接口可以使 array 能够通过属性名访问。


```python
arr = np.rec.array(
    [(1, 2., 'Hello'), (2, 3., "World")], 
    dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')]
)
```


```python
arr.foo
```




    array([1, 2])



普通 array 也可以转换成结构化数组。


```python
arr = np.array([[1, 2], [3, 4]], dtype=("i, i"))
arr
```




    array([[(1, 1), (2, 2)],
           [(3, 3), (4, 4)]], dtype=[('f0', '<i4'), ('f1', '<i4')])




```python
# 两种转换方式
a = np.rec.array(arr)
b = arr.view(np.recarray)
a, b
```




    (rec.array([[(1, 1), (2, 2)],
                [(3, 3), (4, 4)]],
               dtype=[('f0', '<i4'), ('f1', '<i4')]),
     rec.array([[(1, 1), (2, 2)],
                [(3, 3), (4, 4)]],
               dtype=[('f0', '<i4'), ('f1', '<i4')]))




```python
a.shape, b.shape
```




    ((2, 2), (2, 2))




```python
a.f0, b.f1
```




    (array([[1, 2],
            [3, 4]], dtype=int32),
     array([[1, 2],
            [3, 4]], dtype=int32))




```python
a.f0.shape, b.f1.shape
```




    ((2, 2), (2, 2))



另外，结构化数组也可以在类型中加入前缀作为 shape。


```python
parr = np.ones((2, ), dtype=('3i4, (2,3)f4, (2, 2)S2'))
parr
```




    array([([1, 1, 1], [[1., 1., 1.], [1., 1., 1.]], [[b'1', b'1'], [b'1', b'1']]),
           ([1, 1, 1], [[1., 1., 1.], [1., 1., 1.]], [[b'1', b'1'], [b'1', b'1']])],
          dtype=[('f0', '<i4', (3,)), ('f1', '<f4', (2, 3)), ('f2', 'S2', (2, 2))])




```python
# 同时选择多个 field
parr[['f0', 'f2']]
```




    array([([1, 1, 1], [[b'1', b'1'], [b'1', b'1']]),
           ([1, 1, 1], [[b'1', b'1'], [b'1', b'1']])],
          dtype={'names': ['f0', 'f2'], 'formats': [('<i4', (3,)), ('S2', (2, 2))], 'offsets': [0, 36], 'itemsize': 44})




```python
parr['f0'].shape, parr['f1'].shape
```




    ((2, 3), (2, 2, 3))



结构化数组还有一些快捷的操纵方式，具体可查看[文档](https://numpy.org/devdocs/user/basics.rec.html#module-numpy.lib.recfunctions)，此处不再深入介绍。

## 时间数组

`datetime` 是专门处理时间的 API，在处理时间序列时非常有用。为了和 Python 中的 `datetime` 区分，Numpy 中是 `datetime64`，格式是 [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601)。

即，从1970年1月1日0时0分0秒起，常见的单位包括：年（Y）、月（M）、周（W）、日（D）、时（h）、分（m）、秒（s）、微秒（ms），及 NAT（Not a Time）。


```python
t1 = np.datetime64("2022-02-28")
t2 = np.datetime64("2023")
t3 = np.datetime64("12")
```


```python
# 分别到天和年，但是它可没聪明到自己识别月
t1.dtype, t2.dtype, t3.dtype
```




    (dtype('<M8[D]'), dtype('<M8[Y]'), dtype('<M8[Y]'))




```python
# 它会直接给你转成四位数的「年」
t3
```




    numpy.datetime64('0012')



也可以指定具体的单位：


```python
# 从1970年起第一年
np.datetime64(1, "Y")
```




    numpy.datetime64('1971')




```python
np.datetime64('2005-02', 'D')
```




    numpy.datetime64('2005-02-01')




```python
np.datetime64('2005-02', 's')
```




    numpy.datetime64('2005-02-01T00:00:00')




```python
# NAT
np.datetime64("nat")
```




    numpy.datetime64('NaT')



支持Unix时间戳（⚠️ 注意这里的数据类型要指定一个单位）：


```python
np.array([0, 1577836800000], dtype="datetime64[ms]")
```




    array(['1970-01-01T00:00:00.000', '2020-01-01T00:00:00.000'],
          dtype='datetime64[ms]')



还支持`arange`操作：


```python
np.arange("2020-01", "2021-01-02T00:00:00", dtype="datetime64[M]")
```




    array(['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06',
           '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12'],
          dtype='datetime64[M]')



当然也可以转回字符串：


```python
t1
```




    numpy.datetime64('2022-02-28')




```python
np.datetime_as_string(t1)
```




    '2022-02-28'



`timedelta64`是和`timedelta`类似的东西：


```python
# 一天
np.timedelta64(1, "D")
```




    numpy.timedelta64(1,'D')




```python
# 日期相减
np.datetime64("2021-02-28") - np.datetime64("2021-01-31")
```




    numpy.timedelta64(28,'D')




```python
# 最小单位到小时，结果也是小时
np.datetime64("2021-02-28T00") - np.datetime64("2021-01-31")
```




    numpy.timedelta64(672,'h')




```python
np.datetime64("2021-02") - np.datetime64("2020-01")
```




    numpy.timedelta64(13,'M')



当然，时间也可以增加或减少`timedelta`：


```python
np.datetime64("2021-03-01") + np.timedelta64(1, "D")
```




    numpy.datetime64('2021-03-02')




```python
np.datetime64("2021-03-01T00:00:00") + np.timedelta64(1, "h")
```




    numpy.datetime64('2021-03-01T01:00:00')



除了这些基本功能外，关于「工作日」，还有几个好用的API：


```python
# 比如 2022-03-29 是周二，+4天后本来是 2 号，但它会跳过周末，直接到 4 号（下周一）
np.busday_offset("2022-03-29", 4)
```




    numpy.datetime64('2022-04-04')




```python
np.busday_offset("2022-03-29", [3, 4])
```




    array(['2022-04-01', '2022-04-04'], dtype='datetime64[D]')



`busday_offset` 的几个重要参数参数如下：

- NumPy的日期
- offset
- roll：如何处理非有效日期，默认 raise，还可以选择 nat（Not a Time），forward/following（下一个最近的有效日期），backward/preceding（上一个最近的有效日期），modifiedfollowing（下一个最近的有效日期但不跨月），modifiedpreceding（上一个最近的有效日期但不跨月）等
- weekmask：指定每周哪些天是有效的「工作日」
- holidays：假期，NumPy日期格式的无效日期（即非工作日）


```python
# 默认的 roll=raise，也就是抛出异常，4月2日是周六
np.busday_offset("2022-04-02", 0)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-156-b4c681b86e8b> in <module>
          1 # 默认的 roll=raise，也就是抛出异常，4月2日是周六
    ----> 2 np.busday_offset("2022-04-02", 0)
    

    /usr/local/lib/python3.8/site-packages/numpy/core/overrides.py in busday_offset(*args, **kwargs)
    

    ValueError: Non-business day date in busday_offset



```python
# 下一个
np.busday_offset("2022-04-02", 0, "forward")
```




    numpy.datetime64('2022-04-04')




```python
# 上一个
np.busday_offset("2022-04-03", 0, "preceding")
```




    numpy.datetime64('2022-04-01')



4月30日是周六，5月1日是周日：


```python
np.busday_offset("2022-04-30", 0, "preceding")
```




    numpy.datetime64('2022-04-29')




```python
np.busday_offset("2022-05-01", 0, "preceding")
```




    numpy.datetime64('2022-04-29')




```python
np.busday_offset("2022-04-30", 0, "forward")
```




    numpy.datetime64('2022-05-02')




```python
np.busday_offset("2022-05-01", 0, "forward")
```




    numpy.datetime64('2022-05-02')



5月1日跨月，就是5月的第一个有效日（5月2日）：


```python
np.busday_offset("2022-05-01", 0, "modifiedpreceding")
```




    numpy.datetime64('2022-05-02')




```python
np.busday_offset("2022-04-30", 0, "modifiedpreceding")
```




    numpy.datetime64('2022-04-29')



4月30日接下来跨月，选4月最后一个有效日：


```python
np.busday_offset("2022-04-30", 0, "modifiedfollowing")
```




    numpy.datetime64('2022-04-29')




```python
np.busday_offset("2022-05-01", 0, "modifiedfollowing")
```




    numpy.datetime64('2022-05-02')



看下weekmask：


```python
# 必须7个，后3天休息！
weekmask = [1, 1, 1, 1, 0, 0, 0]
```


```python
# 5月5日是周四，加一天就跨过了3天
np.busday_offset("2022-05-05", 1, roll='forward', weekmask=weekmask)
```




    numpy.datetime64('2022-05-09')




```python
# or "1111000"
np.busday_offset("2022-05-05", 1, roll='forward', weekmask="1111000")
```




    numpy.datetime64('2022-05-09')



另外还可以统计两个日期之间的有效日数量，或判断某天是否有效日：


```python
np.is_busday("2022-05-01")
```




    False




```python
# 1号周日，2-4号，3天
np.busday_count("2022-05-01", "2022-05-05")
```




    3




```python
np.busday_count(np.datetime64("2022-05-01"), "2022-05-05")
```




    3



## 数组对象

### ndarray

NumPy 提供了一个 N 维数组类型，即 `ndarray`，描述了**相同类型**「元素」集合。它是偏底层的 array 接口。

所有的 `ndarray` 元素都是同质的，每个元素占用大小相同的内存块，具体大小由「数据类型」决定。`ndarray` 可以共享相同数据。

对象签名如下：

```python
numpy.ndarray(shape, dtype=float, buffer=None, offset=0, strides=None, order=None)
```

- shape：整数元组，表示形状
- dtype：数据类型对象
- buffer：使用 buffer 中的数据填充 `ndarray`
- offset：buffer 中的偏移量
- stride：内存中数据跨度
- order：行为主（C-Style）或列为主（Fortran-Style）


当 buffer 为空时，shape, dtype 和 order 三个参数会被使用；  
当 buffer 不为空时，所有参数都会被使用。


```python
# buffer 为空，结果随机
arr = np.ndarray(shape=(2, 3), dtype=np.int32, order="C")
arr
```




    array([[ 2069159998,  1074974876,  -768170609],
           [ 1074129069, -1684540248,  1073865591]], dtype=int32)




```python
# buffer 不为空
buf = np.array([1, 2, 3, 4], dtype=np.int_)
arr = np.ndarray(
    shape=(2, 2), 
    dtype=np.int_, 
    offset=0,
    buffer=buf,
    strides=(np.int_().itemsize, np.int_().itemsize),
)
arr
```




    array([[1, 2],
           [2, 3]])




```python
# buffer 的 shape 并无关系
buf = np.array([[[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]]], dtype=np.int_, order="C")
arr = np.ndarray(
    shape=(2, 2), 
    dtype=np.int_, 
    offset=0,
    buffer=buf,
    strides=(np.int_().itemsize, np.int_().itemsize),
)
arr
```




    array([[1, 2],
           [2, 3]])




```python
# buf 的 order 有关，原因我们后面解释
buf = np.array([[[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]]], dtype=np.int_, order="F")
arr = np.ndarray(
    shape=(2, 2), 
    dtype=np.int_, 
    offset=0,
    buffer=buf,
    strides=(np.int_().itemsize, np.int_().itemsize),
)
arr
```




    array([[1, 3],
           [3, 5]])



其实，buffer 本质上是 `bytes`，因为这个接口处于相对底层，在这里数据是连续存储的（或者说会尽可能地让其在内存中连续）。我们看个例子：


```python
buf = b"\x01\x02\x03\x04"
```

上面的 `buf` 是一组 16 进制的数（每个数字 4 位，所以一共 32 位），并不不一定是 1、2、3、4 噢。它们的值具体是多少，得看我们如何指定 `dtype`。如果我们指定 `dtype` 为 `int8`，那 `buf` 里就有 4 个数字，如果指定为 `int16`，那 `buf` 里就只有 2 个数字。


```python
np.ndarray(shape=(4, ), dtype=np.int8, buffer=buf)
```




    array([1, 2, 3, 4], dtype=int8)




```python
np.ndarray(shape=(2, ), dtype=np.int16, buffer=buf)
```




    array([ 513, 1027], dtype=int16)



想一下，这两个值是怎么算出来的？

>提示：可以写成：0x 0201  0x 0403

由于我们平时很少用到这个接口，您可能会对其中的一些参数有些困惑。接下来我们稍微解释一下。buffer 为空时没有太多要强调的，重点说一下 buffer 不为空时。

`shape` 和 `dtype` 也比较清晰，`buffer` 刚刚也说明了，主要是还剩下的三个参数：`offset`, `strides` 和 `order`。

`order` 是指采用哪种风格进行存储。计算中，行主序（C Style）和列主序（F Style）是将多维数组存储在线性存储器（例如 RAM）中的方法。在行主序中，一行的连续元素彼此相邻，而在列主序中，一列连续元素彼此相邻。具体可参考：[Row- and column-major order - Wikipedia](https://en.wikipedia.org/wiki/Row-_and_column-major_order)。

⚠️ **需要注意的是**：不同的存储方式会导致计算效率的不同，可以针对具体的场景（处理行多还是列多）选择不同的 Style。


```python
# copy 让 carr 拥有数据，否则只是 view
carr = np.arange(1000000).reshape(1000, 1000).copy()
carr.flags
```




      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False




```python
farr = np.asfortranarray(carr)
farr.flags
```




      C_CONTIGUOUS : False
      F_CONTIGUOUS : True
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False




```python
# 加第一行所有列
# C style 应该比 F style 快一些
%timeit np.sum(carr[0,:])
```

    5.01 µs ± 131 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    


```python
%timeit np.sum(farr[0,:])
```

    7.7 µs ± 16.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    


```python
# 加第一列所有行
# F style 应该比 C style 快一些
%timeit np.sum(farr[:, 0])
```

    4.93 µs ± 34.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    


```python
%timeit np.sum(carr[:, 0])
```

    7.73 µs ± 219 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    

`offset` 和 `strides` 是配合使用的，前者是偏移位置，后者是步幅，它必须与 shape 等长。也就是根据给定的 buffer，生成目标 shape 的 array。至于这么做的原因，主要是和内部存储有关，事实上，`ndarray` 就是通过这两个参数来控制 shape，不同的 shape 其实存储是一样的。


```python
buf = np.arange(1, 9)
buf
```




    array([1, 2, 3, 4, 5, 6, 7, 8])




```python
# 因为咱们是 int64，所以 1 个数字是 64 位，即 8 个 Bytes
buf.strides
```




    (8,)




```python
# 这个例子 偏移了「1」个数字，步幅也正好是「1」
# 结果是 从 2 开始
# strides 两个数字分别控制 行和列 的步幅：从左往右看，每次增加 1 个数字，从上往下看，每次增加 1 个数字
arr1 = np.ndarray(
    shape=(2, 3), 
    dtype=np.int64, 
    offset=8,
    buffer=buf,
    strides=(8, 8),
    order="C")
arr1
```




    array([[2, 3, 4],
           [3, 4, 5]])




```python
# 再来个例子
# 没有偏移，ok，从 1 开始
# 从左到右是列，每次加 1 个数字，从上到下是行，每次增加 2 个数字
# 这里的 dtype 没有关系
arr2 = np.ndarray(
    shape=(3, 2), 
    dtype=np.int8, 
    offset=0,
    buffer=buf,
    strides=(16, 8),
    order="C")
arr2
```




    array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int8)




```python
arr3 = np.ndarray(
    shape=(3, 2), 
    dtype=np.int8, 
    offset=8,
    buffer=buf,
    strides=(16, 8),
    order="C")
arr3
```




    array([[2, 3],
           [4, 5],
           [6, 7]], dtype=int8)



⚠️ 需要注意的是：`offset` 和 `strides` 的数字并不是真实的数字大小，而是占的位数。

我们进一步看一下不同的 shape 的存储情况。


```python
buf = np.arange(1, 9, dtype=np.int8)
buf
```




    array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int8)




```python
# int8 的，每次正好 8 位，即 1 个 Byte
buf.strides
```




    (1,)




```python
bytes(buf)
```




    b'\x01\x02\x03\x04\x05\x06\x07\x08'




```python
# 改一下 shape
buf.shape = 2, 4
```


```python
buf.strides
```




    (4, 1)




```python
# 发现规律了吗？
buf
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8]], dtype=int8)




```python
# 此时再看 内存布局
# 和之前是一样的，也就是说，用 strides 我们就可以给同一个 array 不同的 shape
bytes(buf)
```




    b'\x01\x02\x03\x04\x05\x06\x07\x08'



事实上，无论 shape 怎么变化，内存是完全没变化的，不同的 array 其实就是不同的 strides 方式而已，不同的 strides 表现出不同的 shape。感兴趣的可以进一步尝试。

另外，使用 buffer 创建的 `ndarray` 都使用了同一块内存。


```python
buf = np.arange(1, 9, dtype=np.int8)
```


```python
arr1 = np.ndarray(
    shape=(2, 3), 
    dtype=np.int8, 
    offset=0,
    buffer=buf,
    strides=(1, 1),
    order="C")
arr1
```




    array([[1, 2, 3],
           [2, 3, 4]], dtype=int8)




```python
arr2 = np.ndarray(
    shape=(3, 2), 
    dtype=np.int8, 
    offset=0,
    buffer=buf,
    strides=(1, 1),
    order="C")
arr2
```




    array([[1, 2],
           [2, 3],
           [3, 4]], dtype=int8)




```python
bytes(arr1), bytes(arr2)
```




    (b'\x01\x02\x03\x02\x03\x04', b'\x01\x02\x02\x03\x03\x04')




```python
np.may_share_memory(arr1, arr2), np.may_share_memory(buf, arr1)
```




    (True, True)



其实，无论 arr1 还是 arr2 都是 buf 的一个 view（引用），与此相对的是 copy。

`reshape` 在大多数时候会改变 strides 获取 view，但在数组不连续时（比如转置后）就不能这样操作了，因为转置改变了排列方式（其实就是 C-style 与 F-style 互转）。


```python
arr = np.ones((2, 3))
arr.shape = (3, 2)
```


```python
arr.flags
```




      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False




```python
arr.T.flags
```




      C_CONTIGUOUS : False
      F_CONTIGUOUS : True
      OWNDATA : False
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False




```python
arr_t = arr.T
arr_t.shape = 6
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-236-a5df655be35d> in <module>
          1 arr_t = arr.T
    ----> 2 arr_t.shape = 6
    

    AttributeError: Incompatible shape for in-place modification. Use `.reshape()` to make a copy with the desired shape.


顺带补充，**一般来说**，切片（slicing）会创建 view，索引（indexing）会创建 copy。关于切片和索引我们会在后面进一步介绍。


```python
a = np.arange(6).reshape(2, 3)
a
```




    array([[0, 1, 2],
           [3, 4, 5]])




```python
# view
b = a[:1]
b
```




    array([[0, 1, 2]])




```python
a[:1] = 100
a
```




    array([[100, 100, 100],
           [  3,   4,   5]])




```python
# view 的效果
b
```




    array([[100, 100, 100]])




```python
b.base
```




    array([100, 100, 100,   3,   4,   5])




```python
x = np.arange(9).reshape(3, 3)
x
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
# copy
y = x[[1]]
y
```




    array([[3, 4, 5]])




```python
x[[1]] = 100
x
```




    array([[  0,   1,   2],
           [100, 100, 100],
           [  6,   7,   8]])




```python
# copy 的效果
y
```




    array([[3, 4, 5]])




```python
y.base is None
```




    True



也就是说，使用 buffer 创建 `ndarray` 其实可以理解成一种「切片」。实际上，如果您查看 `np.array` 的接口，就会发现其中有个 `copy` 参数，它默认是 `True`。


```python
buf, arr1, arr2
```




    (array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int8),
     array([[1, 2, 3],
            [2, 3, 4]], dtype=int8),
     array([[1, 2],
            [2, 3],
            [3, 4]], dtype=int8))




```python
buf[0] = 9
```


```python
buf, arr1, arr2
```




    (array([9, 2, 3, 4, 5, 6, 7, 8], dtype=int8),
     array([[9, 2, 3],
            [2, 3, 4]], dtype=int8),
     array([[9, 2],
            [2, 3],
            [3, 4]], dtype=int8))



`strides`　不同时，处理的效率也有差异。当步长增加时，找到对应位置的值会变慢。其原因是，CPU 在处理任务时会将数据从内存读取到缓存，步长小时，需要的传输更少。比如要取 10 个数，连在一起的（步长=1个数字）可以一次取到，但步长大时却要取多次。

>注：CPU 缓存是用于减少处理器访问内存所需平均时间的部件。在金字塔式存储体系中它位于自顶向下的第二层，仅次于 CPU 寄存器。其容量远小于内存，但速度却可以接近处理器的频率。一般会有多级缓存。——维基百科


```python
arr1 = np.ones((1000, 100), dtype=np.int8)
arr2 = np.ones((10000, 100), dtype=np.int8)[::10]
arr1.shape, arr2.shape
```




    ((1000, 100), (1000, 100))




```python
arr1.flags
```




      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False




```python
# OWNDATA=False，意思是这个 array 的数据是从其他地方「借」来的
# 从哪个地方呢？当然就是 `np.ones((10000, 100), dtype=np.int8)` 这里了
arr2.flags
```




      C_CONTIGUOUS : False
      F_CONTIGUOUS : False
      OWNDATA : False
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False




```python
np.any(arr1 == arr2)
```




    True




```python
arr1.strides, arr2.strides
```




    ((100, 1), (1000, 1))




```python
%timeit arr1.sum()
```

    74.8 µs ± 1.55 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    


```python
%timeit arr2.sum()
```

    85.2 µs ± 1.4 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    

最后说明下，我们上面的例子都是用 `int` 类型来说明，其他数据类型类似。

另外，源码中的大致逻辑是：`arrayobject.c` 中的 `array_new` 方法调用了 ctors.c 中的 `PyArray_NewFromDescr_int` 来实现创建一个 `ndarray`。

除了上面提到的可以影响性能，其实不同的调用接口也是有差异的。比如转置操作，一共有三种方法：

- `arr.T`
- `arr.transpose`
- `np.transpose`

其实它们几乎是一样的，只是调用方式不同，性能也表现出不同的差异（很自然嘛）。


```python
rng = np.random.default_rng(42)
```


```python
arr = rng.integers(0, 10, (2, 3))
arr
```




    array([[0, 7, 6],
           [4, 4, 8]])




```python
%timeit arr.T
```

    123 ns ± 4.81 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
    


```python
%timeit arr.transpose()
```

    165 ns ± 14.3 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
    


```python
%timeit np.transpose(arr)
```

    988 ns ± 57.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    

稍微解释一下，`arr.T` 和 `arr.transpose` 是差不多的，后者稍微慢的原因是还有一个 `axes` 参数；而 `np.transpose` 慢则是因为它的调用方式：

```python
def transpose(a, axes=None):
    return _wrapfunc(a, 'transpose', axes)

def _wrapfunc(obj, method, *args, **kwds):
    try:
        return getattr(obj, method)(*args, **kwds)
    except (AttributeError, TypeError):
        return _wrapit(obj, method, *args, **kwds)
```

所以如果可以的话，推荐尽量在 array 上调用方法。

### array

首先要明确，`array` 只是快速创建 `ndarray` 的接口函数，源代码是 `core/src/multiarray/methods.c` 中的 `array_getarray`。这玩意儿其实就调用了上面提到的 `PyArray_NewFromDescr_int`。

`np.array`的参数如下：

- object
- dtype：数据类型，默认None
- copy：是否复制，默认为True
- order：默认K，可取CFKA，C和F之前提到过，K和A，当没有copy时就是原来的顺序；copy=True时：
  - K时，会保留F和C两种，否则会保留最相似的
  - A时，如果输入是F且不是C，则是F；否则是C
- subok，默认False，如果为真时返回传入子类类型
- ndmin：默认0，指定最小维度
- like：默认None，允许创建不属于NumPy的数组

前两个参数就不再赘述了。首先看copy参数：


```python
a1 = np.array([[2, 3], [4, 5]])
```


```python
a2 = np.array(a1, copy=False)
```


```python
a1[0][0] = 0
```


```python
a1, a2
```




    (array([[0, 3],
            [4, 5]]),
     array([[0, 3],
            [4, 5]]))




```python
a3 = np.array(a1, copy=True)
```


```python
a1[0][0] = -1
```


```python
a1, a3
```




    (array([[-1,  3],
            [ 4,  5]]),
     array([[0, 3],
            [4, 5]]))



然后是order参数：


```python
# 没有 copy 时，就是原来的顺序
a1 = np.array([[2, 3], [4, 5]], order="F")
a2 = np.array(a1, order="K", copy=False)
a2.flags
```




      C_CONTIGUOUS : False
      F_CONTIGUOUS : True
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False




```python
# copy=True，order=K，保留最相似的（F），向量是CF
a1 = np.array([[2, 3], [4, 5]], order="F")
a2 = np.array(a1, order="K", copy=True)
a2.flags
```




      C_CONTIGUOUS : False
      F_CONTIGUOUS : True
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False




```python
# copy=True，order=A，如果输入是 F 则是 F；否则（AKC时）是 C
a1 = np.array([[2, 3], [4, 5]], order="A")
a2 = np.array(a1, order="A")
a2.flags
```




      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False



读者可以自行验证其他情况。

再看subok参数：


```python
np.array(np.mat('1 2; 3 4'))
```




    array([[1, 2],
           [3, 4]])




```python
# 返回原来的类型
np.array(np.mat('1 2; 3 4'), subok=True)
```




    matrix([[1, 2],
            [3, 4]])




```python
np.array(np.char.array([2,3]), subok=True)
```




    chararray([b'2', b'3'], dtype='|S1')




```python
# 子类
class A(np.ndarray): pass
```


```python
np.array(A(2), subok=True)
```




    A([9.9e-324, 1.5e-323])



然后是ndmin参数：


```python
# 会自动扩充一个维度
np.array([[2, 3], [4, 5]], ndmin=3).shape
```




    (1, 2, 2)




```python
# 但如果小于本来的维度，则不发生变化
np.array([[2, 3], [4, 5]], ndmin=1).shape
```




    (2, 2)



最后的like参数演示如下：


```python
import dask.array as da
```

    /usr/local/lib/python3.8/site-packages/scipy/__init__.py:138: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.0)
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion} is required for this version of "
    


```python
da.array([2, 3])
```




<table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 16 B </td>
                        <td> 16 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (2,) </td>
                        <td> (2,) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 1 Tasks </td>
                        <td> 1 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> int64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="110" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="60" style="stroke-width:2" />
  <line x1="120" y1="0" x2="120" y2="60" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,60.0 0.0,60.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="80.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >2</text>
  <text x="140.000000" y="30.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,30.000000)">1</text>
</svg>
        </td>
    </tr>
</table>




```python
np.array([2,3], like=da.array([2,3]))
```




<table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 16 B </td>
                        <td> 16 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (2,) </td>
                        <td> (2,) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 1 Tasks </td>
                        <td> 1 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> int64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="110" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="60" style="stroke-width:2" />
  <line x1="120" y1="0" x2="120" y2="60" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,60.0 0.0,60.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="80.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >2</text>
  <text x="140.000000" y="30.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,30.000000)">1</text>
</svg>
        </td>
    </tr>
</table>



## 自定义数组容器


```python
class MyArray:
    
    def __init__(self, lst: list):
        self.list = lst
    
    
    def __array__(self):
        return np.array(self.list)

    
class HisArray:
    
    def __init__(self, lst: list):
        self.list = lst
    
    
    def __array(self):
        return np.array(self.list)
```


```python
a = MyArray([[2, 3], [4, 5]])
```

可以使用`np.asarray`或`np.array`将其转为`array`（会调用`__array__`方法）：


```python
np.asarray(a)
```




    array([[2, 3],
           [4, 5]])




```python
b = HisArray([[2, 3], [4, 5]])
```


```python
np.asarray(b)
```




    array(<__main__.HisArray object at 0x11c007d30>, dtype=object)



或者使用NumPy的API进行操作时，也会调用`__array__`：


```python
np.add(a, 2)
```




    array([[4, 5],
           [6, 7]])




```python
np.add(b, 2)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-290-9e59ee7cfe52> in <module>
    ----> 1 np.add(b, 2)
    

    TypeError: unsupported operand type(s) for +: 'HisArray' and 'int'


可以通过自定义`__array_function__`或`__array_ufunc__`来自**「定义行为」**。

假设需要定义一个自定义类加法，可以通过使用`__add__`或继承`numpy.lib.mixins.NDArrayOperatorsMixin`来实现。


```python
class MyArray:
    
    def __init__(self, lst: list):
        self.list = lst
    
    # 自定义 add 方法
    def __add__(self, v):
        return np.array(self.list) + v
```


```python
a = MyArray([[2, 3], [4, 5]])
```


```python
a + 3
```




    array([[5, 6],
           [7, 8]])




```python
class MyArray(np.lib.mixins.NDArrayOperatorsMixin):
    
    def __init__(self, lst: list):
        self.list = lst
    
    # 继承后用 __array_ufunc__
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return ufunc(np.array(self.list), inputs[1])
```


```python
a = MyArray([[2, 3], [4, 5]])
```


```python
# 这种操作要继承
a * 3
```




    array([[ 6,  9],
           [12, 15]])




```python
# 这个不需要继承 NDArrayOperatorMixin
# 有 __array_ufunc__ 就可以了
np.add(a, 3)
```




    array([[5, 6],
           [7, 8]])




```python
class DiagonalArray:
    
    def __init__(self, N, value):
        self.N = N
        self.v = value
    
    def __array_function__(self, func, types, args, kwargs):
        if func == np.sum: return self.N * self.v
        elif func == np.mean: return self.N / self.v
```


```python
a = DiagonalArray(5, 2)
```


```python
# 调用了 np.mean 执行的是我们自己的逻辑
np.mean(a)
```




    2.5



更多内容可参考：

- https://numpy.org/doc/stable/user/basics.dispatch.html
- https://numpy.org/doc/stable/reference/arrays.classes.html 第一部分

## 子类化与标准子类

ndarray的新实例可以以三种不同的方式出现：

- 显式构造函数调用
- 视图转换
- 模板创建：最明显的地方是对子类数组进行切片。

后两种是ndarray的特性，子类化ndarray的复杂性是由于NumPy必须支持后两种实例创建路径的机制。

子类化适用以下场合：
- 不担心可维护性或自己以外的用户。
- 子类信息忽略或丢失不是什么问题。


```python
# view casting：获取任何子类的 ndarray，并将数组的view作为另一个（指定的）子类返回
class C(np.ndarray): 
    pass
arr = np.zeros((3,))
c_arr = arr.view(C)
type(c_arr)
```




    __main__.C




```python
# 通过切片从模板实例创建新实例
v = c_arr[1:]
type(v)
```




    __main__.C




```python
# 是一个新的实例
v is c_arr 
```




    False



ndarray 用于支持子类中的视图和新模板的机制有两个方面。

- 使用 `ndarray.__new__` 方法进行对象初始化的主要工作，而不是更常见的 `__init__` 方法。
- 使用 `__array_finalize__` 方法允许子类在从模板创建视图和新实例之后进行清理。

首先看下初始化的 `__new__` 方法，这样做的原因是在某些情况下，对于ndarray，我们希望能够返回某个其他类的对象。


```python
class D(C):
    def __new__(cls, *args):
        print('D cls is:', cls)
        print('D args in __new__:', args)
        return C.__new__(C, *args)

    def __init__(self, *args):
        # 当 __new__ 方法返回一个类的对象而不是定义它的类时，该类的 __init__ 方法不会被调用
        print('In D __init__')
```


```python
d = D((1, ))
```

    D cls is: <class '__main__.D'>
    D args in __new__: ((1,),)
    


```python
arr = np.zeros((3, ))
```


```python
d_arr = arr.view(D)
type(d_arr)
```




    __main__.D




```python
d_arr
```




    D([0., 0., 0.])




```python
v = d_arr[1:]
type(v)
```




    __main__.D



这就是ndarray类的子类如何能够返回保留类类型的view（view casting），当执行view时，标准的ndarray机制会这样创建新的ndarray对象：
`obj = ndarray.__new__(subtype, shape, ...)`，`subbtype` 就是子类，所以返回的是子类的类，而不是ndarray的类。


<!-- 不过这又有了新问题，我们还没有这样的`__new__`方法。 -->

接下里是 `__array_finalize__`，它允许子类处理创建的新实例的各种方法，签名是 `__array_finalize__(self, obj)`。因为我们不能依赖 `MySubClass.__new__` 或 `MySubClass.__init__` 来处理视图转换和模板创建。


```python
class E(C):
    def __new__(cls, *args, **kwargs):
        print('In __new__ with class %s' % cls)
        return C.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        # 实际在子类中可能不需要
        print('In __init__ with class %s' % self.__class__)

    def __array_finalize__(self, obj):
        print('In array_finalize:')
        print('   self type is %s' % type(self), "self is ", id(self))
        print('   obj type is %s' % type(obj), "obj is ", id(obj))
```


```python
e = E((3, ))
```

    In __new__ with class <class '__main__.E'>
    In array_finalize:
       self type is <class '__main__.E'> self is  4773873856
       obj type is <class 'NoneType'> obj is  4530430688
    In __init__ with class <class '__main__.E'>
    


```python
arr = np.ones((3, ))  # == obj
e_arr = arr.view(E)
type(e_arr)
```

    In array_finalize:
       self type is <class '__main__.E'> self is  4773875136
       obj type is <class 'numpy.ndarray'> obj is  4765117296
    




    __main__.E




```python
v = e[:1]
type(v)
```

    In array_finalize:
       self type is <class '__main__.E'> self is  4773872832
       obj type is <class '__main__.E'> obj is  4773873856
    




    __main__.E




```python
v is e_arr
```




    False



通过上面的例子可知：

- 从显式构造函数调用时，obj 是 None
- 从视图转换中调用时，obj 可以是 ndarray 的任何子类的实例，包括我们自己的子类
- 在从模板创建中调用时，obj 是我们自己的子类的另一个实例，我们可能会用它来更新新的 self 实例

`__array_finalize__` 是唯一始终看到正在创建的新实例的方法，所以在其他任务中，它是为新对象属性填充实例默认值的最优选择。


```python
# 简单示例
class InfoArray(np.ndarray):

    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        # 创建自定义类型的 ndarray，调用标准 ndarray 构造器，但返回自定义类型
        # 同时会触发 InfoArray.__array_finalize__
        obj = super().__new__(subtype, shape, dtype,
                              buffer, offset, strides, order)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        # self 是从 ndarray.__new__(InfoArray, ...) 来的新对象
        # 因此只有 ndarray.__new__ 构造器给的属性
        
        # 可以通过三种方法调用 ndarray.__new__:
        # 从显式构造函数，如 InfoArray(): obj 是 None
        if obj is None: return
        # 从视图转换，如 arr.view(InfoArray): obj 是 arr，type(obj) 是 InfoArray
        # 从模板创建中调用，如 infoarr[:3]：type(obj) 是 InfoArray
        #    type(obj) is InfoArray
        #
        # ⚠️ 注意：在这里设置 info 的默认值（不是 __new__ 方法中）
        # 因为这个方法可以看到所有默认对象的创建（显式构造函数、视图转换、模板创建）
        self.info = getattr(obj, 'info', None)
        # 不需要返回任何东西
```


```python
obj = InfoArray(shape=(3,))
type(obj)
```




    __main__.InfoArray




```python
obj.info is None
```




    True




```python
obj = InfoArray(shape=(3,), info='information')
obj.info
```




    'information'




```python
arr = np.arange(10)
cast_arr = arr.view(InfoArray) # view casting, arr 没有 info
type(cast_arr)
```




    __main__.InfoArray




```python
cast_arr.info is None
```




    True




```python
v = obj[1:]  # obj 自己有info
type(v)
```




    __main__.InfoArray




```python
v.info
```




    'information'




```python
# 更真实的例子
class RealisticInfoArray(np.ndarray):

    def __new__(cls, input_array, info=None):
        # 输入 array 已经是 ndarray 实例，先将其 cast 到自定义 class
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)
```


```python
arr = np.arange(5)
obj = RealisticInfoArray(arr, info='information')
type(obj)
```




    __main__.RealisticInfoArray




```python
obj.info
```




    'information'




```python
v = obj[1:]
type(v)
```




    __main__.RealisticInfoArray




```python
v.info
```




    'information'



更多可参考：

- https://numpy.org/doc/stable/user/basics.subclassing.html

NumPy内置了一些子类，我们这里主要介绍内存映射文件数组，它一般用于读取或修改具有规则布局的大文件的小段，无需将整个文件读入内存。


```python
filename = "data/memmap.dat"
fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
```


```python
arr = np.arange(12).reshape(3, 4)
fp[:] = arr[:]
```

需要手动 flush 到磁盘（试试不刷会咋样）：


```python
fp.flush()
```

然后就可以读回来了：


```python
newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
```


```python
newfp
```




    memmap([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]], dtype=float32)



读取部分使用offset控制的，offset的大小是dtype的大小的整数倍：


```python
# 32位=4个字节
partfp = np.memmap(filename, mode="r", dtype=np.float32, offset=4)
partfp
```




    memmap([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
           dtype=float32)




```python
partfp = np.memmap(filename, mode="r", dtype=np.float32, offset=2)
partfp
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-534-b8d9cba6aa70> in <module>
    ----> 1 partfp = np.memmap(filename, mode="r", dtype=np.float32, offset=2)
          2 partfp
    

    /usr/local/lib/python3.8/site-packages/numpy/core/memmap.py in __new__(subtype, filename, dtype, mode, offset, shape, order)
        237                 bytes = flen - offset
        238                 if bytes % _dbytes:
    --> 239                     raise ValueError("Size of available data is not a "
        240                             "multiple of the data-type size.")
        241                 size = bytes // _dbytes
    

    ValueError: Size of available data is not a multiple of the data-type size.


除此之外，还有《数据类型：结构化》一节用到的`rec`记录数组、专门用来进行掩码操作的掩码数组等。此处不再赘述，可参考：

- https://numpy.org/doc/stable/reference/arrays.classes.html

## 小结

![](img/core_concepts.png)

## 参考

- [NumPy documentation — NumPy v1.23.dev0 Manual](https://numpy.org/devdocs/index.html)
- [What Is Little-Endian And Big-Endian Byte Ordering? | Engineering Education (EngEd) Program | Section](https://www.section.io/engineering-education/what-is-little-endian-and-big-endian/)
- [Understanding Big and Little Endian Byte Order – BetterExplained](https://betterexplained.com/articles/understanding-big-and-little-endian-byte-order/)


```python

```
