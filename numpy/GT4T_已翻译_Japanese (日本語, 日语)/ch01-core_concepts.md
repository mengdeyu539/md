<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"> <ul class="toc-item"><li><span><span class="toc-item-num">1 &nbsp;&nbsp;</span><ul class="toc-item"><a href="#特殊值" data-toc-modified-id="特殊值-1.1"><span class="toc-item-num">1.1 </a></li><span><span></a><span><span></li></ul></li><li><span><a href="#结构化数组" data-toc-modified-id="结构化数组-3"><span class="toc-item-num">3 &nbsp;&nbsp;</span>構造化配列 </a></span></li><li><span><span class="toc-item-num">4 &nbsp;&nbsp;</span>時間配列 </a><li><span class="toc-item-num">5 &nbsp;&nbsp;</a><ul class="toc-item"><ul class="toc-item"><a href="#ndarray" data-toc-modified-id="ndarray-5.1"><a href="#ndarray" data-toc-modified-id="ndarray-5.1"><ul class="toc-item"><a href="#ndarray" data-toc-modified-id="ndarray-5.1"><ul class="toc-item"><ul class="toc-item"><span class="toc-item-num"><span class="toc-item-num"></span>カスタム配列コンテナ </a></span></li><li><span><a href="#子类化与标准子类" data-toc-modified-id="子类化与标准子类-7"><span class="toc-item-num">7 &nbsp;&nbsp;</span>サブクラス化と標準サブクラス </a></span></li><li><span><span class="toc-item-num">8 &nbsp;&nbsp;</span>概要 </a></span></li><li><span class="toc-item-num">9 &nbsp;&nbsp;</span>参照 </a></span></li></div>



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
    
    Python implementation: CPythonPython version       : 3.8.13IPython version      : 7.23.1
    
    Compiler    : Clang 13.1.6 (clang-1316.0.21.2)OS          : DarwinRelease     : 21.1.0Machine     : x86_64Processor   : i386CPU cores   : 4Architecture: 64bit

    
    


```python
import numpy as np
```



```python
%watermark --iversions
```

    numpy: 1.23.0
    
    

ドキュメントの読み取り手順：

- 🐧はTipを示します
- ⚠️注意事項を示す

## 定数

NumPyには一般的な定数が付属しているので、直接使いやすいです。

### 特殊値



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

### NULL値



```python
# 空值
np.nan
```




    nan





```python
type(np.nan)
```




    float



⚠️ `np.nan` は1つの値であり、2つの `np.nan` は同じタイプに属しているが、等しくないことに注意してください。



```python
np.nan is np.nan
```




    True





```python
np.nan == np.nan
```




    False



 `np.isnan` メソッドで判断することができます。



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



### 無限



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



 `np.isxx` を使って判断できます。



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



## データタイプ


numpyは豊富なデータ型をサポートしており、[官方文档](https://numpy.org/devdocs/user/basics.types.html)では非常に包括的に說明されています。ここではあまり葛藤しないで、全体的な視点から改めて整理してみましょう。実際、私たちがもっと注目しなければならないのは、内蔵されているデータ型オブジェクト `dtype`、つまりこのドキュメント：[Data type objects](https://numpy.org/devdocs/reference/arrays.dtypes.html#arrays-dtypes)です。



```python
# 数据类型 和 数据类型对象
type(np.int8), type(np.dtype(np.int8))
```




    (type, numpy.dtype[int8])



データ型オブジェクトは、配列アイテムに対応する固定サイズのメモリブロック内のバイトを解釈する方法を説明します。主に以下の側面が含まれています（もちろん他にも多くの情報があります）。

- データタイプ
- データサイズ
- データの順序
- 構造化データ型の場合は他のデータ型のコレクション
- データ型がサブ配列の場合、その形状とデータ型


これまでarrayを作成するときはデータ型に関心を持っていませんでした。この場合、numpyは現在の入力に最も適切なデータ型に自働的にマッチし、すべての要素にキャストします。


全体的には次のような種類に分けることができますが、ほとんどの場合、intとfloatの2種類に最も注目すべきです：

- bool： `bool8`, `bool_`,intではありません
- int： `int8/byte`, `int16/short`, `int32`, `int64/longlong`, `int_`
- uint：符号なし型。 `unsigned` を表し、intに対応する
- float： `float16/half`, `float32/single`, `float64/double`, `float_`
- complex：復数、 `complex64`、 `complex128`、 `complex_`
- str： `str0`、 `str_`、ユニコード符号化を表す
- bytes: `bytes_`, `string_`
- datetime/timedelta
- structed array

次の数字は、メモリ内の数字が何桁を占めているかを示します。**一般的にはこの表現が推奨されています**。下線が付いているのはpythonのデータ型を表し、numpyはpythonの型を自動的にそれに変換することができます。また、浮動小数点数は異なる精度と拡張精度をサポートしています。

### タイプ

まずこの図を見てみましょう：

![](https://numpy.org/devdocs/_images/dtype-hierarchy.png)

から：[Scalars—NumPy v1.23.de v0マニュアル](https://numpy.org/devdocs/reference/arrays.scalars.html)

基本的には、datetimeとstructured arrayを除く上記のすべてのタイプをカバーしています。これら2つのタイプは后で別々に說明します。



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



次に整数型を例に説明するが、それ以外は同様である。



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



🐧：アレイを作成するときにデータ型を指定し、統一されたデータ型計算を使用することを推奨します。



```python
%timeit np.arange(100, dtype=np.float32).reshape(10, 10) * np.arange(100, dtype=np.int32).reshape(10, 10)
```

    4.64 µs ± 151 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    


```python
%timeit np.arange(100, dtype=np.int32).reshape(10, 10) * np.arange(100, dtype=np.int32).reshape(10, 10)
```

    2.41 µs ± 70.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    

### サイズ

タイプはとりあえずここを見て、まずサイズを見てみましょう：



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



🐧：再びarrayを作成するときにデータ型を指定することをお勧めします。パフォーマンスだけでなく、メモリを節約することもできます。また、それぞれの配列の範囲を自分に強要して考えて、自分の中で知っています。



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



### シーケンス

これはそれほど直感的ではありません。まず背景知識を理解しましょう。

バイト順序（Endianness）とは、コンピュータサイエンスでは、メモリ内のバイトの順序を指す。バイトの配置には2つの一般的なルールがあります：

- 低いビットを小さいアドレスに置き、高いビットを大きいアドレスに置くことをリトルエンディアン（little-endian）と呼びます。
- それとは逆のものがビッグエンディアン（big-endian）である。


![](https://qnimg.lovevivian.cn/cs-endian-1.jpg)

写真：[Endianness  -  Wikipedia](https://en.wikipedia.org/wiki/Endianness)


私たちがよく使用するx86コンピュータは、メモリアドレスは一般的に低いビットから高いビットに向かって徐々に増加しているので、私たちのバイナリ（またはその他の進数）は上位ビットが前に、下位ビットが後にあるので、小さいエンディアンシーケンスを使用するのは自然で、プログラミングも容易です。しかし、たまたま人が読むと、ちょうど逆になります。

上の画像を例にとると、0A0B0C0Dは自然順序であり、0Dは低位であり、小エンド順序では低位アドレスに配置されています。0Aは上位ビットであり、ビッグエンディアンでは下位ビットに配置される。


 `numpy` では、dtypeの各タイプを1文字で表すことができますが、文字で表す場合はバイト順序を増加させることができます。

サポートされている文字は以下のように表されています（大文字と小文字はそれぞれ符号なしと符号付きを意味します）：


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


さらにいくつかの複雑なタイプがあります：

-  `c`：複素浮動小数点
- `m/M`: timedelta / datetime
-  `O`：Pythonオブジェクト
-  `U`：Unicode文字列
- `V`: void
-  `S/a`：ゼロ終了バイト (推奨されません)

バイトシーケンスには、次のような種類があります：

Character | Byte order | Size 
-----------|-----------|-------
`=`       | native     | standard
`<`       | little-endian     | standard
`>`       | big-endian     | standard

デフォルトはい `=`。

上記の一部はhttps://docs.python.org/3/library/struct.htmlから参照されています。



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



バイト順序は **異なるデバイス**(バイト順序が異なる) 間のデータの相互作用によく使用され、互いに変換することもできます。

## 構造化配列

構造化配列とは、データ型が **グループ**(1つだけではなく) 異なるタイプの配列であり、計算時に復数のタイプのデータをまとめなければならない場合によく使用されます。



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



 `dtype` を取り除くと、以前に說明したものと同じようになります（統一型に変更します）。



```python
np.array(
    [('Rex', 9, 81.0), ('Fido', 3, 27.0)])
```




    array([['Rex', '9', '81.0'],
           ['Fido', '3', '27.0']], dtype='<U32')



各要素は構造化されたデータのセットなので、構造化配列とも呼ばれます。

⚠️注： `dtype` の各tupleは要素のうちの1つに対応します。例えば、上記の例では、最初の要素がU10型であり、それぞれのtupleの最初の要素がU10型であることを示しています。



```python
arr[0]
```




    ('Rex', 9, 81.)



また、多次元の構造化配列を作成することもできますが、この多次元は通常の多次元とは異なり、各要素を複数回繰り返します、つまり、それぞれのarrayのタイプは実はまだ一致していて、それから複数のarrayになっています。



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
           [ 4,  5,  7],[ 7,  8, 11],[10, 11, 12]], dtype=int32)





```python
marr['u']
```




    array([[ 1,  2,  5],
           [ 4,  5,  7],[ 7,  8, 11],[10, 11, 12]])





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



 `zeros` および `ones` は、通常のインタフェースと同様に、多次元（構造化）配列を迅速に作成できます。



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



 `rec` インターフェースは、arrayはプロパティ名でアクセスできます。



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



通常のarrayを構造化配列に変換することもできます。



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



また、構造化配列は、型にshapeとして接頭辞を付けることもできます。



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



構造化配列の簡単な操作方法もあります。[文档](https://numpy.org/devdocs/user/basics.rec.html#module-numpy.lib.recfunctions)を参照してください。ここでは詳しくは說明しません。

## タイム配列

 `datetime` は、時間に特化したAPIであり、時系列を扱う際に非常に役立ちます。Pythonの `datetime` と区別するために、Numpyでは `datetime64`、フォーマットは[ISO 8601](https://en.wikipedia.org/wiki/ISO_8601)です。

すなわち、1970年1月1日0時0分0秒以降、一般的な単位は、年（Y）、月（M）、週（W）、日（D）、時間（h）、分（m）、秒（s）、マイクロ秒（ms）、およびNAT (Not a Time）である。



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



特定の単位を指定することもできます：



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



Unixタイムスタンプをサポートしています（⚠️ここではデータ型は単位を指定することに注意してください）：



```python
np.array([0, 1577836800000], dtype="datetime64[ms]")
```




    array(['1970-01-01T00:00:00.000', '2020-01-01T00:00:00.000'],
          dtype='datetime64[ms]')



 `arange` アクションもサポートされています：



```python
np.arange("2020-01", "2021-01-02T00:00:00", dtype="datetime64[M]")
```




    array(['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06',
           '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12'],
          dtype='datetime64[M]')



もちろん文字列に戻ることもできます：



```python
t1
```




    numpy.datetime64('2022-02-28')





```python
np.datetime_as_string(t1)
```




    '2022-02-28'



 `timedelta64` は `timedelta` と似たようなものです。



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



もちろん、時間は増加または減少することもできます `timedelta`：



```python
np.datetime64("2021-03-01") + np.timedelta64(1, "D")
```




    numpy.datetime64('2021-03-02')





```python
np.datetime64("2021-03-01T00:00:00") + np.timedelta64(1, "h")
```




    numpy.datetime64('2021-03-01T01:00:00')



これらの基本的な机能に加えて、平日に関するいくつかの便利なAPIがあります：



```python
# 比如 2022-03-29 是周二，+4天后本来是 2 号，但它会跳过周末，直接到 4 号（下周一）
np.busday_offset("2022-03-29", 4)
```




    numpy.datetime64('2022-04-04')





```python
np.busday_offset("2022-03-29", [3, 4])
```




    array(['2022-04-01', '2022-04-04'], dtype='datetime64[D]')



 `busday_offset` のいくつかの重要な引数引数は次のとおりです：

- NumPyの日付
- offset
- roll：有効でない日付を処理する方法、デフォルトのraise、またnat (Not a Time)、forward/following (次の最近有効日)、backward/preceding (前の最近有効日)、modifiedfollowing (次の最近有効日が月を越えない)、modifiedpreceding (前の最近有効日が月を越えない) などを選択することができます。
- weekmask：週のどの日が有効な営業日であるかを指定する
- holidays：休日、NumPy日付形式の無効な日付（つまり、非営業日）



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



4月30日は土曜日、5月1日は日曜日：



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



5月1日は月をまたぐと、5月の最初の有効日（5月2日）です。



```python
np.busday_offset("2022-05-01", 0, "modifiedpreceding")
```




    numpy.datetime64('2022-05-02')





```python
np.busday_offset("2022-04-30", 0, "modifiedpreceding")
```




    numpy.datetime64('2022-04-29')



4月30日以降は月をまたいで、4月の最后の有効日を選択します：



```python
np.busday_offset("2022-04-30", 0, "modifiedfollowing")
```




    numpy.datetime64('2022-04-29')





```python
np.busday_offset("2022-05-01", 0, "modifiedfollowing")
```




    numpy.datetime64('2022-05-02')



weekmaskをご覧ください。



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



また、2つの日付間の有効日の数をカウントしたり、特定の日付が有効かどうかを判断したりすることもできます：



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



## 配列オブジェクト

### ndarray

NumPyはN次元配列型、 `ndarray` を提供し、**同じタイプ**要素の集合を記述します。それは下層ですarrayインタフェース。

すべての `ndarray` 要素は同質であり、各要素はデータ型によって同じサイズのメモリブロックを占有します。 `ndarray` 同じデータを共有できます。

オブジェクトは次のように署名されます：


```python
numpy.ndarray(shape, dtype=float, buffer=None, offset=0, strides=None, order=None)
```

- shape：形状を表す整数タプル
- dtype：データ型オブジェクト
- バッファ：バッファ内のデータでパディングします `ndarray`
- offset：バッファ内のオフセット
- stride：メモリ内データのスパン
- オーダー：ローベース（C-Style）またはカラムベース（Fortran-Style）


bufferが空の場合、shape、dtype、orderの3つのパラメータが使用されます。バッファが空でない場合は、すべての引数が使用されます。



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



実際、バッファは本質的に `bytes` です。このインタフェースは比較的下位にあり、ここではデータが連続的に保存されている（あるいはできるだけメモリ内で連続的に保存されている）ためです。例を見てみましょう：



```python
buf = b"\x01\x02\x03\x04"
```

上記の `buf` は16進数のセットであり（各数字は4桁であり、合計は32桁である）、必ずしも1、2、3、4であるとは限りません。それらの値は、私たちがどのように指定するかによって決まります `dtype`。 `dtype` を `int8` として指定すれば、 `buf` には4つの数字があり、 `int16` として指定すれば、 `buf` には2つの数字しかありません。



```python
np.ndarray(shape=(4, ), dtype=np.int8, buffer=buf)
```




    array([1, 2, 3, 4], dtype=int8)





```python
np.ndarray(shape=(2, ), dtype=np.int16, buffer=buf)
```




    array([ 513, 1027], dtype=int16)



考えてみてください、この2つの値はどうやって計算されますか？

> ヒント：書くことができます：0x 0201 0x 0403

私たちは普段このインタフェースをあまり使用しないので、いくつかのパラメータについて少し混乱するかもしれません。次に少し説明していきます。バッファが空のときに強調すべきことはあまりありませんが、バッファが空でないときに焦点を当ててください。

 `shape` と `dtype` もはっきりしています。 `buffer` 先ほど説明しましたが、主に残りの3つのパラメータ： `offset`、 `strides`、 `order` です。

 `order` は、どのスタイルのストレージを使用するかを指します。計算において、行主順序（C Style）と列主順序（F Style）は、多次元配列をRAMなどの線形メモリに格納する方法です。行主順序では、1行の連続要素が互いに隣接し、列主順序では、1列の連続要素が互いに隣接する。[Row- and column-major order  -  Wikipedia](https://en.wikipedia.org/wiki/Row-_and_column-major_order)を参照してください。

⚠️**注意しなければならないのは**：ストレージ方式によって計算効率が異なり、特定のシナリオ（行が多いか列が多いか）に応じて異なるStyleを選択することができます。



```python
# copy 让 carr 拥有数据，否则只是 view
carr = np.arange(1000000).reshape(1000, 1000).copy()
carr.flags
```




      C_CONTIGUOUS : TrueF_CONTIGUOUS : FalseOWNDATA : TrueWRITEABLE : TrueALIGNED : TrueWRITEBACKIFCOPY : False





```python
farr = np.asfortranarray(carr)
farr.flags
```




      C_CONTIGUOUS : FalseF_CONTIGUOUS : TrueOWNDATA : TrueWRITEABLE : TrueALIGNED : TrueWRITEBACKIFCOPY : False





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
    

 `offset` と `strides` が組み合わせて使用されます。前者はオフセット位置、後者はストライドであり、それはshapeと同じ長さでなければなりません。つまり、与えられたbufferに基づいて、ターゲットshapeのarrayを生成する。これの理由は主に内部ストレージに関係しています。実際、 `ndarray` はこの2つのパラメータによってshapeを制御しています。異なるshapeは実際に同じストレージを持っています。



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
           [3, 4],[5, 6]], dtype=int8)





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
           [4, 5],[6, 7]], dtype=int8)



⚠️注意しなければならないのは、 `offset` と `strides` の数字は本当の数字のサイズではなく、占める桁数であることです。

さらに様々なものを見てみましょうshapeの保存状況。



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



実際、shapeがどんなに変わっても、メモリはまったく変わらず、異なるarrayは実は異なるstrides方式にすぎず、異なるstridesは異なるshapeを表現します。興味のある方はさらに試してみてください。

また、bufferを使用して作成された `ndarray` はすべて同じメモリを使用しています。



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
           [2, 3],[3, 4]], dtype=int8)





```python
bytes(arr1), bytes(arr2)
```




    (b'\x01\x02\x03\x02\x03\x04', b'\x01\x02\x02\x03\x03\x04')





```python
np.may_share_memory(arr1, arr2), np.may_share_memory(buf, arr1)
```




    (True, True)



実は、arr1もarr2もbufの一つのview（参照）であり、これに対してcopyである。

 `reshape` ほとんどの場合、viewを取得するためにstridesを変更しますが、配列が連続していない場合（例えば、トランスポート后）はそれはできません。トランスポートによって配列が変わります（実際には、C-styleとF-styleが互いに変換されます）。



```python
arr = np.ones((2, 3))
arr.shape = (3, 2)
```



```python
arr.flags
```




      C_CONTIGUOUS : TrueF_CONTIGUOUS : FalseOWNDATA : TrueWRITEABLE : TrueALIGNED : TrueWRITEBACKIFCOPY : False





```python
arr.T.flags
```




      C_CONTIGUOUS : FalseF_CONTIGUOUS : TrueOWNDATA : FalseWRITEABLE : TrueALIGNED : TrueWRITEBACKIFCOPY : False





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


ところで、**一般に**、スライスはviewを作成し、インデックスはcopyを作成します。スライスとインデックスについては後述します。



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
           [3, 4, 5],[6, 7, 8]])





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
           [100, 100, 100],[  6,   7,   8]])





```python
# copy 的效果
y
```




    array([[3, 4, 5]])





```python
y.base is None
```




    True



つまり、bufferを使用して `ndarray` を作成することは、実際にはスライスとして理解されます。実際、 `np.array` のインタフェースを見ると、 `copy` パラメータがあり、デフォルトは `True` です。



```python
buf, arr1, arr2
```




    (array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int8),
     array([[1, 2, 3],
            [2, 3, 4]], dtype=int8),
     array([[1, 2],
            [2, 3],[3, 4]], dtype=int8))





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
            [2, 3],[3, 4]], dtype=int8))



 `strides` 処理効率は同時によって異なります。ステップサイズが増えると、対応する位置の値を見つけるのが遅くなります。その理由は、CPUがタスクを処理する際にメモリからキャッシュにデータを読み出し、ステップが小さいほど転送が少ないからです。たとえば10個の数を取ると、連結したもの（ステップサイズ=1個の数字）は一度に取ることができますが、ステップが大きくなると何度も取らなければなりません。

> 注：CPUキャッシュは、プロセッサがメモリにアクセスするのにかかる平均時間を短縮するためのコンポーネントです。ピラミッド型ストレージアーキテクチャでは、CPUレジスタに次ぐトップダウンの2番目のレイヤに位置します。容量はメモリよりはるかに小さいが、速度はプロセッサの周波数に近い。通常、複数のレベルのキャッシュがあります。——ウィキペディア



```python
arr1 = np.ones((1000, 100), dtype=np.int8)
arr2 = np.ones((10000, 100), dtype=np.int8)[::10]
arr1.shape, arr2.shape
```




    ((1000, 100), (1000, 100))





```python
arr1.flags
```




      C_CONTIGUOUS : TrueF_CONTIGUOUS : FalseOWNDATA : TrueWRITEABLE : TrueALIGNED : TrueWRITEBACKIFCOPY : False





```python
# OWNDATA=False，意思是这个 array 的数据是从其他地方「借」来的
# 从哪个地方呢？当然就是 `np.ones((10000, 100), dtype=np.int8)` 这里了
arr2.flags
```




      C_CONTIGUOUS : FalseF_CONTIGUOUS : FalseOWNDATA : FalseWRITEABLE : TrueALIGNED : TrueWRITEBACKIFCOPY : False





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
    

最後に説明しますが、上の例はすべて使用しています。 `int` 型で説明しますが、他のデータ型も同様です。

さらに、ソースコードの大まかなロジックは、 `arrayobject.c` の `array_new` メソッドがctors.cの `PyArray_NewFromDescr_int` を呼び出し、 `ndarray` を作成します。

上記のパフォーマンスに影響を与えることに加えて、実際には呼び出しインタフェースによっても違いがあります。例えば、トランスポート操作には、次の3つの方法があります：

- `arr.T`
- `arr.transpose`
- `np.transpose`

実はそれらはほぼ同じですが、呼び出し方法が異なり、パフォーマンスにも違いがあります（当然ですね）。



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
    

少し説明すると、 `arr.T` と `arr.transpose` はほぼ同じですが、後者が少し遅い理由は、 `axes` パラメータがあるからです。 `np.transpose` が遅いのは、呼び出し方法のためです。


```python
def transpose(a, axes=None):
    return _wrapfunc(a, 'transpose', axes)

def _wrapfunc(obj, method, *args, **kwds):
    try:
        return getattr(obj, method)(*args, **kwds)
    except (AttributeError, TypeError):
        return _wrapit(obj, method, *args, **kwds)
```

だからできれば、できるだけarrayでメソッドを呼ぶことをおすすめします。

### array

まず、 `array` は `ndarray` のインターフェース関数を素早く作成するだけで、ソースコードは `core/src/multiarray/methods.c` の `array_getarray` です。これは実は上記の `PyArray_NewFromDescr_int` を呼び出しています。

 `np.array` の引数は次のとおりです：

- object
- dtype：データ型、デフォルトNone
- copy：コピーするかどうか、デフォルトはTrue
- order：デフォルトのK、CFKA、CとFが取得できます。前に述べたように、KとAは、コピーがないときは元の順序です。copy=True時間：
  - Kの場合、FとCの2種類が残り、そうでない場合は最も類似したものが残ります。
  - Aの場合、入力がFであり、Cでない場合はF;そうでなければC
- subok、デフォルトはFalse、Trueの場合は受信したサブクラス型を返します。
- ndmin：デフォルト0、最小次元を指定します
- like：デフォルトのNone、NumPyに属していない配列を作成できます。

最初の2つのパラメータは言及しません。まずcopyパラメータを見てみましょう。



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



次に、orderパラメータがあります：



```python
# 没有 copy 时，就是原来的顺序
a1 = np.array([[2, 3], [4, 5]], order="F")
a2 = np.array(a1, order="K", copy=False)
a2.flags
```




      C_CONTIGUOUS : FalseF_CONTIGUOUS : TrueOWNDATA : TrueWRITEABLE : TrueALIGNED : TrueWRITEBACKIFCOPY : False





```python
# copy=True，order=K，保留最相似的（F），向量是CF
a1 = np.array([[2, 3], [4, 5]], order="F")
a2 = np.array(a1, order="K", copy=True)
a2.flags
```




      C_CONTIGUOUS : FalseF_CONTIGUOUS : TrueOWNDATA : TrueWRITEABLE : TrueALIGNED : TrueWRITEBACKIFCOPY : False





```python
# copy=True，order=A，如果输入是 F 则是 F；否则（AKC时）是 C
a1 = np.array([[2, 3], [4, 5]], order="A")
a2 = np.array(a1, order="A")
a2.flags
```




      C_CONTIGUOUS : TrueF_CONTIGUOUS : FalseOWNDATA : TrueWRITEABLE : TrueALIGNED : TrueWRITEBACKIFCOPY : False



読者は自分で他の状況を検証することができます。

subokパラメータを見てみましょう。



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



次にndmin引数があります：



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



最后のlikeパラメータは以下のように示されます。



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



## カスタム配列コンテナ



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

 `np.asarray` または `np.array` を使用して `array` に変換できます（ `__array__` メソッドが呼び出されます）：



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



 `__array__` は、NumPyのAPIを使用して操作するときに呼び出されます：



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


 `__array_function__` または `__array_ufunc__` から**動作の定義**をカスタマイズすることができます。

カスタムクラス加算を定義する必要があるとします。これは、 `__add__` または継承 `numpy.lib.mixins.NDArrayOperatorsMixin` を使用することで実現できます。



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



より多くの内容を参照してください：

- https://numpy.org/doc/stable/user/basics.dispatch.html
-  https://numpy.org/doc/stable/reference/arrays.classes.html 第1部

## サブクラス化と標準サブクラス

ndarrayの新しいインスタンスは3つの異なる方法で表示されます。

- 明示的なコンストラクタ呼び出し
- ビュー変換
- テンプレート作成：最も明らかなところはサブクラスの配列をスライスすることです。

後者の2つはndarrayの特徴であり、ndarrayをサブクラス化する複雑さは、NumPyが後者の2つのインスタンス作成パスのメカニズムをサポートしなければならないからです。

サブクラス化は、次の場合に適用されます：
- メンテナンス性や自分以外のユーザーの心配はありません。
- サブクラス情報が無視されたり失われたりするのは問題ではありません。



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



ndarrayがサブクラス内のビューと新しいテンプレートをサポートするために使用するメカニズムには2つの側面があります。

- オブジェクト初期化の主な作業は、より一般的な `__init__` メソッドではなく、 `ndarray.__new__` メソッドを使用します。
-  `__array_finalize__` メソッドを使用すると、テンプレートからビューと新しいインスタンスを作成した後にサブクラスをクリーンアップできます。

最初に初期化された `__new__` メソッドを見てみましょう。これは、ndarrayの場合には、別のクラスのオブジェクトを返したい場合があるからです。



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



このようにndarrayクラスのサブクラスは、クラス型を保持したview (view casting) を返すことができます。viewが実行されると、標准的なndarrayメカニズムは次のように新しいndarrayオブジェクトを作成します。 `obj = ndarray.__new__(subtype, shape,...)`、 `subbtype` はサブクラスなので、ndarrayのクラスではなく、サブクラスのクラスが返されます。


 <! - -しかし、これは新しい問題があります。私たちはまだこのような `__new__` メソッドを持っていません。 - ->

次は `__array_finalize__` です。これは、サブクラスが作成した新しいインスタンスを処理するためのさまざまなメソッドを許可し、署名は `__array_finalize__(self, obj)` です。ビュー変換やテンプレート作成には `MySubClass.__new__` または `MySubClass.__init__` に依存することはできないからです。



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
       self type is <class '__main__.E'> self is  4773873856obj type is <class 'NoneType'> obj is  4530430688
    In __init__ with class <class '__main__.E'>

    


```python
arr = np.ones((3, ))  # == obj
e_arr = arr.view(E)
type(e_arr)
```

    In array_finalize:
       self type is <class '__main__.E'> self is  4773875136obj type is <class 'numpy.ndarray'> obj is  4765117296
    




    __main__.E





```python
v = e[:1]
type(v)
```

    In array_finalize:
       self type is <class '__main__.E'> self is  4773872832obj type is <class '__main__.E'> obj is  4773873856
    




    __main__.E





```python
v is e_arr
```




    False



上記の例を見ると、次のように分かる：

- 明示的なコンストラクタから呼び出すと、objはNoneです。
- ビュー変換から呼び出された場合、objは自分のサブクラスを含むndarrayの任意のサブクラスのインスタンスであることができます。
- テンプレート作成から呼び出されたとき、objは自分のサブクラスの別のインスタンスであり、新しいselfインスタンスを更新するために使用する可能性があります。

 `__array_finalize__` は、作成中の新しいインスタンスを常に表示する唯一の方法です。したがって、他のタスクでは、新しいオブジェクトプロパティのインスタンスのデフォルト値を入力するのに最適です。



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



より多くは参照できます：

- https://numpy.org/doc/stable/user/basics.subclassing.html

NumPyにはいくつかのサブクラスが組み込まれています。ここでは主にメモリマッピングファイル配列を紹介します。これは一般的に、ファイル全体をメモリに読み込む必要はなく、規則的なレイアウトを持つ大きなファイルの小さなセグメントを読み込んだり修正するために使用されます。



```python
filename = "data/memmap.dat"
fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
```



```python
arr = np.arange(12).reshape(3, 4)
fp[:] = arr[:]
```

手動でディスクにフラッシュする必要があります（フラッシュしないと試してみてください）：



```python
fp.flush()
```

そしてそれを読み返すことができます：



```python
newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
```



```python
newfp
```




    memmap([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],[ 8.,  9., 10., 11.]], dtype=float32)



読み取り部分は、dtypeのサイズの整数倍のoffsetを使用して制御されます：



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
        237                 bytes = flen - offset238                 if bytes % _dbytes:
    --> 239                     raise ValueError("Size of available data is not a "
        240                             "multiple of the data-type size.")241                 size = bytes // _dbytes
    

    ValueError: Size of available data is not a multiple of the data-type size.


これに加えて、「データ型：構造化」のセクションで使用されている `rec` レコード配列、マスキング操作に特化したマスキング配列などがあります。ここでは説明しないが、以下を参照してください：

- https://numpy.org/doc/stable/reference/arrays.classes.html

## まとめ

![](img/core_concepts.png)

## 参考

-  [NumPyドキュメント—NumPy v1.23.de v0マニュアル](https://numpy.org/devdocs/index.html)
- [What Is Little-Endian And Big-Endian Byte Ordering? | Engineering Education (EngEd) Program | Section](https://www.section.io/engineering-education/what-is-little-endian-and-big-endian/)
-  [Understanding Big and Little Endian Byte Order-BetterExplained](https://betterexplained.com/articles/understanding-big-and-little-endian-byte-order/)



```python

```
