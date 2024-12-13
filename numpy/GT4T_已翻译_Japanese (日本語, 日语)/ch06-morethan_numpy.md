<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"> <ul class="toc-item"><li><span><span class="toc-item-num">1 &nbsp;&nbsp;</span>Numba </a></span><ul class="toc-item"><a href="#jit与njit" data-toc-modified-id="jit与njit-1.1"><span class="toc-item-num">1.1 &nbsp;&nbsp;</span><li><span><li><span><a href="#Loops" data-toc-modified-id="Loops-1.2"><li><li><a href="#Loops" data-toc-modified-id="Loops-1.2"><li><li><a href="#Loops" data-toc-modified-id="Loops-1.2"><li><a href="#Loops" data-toc-modified-id="Loops-1.2"><li><li></span></a></span></a></a><li></a><li><li></a>/> jit </a></span></li><li><span><span class="toc-item-num">2.3 &nbsp;&nbsp;</span>grad </a></li><li><a href="#vmap" data-toc-modified-id="vmap-2.4"></span>vmap </a></span></span></li><gtt r="472"/><span class="toc-item-num">4.1 &nbsp;&nbsp;</span>cupy.ndarray </a></span><li><a href="#Device" data-toc-modified-id="Device-4.2"><span class="toc-item-num">4.2 &nbsp;&nbsp;<li><a href="#Data-Transfer" data-toc-modified-id="Data-Transfer-4.3">4.3 &nbsp;&nbsp;</span>Data Transfer </a>"535"/> </ul></li><li><a href="#Dask" data-toc-modified-id="Dask-6"><span class="toc-item-num">6 &nbsp;&nbsp;</span>Dask </a><span><a href="#创建" data-toc-modified-id="创建-6.1"><span class="toc-item-num">6.1 </a></span></span></span><li><li></span><li></span></span></span></span>> </span>> </span></span>> </span>> </span></span>> </span></span></span>> >> </span>> >> >> >="600"/> </li></ul><li><span><span class="toc-item-num">7 &nbsp;&nbsp;</span>Xarray </span><ul class="toc-item"><a href="#创建" data-toc-modified-id="创建-7.1"><span class="toc-item-num">7.1 </a></span></li><li><span><a href="#索引" data-toc-modified-id="索引-7.2"></span><a href="#索引" data-toc-modified-id="索引-7.2"><a href="#索引" data-toc-modified-id="索引-7.2"></span><a href="#索引" data-toc-modified-id="索引-7.2"><span class="toc-item-num"></div>



```python
import numpy as np
np.__version__
```




    '1.22.4'



ドキュメントの読み取り手順：

- 🐧はTipを示します
- ⚠️注意事項を示す

このセクションでは、NumPyに関連する高性能、分散型数値計算の用法とツールについて説明します。これらのインストールは比較的簡単で、ドキュメントを参照してください。ここでは、各ツールが何をしているのか、どのような特徴があるのか、いつ使う必要があるのかに焦点を当てて紹介します。

## Numba


文書：[Numbaドキュメント—Numba 0.55.2+0.g2298ad618.dirty-py3.7-linux-x86_64.eggドキュメント](https://numba.readthedocs.io/en/stable/index.html)

NumbaはPython用のインタイムコンパイラであり、NumPy配列や関数、ループを使用するコードに最適です。最も一般的な使い方は、デコレーターを通じてです。Numbaのデコレータが呼び出されると、それは実行のための即時のマシンコードにコンパイルされ、そのコードの全部または一部がマシンコードの速度で実行される。

要約すると、Numbaを使用するには、次のような状況が適しています：

- たくさんの数学計算
- Numpyをたくさん使っています
- ループがたくさんあります

その原理は、装飾関数のPythonバイトコードを読み取り、関数入力のパラメータ型情報と組み合わせて、コードを分析・最適化した後、LLVMコンパイラを用いてCPUカスタマイズに合わせた関数のマシンコードバージョンを生成するというものです。その後の呼び出しでは、コンパイルされたバージョンが使用されます。



```python
from numba import jit
```



```python
def func_normal(a):
    x = np.median(a)
    y = np.max(a)
    t = x / y;
    z = x * np.sqrt(1 + t * t)
    m = 0.0
    for i in range(a.shape[0]):
        m += np.tanh(a[i, i])
        m /= z
    return a + m
```



```python
a = np.arange(100, dtype=np.int32).reshape(10, 10)
```



```python
%timeit func_normal(a)
```

    68.2 µs ± 2.32 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    


```python
@jit(nopython=True)
def func_numba(a):
    x = np.median(a)
    y = np.max(a)
    t = x / y;
    z = x * np.sqrt(1 + t * t)
    m = 0.0
    for i in range(a.shape[0]):
        m += np.tanh(a[i, i])
        m /= z
    return a + m
```



```python
prebuild = func_numba(a)
```



```python
%timeit func_numba(a)
```

    1.68 µs ± 33.8 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    

パフォーマンスの向上が明らかに分かる。

### jit vs njit

 `Numba` 2つのモードがあります：

-  `nopython` モード： `@jit(nopython=True)` または `@njit` デコレータで飾ります。このモードでは、関数は完全にコンパイルモードで実行され、Pythonインタプリタの関与は必要ありません。これは `Numba` の推奨方法でもあります。
-  `object` パターン： `@jit` で直接装飾した場合、 `nopython` パターンが失敗した場合、 `object` パターンを使用してコンパイルされ、Numba可能なコードの一部はマシンコードで実行され、残りはPythonコンパイラで実行されます。



```python
from numba import njit
import pandas as pd
```



```python
@njit
def jit_fail(x):
    df = pd.DataFrame(x)
    df += 1
    cov = df.cov()
    return cov
```



```python
x = {'a': [1, 2, 3], 'b': [20, 30, 40]}
```



```python
jit_fail(x)
```


    ---------------------------------------------------------------------------

    TypingError                               Traceback (most recent call last)

    <ipython-input-6-1a477a3676a1> in <module>
    ----> 1 jit_fail(x)
    

    /usr/local/lib/python3.8/site-packages/numba/core/dispatcher.py in _compile_for_args(self, *args, **kws)
        399                 e.patch_message(msg)400 
    --> 401             error_rewrite(e, 'typing')
        402         except errors.UnsupportedError as e:403             # Something unsupported is present in the user code, add help info
    

    /usr/local/lib/python3.8/site-packages/numba/core/dispatcher.py in error_rewrite(e, issue_type)
        342                 raise e343             else:
    --> 344                 reraise(type(e), e, None)
        345 346         argtypes = []
    

    /usr/local/lib/python3.8/site-packages/numba/core/utils.py in reraise(tp, value, tb)
         78         value = tp()79     if value.__traceback__ is not tb:
    ---> 80         raise value.with_traceback(tb)
         81     raise value82 
    

    TypingError: Failed in nopython mode pipeline (step: nopython frontend)non-precise type pyobject[1] During: typing of argument at <ipython-input-4-481c7b069f0d> (3)
    
    File "<ipython-input-4-481c7b069f0d>", line 3:def jit_fail(x):
        df = pd.DataFrame(x)
        ^
    
    This error may have been caused by the following argument(s):
    - argument 0: cannot determine Numba type of <class 'dict'>

    



```python
@jit
def jit_succ(x):
    df = pd.DataFrame(x)
    df += 1
    cov = df.cov()
    return cov
```



```python
# 会有警告
cov = jit_succ(x)
```

    <ipython-input-7-cb5f8409f203>:1: NumbaWarning: [1mCompilation is falling back to object mode WITH looplifting enabled because Function "jit_succ" failed type inference due to: [1m[1mnon-precise type pyobject[0m[0m[1m[1] During: typing of argument at <ipython-input-7-cb5f8409f203> (3)[0m[1mFile "<ipython-input-7-cb5f8409f203>", line 3:[0m[1mdef jit_succ(x):[1m    df = pd.DataFrame(x)[0m    [1m^[0m[0m[0m
      @jit
    /usr/local/lib/python3.8/site-packages/numba/core/object_mode_passes.py:177: NumbaWarning: [1mFunction "jit_succ" was compiled in object mode without forceobj=True.[1mFile "<ipython-input-7-cb5f8409f203>", line 2:[0m[1m@jit[1mdef jit_succ(x):[0m[1m^[0m[0m[0m
      warnings.warn(errors.NumbaWarning(warn_msg,
    /usr/local/lib/python3.8/site-packages/numba/core/object_mode_passes.py:187: NumbaDeprecationWarning: [1mFall-back from the nopython compilation path to the object mode compilation path has been detected, this is deprecated behaviour.
    
    For more information visit http://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit[1mFile "<ipython-input-7-cb5f8409f203>", line 2:[0m[1m@jit[1mdef jit_succ(x):[0m[1m^[0m[0m[0m
      warnings.warn(errors.NumbaDeprecationWarning(msg,

    


```python
%timeit jit_succ(x)
```

    559µs±16µs per loop (mean±std.dev.of 7 runs,1000 loops each)

    


```python
def func(x):
    df = pd.DataFrame(x)
    df += 1
    cov = df.cov()
    return cov
```



```python
%timeit func(x)
```

    514µs±29.8µs per loop (mean±std.dev.of 7 runs,1000 loops each)
    

このとき、パフォーマンスはほぼ同じですが、 `Numba` は、コンパイル最適化が可能かどうかを判断しなければならないため、むしろ遅くなります。

 [官方文档](https://numba.readthedocs.io/en/stable/user/5minguide.html#other-things-of-interest)他にもいくつかの机能がありますが、私たちは主にパフォーマンスに関連するものに焦点を当てています。

### Loops

 `Numba` ループは次のように最適化できます：



```python
def get_primes(x):
    res = []
    for v in range(x+1):
        if v < 2:
            continue
        flag = True
        for i in range(2, int(np.sqrt(v)) + 1):
            if v % i == 0:
                flag = False
        if flag:
            res.append(v)
    return res
```



```python
is_prime(10)
```




    [2, 3, 5, 7]





```python
x = 100000
```



```python
%timeit get_primes(x)
```

    1.28 s ± 48 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    


```python
@njit
def jit_get_primes(x):
    res = []
    for v in range(x+1):
        if v < 2:
            continue
        flag = True
        for i in range(2, int(np.sqrt(v)) + 1):
            if v % i == 0:
                flag = False
        if flag:
            res.append(v)
    return res
```



```python
prebuild = jit_get_primes(x)
```



```python
%timeit jit_get_primes(x)
```

    77.7 ms ± 2.16 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    

効果ははっきりしていることがわかりますが、できるだけ性能を比較して、心の中にあるようにしてください。

### FastMath

いくつかの厳格な (IEEE754) 値を緩和することで、さらにパフォーマンスを向上させることができます。

> IEEEバイナリ浮動小数点演算標准（IEEE 754）は、1980年代以来最も広く使用されている浮動小数点演算標准であり、多くのCPUおよび浮動小数点演算装置で採用されています。この標准は、浮動小数点数を表すフォーマット（負のゼロ - 0を含む）と異常値（デノーマル数）、いくつかの特別な数値（（無限（Inf）と非数値（NaN））、そしてこれらの数値の「浮動小数点演算子」を定義しています。また、数値の丸めの4つのルールと5つの例外（例外が発生するタイミングと処理方法を含む）を指定しています。

公式ドキュメントの例を参照してください：



```python
@njit(fastmath=False)
def do_sum(A):
    acc = 0.
    # without fastmath, this loop must accumulate in strict order
    for x in A:
        acc += np.sqrt(x)
    return acc

@njit(fastmath=True)
def do_sum_fast(A):
    acc = 0.
    # with fastmath, the reduction can be vectorized as floating point
    # reassociation is permitted.
    for x in A:
        acc += np.sqrt(x)
    return acc
```



```python
a = np.arange(40000)
```



```python
prebuild1 = do_sum(a)
prebuild2 = do_sum_fast(a)
```



```python
prebuild1, prebuild2
```




    (5333233.1256554425, 5333233.1256554425)





```python
%timeit do_sum(a)
```

    70 µs ± 678 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    


```python
%timeit do_sum_fast(a)
```

    53.1 µs ± 1.58 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    

デフォルトでは、コンパイラは浮動小数点式の再関連付けなどの浮動小数点最適化に関して厳しい制限を受けています。このような最適化は結果が変化する可能性があるためです。例えば：

- (10000001.0f * 10000001.0f) / 10000001.0f == 10000000.0f
- 10000001.0f * (10000001.0f / 10000001.0f) == 10000001.0f

カッコ内の最初の式は32ビットの精度を超え、丸められます。

この分野の詳細については、以下を参照してください：

- [SIMD vectorization](https://arcb.csc.ncsu.edu/~mueller/cluster/ps3/SDK3.0/docs/accessibility/sdkpt/cbet_1simdvector.html)
- [Floating Point Optimization](https://software-dl.ti.com/ccs/esd/documents/sdto_cgt_floating_point_optimization.html)

### Parallel



```python
from numba import prange
```



```python
@njit(parallel=True)
def do_sum_parallel(A):
    # each thread can accumulate its own partial sum, 
    # and then a cross
    # thread reduction is performed to obtain the result to return
    n = len(A)
    acc = 0.
    for i in prange(n):
        acc += np.sqrt(A[i])
    return acc

@njit(parallel=True, fastmath=True)
def do_sum_parallel_fast(A):
    n = len(A)
    acc = 0.
    for i in prange(n):
        acc += np.sqrt(A[i])
    return acc
```



```python
prebuild1 = do_sum_parallel(a)
prebuild2 = do_sum_parallel_fast(a)
prebuild1, prebuild2
```




    (5333233.125655441, 5333233.125655442)





```python
%timeit do_sum_parallel(a)
```

    108 µs ± 1.3 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    


```python
%timeit do_sum_parallel_fast(a)
```

    95.5 µs ± 3.72 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    

## JAX


文書：[JAX Quickstart-JAXドキュメンテーション](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)

JAXは、CPU、GPU、TPU上で実行されるNumPyです。

- JAXは便利なNumPyインスピレーションのインターフェースを提供します。
- ダック型によって、JAX配列は通常、NumPy配列を直接置き換えることができます。
- NumPy配列とは異なり、JAX配列は不変です。

### NumPyを置き換える



```python
import jax.numpy as jnp
```



```python
jnp.arange(10)
```

    WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
    




    DeviceArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)





```python
list(_)
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]





```python
from jax import random
```



```python
key = random.PRNGKey(42)
```



```python
a = random.normal(key, (2, 3))
a
```




    DeviceArray([[ 0.61226517,  1.1225883 ,  1.1373315 ],
                 [-0.81273264, -0.8904051 ,  0.12623137]], dtype=float32)





```python
b = random.normal(key, (3, 2))
b
```




    DeviceArray([[ 0.61226517,  1.1225882 ],
                 [ 1.1373315 , -0.8127326 ],[-0.8904051 ,  0.12623137]], dtype=float32)





```python
jnp.dot(a, b)
```




    DeviceArray([[ 0.63893783, -0.08147554],
                 [-1.6226908 , -0.1727684 ]], dtype=float32)





```python
np.dot(a, b)
```




    array([[ 0.63893783, -0.08147555],
           [-1.6226908 , -0.1727684 ]], dtype=float32)



### jit

 `jit` 主に加速に使用されます。



```python
from jax import jit
```



```python
def func_normal(a):
    x = jnp.median(a)
    y = jnp.max(a)
    t = x / y;
    z = x * jnp.sqrt(1 + t * t)
    m = 0.0
    for i in range(a.shape[0]):
        m += jnp.tanh(a[i, i])
        m /= z
    return a + m
```



```python
a = np.arange(100, dtype=np.int32).reshape(10, 10)
```



```python
pre = jit(func_normal)(a)
```



```python
%timeit func_normal(a)
```

    5.83 ms ± 57.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    


```python
%timeit jit(func_normal)(a)
```

    42.5 µs ± 2.45 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    

### grad

 `grad` は、微分を計算するために使用されます。



```python
from jax import grad
```

Sigmoid関数を例に挙げて下さい。

$$
f(x) = \frac{1}{1+e^{-x}}
$$

その微分は `f(x) * (1-f(x))` です。



```python
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))
```



```python
def dsigmoid(x):
    x = sigmoid(x)
    return x * (1-x)
```



```python
dersigmoid = grad(sigmoid)
```



```python
dersigmoid(2.)
```




    DeviceArray(0.10499357, dtype=float32)





```python
dsigmoid(2.)
```




    DeviceArray(0.10499363, dtype=float32)



### vmap

 `vmap` 自動ベクトル化またはバッチ化に使用されます。公式ドキュメントを例に挙げてみましょう。



```python
mat = random.normal(key, (150, 100))
```



```python
batched_x = random.normal(key, (10, 100))
```

まずシンプルループ版を見てみましょう：



```python
def apply_matrix(v):
    return jnp.dot(mat, v)
```



```python
def naive_batched(v_batched):
    return jnp.stack([apply_matrix(v) for v in v_batched])
```



```python
naive_batched(batched_x).shape
```




    (10, 150)





```python
%timeit naive_batched(batched_x).block_until_ready()
```

    3.78 ms ± 168 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    

次は行列乗算（手動バッチ）版です。



```python
def batched_apply_matrix(v_batched):
    return jnp.dot(v_batched, mat.T)
```



```python
%timeit batched_apply_matrix(batched_x).block_until_ready()
```

    226 µs ± 24.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    

最後には `vmap`：



```python
from jax import vmap, jit
```



```python
@jit
def vmap_apply_matrix(v_batched):
    return vmap(apply_matrix)(v_batched)
```



```python
%timeit vmap_apply_matrix(batched_x).block_until_ready()
```

    18.9 µs ± 545 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    

これは行列乗算ができないときに非常に便利です。

最后に、3つのメソッド、さらには `jnp` を単独で使用することも、組み合わせて使用することもできることを特筆すべきである。実際に使用する際には自分のニーズに合わせて柔軟に組み合わせることができます。

ここでは簡単に紹介しますが、詳しくはドキュメントをさらに読むことができます。

## Cython

 [Welcome to Cython Documentation-Cython 3.0.0a10ドキュメント](https://cython.readthedocs.io/en/latest/)

Cythonはこの章で特別なもので、C拡張子をPythonと同じくらい簡単に書くプログラミング言語です。Pythonのスーパーセットになることを目的としており、高度なオブジェクト指向、ダイナミックなプログラミングを提供しています。Cythonコードは最適化されたC/C++コードに翻訳され、Python拡張モジュールにコンパイルされます。プログラム実行をC言語と緊密に統合するだけでなく、Pythonの開発性を維持します。

最も簡単な例を見てみましょう：



```python
# 加载扩展
%load_ext Cython
```



```cython
%%cython

cdef int a = 0
for i in range(10):
    a += i
print(a)
```

    45
    

### annotate

コード解析は `annotate` オプションを使用して表示できます：



```cython
%%cython --annotate

cdef int a = 0
for i in range(10):
    a += i
print(a)
```

    45
    




<!DOCTYPE html>
<!-- Generated by Cython 0.29.30 -->
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <title>Cython: _cython_magic_5ab44e0a8b28fe78d74e9d2f9669b963.pyx</title>
    <style type="text/css">

body.cython { font-family: courier; font-size: 12; }

.cython.tag  {  }.cython.line { margin: 0em }.cython.code { font-size: 9; color: #444444; display: none; margin: 0px 0px 0px 8px; border-left: 8px none; }

.cython.line .run { background-color: #B0FFB0; }.cython.line .mis { background-color: #FFB0B0; }.cython.code.run  { border-left: 8px solid #B0FFB0; }.cython.code.mis  { border-left: 8px solid #FFB0B0; }

.cython.code .py_c_api  { color: red; }.cython.code .py_macro_api  { color: #FF7000; }.cython.code .pyx_c_api  { color: #FF3000; }.cython.code .pyx_macro_api  { color: #FF7000; }.cython.code .refnanny  { color: #FFA000; }.cython.code .trace  { color: #FFA000; }.cython.code .error_goto  { color: #FFA000; }

.cython.code .coerce  { color: #008000; border: 1px dotted #008000 }.cython.code .py_attr { color: #FF0000; font-weight: bold; }.cython.code .c_attr  { color: #0000FF; }.cython.code .py_call { color: #FF0000; font-weight: bold; }.cython.code .c_call  { color: #0000FF; }

.cython.score-0 {background-color: #FFFFff;}.cython.score-1 {background-color: #FFFFe7;}.cython.score-2 {background-color: #FFFFd4;}.cython.score-3 {background-color: #FFFFc4;}.cython.score-4 {background-color: #FFFFb6;}.cython.score-5 {background-color: #FFFFaa;}.cython.score-6 {background-color: #FFFF9f;}.cython.score-7 {background-color: #FFFF96;}.cython.score-8 {background-color: #FFFF8d;}.cython.score-9 {background-color: #FFFF86;}.cython.score-10 {background-color: #FFFF7f;}.cython.score-11 {background-color: #FFFF79;}.cython.score-12 {background-color: #FFFF73;}.cython.score-13 {background-color: #FFFF6e;}.cython.score-14 {background-color: #FFFF6a;}.cython.score-15 {background-color: #FFFF66;}.cython.score-16 {background-color: #FFFF62;}.cython.score-17 {background-color: #FFFF5e;}.cython.score-18 {background-color: #FFFF5b;}.cython.score-19 {background-color: #FFFF57;}.cython.score-20 {background-color: #FFFF55;}.cython.score-21 {background-color: #FFFF52;}.cython.score-22 {background-color: #FFFF4f;}.cython.score-23 {background-color: #FFFF4d;}.cython.score-24 {background-color: #FFFF4b;}.cython.score-25 {background-color: #FFFF48;}.cython.score-26 {background-color: #FFFF46;}.cython.score-27 {background-color: #FFFF44;}.cython.score-28 {background-color: #FFFF43;}.cython.score-29 {background-color: #FFFF41;}.cython.score-30 {background-color: #FFFF3f;}.cython.score-31 {background-color: #FFFF3e;}.cython.score-32 {background-color: #FFFF3c;}.cython.score-33 {background-color: #FFFF3b;}.cython.score-34 {background-color: #FFFF39;}.cython.score-35 {background-color: #FFFF38;}.cython.score-36 {background-color: #FFFF37;}.cython.score-37 {background-color: #FFFF36;}.cython.score-38 {background-color: #FFFF35;}.cython.score-39 {background-color: #FFFF34;}.cython.score-40 {background-color: #FFFF33;}.cython.score-41 {background-color: #FFFF32;}.cython.score-42 {background-color: #FFFF31;}.cython.score-43 {background-color: #FFFF30;}.cython.score-44 {background-color: #FFFF2f;}.cython.score-45 {background-color: #FFFF2e;}.cython.score-46 {background-color: #FFFF2d;}.cython.score-47 {background-color: #FFFF2c;}.cython.score-48 {background-color: #FFFF2b;}.cython.score-49 {background-color: #FFFF2b;}.cython.score-50 {background-color: #FFFF2a;}.cython.score-51 {background-color: #FFFF29;}.cython.score-52 {background-color: #FFFF29;}.cython.score-53 {background-color: #FFFF28;}.cython.score-54 {background-color: #FFFF27;}.cython.score-55 {background-color: #FFFF27;}.cython.score-56 {background-color: #FFFF26;}.cython.score-57 {background-color: #FFFF26;}.cython.score-58 {background-color: #FFFF25;}.cython.score-59 {background-color: #FFFF24;}.cython.score-60 {background-color: #FFFF24;}.cython.score-61 {background-color: #FFFF23;}.cython.score-62 {background-color: #FFFF23;}.cython.score-63 {background-color: #FFFF22;}.cython.score-64 {background-color: #FFFF22;}.cython.score-65 {background-color: #FFFF22;}.cython.score-66 {background-color: #FFFF21;}.cython.score-67 {background-color: #FFFF21;}.cython.score-68 {background-color: #FFFF20;}.cython.score-69 {background-color: #FFFF20;}.cython.score-70 {background-color: #FFFF1f;}.cython.score-71 {background-color: #FFFF1f;}.cython.score-72 {background-color: #FFFF1f;}.cython.score-73 {background-color: #FFFF1e;}.cython.score-74 {background-color: #FFFF1e;}.cython.score-75 {background-color: #FFFF1e;}.cython.score-76 {background-color: #FFFF1d;}.cython.score-77 {background-color: #FFFF1d;}.cython.score-78 {background-color: #FFFF1c;}.cython.score-79 {background-color: #FFFF1c;}.cython.score-80 {background-color: #FFFF1c;}.cython.score-81 {background-color: #FFFF1c;}.cython.score-82 {background-color: #FFFF1b;}.cython.score-83 {background-color: #FFFF1b;}.cython.score-84 {background-color: #FFFF1b;}.cython.score-85 {background-color: #FFFF1a;}.cython.score-86 {background-color: #FFFF1a;}.cython.score-87 {background-color: #FFFF1a;}.cython.score-88 {background-color: #FFFF1a;}.cython.score-89 {background-color: #FFFF19;}.cython.score-90 {background-color: #FFFF19;}.cython.score-91 {background-color: #FFFF19;}.cython.score-92 {background-color: #FFFF19;}.cython.score-93 {background-color: #FFFF18;}.cython.score-94 {background-color: #FFFF18;}.cython.score-95 {background-color: #FFFF18;}.cython.score-96 {background-color: #FFFF18;}.cython.score-97 {background-color: #FFFF17;}.cython.score-98 {background-color: #FFFF17;}.cython.score-99 {background-color: #FFFF17;}.cython.score-100 {background-color: #FFFF17;}.cython.score-101 {background-color: #FFFF16;}.cython.score-102 {background-color: #FFFF16;}.cython.score-103 {background-color: #FFFF16;}.cython.score-104 {background-color: #FFFF16;}.cython.score-105 {background-color: #FFFF16;}.cython.score-106 {background-color: #FFFF15;}.cython.score-107 {background-color: #FFFF15;}.cython.score-108 {background-color: #FFFF15;}.cython.score-109 {background-color: #FFFF15;}.cython.score-110 {background-color: #FFFF15;}.cython.score-111 {background-color: #FFFF15;}.cython.score-112 {background-color: #FFFF14;}.cython.score-113 {background-color: #FFFF14;}.cython.score-114 {background-color: #FFFF14;}.cython.score-115 {background-color: #FFFF14;}.cython.score-116 {background-color: #FFFF14;}.cython.score-117 {background-color: #FFFF14;}.cython.score-118 {background-color: #FFFF13;}.cython.score-119 {background-color: #FFFF13;}.cython.score-120 {background-color: #FFFF13;}.cython.score-121 {background-color: #FFFF13;}.cython.score-122 {background-color: #FFFF13;}.cython.score-123 {background-color: #FFFF13;}.cython.score-124 {background-color: #FFFF13;}.cython.score-125 {background-color: #FFFF12;}.cython.score-126 {background-color: #FFFF12;}.cython.score-127 {background-color: #FFFF12;}.cython.score-128 {background-color: #FFFF12;}.cython.score-129 {background-color: #FFFF12;}.cython.score-130 {background-color: #FFFF12;}.cython.score-131 {background-color: #FFFF12;}.cython.score-132 {background-color: #FFFF11;}.cython.score-133 {background-color: #FFFF11;}.cython.score-134 {background-color: #FFFF11;}.cython.score-135 {background-color: #FFFF11;}.cython.score-136 {background-color: #FFFF11;}.cython.score-137 {background-color: #FFFF11;}.cython.score-138 {background-color: #FFFF11;}.cython.score-139 {background-color: #FFFF11;}.cython.score-140 {background-color: #FFFF11;}.cython.score-141 {background-color: #FFFF10;}.cython.score-142 {background-color: #FFFF10;}.cython.score-143 {background-color: #FFFF10;}.cython.score-144 {background-color: #FFFF10;}.cython.score-145 {background-color: #FFFF10;}.cython.score-146 {background-color: #FFFF10;}.cython.score-147 {background-color: #FFFF10;}.cython.score-148 {background-color: #FFFF10;}.cython.score-149 {background-color: #FFFF10;}.cython.score-150 {background-color: #FFFF0f;}.cython.score-151 {background-color: #FFFF0f;}.cython.score-152 {background-color: #FFFF0f;}.cython.score-153 {background-color: #FFFF0f;}.cython.score-154 {background-color: #FFFF0f;}.cython.score-155 {background-color: #FFFF0f;}.cython.score-156 {background-color: #FFFF0f;}.cython.score-157 {background-color: #FFFF0f;}.cython.score-158 {background-color: #FFFF0f;}.cython.score-159 {background-color: #FFFF0f;}.cython.score-160 {background-color: #FFFF0f;}.cython.score-161 {background-color: #FFFF0e;}.cython.score-162 {background-color: #FFFF0e;}.cython.score-163 {background-color: #FFFF0e;}.cython.score-164 {background-color: #FFFF0e;}.cython.score-165 {background-color: #FFFF0e;}.cython.score-166 {background-color: #FFFF0e;}.cython.score-167 {background-color: #FFFF0e;}.cython.score-168 {background-color: #FFFF0e;}.cython.score-169 {background-color: #FFFF0e;}.cython.score-170 {background-color: #FFFF0e;}.cython.score-171 {background-color: #FFFF0e;}.cython.score-172 {background-color: #FFFF0e;}.cython.score-173 {background-color: #FFFF0d;}.cython.score-174 {background-color: #FFFF0d;}.cython.score-175 {background-color: #FFFF0d;}.cython.score-176 {background-color: #FFFF0d;}.cython.score-177 {background-color: #FFFF0d;}.cython.score-178 {background-color: #FFFF0d;}.cython.score-179 {background-color: #FFFF0d;}.cython.score-180 {background-color: #FFFF0d;}.cython.score-181 {background-color: #FFFF0d;}.cython.score-182 {background-color: #FFFF0d;}.cython.score-183 {background-color: #FFFF0d;}.cython.score-184 {background-color: #FFFF0d;}.cython.score-185 {background-color: #FFFF0d;}.cython.score-186 {background-color: #FFFF0d;}.cython.score-187 {background-color: #FFFF0c;}.cython.score-188 {background-color: #FFFF0c;}.cython.score-189 {background-color: #FFFF0c;}.cython.score-190 {background-color: #FFFF0c;}.cython.score-191 {background-color: #FFFF0c;}.cython.score-192 {background-color: #FFFF0c;}.cython.score-193 {background-color: #FFFF0c;}.cython.score-194 {background-color: #FFFF0c;}.cython.score-195 {background-color: #FFFF0c;}.cython.score-196 {background-color: #FFFF0c;}.cython.score-197 {background-color: #FFFF0c;}.cython.score-198 {background-color: #FFFF0c;}.cython.score-199 {background-color: #FFFF0c;}.cython.score-200 {background-color: #FFFF0c;}.cython.score-201 {background-color: #FFFF0c;}.cython.score-202 {background-color: #FFFF0c;}.cython.score-203 {background-color: #FFFF0b;}.cython.score-204 {background-color: #FFFF0b;}.cython.score-205 {background-color: #FFFF0b;}.cython.score-206 {background-color: #FFFF0b;}.cython.score-207 {background-color: #FFFF0b;}.cython.score-208 {background-color: #FFFF0b;}.cython.score-209 {background-color: #FFFF0b;}.cython.score-210 {background-color: #FFFF0b;}.cython.score-211 {background-color: #FFFF0b;}.cython.score-212 {background-color: #FFFF0b;}.cython.score-213 {background-color: #FFFF0b;}.cython.score-214 {background-color: #FFFF0b;}.cython.score-215 {background-color: #FFFF0b;}.cython.score-216 {background-color: #FFFF0b;}.cython.score-217 {background-color: #FFFF0b;}.cython.score-218 {background-color: #FFFF0b;}.cython.score-219 {background-color: #FFFF0b;}.cython.score-220 {background-color: #FFFF0b;}.cython.score-221 {background-color: #FFFF0b;}.cython.score-222 {background-color: #FFFF0a;}.cython.score-223 {background-color: #FFFF0a;}.cython.score-224 {background-color: #FFFF0a;}.cython.score-225 {background-color: #FFFF0a;}.cython.score-226 {background-color: #FFFF0a;}.cython.score-227 {background-color: #FFFF0a;}.cython.score-228 {background-color: #FFFF0a;}.cython.score-229 {background-color: #FFFF0a;}.cython.score-230 {background-color: #FFFF0a;}.cython.score-231 {background-color: #FFFF0a;}.cython.score-232 {background-color: #FFFF0a;}.cython.score-233 {background-color: #FFFF0a;}.cython.score-234 {background-color: #FFFF0a;}.cython.score-235 {background-color: #FFFF0a;}.cython.score-236 {background-color: #FFFF0a;}.cython.score-237 {background-color: #FFFF0a;}.cython.score-238 {background-color: #FFFF0a;}.cython.score-239 {background-color: #FFFF0a;}.cython.score-240 {background-color: #FFFF0a;}.cython.score-241 {background-color: #FFFF0a;}.cython.score-242 {background-color: #FFFF0a;}.cython.score-243 {background-color: #FFFF0a;}.cython.score-244 {background-color: #FFFF0a;}.cython.score-245 {background-color: #FFFF0a;}.cython.score-246 {background-color: #FFFF09;}.cython.score-247 {background-color: #FFFF09;}.cython.score-248 {background-color: #FFFF09;}.cython.score-249 {background-color: #FFFF09;}.cython.score-250 {background-color: #FFFF09;}.cython.score-251 {background-color: #FFFF09;}.cython.score-252 {background-color: #FFFF09;}.cython.score-253 {background-color: #FFFF09;}.cython.score-254 {background-color: #FFFF09;}pre { line-height: 125%; }td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }.cython .hll { background-color: #ffffcc }.cython { background: #f8f8f8; }.cython .c { color: #3D7B7B; font-style: italic } /* Comment */.cython .err { border: 1px solid #FF0000 } /* Error */.cython .k { color: #008000; font-weight: bold } /* Keyword */.cython .o { color: #666666 } /* Operator */.cython .ch { color: #3D7B7B; font-style: italic } /* Comment.Hashbang */.cython .cm { color: #3D7B7B; font-style: italic } /* Comment.Multiline */.cython .cp { color: #9C6500 } /* Comment.Preproc */.cython .cpf { color: #3D7B7B; font-style: italic } /* Comment.PreprocFile */.cython .c1 { color: #3D7B7B; font-style: italic } /* Comment.Single */.cython .cs { color: #3D7B7B; font-style: italic } /* Comment.Special */.cython .gd { color: #A00000 } /* Generic.Deleted */.cython .ge { font-style: italic } /* Generic.Emph */.cython .gr { color: #E40000 } /* Generic.Error */.cython .gh { color: #000080; font-weight: bold } /* Generic.Heading */.cython .gi { color: #008400 } /* Generic.Inserted */.cython .go { color: #717171 } /* Generic.Output */.cython .gp { color: #000080; font-weight: bold } /* Generic.Prompt */.cython .gs { font-weight: bold } /* Generic.Strong */.cython .gu { color: #800080; font-weight: bold } /* Generic.Subheading */.cython .gt { color: #0044DD } /* Generic.Traceback */.cython .kc { color: #008000; font-weight: bold } /* Keyword.Constant */.cython .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */.cython .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */.cython .kp { color: #008000 } /* Keyword.Pseudo */.cython .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */.cython .kt { color: #B00040 } /* Keyword.Type */.cython .m { color: #666666 } /* Literal.Number */.cython .s { color: #BA2121 } /* Literal.String */.cython .na { color: #687822 } /* Name.Attribute */.cython .nb { color: #008000 } /* Name.Builtin */.cython .nc { color: #0000FF; font-weight: bold } /* Name.Class */.cython .no { color: #880000 } /* Name.Constant */.cython .nd { color: #AA22FF } /* Name.Decorator */.cython .ni { color: #717171; font-weight: bold } /* Name.Entity */.cython .ne { color: #CB3F38; font-weight: bold } /* Name.Exception */.cython .nf { color: #0000FF } /* Name.Function */.cython .nl { color: #767600 } /* Name.Label */.cython .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */.cython .nt { color: #008000; font-weight: bold } /* Name.Tag */.cython .nv { color: #19177C } /* Name.Variable */.cython .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */.cython .w { color: #bbbbbb } /* Text.Whitespace */.cython .mb { color: #666666 } /* Literal.Number.Bin */.cython .mf { color: #666666 } /* Literal.Number.Float */.cython .mh { color: #666666 } /* Literal.Number.Hex */.cython .mi { color: #666666 } /* Literal.Number.Integer */.cython .mo { color: #666666 } /* Literal.Number.Oct */.cython .sa { color: #BA2121 } /* Literal.String.Affix */.cython .sb { color: #BA2121 } /* Literal.String.Backtick */.cython .sc { color: #BA2121 } /* Literal.String.Char */.cython .dl { color: #BA2121 } /* Literal.String.Delimiter */.cython .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */.cython .s2 { color: #BA2121 } /* Literal.String.Double */.cython .se { color: #AA5D1F; font-weight: bold } /* Literal.String.Escape */.cython .sh { color: #BA2121 } /* Literal.String.Heredoc */.cython .si { color: #A45A77; font-weight: bold } /* Literal.String.Interpol */.cython .sx { color: #008000 } /* Literal.String.Other */.cython .sr { color: #A45A77 } /* Literal.String.Regex */.cython .s1 { color: #BA2121 } /* Literal.String.Single */.cython .ss { color: #19177C } /* Literal.String.Symbol */.cython .bp { color: #008000 } /* Name.Builtin.Pseudo */.cython .fm { color: #0000FF } /* Name.Function.Magic */.cython .vc { color: #19177C } /* Name.Variable.Class */.cython .vg { color: #19177C } /* Name.Variable.Global */.cython .vi { color: #19177C } /* Name.Variable.Instance */.cython .vm { color: #19177C } /* Name.Variable.Magic */.cython .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
</head>
<body class="cython">
<p><span style="border-bottom: solid 1px grey;">Generated by Cython 0.29.30</span></p>
<p>
    <span style="background-color: #FFFF00">Yellow lines</span> hint at Python interaction.<br />
    Click on a line that starts with a "<code>+</code>" to see the C code that Cython generated for it.
</p>
<div class="cython"><pre class="cython line score-0">&#xA0;<span class="">1</span>: </pre>
<pre class="cython line score-0" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">2</span>: <span class="k">cdef</span> <span class="kt">int</span> <span class="nf">a</span> <span class="o">=</span> <span class="mf">0</span></pre>
<pre class='cython code score-0 '>  __pyx_v_46_cython_magic_5ab44e0a8b28fe78d74e9d2f9669b963_a = 0;
</pre><pre class="cython line score-8" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">3</span>: <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mf">10</span><span class="p">):</span></pre>
<pre class='cython code score-8 '>  for (__pyx_t_1 = 0; __pyx_t_1 &lt; 10; __pyx_t_1+=1) {
    __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyInt_From_long</span>(__pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 3, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
    if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_i, __pyx_t_2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 3, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
</pre><pre class="cython line score-19" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">4</span>:     <span class="n">a</span> <span class="o">+=</span> <span class="n">i</span></pre>
<pre class='cython code score-19 '>    __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyInt_From_int</span>(__pyx_v_46_cython_magic_5ab44e0a8b28fe78d74e9d2f9669b963_a);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
    <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_3, __pyx_n_s_i);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
    __pyx_t_4 = <span class='py_c_api'>PyNumber_InPlaceAdd</span>(__pyx_t_2, __pyx_t_3);<span class='error_goto'> if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_4);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyInt_As_int</span>(__pyx_t_4); if (unlikely((__pyx_t_5 == (int)-1) &amp;&amp; <span class='py_c_api'>PyErr_Occurred</span>())) <span class='error_goto'>__PYX_ERR(0, 4, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_4); __pyx_t_4 = 0;
    __pyx_v_46_cython_magic_5ab44e0a8b28fe78d74e9d2f9669b963_a = __pyx_t_5;
  }
</pre><pre class="cython line score-6" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">5</span>: <span class="k">print</span><span class="p">(</span><span class="n">a</span><span class="p">)</span></pre>
<pre class='cython code score-6 '>  __pyx_t_4 = <span class='pyx_c_api'>__Pyx_PyInt_From_int</span>(__pyx_v_46_cython_magic_5ab44e0a8b28fe78d74e9d2f9669b963_a);<span class='error_goto'> if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_4);
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_CallOneArg</span>(__pyx_builtin_print, __pyx_t_4);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_4); __pyx_t_4 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
</pre></div></body></html>



以下は純粋なPython版を示していますが、この場合はタイプタグが必要です。



```cython
%%cython --annotate

a: cython.int = 0
for i in range(10):
    a += i
print(a)
```

    45
    




<!DOCTYPE html>
<!-- Generated by Cython 0.29.30 -->
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <title>Cython: _cython_magic_ba8df6c4aec618ea3f81dafd9d00c6a3.pyx</title>
    <style type="text/css">

body.cython { font-family: courier; font-size: 12; }

.cython.tag  {  }.cython.line { margin: 0em }.cython.code { font-size: 9; color: #444444; display: none; margin: 0px 0px 0px 8px; border-left: 8px none; }

.cython.line .run { background-color: #B0FFB0; }.cython.line .mis { background-color: #FFB0B0; }.cython.code.run  { border-left: 8px solid #B0FFB0; }.cython.code.mis  { border-left: 8px solid #FFB0B0; }

.cython.code .py_c_api  { color: red; }.cython.code .py_macro_api  { color: #FF7000; }.cython.code .pyx_c_api  { color: #FF3000; }.cython.code .pyx_macro_api  { color: #FF7000; }.cython.code .refnanny  { color: #FFA000; }.cython.code .trace  { color: #FFA000; }.cython.code .error_goto  { color: #FFA000; }

.cython.code .coerce  { color: #008000; border: 1px dotted #008000 }.cython.code .py_attr { color: #FF0000; font-weight: bold; }.cython.code .c_attr  { color: #0000FF; }.cython.code .py_call { color: #FF0000; font-weight: bold; }.cython.code .c_call  { color: #0000FF; }

.cython.score-0 {background-color: #FFFFff;}.cython.score-1 {background-color: #FFFFe7;}.cython.score-2 {background-color: #FFFFd4;}.cython.score-3 {background-color: #FFFFc4;}.cython.score-4 {background-color: #FFFFb6;}.cython.score-5 {background-color: #FFFFaa;}.cython.score-6 {background-color: #FFFF9f;}.cython.score-7 {background-color: #FFFF96;}.cython.score-8 {background-color: #FFFF8d;}.cython.score-9 {background-color: #FFFF86;}.cython.score-10 {background-color: #FFFF7f;}.cython.score-11 {background-color: #FFFF79;}.cython.score-12 {background-color: #FFFF73;}.cython.score-13 {background-color: #FFFF6e;}.cython.score-14 {background-color: #FFFF6a;}.cython.score-15 {background-color: #FFFF66;}.cython.score-16 {background-color: #FFFF62;}.cython.score-17 {background-color: #FFFF5e;}.cython.score-18 {background-color: #FFFF5b;}.cython.score-19 {background-color: #FFFF57;}.cython.score-20 {background-color: #FFFF55;}.cython.score-21 {background-color: #FFFF52;}.cython.score-22 {background-color: #FFFF4f;}.cython.score-23 {background-color: #FFFF4d;}.cython.score-24 {background-color: #FFFF4b;}.cython.score-25 {background-color: #FFFF48;}.cython.score-26 {background-color: #FFFF46;}.cython.score-27 {background-color: #FFFF44;}.cython.score-28 {background-color: #FFFF43;}.cython.score-29 {background-color: #FFFF41;}.cython.score-30 {background-color: #FFFF3f;}.cython.score-31 {background-color: #FFFF3e;}.cython.score-32 {background-color: #FFFF3c;}.cython.score-33 {background-color: #FFFF3b;}.cython.score-34 {background-color: #FFFF39;}.cython.score-35 {background-color: #FFFF38;}.cython.score-36 {background-color: #FFFF37;}.cython.score-37 {background-color: #FFFF36;}.cython.score-38 {background-color: #FFFF35;}.cython.score-39 {background-color: #FFFF34;}.cython.score-40 {background-color: #FFFF33;}.cython.score-41 {background-color: #FFFF32;}.cython.score-42 {background-color: #FFFF31;}.cython.score-43 {background-color: #FFFF30;}.cython.score-44 {background-color: #FFFF2f;}.cython.score-45 {background-color: #FFFF2e;}.cython.score-46 {background-color: #FFFF2d;}.cython.score-47 {background-color: #FFFF2c;}.cython.score-48 {background-color: #FFFF2b;}.cython.score-49 {background-color: #FFFF2b;}.cython.score-50 {background-color: #FFFF2a;}.cython.score-51 {background-color: #FFFF29;}.cython.score-52 {background-color: #FFFF29;}.cython.score-53 {background-color: #FFFF28;}.cython.score-54 {background-color: #FFFF27;}.cython.score-55 {background-color: #FFFF27;}.cython.score-56 {background-color: #FFFF26;}.cython.score-57 {background-color: #FFFF26;}.cython.score-58 {background-color: #FFFF25;}.cython.score-59 {background-color: #FFFF24;}.cython.score-60 {background-color: #FFFF24;}.cython.score-61 {background-color: #FFFF23;}.cython.score-62 {background-color: #FFFF23;}.cython.score-63 {background-color: #FFFF22;}.cython.score-64 {background-color: #FFFF22;}.cython.score-65 {background-color: #FFFF22;}.cython.score-66 {background-color: #FFFF21;}.cython.score-67 {background-color: #FFFF21;}.cython.score-68 {background-color: #FFFF20;}.cython.score-69 {background-color: #FFFF20;}.cython.score-70 {background-color: #FFFF1f;}.cython.score-71 {background-color: #FFFF1f;}.cython.score-72 {background-color: #FFFF1f;}.cython.score-73 {background-color: #FFFF1e;}.cython.score-74 {background-color: #FFFF1e;}.cython.score-75 {background-color: #FFFF1e;}.cython.score-76 {background-color: #FFFF1d;}.cython.score-77 {background-color: #FFFF1d;}.cython.score-78 {background-color: #FFFF1c;}.cython.score-79 {background-color: #FFFF1c;}.cython.score-80 {background-color: #FFFF1c;}.cython.score-81 {background-color: #FFFF1c;}.cython.score-82 {background-color: #FFFF1b;}.cython.score-83 {background-color: #FFFF1b;}.cython.score-84 {background-color: #FFFF1b;}.cython.score-85 {background-color: #FFFF1a;}.cython.score-86 {background-color: #FFFF1a;}.cython.score-87 {background-color: #FFFF1a;}.cython.score-88 {background-color: #FFFF1a;}.cython.score-89 {background-color: #FFFF19;}.cython.score-90 {background-color: #FFFF19;}.cython.score-91 {background-color: #FFFF19;}.cython.score-92 {background-color: #FFFF19;}.cython.score-93 {background-color: #FFFF18;}.cython.score-94 {background-color: #FFFF18;}.cython.score-95 {background-color: #FFFF18;}.cython.score-96 {background-color: #FFFF18;}.cython.score-97 {background-color: #FFFF17;}.cython.score-98 {background-color: #FFFF17;}.cython.score-99 {background-color: #FFFF17;}.cython.score-100 {background-color: #FFFF17;}.cython.score-101 {background-color: #FFFF16;}.cython.score-102 {background-color: #FFFF16;}.cython.score-103 {background-color: #FFFF16;}.cython.score-104 {background-color: #FFFF16;}.cython.score-105 {background-color: #FFFF16;}.cython.score-106 {background-color: #FFFF15;}.cython.score-107 {background-color: #FFFF15;}.cython.score-108 {background-color: #FFFF15;}.cython.score-109 {background-color: #FFFF15;}.cython.score-110 {background-color: #FFFF15;}.cython.score-111 {background-color: #FFFF15;}.cython.score-112 {background-color: #FFFF14;}.cython.score-113 {background-color: #FFFF14;}.cython.score-114 {background-color: #FFFF14;}.cython.score-115 {background-color: #FFFF14;}.cython.score-116 {background-color: #FFFF14;}.cython.score-117 {background-color: #FFFF14;}.cython.score-118 {background-color: #FFFF13;}.cython.score-119 {background-color: #FFFF13;}.cython.score-120 {background-color: #FFFF13;}.cython.score-121 {background-color: #FFFF13;}.cython.score-122 {background-color: #FFFF13;}.cython.score-123 {background-color: #FFFF13;}.cython.score-124 {background-color: #FFFF13;}.cython.score-125 {background-color: #FFFF12;}.cython.score-126 {background-color: #FFFF12;}.cython.score-127 {background-color: #FFFF12;}.cython.score-128 {background-color: #FFFF12;}.cython.score-129 {background-color: #FFFF12;}.cython.score-130 {background-color: #FFFF12;}.cython.score-131 {background-color: #FFFF12;}.cython.score-132 {background-color: #FFFF11;}.cython.score-133 {background-color: #FFFF11;}.cython.score-134 {background-color: #FFFF11;}.cython.score-135 {background-color: #FFFF11;}.cython.score-136 {background-color: #FFFF11;}.cython.score-137 {background-color: #FFFF11;}.cython.score-138 {background-color: #FFFF11;}.cython.score-139 {background-color: #FFFF11;}.cython.score-140 {background-color: #FFFF11;}.cython.score-141 {background-color: #FFFF10;}.cython.score-142 {background-color: #FFFF10;}.cython.score-143 {background-color: #FFFF10;}.cython.score-144 {background-color: #FFFF10;}.cython.score-145 {background-color: #FFFF10;}.cython.score-146 {background-color: #FFFF10;}.cython.score-147 {background-color: #FFFF10;}.cython.score-148 {background-color: #FFFF10;}.cython.score-149 {background-color: #FFFF10;}.cython.score-150 {background-color: #FFFF0f;}.cython.score-151 {background-color: #FFFF0f;}.cython.score-152 {background-color: #FFFF0f;}.cython.score-153 {background-color: #FFFF0f;}.cython.score-154 {background-color: #FFFF0f;}.cython.score-155 {background-color: #FFFF0f;}.cython.score-156 {background-color: #FFFF0f;}.cython.score-157 {background-color: #FFFF0f;}.cython.score-158 {background-color: #FFFF0f;}.cython.score-159 {background-color: #FFFF0f;}.cython.score-160 {background-color: #FFFF0f;}.cython.score-161 {background-color: #FFFF0e;}.cython.score-162 {background-color: #FFFF0e;}.cython.score-163 {background-color: #FFFF0e;}.cython.score-164 {background-color: #FFFF0e;}.cython.score-165 {background-color: #FFFF0e;}.cython.score-166 {background-color: #FFFF0e;}.cython.score-167 {background-color: #FFFF0e;}.cython.score-168 {background-color: #FFFF0e;}.cython.score-169 {background-color: #FFFF0e;}.cython.score-170 {background-color: #FFFF0e;}.cython.score-171 {background-color: #FFFF0e;}.cython.score-172 {background-color: #FFFF0e;}.cython.score-173 {background-color: #FFFF0d;}.cython.score-174 {background-color: #FFFF0d;}.cython.score-175 {background-color: #FFFF0d;}.cython.score-176 {background-color: #FFFF0d;}.cython.score-177 {background-color: #FFFF0d;}.cython.score-178 {background-color: #FFFF0d;}.cython.score-179 {background-color: #FFFF0d;}.cython.score-180 {background-color: #FFFF0d;}.cython.score-181 {background-color: #FFFF0d;}.cython.score-182 {background-color: #FFFF0d;}.cython.score-183 {background-color: #FFFF0d;}.cython.score-184 {background-color: #FFFF0d;}.cython.score-185 {background-color: #FFFF0d;}.cython.score-186 {background-color: #FFFF0d;}.cython.score-187 {background-color: #FFFF0c;}.cython.score-188 {background-color: #FFFF0c;}.cython.score-189 {background-color: #FFFF0c;}.cython.score-190 {background-color: #FFFF0c;}.cython.score-191 {background-color: #FFFF0c;}.cython.score-192 {background-color: #FFFF0c;}.cython.score-193 {background-color: #FFFF0c;}.cython.score-194 {background-color: #FFFF0c;}.cython.score-195 {background-color: #FFFF0c;}.cython.score-196 {background-color: #FFFF0c;}.cython.score-197 {background-color: #FFFF0c;}.cython.score-198 {background-color: #FFFF0c;}.cython.score-199 {background-color: #FFFF0c;}.cython.score-200 {background-color: #FFFF0c;}.cython.score-201 {background-color: #FFFF0c;}.cython.score-202 {background-color: #FFFF0c;}.cython.score-203 {background-color: #FFFF0b;}.cython.score-204 {background-color: #FFFF0b;}.cython.score-205 {background-color: #FFFF0b;}.cython.score-206 {background-color: #FFFF0b;}.cython.score-207 {background-color: #FFFF0b;}.cython.score-208 {background-color: #FFFF0b;}.cython.score-209 {background-color: #FFFF0b;}.cython.score-210 {background-color: #FFFF0b;}.cython.score-211 {background-color: #FFFF0b;}.cython.score-212 {background-color: #FFFF0b;}.cython.score-213 {background-color: #FFFF0b;}.cython.score-214 {background-color: #FFFF0b;}.cython.score-215 {background-color: #FFFF0b;}.cython.score-216 {background-color: #FFFF0b;}.cython.score-217 {background-color: #FFFF0b;}.cython.score-218 {background-color: #FFFF0b;}.cython.score-219 {background-color: #FFFF0b;}.cython.score-220 {background-color: #FFFF0b;}.cython.score-221 {background-color: #FFFF0b;}.cython.score-222 {background-color: #FFFF0a;}.cython.score-223 {background-color: #FFFF0a;}.cython.score-224 {background-color: #FFFF0a;}.cython.score-225 {background-color: #FFFF0a;}.cython.score-226 {background-color: #FFFF0a;}.cython.score-227 {background-color: #FFFF0a;}.cython.score-228 {background-color: #FFFF0a;}.cython.score-229 {background-color: #FFFF0a;}.cython.score-230 {background-color: #FFFF0a;}.cython.score-231 {background-color: #FFFF0a;}.cython.score-232 {background-color: #FFFF0a;}.cython.score-233 {background-color: #FFFF0a;}.cython.score-234 {background-color: #FFFF0a;}.cython.score-235 {background-color: #FFFF0a;}.cython.score-236 {background-color: #FFFF0a;}.cython.score-237 {background-color: #FFFF0a;}.cython.score-238 {background-color: #FFFF0a;}.cython.score-239 {background-color: #FFFF0a;}.cython.score-240 {background-color: #FFFF0a;}.cython.score-241 {background-color: #FFFF0a;}.cython.score-242 {background-color: #FFFF0a;}.cython.score-243 {background-color: #FFFF0a;}.cython.score-244 {background-color: #FFFF0a;}.cython.score-245 {background-color: #FFFF0a;}.cython.score-246 {background-color: #FFFF09;}.cython.score-247 {background-color: #FFFF09;}.cython.score-248 {background-color: #FFFF09;}.cython.score-249 {background-color: #FFFF09;}.cython.score-250 {background-color: #FFFF09;}.cython.score-251 {background-color: #FFFF09;}.cython.score-252 {background-color: #FFFF09;}.cython.score-253 {background-color: #FFFF09;}.cython.score-254 {background-color: #FFFF09;}pre { line-height: 125%; }td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }.cython .hll { background-color: #ffffcc }.cython { background: #f8f8f8; }.cython .c { color: #3D7B7B; font-style: italic } /* Comment */.cython .err { border: 1px solid #FF0000 } /* Error */.cython .k { color: #008000; font-weight: bold } /* Keyword */.cython .o { color: #666666 } /* Operator */.cython .ch { color: #3D7B7B; font-style: italic } /* Comment.Hashbang */.cython .cm { color: #3D7B7B; font-style: italic } /* Comment.Multiline */.cython .cp { color: #9C6500 } /* Comment.Preproc */.cython .cpf { color: #3D7B7B; font-style: italic } /* Comment.PreprocFile */.cython .c1 { color: #3D7B7B; font-style: italic } /* Comment.Single */.cython .cs { color: #3D7B7B; font-style: italic } /* Comment.Special */.cython .gd { color: #A00000 } /* Generic.Deleted */.cython .ge { font-style: italic } /* Generic.Emph */.cython .gr { color: #E40000 } /* Generic.Error */.cython .gh { color: #000080; font-weight: bold } /* Generic.Heading */.cython .gi { color: #008400 } /* Generic.Inserted */.cython .go { color: #717171 } /* Generic.Output */.cython .gp { color: #000080; font-weight: bold } /* Generic.Prompt */.cython .gs { font-weight: bold } /* Generic.Strong */.cython .gu { color: #800080; font-weight: bold } /* Generic.Subheading */.cython .gt { color: #0044DD } /* Generic.Traceback */.cython .kc { color: #008000; font-weight: bold } /* Keyword.Constant */.cython .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */.cython .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */.cython .kp { color: #008000 } /* Keyword.Pseudo */.cython .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */.cython .kt { color: #B00040 } /* Keyword.Type */.cython .m { color: #666666 } /* Literal.Number */.cython .s { color: #BA2121 } /* Literal.String */.cython .na { color: #687822 } /* Name.Attribute */.cython .nb { color: #008000 } /* Name.Builtin */.cython .nc { color: #0000FF; font-weight: bold } /* Name.Class */.cython .no { color: #880000 } /* Name.Constant */.cython .nd { color: #AA22FF } /* Name.Decorator */.cython .ni { color: #717171; font-weight: bold } /* Name.Entity */.cython .ne { color: #CB3F38; font-weight: bold } /* Name.Exception */.cython .nf { color: #0000FF } /* Name.Function */.cython .nl { color: #767600 } /* Name.Label */.cython .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */.cython .nt { color: #008000; font-weight: bold } /* Name.Tag */.cython .nv { color: #19177C } /* Name.Variable */.cython .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */.cython .w { color: #bbbbbb } /* Text.Whitespace */.cython .mb { color: #666666 } /* Literal.Number.Bin */.cython .mf { color: #666666 } /* Literal.Number.Float */.cython .mh { color: #666666 } /* Literal.Number.Hex */.cython .mi { color: #666666 } /* Literal.Number.Integer */.cython .mo { color: #666666 } /* Literal.Number.Oct */.cython .sa { color: #BA2121 } /* Literal.String.Affix */.cython .sb { color: #BA2121 } /* Literal.String.Backtick */.cython .sc { color: #BA2121 } /* Literal.String.Char */.cython .dl { color: #BA2121 } /* Literal.String.Delimiter */.cython .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */.cython .s2 { color: #BA2121 } /* Literal.String.Double */.cython .se { color: #AA5D1F; font-weight: bold } /* Literal.String.Escape */.cython .sh { color: #BA2121 } /* Literal.String.Heredoc */.cython .si { color: #A45A77; font-weight: bold } /* Literal.String.Interpol */.cython .sx { color: #008000 } /* Literal.String.Other */.cython .sr { color: #A45A77 } /* Literal.String.Regex */.cython .s1 { color: #BA2121 } /* Literal.String.Single */.cython .ss { color: #19177C } /* Literal.String.Symbol */.cython .bp { color: #008000 } /* Name.Builtin.Pseudo */.cython .fm { color: #0000FF } /* Name.Function.Magic */.cython .vc { color: #19177C } /* Name.Variable.Class */.cython .vg { color: #19177C } /* Name.Variable.Global */.cython .vi { color: #19177C } /* Name.Variable.Instance */.cython .vm { color: #19177C } /* Name.Variable.Magic */.cython .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
</head>
<body class="cython">
<p><span style="border-bottom: solid 1px grey;">Generated by Cython 0.29.30</span></p>
<p>
    <span style="background-color: #FFFF00">Yellow lines</span> hint at Python interaction.<br />
    Click on a line that starts with a "<code>+</code>" to see the C code that Cython generated for it.
</p>
<div class="cython"><pre class="cython line score-0">&#xA0;<span class="">1</span>: </pre>
<pre class="cython line score-5" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">2</span>: <span class="n">a</span><span class="p">:</span> <span class="n">cython</span><span class="o">.</span><span class="n">int</span> <span class="o">=</span> <span class="mf">0</span></pre>
<pre class='cython code score-5 '>  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_a, __pyx_int_0) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 2, __pyx_L1_error)</span>
</pre><pre class="cython line score-8" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">3</span>: <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mf">10</span><span class="p">):</span></pre>
<pre class='cython code score-8 '>  for (__pyx_t_1 = 0; __pyx_t_1 &lt; 10; __pyx_t_1+=1) {
    __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyInt_From_long</span>(__pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 3, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
    if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_i, __pyx_t_2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 3, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
</pre><pre class="cython line score-17" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">4</span>:     <span class="n">a</span> <span class="o">+=</span> <span class="n">i</span></pre>
<pre class='cython code score-17 '>    <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_2, __pyx_n_s_a);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
    <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_3, __pyx_n_s_i);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
    __pyx_t_4 = <span class='py_c_api'>PyNumber_InPlaceAdd</span>(__pyx_t_2, __pyx_t_3);<span class='error_goto'> if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_4);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
    if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_a, __pyx_t_4) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 4, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_4); __pyx_t_4 = 0;
  }
</pre><pre class="cython line score-6" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">5</span>: <span class="k">print</span><span class="p">(</span><span class="n">a</span><span class="p">)</span></pre>
<pre class='cython code score-6 '>  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_4, __pyx_n_s_a);<span class='error_goto'> if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_4);
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_CallOneArg</span>(__pyx_builtin_print, __pyx_t_4);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_4); __pyx_t_4 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
</pre></div></body></html>



もちろん、純粋なPythonコードであっても、Cythonを使って先にコンパイルしてパフォーマンスを向上させることができます。ただし、パフォーマンスが重要なコードでは、静的型宣言を追加するのがよく便利です。



```cython
%%cython --annotate

a = 0
for i in range(10):
    a += i
print(a)
```

    45
    




<!DOCTYPE html>
<!-- Generated by Cython 0.29.30 -->
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <title>Cython: _cython_magic_bce3a6fd2b62e2d632d45840f0830867.pyx</title>
    <style type="text/css">

body.cython { font-family: courier; font-size: 12; }

.cython.tag  {  }.cython.line { margin: 0em }.cython.code { font-size: 9; color: #444444; display: none; margin: 0px 0px 0px 8px; border-left: 8px none; }

.cython.line .run { background-color: #B0FFB0; }.cython.line .mis { background-color: #FFB0B0; }.cython.code.run  { border-left: 8px solid #B0FFB0; }.cython.code.mis  { border-left: 8px solid #FFB0B0; }

.cython.code .py_c_api  { color: red; }.cython.code .py_macro_api  { color: #FF7000; }.cython.code .pyx_c_api  { color: #FF3000; }.cython.code .pyx_macro_api  { color: #FF7000; }.cython.code .refnanny  { color: #FFA000; }.cython.code .trace  { color: #FFA000; }.cython.code .error_goto  { color: #FFA000; }

.cython.code .coerce  { color: #008000; border: 1px dotted #008000 }.cython.code .py_attr { color: #FF0000; font-weight: bold; }.cython.code .c_attr  { color: #0000FF; }.cython.code .py_call { color: #FF0000; font-weight: bold; }.cython.code .c_call  { color: #0000FF; }

.cython.score-0 {background-color: #FFFFff;}.cython.score-1 {background-color: #FFFFe7;}.cython.score-2 {background-color: #FFFFd4;}.cython.score-3 {background-color: #FFFFc4;}.cython.score-4 {background-color: #FFFFb6;}.cython.score-5 {background-color: #FFFFaa;}.cython.score-6 {background-color: #FFFF9f;}.cython.score-7 {background-color: #FFFF96;}.cython.score-8 {background-color: #FFFF8d;}.cython.score-9 {background-color: #FFFF86;}.cython.score-10 {background-color: #FFFF7f;}.cython.score-11 {background-color: #FFFF79;}.cython.score-12 {background-color: #FFFF73;}.cython.score-13 {background-color: #FFFF6e;}.cython.score-14 {background-color: #FFFF6a;}.cython.score-15 {background-color: #FFFF66;}.cython.score-16 {background-color: #FFFF62;}.cython.score-17 {background-color: #FFFF5e;}.cython.score-18 {background-color: #FFFF5b;}.cython.score-19 {background-color: #FFFF57;}.cython.score-20 {background-color: #FFFF55;}.cython.score-21 {background-color: #FFFF52;}.cython.score-22 {background-color: #FFFF4f;}.cython.score-23 {background-color: #FFFF4d;}.cython.score-24 {background-color: #FFFF4b;}.cython.score-25 {background-color: #FFFF48;}.cython.score-26 {background-color: #FFFF46;}.cython.score-27 {background-color: #FFFF44;}.cython.score-28 {background-color: #FFFF43;}.cython.score-29 {background-color: #FFFF41;}.cython.score-30 {background-color: #FFFF3f;}.cython.score-31 {background-color: #FFFF3e;}.cython.score-32 {background-color: #FFFF3c;}.cython.score-33 {background-color: #FFFF3b;}.cython.score-34 {background-color: #FFFF39;}.cython.score-35 {background-color: #FFFF38;}.cython.score-36 {background-color: #FFFF37;}.cython.score-37 {background-color: #FFFF36;}.cython.score-38 {background-color: #FFFF35;}.cython.score-39 {background-color: #FFFF34;}.cython.score-40 {background-color: #FFFF33;}.cython.score-41 {background-color: #FFFF32;}.cython.score-42 {background-color: #FFFF31;}.cython.score-43 {background-color: #FFFF30;}.cython.score-44 {background-color: #FFFF2f;}.cython.score-45 {background-color: #FFFF2e;}.cython.score-46 {background-color: #FFFF2d;}.cython.score-47 {background-color: #FFFF2c;}.cython.score-48 {background-color: #FFFF2b;}.cython.score-49 {background-color: #FFFF2b;}.cython.score-50 {background-color: #FFFF2a;}.cython.score-51 {background-color: #FFFF29;}.cython.score-52 {background-color: #FFFF29;}.cython.score-53 {background-color: #FFFF28;}.cython.score-54 {background-color: #FFFF27;}.cython.score-55 {background-color: #FFFF27;}.cython.score-56 {background-color: #FFFF26;}.cython.score-57 {background-color: #FFFF26;}.cython.score-58 {background-color: #FFFF25;}.cython.score-59 {background-color: #FFFF24;}.cython.score-60 {background-color: #FFFF24;}.cython.score-61 {background-color: #FFFF23;}.cython.score-62 {background-color: #FFFF23;}.cython.score-63 {background-color: #FFFF22;}.cython.score-64 {background-color: #FFFF22;}.cython.score-65 {background-color: #FFFF22;}.cython.score-66 {background-color: #FFFF21;}.cython.score-67 {background-color: #FFFF21;}.cython.score-68 {background-color: #FFFF20;}.cython.score-69 {background-color: #FFFF20;}.cython.score-70 {background-color: #FFFF1f;}.cython.score-71 {background-color: #FFFF1f;}.cython.score-72 {background-color: #FFFF1f;}.cython.score-73 {background-color: #FFFF1e;}.cython.score-74 {background-color: #FFFF1e;}.cython.score-75 {background-color: #FFFF1e;}.cython.score-76 {background-color: #FFFF1d;}.cython.score-77 {background-color: #FFFF1d;}.cython.score-78 {background-color: #FFFF1c;}.cython.score-79 {background-color: #FFFF1c;}.cython.score-80 {background-color: #FFFF1c;}.cython.score-81 {background-color: #FFFF1c;}.cython.score-82 {background-color: #FFFF1b;}.cython.score-83 {background-color: #FFFF1b;}.cython.score-84 {background-color: #FFFF1b;}.cython.score-85 {background-color: #FFFF1a;}.cython.score-86 {background-color: #FFFF1a;}.cython.score-87 {background-color: #FFFF1a;}.cython.score-88 {background-color: #FFFF1a;}.cython.score-89 {background-color: #FFFF19;}.cython.score-90 {background-color: #FFFF19;}.cython.score-91 {background-color: #FFFF19;}.cython.score-92 {background-color: #FFFF19;}.cython.score-93 {background-color: #FFFF18;}.cython.score-94 {background-color: #FFFF18;}.cython.score-95 {background-color: #FFFF18;}.cython.score-96 {background-color: #FFFF18;}.cython.score-97 {background-color: #FFFF17;}.cython.score-98 {background-color: #FFFF17;}.cython.score-99 {background-color: #FFFF17;}.cython.score-100 {background-color: #FFFF17;}.cython.score-101 {background-color: #FFFF16;}.cython.score-102 {background-color: #FFFF16;}.cython.score-103 {background-color: #FFFF16;}.cython.score-104 {background-color: #FFFF16;}.cython.score-105 {background-color: #FFFF16;}.cython.score-106 {background-color: #FFFF15;}.cython.score-107 {background-color: #FFFF15;}.cython.score-108 {background-color: #FFFF15;}.cython.score-109 {background-color: #FFFF15;}.cython.score-110 {background-color: #FFFF15;}.cython.score-111 {background-color: #FFFF15;}.cython.score-112 {background-color: #FFFF14;}.cython.score-113 {background-color: #FFFF14;}.cython.score-114 {background-color: #FFFF14;}.cython.score-115 {background-color: #FFFF14;}.cython.score-116 {background-color: #FFFF14;}.cython.score-117 {background-color: #FFFF14;}.cython.score-118 {background-color: #FFFF13;}.cython.score-119 {background-color: #FFFF13;}.cython.score-120 {background-color: #FFFF13;}.cython.score-121 {background-color: #FFFF13;}.cython.score-122 {background-color: #FFFF13;}.cython.score-123 {background-color: #FFFF13;}.cython.score-124 {background-color: #FFFF13;}.cython.score-125 {background-color: #FFFF12;}.cython.score-126 {background-color: #FFFF12;}.cython.score-127 {background-color: #FFFF12;}.cython.score-128 {background-color: #FFFF12;}.cython.score-129 {background-color: #FFFF12;}.cython.score-130 {background-color: #FFFF12;}.cython.score-131 {background-color: #FFFF12;}.cython.score-132 {background-color: #FFFF11;}.cython.score-133 {background-color: #FFFF11;}.cython.score-134 {background-color: #FFFF11;}.cython.score-135 {background-color: #FFFF11;}.cython.score-136 {background-color: #FFFF11;}.cython.score-137 {background-color: #FFFF11;}.cython.score-138 {background-color: #FFFF11;}.cython.score-139 {background-color: #FFFF11;}.cython.score-140 {background-color: #FFFF11;}.cython.score-141 {background-color: #FFFF10;}.cython.score-142 {background-color: #FFFF10;}.cython.score-143 {background-color: #FFFF10;}.cython.score-144 {background-color: #FFFF10;}.cython.score-145 {background-color: #FFFF10;}.cython.score-146 {background-color: #FFFF10;}.cython.score-147 {background-color: #FFFF10;}.cython.score-148 {background-color: #FFFF10;}.cython.score-149 {background-color: #FFFF10;}.cython.score-150 {background-color: #FFFF0f;}.cython.score-151 {background-color: #FFFF0f;}.cython.score-152 {background-color: #FFFF0f;}.cython.score-153 {background-color: #FFFF0f;}.cython.score-154 {background-color: #FFFF0f;}.cython.score-155 {background-color: #FFFF0f;}.cython.score-156 {background-color: #FFFF0f;}.cython.score-157 {background-color: #FFFF0f;}.cython.score-158 {background-color: #FFFF0f;}.cython.score-159 {background-color: #FFFF0f;}.cython.score-160 {background-color: #FFFF0f;}.cython.score-161 {background-color: #FFFF0e;}.cython.score-162 {background-color: #FFFF0e;}.cython.score-163 {background-color: #FFFF0e;}.cython.score-164 {background-color: #FFFF0e;}.cython.score-165 {background-color: #FFFF0e;}.cython.score-166 {background-color: #FFFF0e;}.cython.score-167 {background-color: #FFFF0e;}.cython.score-168 {background-color: #FFFF0e;}.cython.score-169 {background-color: #FFFF0e;}.cython.score-170 {background-color: #FFFF0e;}.cython.score-171 {background-color: #FFFF0e;}.cython.score-172 {background-color: #FFFF0e;}.cython.score-173 {background-color: #FFFF0d;}.cython.score-174 {background-color: #FFFF0d;}.cython.score-175 {background-color: #FFFF0d;}.cython.score-176 {background-color: #FFFF0d;}.cython.score-177 {background-color: #FFFF0d;}.cython.score-178 {background-color: #FFFF0d;}.cython.score-179 {background-color: #FFFF0d;}.cython.score-180 {background-color: #FFFF0d;}.cython.score-181 {background-color: #FFFF0d;}.cython.score-182 {background-color: #FFFF0d;}.cython.score-183 {background-color: #FFFF0d;}.cython.score-184 {background-color: #FFFF0d;}.cython.score-185 {background-color: #FFFF0d;}.cython.score-186 {background-color: #FFFF0d;}.cython.score-187 {background-color: #FFFF0c;}.cython.score-188 {background-color: #FFFF0c;}.cython.score-189 {background-color: #FFFF0c;}.cython.score-190 {background-color: #FFFF0c;}.cython.score-191 {background-color: #FFFF0c;}.cython.score-192 {background-color: #FFFF0c;}.cython.score-193 {background-color: #FFFF0c;}.cython.score-194 {background-color: #FFFF0c;}.cython.score-195 {background-color: #FFFF0c;}.cython.score-196 {background-color: #FFFF0c;}.cython.score-197 {background-color: #FFFF0c;}.cython.score-198 {background-color: #FFFF0c;}.cython.score-199 {background-color: #FFFF0c;}.cython.score-200 {background-color: #FFFF0c;}.cython.score-201 {background-color: #FFFF0c;}.cython.score-202 {background-color: #FFFF0c;}.cython.score-203 {background-color: #FFFF0b;}.cython.score-204 {background-color: #FFFF0b;}.cython.score-205 {background-color: #FFFF0b;}.cython.score-206 {background-color: #FFFF0b;}.cython.score-207 {background-color: #FFFF0b;}.cython.score-208 {background-color: #FFFF0b;}.cython.score-209 {background-color: #FFFF0b;}.cython.score-210 {background-color: #FFFF0b;}.cython.score-211 {background-color: #FFFF0b;}.cython.score-212 {background-color: #FFFF0b;}.cython.score-213 {background-color: #FFFF0b;}.cython.score-214 {background-color: #FFFF0b;}.cython.score-215 {background-color: #FFFF0b;}.cython.score-216 {background-color: #FFFF0b;}.cython.score-217 {background-color: #FFFF0b;}.cython.score-218 {background-color: #FFFF0b;}.cython.score-219 {background-color: #FFFF0b;}.cython.score-220 {background-color: #FFFF0b;}.cython.score-221 {background-color: #FFFF0b;}.cython.score-222 {background-color: #FFFF0a;}.cython.score-223 {background-color: #FFFF0a;}.cython.score-224 {background-color: #FFFF0a;}.cython.score-225 {background-color: #FFFF0a;}.cython.score-226 {background-color: #FFFF0a;}.cython.score-227 {background-color: #FFFF0a;}.cython.score-228 {background-color: #FFFF0a;}.cython.score-229 {background-color: #FFFF0a;}.cython.score-230 {background-color: #FFFF0a;}.cython.score-231 {background-color: #FFFF0a;}.cython.score-232 {background-color: #FFFF0a;}.cython.score-233 {background-color: #FFFF0a;}.cython.score-234 {background-color: #FFFF0a;}.cython.score-235 {background-color: #FFFF0a;}.cython.score-236 {background-color: #FFFF0a;}.cython.score-237 {background-color: #FFFF0a;}.cython.score-238 {background-color: #FFFF0a;}.cython.score-239 {background-color: #FFFF0a;}.cython.score-240 {background-color: #FFFF0a;}.cython.score-241 {background-color: #FFFF0a;}.cython.score-242 {background-color: #FFFF0a;}.cython.score-243 {background-color: #FFFF0a;}.cython.score-244 {background-color: #FFFF0a;}.cython.score-245 {background-color: #FFFF0a;}.cython.score-246 {background-color: #FFFF09;}.cython.score-247 {background-color: #FFFF09;}.cython.score-248 {background-color: #FFFF09;}.cython.score-249 {background-color: #FFFF09;}.cython.score-250 {background-color: #FFFF09;}.cython.score-251 {background-color: #FFFF09;}.cython.score-252 {background-color: #FFFF09;}.cython.score-253 {background-color: #FFFF09;}.cython.score-254 {background-color: #FFFF09;}pre { line-height: 125%; }td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }.cython .hll { background-color: #ffffcc }.cython { background: #f8f8f8; }.cython .c { color: #3D7B7B; font-style: italic } /* Comment */.cython .err { border: 1px solid #FF0000 } /* Error */.cython .k { color: #008000; font-weight: bold } /* Keyword */.cython .o { color: #666666 } /* Operator */.cython .ch { color: #3D7B7B; font-style: italic } /* Comment.Hashbang */.cython .cm { color: #3D7B7B; font-style: italic } /* Comment.Multiline */.cython .cp { color: #9C6500 } /* Comment.Preproc */.cython .cpf { color: #3D7B7B; font-style: italic } /* Comment.PreprocFile */.cython .c1 { color: #3D7B7B; font-style: italic } /* Comment.Single */.cython .cs { color: #3D7B7B; font-style: italic } /* Comment.Special */.cython .gd { color: #A00000 } /* Generic.Deleted */.cython .ge { font-style: italic } /* Generic.Emph */.cython .gr { color: #E40000 } /* Generic.Error */.cython .gh { color: #000080; font-weight: bold } /* Generic.Heading */.cython .gi { color: #008400 } /* Generic.Inserted */.cython .go { color: #717171 } /* Generic.Output */.cython .gp { color: #000080; font-weight: bold } /* Generic.Prompt */.cython .gs { font-weight: bold } /* Generic.Strong */.cython .gu { color: #800080; font-weight: bold } /* Generic.Subheading */.cython .gt { color: #0044DD } /* Generic.Traceback */.cython .kc { color: #008000; font-weight: bold } /* Keyword.Constant */.cython .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */.cython .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */.cython .kp { color: #008000 } /* Keyword.Pseudo */.cython .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */.cython .kt { color: #B00040 } /* Keyword.Type */.cython .m { color: #666666 } /* Literal.Number */.cython .s { color: #BA2121 } /* Literal.String */.cython .na { color: #687822 } /* Name.Attribute */.cython .nb { color: #008000 } /* Name.Builtin */.cython .nc { color: #0000FF; font-weight: bold } /* Name.Class */.cython .no { color: #880000 } /* Name.Constant */.cython .nd { color: #AA22FF } /* Name.Decorator */.cython .ni { color: #717171; font-weight: bold } /* Name.Entity */.cython .ne { color: #CB3F38; font-weight: bold } /* Name.Exception */.cython .nf { color: #0000FF } /* Name.Function */.cython .nl { color: #767600 } /* Name.Label */.cython .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */.cython .nt { color: #008000; font-weight: bold } /* Name.Tag */.cython .nv { color: #19177C } /* Name.Variable */.cython .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */.cython .w { color: #bbbbbb } /* Text.Whitespace */.cython .mb { color: #666666 } /* Literal.Number.Bin */.cython .mf { color: #666666 } /* Literal.Number.Float */.cython .mh { color: #666666 } /* Literal.Number.Hex */.cython .mi { color: #666666 } /* Literal.Number.Integer */.cython .mo { color: #666666 } /* Literal.Number.Oct */.cython .sa { color: #BA2121 } /* Literal.String.Affix */.cython .sb { color: #BA2121 } /* Literal.String.Backtick */.cython .sc { color: #BA2121 } /* Literal.String.Char */.cython .dl { color: #BA2121 } /* Literal.String.Delimiter */.cython .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */.cython .s2 { color: #BA2121 } /* Literal.String.Double */.cython .se { color: #AA5D1F; font-weight: bold } /* Literal.String.Escape */.cython .sh { color: #BA2121 } /* Literal.String.Heredoc */.cython .si { color: #A45A77; font-weight: bold } /* Literal.String.Interpol */.cython .sx { color: #008000 } /* Literal.String.Other */.cython .sr { color: #A45A77 } /* Literal.String.Regex */.cython .s1 { color: #BA2121 } /* Literal.String.Single */.cython .ss { color: #19177C } /* Literal.String.Symbol */.cython .bp { color: #008000 } /* Name.Builtin.Pseudo */.cython .fm { color: #0000FF } /* Name.Function.Magic */.cython .vc { color: #19177C } /* Name.Variable.Class */.cython .vg { color: #19177C } /* Name.Variable.Global */.cython .vi { color: #19177C } /* Name.Variable.Instance */.cython .vm { color: #19177C } /* Name.Variable.Magic */.cython .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
</head>
<body class="cython">
<p><span style="border-bottom: solid 1px grey;">Generated by Cython 0.29.30</span></p>
<p>
    <span style="background-color: #FFFF00">Yellow lines</span> hint at Python interaction.<br />
    Click on a line that starts with a "<code>+</code>" to see the C code that Cython generated for it.
</p>
<div class="cython"><pre class="cython line score-0">&#xA0;<span class="">1</span>: </pre>
<pre class="cython line score-5" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">2</span>: <span class="n">a</span> <span class="o">=</span> <span class="mf">0</span></pre>
<pre class='cython code score-5 '>  if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_a, __pyx_int_0) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 2, __pyx_L1_error)</span>
</pre><pre class="cython line score-8" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">3</span>: <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mf">10</span><span class="p">):</span></pre>
<pre class='cython code score-8 '>  for (__pyx_t_1 = 0; __pyx_t_1 &lt; 10; __pyx_t_1+=1) {
    __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyInt_From_long</span>(__pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 3, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
    if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_i, __pyx_t_2) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 3, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
</pre><pre class="cython line score-17" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">4</span>:     <span class="n">a</span> <span class="o">+=</span> <span class="n">i</span></pre>
<pre class='cython code score-17 '>    <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_2, __pyx_n_s_a);<span class='error_goto'> if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
    <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_3, __pyx_n_s_i);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
    __pyx_t_4 = <span class='py_c_api'>PyNumber_InPlaceAdd</span>(__pyx_t_2, __pyx_t_3);<span class='error_goto'> if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 4, __pyx_L1_error)</span>
    <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_4);
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
    if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_a, __pyx_t_4) &lt; 0) <span class='error_goto'>__PYX_ERR(0, 4, __pyx_L1_error)</span>
    <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_4); __pyx_t_4 = 0;
  }
</pre><pre class="cython line score-6" onclick="(function(s){s.display=s.display==='block'?'none':'block'})(this.nextElementSibling.style)">+<span class="">5</span>: <span class="k">print</span><span class="p">(</span><span class="n">a</span><span class="p">)</span></pre>
<pre class='cython code score-6 '>  <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_t_4, __pyx_n_s_a);<span class='error_goto'> if (unlikely(!__pyx_t_4)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_4);
  __pyx_t_3 = <span class='pyx_c_api'>__Pyx_PyObject_CallOneArg</span>(__pyx_builtin_print, __pyx_t_4);<span class='error_goto'> if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 5, __pyx_L1_error)</span>
  <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_4); __pyx_t_4 = 0;
  <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
</pre></div></body></html>



### cfunc/cdef

Python関数呼び出しは時間がかかる可能性があります。Cythonでは、呼び出す前にPythonオブジェクト間の変換が必要になる可能性があるため、二重になる可能性があります。したがって、CythonはCスタイルの関数を宣言する方法、Cython固有の `cdef` 文、およびPython構文でCスタイルの関数を宣言するための `@cfunc` デコレータを提供しています。どちらの方法でも同じCコードが生成されます。



```python
import cython
```



```python
@cython.cfunc
@cython.exceptval(-2, check=True)
def f(x: cython.double) -> cython.double:
    return x ** 2 - x
```



```cython
%%cython

cdef double f(double x) except? -2:
    return x ** 2 - x
```

### パフォーマンス比較

次に、実際の例を用いて性能を比較します。



```python
# 标准Python版
def get_primes(num):
    res = [0] * 1000
    v = 2
    len_res = 0
    while len_res < num:
        flag = True
        for i in range(2, int(np.sqrt(v)) + 1):        
            if v % i == 0:
                flag = False
        if flag:
            res[len_res] = v
            len_res += 1
        v += 1
    return res
```



```cython
%%cython

cdef extern from "math.h":
    double sqrt(double x)

def cython_get_primes(int num):
    cdef int i, n, v=2, len_res=0
    cdef flag
    cdef int res[1000]
    
    while len_res < num:
        flag = True
        n = int(sqrt(v)) + 1
        for i in range(2, n):
            if v % i == 0:
                flag = False
        if flag:
            res[len_res] = v
            len_res += 1
        v += 1
    return res
```



```python
%timeit ps1 = cython_get_primes(1000)
```

    2.12 ms ± 157 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    


```python
%timeit ps2 = get_primes(1000)
```

    37.4 ms ± 471 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

    


```python
ps1 == ps2
```




    True



直接コンパイルされたPythonコードを比較して、codeディレクトリで実行します：


```bash
python3 python setup.py build_ext --inplace
```



```python
# 导入
%cd code
```

    /Users/Yam/Yam/powerful-numpy/src/skilled/code

    


```python
import primes
```



```python
ps3 = primes.get_primes(1000)
```



```python
ps3 ==  ps2
```




    True





```python
%timeit primes.get_primes(1000)
```

    23.1 ms ± 1.14 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    

## CuPy

文書：[CuPy–NumPy&amp;SciPy for GPU–CuPy 10.5.0ドキュメント](https://docs.cupy.dev/en/stable/)

Cupyは、Pythonを使用したGPU加速計算のためのNumPy/SciPy互換配列ライブラリです。CuPyは、NVIDIA CUDAまたはAMD ROCmプラットフォーム上で実行される既存のNumPy/SciPyコードの代替として机能します。その主な目的は、Pythonユーザーに基礎となるGPUテクノロジーを深く理解する必要がなく、GPUアクセラレーション能力を提供することです。

⚠️なお、このセクションの内容にはcuda環境が必要です。



```python
import cupy as cp
cp.__version__
```




    '10.5.0'



### cupy.ndarray



```python
x_gpu = cp.array([1,2,3])
```



```python
x_gpu
```




    array([1, 2, 3])





```python
type(x_gpu)
```




    cupy._core.core.ndarray



 `cuda.ndarray` と `np.ndarray` の主な違いは、CuPyが現在のデバイス（特定のGPUカード）に配列を割り当てることです。他のAPIはNumPyとほとんど変わらない。NumPyに精通すれば、CuPyに精通することになります。



```python
rng = cp.random.default_rng(42)
```



```python
rng.integers(0, 10, (2,3))
```




    array([[5, 4, 0],
           [7, 2, 3]])



### Device

これはCuPyの比較的重要な概念です。現在のデバイスです。これはデフォルトのGPUデバイスであり、配列の割り当て、操作、計算が実行されます。



```python
x_gpu0 = cp.array([1,2,3])
```



```python
x_gpu0.device
```




    <CUDA Device 0>





```python
with cp.cuda.Device(1):
    x_gpu1 = cp.array([1,2,3])
```



```python
x_gpu1.device
```




    <CUDA Device 1>



ここでは2枚（またはそれ以上）のカードが必要です。たとえば、存在しないカードをもう一つ持ってきます。



```python
with cp.cuda.Device(2):
    x_gpu2 = cp.array([1,2,3])
```


    ---------------------------------------------------------------------------

    CUDARuntimeError                          Traceback (most recent call last)

    Input In [12], in <cell line: 1>()----> 1 with cp.cuda.Device(2):
          2     x_gpu2 = cp.array([1,2,3])
    

    File cupy/cuda/device.pyx:184, in cupy.cuda.device.Device.__enter__()
    

    File cupy_backends/cuda/api/runtime.pyx:365, in cupy_backends.cuda.api.runtime.setDevice()
    

    File cupy_backends/cuda/api/runtime.pyx:142, in cupy_backends.cuda.api.runtime.check_status()
    

    CUDARuntimeError: cudaErrorInvalidDevice: invalid device ordinal


### Data Transfer

主にGPUカードとホスト（カードがマウントされているホスト）間の転送を指します。



```python
x_cpu = np.array([1,2,3])
```



```python
type(x_cpu)
```




    numpy.ndarray





```python
# 移动到GPU上
x_gpu = cp.asarray(x_cpu)
```



```python
type(x_gpu)
```




    cupy._core.core.ndarray



 `cp.asarray` GPUカード間で移動することもできます。



```python
with cp.cuda.Device(1):
    x_gpu2 = cp.asarray(x_gpu)
```



```python
x_gpu2.device
```




    <CUDA Device 1>



 `cp.asarray` データはコピーされません。コピーが必要な場合は `cp.array(arr, dtype, copy=True)` を使用してください。これは実際には `cp.array(a, dtype, copy=False)` と同等です。

 `copy=True` 新しい配列が返され、そうでない場合はオブジェクトが返されます。



```python
arr = cp.array([1,2,3])
cp.asarray(arr) is arr
```




    True





```python
# 从GPU到Host
x_cpu2 = cp.asnumpy(x_gpu2)
x_cpu2
```




    array([1, 2, 3])





```python
type(x_cpu2)
```




    numpy.ndarray





```python
# 或者使用`get`方法
x_gpu2.get()
```




    array([1, 2, 3])





```python
type(_)
```




    numpy.ndarray



 `cp.asnumpy` はNumPy配列（ホスト上）を返し、 `cp.asarray` はCuPy配列（現在のカード上）を返します。どちらのメソッドも任意の入力（cpまたはnpの配列）を受け入れることができます。

### Memory

GPUプログラミングにおいて、メモリ管理は比較的重要な部分です。CuPyは、メモリプールを使用してメモリを管理します。これには、次の2種類があります：

- Deviceメモリプール（GPUメモリ）、GPUメモリを割り当てるときに使用される
- Pinnedメモリプール（非スワップCPUメモリ）、CPUからGPUへのデータ転送時に使用される



```python
mempool = cp.get_default_memory_pool()
pinpool = cp.get_default_pinned_memory_pool()
```



```python
# 400bytes CPU内存
a_cpu = np.arange(100, dtype=np.float32)
```



```python
a_cpu.nbytes
```




    400





```python
mempool.used_bytes()
```




    0





```python
mempool.total_bytes()
```




    0





```python
pinpool.n_free_blocks()
```




    0



CPUからGPUへ、転送が完了するとpinned memoryが解放されます。

実際に割り当てられたサイズは、要求されたサイズよりも大きい値に切り捨てられることに注意してください。



```python
a = cp.array(a_cpu)
```



```python
a.nbytes
```




    400





```python
mempool.used_bytes()
```




    512





```python
mempool.total_bytes()
```




    512





```python
pinpool.n_free_blocks()
```




    1



配列がドメインを超えると、GPUメモリが解放されます。



```python
a = None
```



```python
mempool.used_bytes()
```




    0





```python
mempool.total_bytes()
```




    512





```python
pinpool.n_free_blocks()
```




    1



 `free_all_blocks` を使用してメモリプールをクリーンアップします。



```python
mempool.free_all_blocks()
```



```python
mempool.used_bytes()
```




    0





```python
mempool.total_bytes()
```




    0





```python
pinpool.free_all_blocks()
```



```python
pinpool.n_free_blocks()
```




    0



CUDAプログラミングにおける `threads`、 `blocks`、 `grids` は、次の3つの重要な概念です。

- thread：threadは単一のGPUコア上で実行される一連の命令です。
- block：復数のスレッドがGPU上でブロックの抽象単位で実行される
- ブロックのブロックは、グリッドとも呼ばれます。

GPUのメモリをハードに制限することもできます：


```bash
export CUPY_GPU_MEMORY_LIMIT="1073741824"

# or

export CUPY_GPU_MEMORY_LIMIT="50%"
```

または、組み込みのメソッドを使用します：



```python
mempool = cp.get_default_memory_pool()
```



```python
with cp.cuda.Device(0):
    mempool.set_limit(size=1024**3)
```



```python
cp.get_default_memory_pool().get_limit()
```




    1073741824



APIを介してメモリプールをカスタマイズまたは変更することもできます。詳細はドキュメントを参照してください：

 [Memory Management—CuPy 10.5.0ドキュメント](https://docs.cupy.dev/en/stable/user_guide/memory.html#changing-memory-pool)

CuPyとNumPyの動作には、以下のような微妙な違いがあります。

- 浮動小数点から整数への変換
- ランダム手法
- オーバーボードインデックス
- 繰り返しインデックス処理
- 0次元配列
- マトリックスタイプ
- データタイプ
- UFUNC
- ランダムシード
- NaN処理

詳細については、ドキュメントを参照してください：[Differences between CuPy and NumPy—CuPy 10.5.0 documentation](https://docs.cupy.dev/en/stable/user_guide/difference.html)



```python

```

CuPyは、NumPy、Numba、PyTorchなどの多くのライブラリと組み合わせることができます。詳細はドキュメントを参照してください。

 [Interoperability—CuPy 10.5.0ドキュメント](https://docs.cupy.dev/en/stable/user_guide/interoperability.html)

このセクションはcudaプログラミングに関連しています。私たちはより柔軟な制御を必要とするかもしれません。[PyCuda](https://github.com/inducer/pycuda)を検討してください。

## Sparse

文書：[スパース—sparse 0.13.0+0.g0b7dfeb.dirtyドキュメント](https://sparse.pydata.org/en/stable/)

 `Sparse` 任意の次元のスパース配列はNumPyとscipy.sparseで実装されています。

主なデータ構造参照は、疎行列のCoordinate List (COO) レイアウトに従い、復数の次元に拡張します。

|dmi1|dim2|dim3|...|data|
|----|----|----|---|----|
|0   |0   |0   |.  |10  |
|0   |0   |3   |.  |13  |
|0   |2   |2   |.  |9   |
|3   |1   |4   |.  |21  |

ストレージを除いて、配列関連のすべての操作（トランスポーズ、リシェイプ、スライス、乗算など）は再実装する必要があります。

さらに、このウェアハウスには、いくつかのデータ構造が含まれています。例えば、Dictionary of Keys (DOK) フォーマットは、任意の数の次元に一般化することができます。DOKは書き込みと操作に適していますが、他の操作はサポートされていません。一般的なベストプラクティスは、DOKを使用して配列を書き、別の形式に変換して別の操作を実行することです。

Compressed Sparse Row/Column (CSR/CSC）フォーマットもサポートされています。

### 作成



```python
import sparse as se
se.__version__
```




    '0.13.0'





```python
rng = np.random.default_rng(42)
a = rng.random((100, 100, 100))
```



```python
# 构造一个稀疏矩阵，90%为0
a[a<0.9] = 0
```



```python
s = se.COO(a)
```



```python
s.nbytes
```




    3205760





```python
a.nbytes
```




    8000000





```python
s
```




<table><tbody><tr><th style="text-align: left">Format</th><td style="text-align: left">coo</td></tr><tr><th style="text-align: left">Data Type</th><td style="text-align: left">float64</td></tr><tr><th style="text-align: left">Shape</th><td style="text-align: left">(100, 100, 100)</td></tr><tr><th style="text-align: left">nnz</th><td style="text-align: left">100180</td></tr><tr><th style="text-align: left">Density</th><td style="text-align: left">0.10018</td></tr><tr><th style="text-align: left">Read-only</th><td style="text-align: left">True</td></tr><tr><th style="text-align: left">Size</th><td style="text-align: left">3.1M</td></tr><tr><th style="text-align: left">Storage ratio</th><td style="text-align: left">0.4</td></tr></tbody></table>





```python
s.data
```




    array([0.97562235, 0.92676499, 0.97069802, ..., 0.98212211, 0.95084277,
           0.98694171])





```python
s.coords
```




    array([[ 0,  0,  0, ..., 99, 99, 99],
           [ 0,  0,  0, ..., 99, 99, 99],[ 5, 11, 22, ..., 94, 95, 98]])



座標と値から直接作成することもできます：



```python
coords = [[0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4]]
data = [10, 20, 30, 40, 50]
s = se.COO(coords, data, shape=(5, 5))
```



```python
s
```




<table><tbody><tr><th style="text-align: left">Format</th><td style="text-align: left">coo</td></tr><tr><th style="text-align: left">Data Type</th><td style="text-align: left">int64</td></tr><tr><th style="text-align: left">Shape</th><td style="text-align: left">(5, 5)</td></tr><tr><th style="text-align: left">nnz</th><td style="text-align: left">5</td></tr><tr><th style="text-align: left">Density</th><td style="text-align: left">0.2</td></tr><tr><th style="text-align: left">Read-only</th><td style="text-align: left">True</td></tr><tr><th style="text-align: left">Size</th><td style="text-align: left">120</td></tr><tr><th style="text-align: left">Storage ratio</th><td style="text-align: left">0.6</td></tr></tbody></table>





```python
s.todense()
```




    array([[10,  0,  0,  0,  0],
           [ 0, 20,  0,  0,  0],[ 0,  0, 30,  0,  0],[ 0,  0,  0, 40,  0],[ 0,  0,  0,  0, 50]])



次元はさらに任意であることもできます：



```python
coords = [[0, 3, 2, 1], [4, 1, 2, 0]]
data = [1, 4, 2, 1]
s = se.COO(coords, data, shape=(6, 5))
s.todense()
```




    array([[0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0],[0, 0, 2, 0, 0],[0, 4, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]])





```python
# 指定填充值
s = se.COO(coords, data, shape=(5,5), fill_value=-1)
s.todense()
```




    array([[-1, -1, -1, -1,  1],
           [ 1, -1, -1, -1, -1],[-1, -1,  2, -1, -1],[-1,  4, -1, -1, -1],[-1, -1, -1, -1, -1]])



SciPyの疎行列から、NumPyの配列から生成します：

- `se.COO.from_scipy_sparse(x)`
- `se.COO.from_numpy(x)`

ランダムもサポートされています：



```python
s = se.random((5, 5), density=0.1)
s
```




<table><tbody><tr><th style="text-align: left">Format</th><td style="text-align: left">coo</td></tr><tr><th style="text-align: left">Data Type</th><td style="text-align: left">float64</td></tr><tr><th style="text-align: left">Shape</th><td style="text-align: left">(5, 5)</td></tr><tr><th style="text-align: left">nnz</th><td style="text-align: left">2</td></tr><tr><th style="text-align: left">Density</th><td style="text-align: left">0.08</td></tr><tr><th style="text-align: left">Read-only</th><td style="text-align: left">True</td></tr><tr><th style="text-align: left">Size</th><td style="text-align: left">20</td></tr><tr><th style="text-align: left">Storage ratio</th><td style="text-align: left">0.1</td></tr></tbody></table>





```python
s.todense()
```




    array([[0.        , 0.        , 0.1182219 , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],[0.        , 0.        , 0.        , 0.        , 0.        ],[0.        , 0.        , 0.        , 0.        , 0.        ],[0.        , 0.        , 0.        , 0.        , 0.65295601]])





```python
s.data
```




    array([0.1182219 , 0.65295601])





```python
s.coords
```




    array([[0, 4],
           [2, 4]], dtype=uint8)



または辞書を入力して作成します：



```python
d = {(0, 0, 0): 1, (1, 2, 3): 2, (1, 1, 0): 3}
s = se.COO(d)
```



```python
s.shape
```




    (2, 3, 4)





```python
s
```




<table><tbody><tr><th style="text-align: left">Format</th><td style="text-align: left">coo</td></tr><tr><th style="text-align: left">Data Type</th><td style="text-align: left">int64</td></tr><tr><th style="text-align: left">Shape</th><td style="text-align: left">(2, 3, 4)</td></tr><tr><th style="text-align: left">nnz</th><td style="text-align: left">3</td></tr><tr><th style="text-align: left">Density</th><td style="text-align: left">0.125</td></tr><tr><th style="text-align: left">Read-only</th><td style="text-align: left">True</td></tr><tr><th style="text-align: left">Size</th><td style="text-align: left">96</td></tr><tr><th style="text-align: left">Storage ratio</th><td style="text-align: left">0.5</td></tr></tbody></table>



配列を使用することもできます：



```python
L = [((0, 0), 1),
     ((1, 1), 2),
     ((0, 0), 3)]
```



```python
s = se.COO(L)
s.todense()
```




    array([[4, 0],
           [0, 2]])



またはDOKから変換します：



```python
s1 = se.DOK((5, 5))
s1
```




<table><tbody><tr><th style="text-align: left">Format</th><td style="text-align: left">dok</td></tr><tr><th style="text-align: left">Data Type</th><td style="text-align: left">float64</td></tr><tr><th style="text-align: left">Shape</th><td style="text-align: left">(5, 5)</td></tr><tr><th style="text-align: left">nnz</th><td style="text-align: left">0</td></tr><tr><th style="text-align: left">Density</th><td style="text-align: left">0.0</td></tr><tr><th style="text-align: left">Read-only</th><td style="text-align: left">False</td></tr><tr><th style="text-align: left">Size</th><td style="text-align: left">0</td></tr><tr><th style="text-align: left">Storage ratio</th><td style="text-align: left">0.0</td></tr></tbody></table>





```python
s1[1:3, 1:3] = [[4,5],[6,7]]
```



```python
s1.todense()
```




    array([[0., 0., 0., 0., 0.],
           [0., 4., 5., 0., 0.],[0., 6., 7., 0., 0.],[0., 0., 0., 0., 0.],[0., 0., 0., 0., 0.]])





```python
s1.coords
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Input In [59], in <cell line: 1>()----> 1 s1.coords
    

    AttributeError: 'DOK' object has no attribute 'coords'




```python
s1.data
```




    {(1, 1): 4.0, (1, 2): 5.0, (2, 1): 6.0, (2, 2): 7.0}





```python
s2 = s1.asformat("coo")
```



```python
s2.todense()
```




    array([[0., 0., 0., 0., 0.],
           [0., 4., 5., 0., 0.],[0., 6., 7., 0., 0.],[0., 0., 0., 0., 0.],[0., 0., 0., 0., 0.]])





```python
s2.coords
```




    array([[1, 1, 2, 2],
           [1, 2, 1, 2]])





```python
s2.data
```




    array([4., 5., 6., 7.])





```python
# 这样也可以转换
s3 = se.COO(s1)
s3.todense()
```




    array([[0., 0., 0., 0., 0.],
           [0., 4., 5., 0., 0.],[0., 6., 7., 0., 0.],[0., 0., 0., 0., 0.],[0., 0., 0., 0., 0.]])



### 変換

 `COO` オブジェクトは以下を含む他のフォーマットに変換できます：

-  `COO.todense`：NumPy配列に変換
-  `COO.mayhbe_densify`：特定の条件に基づいてNumPy配列に変換する
-  `COO.to_scipy_sparse`：配列が2次元である場合、 `spicy.sparse.coo_matrix` に変換されます。
-  `COO.tocsr`：配列が2次元である場合は、 `scipy.sparse.csr_matrix` に変換します。
-  `COO.tocsc`：配列が2次元である場合は、 `scipy.sparse.csc_matrix` に変換します。

2番目のAPIに焦点を当ててみましょう。max_size（出力要素の最大数、デフォルト1000）とmin_density（出力の最小密度、デフォルト0.25）の2つのパラメータを受け入れ、スパース配列が2つの条件を満たさないときに例外をスローします。



```python
x = np.zeros((5, 5), dtype=np.uint8)
x[2, :] = 1
s = se.COO.from_numpy(x)
```



```python
x
```




    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],[1, 1, 1, 1, 1],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]], dtype=uint8)





```python
# 25满足，0.9不满足
s.maybe_densify(max_size=25, min_density=0.21)
```




    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],[1, 1, 1, 1, 1],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]], dtype=uint8)





```python
# 24不满足，0.1满足
s.maybe_densify(max_size=24, min_density=0.1)
```




    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],[1, 1, 1, 1, 1],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]], dtype=uint8)





```python
# 都满足
s.maybe_densify(max_size=25, min_density=0.1)
```




    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],[1, 1, 1, 1, 1],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]], dtype=uint8)





```python
# 都不满足
s.maybe_densify(max_size=24, min_density=0.21)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Input In [149], in <cell line: 2>()
          1 # 都不满足
    ----> 2 s.maybe_densify(max_size=24, min_density=0.21)
    

    File /home/env/anaconda3/envs/tf29/lib/python3.8/site-packages/sparse/_coo/core.py:1379, in COO.maybe_densify(self, max_size, min_density)
       1377     return self.todense()1378 else:
    -> 1379     raise ValueError(
       1380         "Operation would require converting " "large sparse array to dense"1381     )
    

    ValueError: Operation would require converting large sparse array to dense


### 計算



```python
s = se.random((3, 3, 3), density=0.1)
s.todense()
```




    array([[[0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        ],[0.        , 0.        , 0.        ]],
    
           [[0.        , 0.        , 0.        ],
            [0.82191339, 0.        , 0.        ],[0.        , 0.        , 0.        ]],
    
           [[0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        ],[0.        , 0.312388  , 0.        ]]])





```python
y = np.sin(s) + s.T * 1
y
```




<table><tbody><tr><th style="text-align: left">Format</th><td style="text-align: left">coo</td></tr><tr><th style="text-align: left">Data Type</th><td style="text-align: left">float64</td></tr><tr><th style="text-align: left">Shape</th><td style="text-align: left">(3, 3, 3)</td></tr><tr><th style="text-align: left">nnz</th><td style="text-align: left">4</td></tr><tr><th style="text-align: left">Density</th><td style="text-align: left">0.14814814814814814</td></tr><tr><th style="text-align: left">Read-only</th><td style="text-align: left">True</td></tr><tr><th style="text-align: left">Size</th><td style="text-align: left">44</td></tr><tr><th style="text-align: left">Storage ratio</th><td style="text-align: left">0.2</td></tr></tbody></table>





```python
y.todense()
```




    array([[[0.        , 0.        , 0.        ],
            [0.        , 0.82191339, 0.        ],[0.        , 0.        , 0.        ]],
    
           [[0.        , 0.        , 0.        ],
            [0.73244985, 0.        , 0.        ],[0.        , 0.        , 0.312388  ]],
    
           [[0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        ],[0.        , 0.30733194, 0.        ]]])



より多くの内容を参照してください：

 [Operations on COO and GCXS arrays—sparse 0.13.0+0.g0b7dfeb.dirty documentation](https://sparse.pydata.org/en/stable/operations.html)

## Dask

文書：[Dask—Dask documentation](https://docs.dask.org/en/latest/)

Daskは主に并列計算に使用されています。2つの部分を含んでいます：

- コンピューティングに最適化された動的タスクスケジューリングAirflow、Luigi、Celery、Makeと同様ですが、インタラクティブなコンピューティングワークロードに最適化されています。
- パラレル配列、DataFrame、リストなどのビッグデータコレクションは、NumPy、Pandas、Pythonイテレータなどの一般的なインタフェースをメモリより大きい環境や分散環境に拡張します。これらの並列コレクションは、動的タスクスケジューラ上で実行されます。

全体的なアーキテクチャは以下


![](./img/dask-overview.svg)

Daskは多くのデータ集合を処理することができますが、ここでは配列（Array）を例に挙げます。Daskについては、第1節『配列オブジェクト』でも少し触れています。



```python
import dask.array as da
```



```python
a = np.array([2,3], like=da.array([1,1]))
a
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
a.compute()
```




    array([2, 3])



### 作成



```python
data = np.arange(100_000).reshape(200, 500)
```



```python
a = da.from_array(data, chunks=(100, 100))
a
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
                        <td> 781.25 kiB </td>
                        <td> 78.12 kiB </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (200, 500) </td>
                        <td> (100, 100) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 10 Tasks </td>
                        <td> 10 Chunks </td>
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
        <svg width="170" height="98" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="24" x2="120" y2="24" />
  <line x1="0" y1="48" x2="120" y2="48" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="48" style="stroke-width:2" />
  <line x1="24" y1="0" x2="24" y2="48" />
  <line x1="48" y1="0" x2="48" y2="48" />
  <line x1="72" y1="0" x2="72" y2="48" />
  <line x1="96" y1="0" x2="96" y2="48" />
  <line x1="120" y1="0" x2="120" y2="48" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,48.0 0.0,48.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="68.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >500</text>
  <text x="140.000000" y="24.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,140.000000,24.000000)">200</text>
</svg>
        </td>
    </tr>
</table>





```python
# 自动chunk
b = da.from_array(data)
b
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
                        <td> 781.25 kiB </td>
                        <td> 781.25 kiB </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (200, 500) </td>
                        <td> (200, 500) </td>
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
        <svg width="170" height="98" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="48" x2="120" y2="48" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="48" style="stroke-width:2" />
  <line x1="120" y1="0" x2="120" y2="48" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,48.0 0.0,48.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="68.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >500</text>
  <text x="140.000000" y="24.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,140.000000,24.000000)">200</text>
</svg>
        </td>
    </tr>
</table>



### 索引



```python
a[0]
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
                        <td> 3.91 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (500,) </td>
                        <td> (100,) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 15 Tasks </td>
                        <td> 5 Chunks </td>
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
        <svg width="170" height="75" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="25" x2="120" y2="25" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="25" style="stroke-width:2" />
  <line x1="24" y1="0" x2="24" y2="25" />
  <line x1="48" y1="0" x2="48" y2="25" />
  <line x1="72" y1="0" x2="72" y2="25" />
  <line x1="96" y1="0" x2="96" y2="25" />
  <line x1="120" y1="0" x2="120" y2="25" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,25.412616514582485 0.0,25.412616514582485" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="45.412617" font-size="1.0rem" font-weight="100" text-anchor="middle" >500</text>
  <text x="140.000000" y="12.706308" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,12.706308)">1</text>
</svg>
        </td>
    </tr>
</table>





```python
a[:50, 200]
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
                        <td> 400 B </td>
                        <td> 400 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (50,) </td>
                        <td> (50,) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Tasks </td>
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
        <svg width="170" height="79" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="29" x2="120" y2="29" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="29" style="stroke-width:2" />
  <line x1="120" y1="0" x2="120" y2="29" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,29.030629010473877 0.0,29.030629010473877" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="49.030629" font-size="1.0rem" font-weight="100" text-anchor="middle" >50</text>
  <text x="140.000000" y="14.515315" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,14.515315)">1</text>
</svg>
        </td>
    </tr>
</table>



### 計算

Daskは不活性に推定され、計算用のDaskタスクグラフが生成され、結果が要求されたときに計算されます。計算はNumPyと同様で、NumPyを組み合わせることもできます。



```python
a.compute()
```




    array([[    0,     1,     2, ...,   497,   498,   499],
           [  500,   501,   502, ...,   997,   998,   999],[ 1000,  1001,  1002, ...,  1497,  1498,  1499],...,[98500, 98501, 98502, ..., 98997, 98998, 98999],[99000, 99001, 99002, ..., 99497, 99498, 99499],[99500, 99501, 99502, ..., 99997, 99998, 99999]])





```python
a.mean()
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
                        <td> 8 B </td>
                        <td> 8.0 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> () </td>
                        <td> () </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 26 Tasks </td>
                        <td> 1 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>

        </td>
    </tr>
</table>





```python
a.mean().compute()
```




    49999.5





```python
np.sin(a)
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
                        <td> 781.25 kiB </td>
                        <td> 78.12 kiB </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (200, 500) </td>
                        <td> (100, 100) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 20 Tasks </td>
                        <td> 10 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="98" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="24" x2="120" y2="24" />
  <line x1="0" y1="48" x2="120" y2="48" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="48" style="stroke-width:2" />
  <line x1="24" y1="0" x2="24" y2="48" />
  <line x1="48" y1="0" x2="48" y2="48" />
  <line x1="72" y1="0" x2="72" y2="48" />
  <line x1="96" y1="0" x2="96" y2="48" />
  <line x1="120" y1="0" x2="120" y2="48" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,48.0 0.0,48.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="68.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >500</text>
  <text x="140.000000" y="24.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,140.000000,24.000000)">200</text>
</svg>
        </td>
    </tr>
</table>





```python
_.compute()
```




    array([[ 0.        ,  0.84147098,  0.90929743, ...,  0.58781939,
             0.99834363,  0.49099533],
           [-0.46777181, -0.9964717 , -0.60902011, ..., -0.89796748,
            -0.85547315, -0.02646075],
           [ 0.82687954,  0.9199906 ,  0.16726654, ...,  0.99951642,
             0.51387502, -0.4442207 ],
           ...,[-0.99720859, -0.47596473,  0.48287891, ..., -0.76284376,
             0.13191447,  0.90539115],
           [ 0.84645538,  0.00929244, -0.83641393, ...,  0.37178568,
            -0.5802765 , -0.99883514],
           [-0.49906936,  0.45953849,  0.99564877, ...,  0.10563876,
             0.89383946,  0.86024828]])



### タスクマップ



```python
c = a.max(axis=1)[::-1] + 10
```



```python
c
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
                        <td> 1.56 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (200,) </td>
                        <td> (100,) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 30 Tasks </td>
                        <td> 2 Chunks </td>
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
        <svg width="170" height="75" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="25" x2="120" y2="25" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="25" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="25" />
  <line x1="120" y1="0" x2="120" y2="25" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,25.412616514582485 0.0,25.412616514582485" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="45.412617" font-size="1.0rem" font-weight="100" text-anchor="middle" >200</text>
  <text x="140.000000" y="12.706308" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,12.706308)">1</text>
</svg>
        </td>
    </tr>
</table>





```python
c.dask
```




<div>
    <div>
        <div style="width: 52px; height: 52px; position: absolute;">
            <svg width="76" height="71" viewBox="0 0 76 71" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="61.5" cy="36.5" r="13.5" style="stroke: var(--jp-ui-font-color2, #1D1D1D); fill: var(--jp-layout-color1, #F2F2F2);" stroke-width="2"/>
                <circle cx="14.5" cy="14.5" r="13.5" style="stroke: var(--jp-ui-font-color2, #1D1D1D); fill: var(--jp-layout-color1, #F2F2F2);" stroke-width="2"/>
                <circle cx="14.5" cy="56.5" r="13.5" style="stroke: var(--jp-ui-font-color2, #1D1D1D); fill: var(--jp-layout-color1, #F2F2F2);" stroke-width="2"/>
                <path d="M28 16L30.5 16C33.2614 16 35.5 18.2386 35.5 21L35.5 32.0001C35.5 34.7615 37.7386 37.0001 40.5 37.0001L43 37.0001" style="stroke: var(--jp-ui-font-color2, #1D1D1D);" stroke-width="1.5"/>
                <path d="M40.5 37L40.5 37.75L40.5 37.75L40.5 37ZM35.5 42L36.25 42L35.5 42ZM35.5 52L34.75 52L35.5 52ZM30.5 57L30.5 57.75L30.5 57ZM41.5001 36.25L40.5 36.25L40.5 37.75L41.5001 37.75L41.5001 36.25ZM34.75 42L34.75 52L36.25 52L36.25 42L34.75 42ZM30.5 56.25L28.0001 56.25L28.0001 57.75L30.5 57.75L30.5 56.25ZM34.75 52C34.75 54.3472 32.8472 56.25 30.5 56.25L30.5 57.75C33.6756 57.75 36.25 55.1756 36.25 52L34.75 52ZM40.5 36.25C37.3244 36.25 34.75 38.8243 34.75 42L36.25 42C36.25 39.6528 38.1528 37.75 40.5 37.75L40.5 36.25Z" style="fill: var(--jp-ui-font-color2, #1D1D1D);"/>
                <circle cx="28" cy="16" r="2.25" fill="#E5E5E5" style="stroke: var(--jp-ui-font-color2, #1D1D1D);" stroke-width="1.5"/>
                <circle cx="28" cy="57" r="2.25" fill="#E5E5E5" style="stroke: var(--jp-ui-font-color2, #1D1D1D);" stroke-width="1.5"/>
                <path d="M45.25 36.567C45.5833 36.7594 45.5833 37.2406 45.25 37.433L42.25 39.1651C41.9167 39.3575 41.5 39.117 41.5 38.7321V35.2679C41.5 34.883 41.9167 34.6425 42.25 34.8349L45.25 36.567Z" style="fill: var(--jp-ui-font-color2, #1D1D1D);"/>
            </svg>
        </div>
        <div style="margin-left: 64px;">
            <h3 style="margin-bottom: 0px;">HighLevelGraph</h3>
            <p style="color: var(--jp-ui-font-color2, #5D5851); margin-bottom:0px;">
                HighLevelGraph with 6 layers and 30 keys from all layers.
            </p>

            <div style="">
    <svg width="24" height="24" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="position: absolute;">

        <circle cx="16" cy="16" r="14" fill="#8F8F8F" style="stroke: var(--jp-ui-font-color2, #1D1D1D);" stroke-width="2"/>

    </svg>

    <details style="margin-left: 32px;">
        <summary style="margin-bottom: 10px; margin-top: 10px;">
            <h4 style="display: inline;">Layer1: array</h4>
        </summary>
        <p style="color: var(--jp-ui-font-color2, #5D5851); margin: -0.25em 0px 0px 0px;">
            array-5f1b3c0ca172b03296ed6d431d3c9df7
        </p>

        <table>
        <tr>
            <td>
                <table>

                    <tr>
                        <th style="text-align: left; width: 150px;">layer_type</th>
                        <td style="text-align: left;">MaterializedLayer</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">is_materialized</th>
                        <td style="text-align: left;">True</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">number of outputs</th>
                        <td style="text-align: left;">10</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">shape</th>
                        <td style="text-align: left;">(200, 500)</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">dtype</th>
                        <td style="text-align: left;">int64</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">chunksize</th>
                        <td style="text-align: left;">(100, 100)</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">type</th>
                        <td style="text-align: left;">dask.array.core.Array</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">chunk_type</th>
                        <td style="text-align: left;">numpy.ndarray</td>
                    </tr>

                </table>
            </td>
            <td>
                <svg width="250" height="130" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="200" y2="0" style="stroke-width:2" />
  <line x1="0" y1="40" x2="200" y2="40" />
  <line x1="0" y1="80" x2="200" y2="80" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="80" style="stroke-width:2" />
  <line x1="40" y1="0" x2="40" y2="80" />
  <line x1="80" y1="0" x2="80" y2="80" />
  <line x1="120" y1="0" x2="120" y2="80" />
  <line x1="160" y1="0" x2="160" y2="80" />
  <line x1="200" y1="0" x2="200" y2="80" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 200.0,0.0 200.0,80.0 0.0,80.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="100.000000" y="100.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >500</text>
  <text x="220.000000" y="40.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,220.000000,40.000000)">200</text>
</svg>
            </td>
        </tr>
        </table>

    </details>
</div>

            <div style="">
    <svg width="24" height="24" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="position: absolute;">

        <circle cx="16" cy="16" r="14" style="stroke: var(--jp-ui-font-color2, #1D1D1D); fill: var(--jp-layout-color1, #F2F2F2);" stroke-width="2" />

    </svg>

    <details style="margin-left: 32px;">
        <summary style="margin-bottom: 10px; margin-top: 10px;">
            <h4 style="display: inline;">Layer2: amax</h4>
        </summary>
        <p style="color: var(--jp-ui-font-color2, #5D5851); margin: -0.25em 0px 0px 0px;">
            amax-857d92e422c65e5a56883f4ef7335711
        </p>

        <table>
        <tr>
            <td>
                <table>

                    <tr>
                        <th style="text-align: left; width: 150px;">layer_type</th>
                        <td style="text-align: left;">Blockwise</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">is_materialized</th>
                        <td style="text-align: left;">False</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">number of outputs</th>
                        <td style="text-align: left;">10</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">shape</th>
                        <td style="text-align: left;">(200, 500)</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">dtype</th>
                        <td style="text-align: left;">int64</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">chunksize</th>
                        <td style="text-align: left;">(100, 100)</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">type</th>
                        <td style="text-align: left;">dask.array.core.Array</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">chunk_type</th>
                        <td style="text-align: left;">numpy.ndarray</td>
                    </tr>

                </table>
            </td>
            <td>
                <svg width="250" height="130" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="200" y2="0" style="stroke-width:2" />
  <line x1="0" y1="40" x2="200" y2="40" />
  <line x1="0" y1="80" x2="200" y2="80" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="80" style="stroke-width:2" />
  <line x1="40" y1="0" x2="40" y2="80" />
  <line x1="80" y1="0" x2="80" y2="80" />
  <line x1="120" y1="0" x2="120" y2="80" />
  <line x1="160" y1="0" x2="160" y2="80" />
  <line x1="200" y1="0" x2="200" y2="80" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 200.0,0.0 200.0,80.0 0.0,80.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="100.000000" y="100.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >500</text>
  <text x="220.000000" y="40.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,220.000000,40.000000)">200</text>
</svg>
            </td>
        </tr>
        </table>

    </details>
</div>

            <div style="">
    <svg width="24" height="24" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="position: absolute;">

        <circle cx="16" cy="16" r="14" fill="#8F8F8F" style="stroke: var(--jp-ui-font-color2, #1D1D1D);" stroke-width="2"/>

    </svg>

    <details style="margin-left: 32px;">
        <summary style="margin-bottom: 10px; margin-top: 10px;">
            <h4 style="display: inline;">Layer3: amax-partial</h4>
        </summary>
        <p style="color: var(--jp-ui-font-color2, #5D5851); margin: -0.25em 0px 0px 0px;">
            amax-partial-2116d1054340ff0f1dc9241dc0bd7ee7
        </p>

        <table>
        <tr>
            <td>
                <table>

                    <tr>
                        <th style="text-align: left; width: 150px;">layer_type</th>
                        <td style="text-align: left;">MaterializedLayer</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">is_materialized</th>
                        <td style="text-align: left;">True</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">number of outputs</th>
                        <td style="text-align: left;">4</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">shape</th>
                        <td style="text-align: left;">(200, 2)</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">dtype</th>
                        <td style="text-align: left;">int64</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">chunksize</th>
                        <td style="text-align: left;">(100, 1)</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">type</th>
                        <td style="text-align: left;">dask.array.core.Array</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">chunk_type</th>
                        <td style="text-align: left;">numpy.ndarray</td>
                    </tr>

                </table>
            </td>
            <td>
                <svg width="92" height="250" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="42" y2="0" style="stroke-width:2" />
  <line x1="0" y1="100" x2="42" y2="100" />
  <line x1="0" y1="200" x2="42" y2="200" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="200" style="stroke-width:2" />
  <line x1="21" y1="0" x2="21" y2="200" />
  <line x1="42" y1="0" x2="42" y2="200" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 42.354360857637474,0.0 42.354360857637474,200.0 0.0,200.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="21.177180" y="220.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >2</text>
  <text x="62.354361" y="100.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,62.354361,100.000000)">200</text>
</svg>
            </td>
        </tr>
        </table>

    </details>
</div>

            <div style="">
    <svg width="24" height="24" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="position: absolute;">

        <circle cx="16" cy="16" r="14" fill="#8F8F8F" style="stroke: var(--jp-ui-font-color2, #1D1D1D);" stroke-width="2"/>

    </svg>

    <details style="margin-left: 32px;">
        <summary style="margin-bottom: 10px; margin-top: 10px;">
            <h4 style="display: inline;">Layer4: amax-aggregate</h4>
        </summary>
        <p style="color: var(--jp-ui-font-color2, #5D5851); margin: -0.25em 0px 0px 0px;">
            amax-aggregate-9d9c993907fb6615495aa49e190fe954
        </p>

        <table>
        <tr>
            <td>
                <table>

                    <tr>
                        <th style="text-align: left; width: 150px;">layer_type</th>
                        <td style="text-align: left;">MaterializedLayer</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">is_materialized</th>
                        <td style="text-align: left;">True</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">number of outputs</th>
                        <td style="text-align: left;">2</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">shape</th>
                        <td style="text-align: left;">(200,)</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">dtype</th>
                        <td style="text-align: left;">int64</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">chunksize</th>
                        <td style="text-align: left;">(100,)</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">type</th>
                        <td style="text-align: left;">dask.array.core.Array</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">chunk_type</th>
                        <td style="text-align: left;">numpy.ndarray</td>
                    </tr>

                </table>
            </td>
            <td>
                <svg width="250" height="92" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="200" y2="0" style="stroke-width:2" />
  <line x1="0" y1="42" x2="200" y2="42" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="42" style="stroke-width:2" />
  <line x1="100" y1="0" x2="100" y2="42" />
  <line x1="200" y1="0" x2="200" y2="42" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 200.0,0.0 200.0,42.354360857637474 0.0,42.354360857637474" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="100.000000" y="62.354361" font-size="1.0rem" font-weight="100" text-anchor="middle" >200</text>
  <text x="220.000000" y="21.177180" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,220.000000,21.177180)">1</text>
</svg>
            </td>
        </tr>
        </table>

    </details>
</div>

            <div style="">
    <svg width="24" height="24" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="position: absolute;">

        <circle cx="16" cy="16" r="14" fill="#8F8F8F" style="stroke: var(--jp-ui-font-color2, #1D1D1D);" stroke-width="2"/>

    </svg>

    <details style="margin-left: 32px;">
        <summary style="margin-bottom: 10px; margin-top: 10px;">
            <h4 style="display: inline;">Layer5: getitem</h4>
        </summary>
        <p style="color: var(--jp-ui-font-color2, #5D5851); margin: -0.25em 0px 0px 0px;">
            getitem-a612729f5a152786f84084f0fe628918
        </p>

        <table>
        <tr>
            <td>
                <table>

                    <tr>
                        <th style="text-align: left; width: 150px;">layer_type</th>
                        <td style="text-align: left;">MaterializedLayer</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">is_materialized</th>
                        <td style="text-align: left;">True</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">number of outputs</th>
                        <td style="text-align: left;">2</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">shape</th>
                        <td style="text-align: left;">(200,)</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">dtype</th>
                        <td style="text-align: left;">int64</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">chunksize</th>
                        <td style="text-align: left;">(100,)</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">type</th>
                        <td style="text-align: left;">dask.array.core.Array</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">chunk_type</th>
                        <td style="text-align: left;">numpy.ndarray</td>
                    </tr>

                </table>
            </td>
            <td>
                <svg width="250" height="92" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="200" y2="0" style="stroke-width:2" />
  <line x1="0" y1="42" x2="200" y2="42" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="42" style="stroke-width:2" />
  <line x1="100" y1="0" x2="100" y2="42" />
  <line x1="200" y1="0" x2="200" y2="42" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 200.0,0.0 200.0,42.354360857637474 0.0,42.354360857637474" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="100.000000" y="62.354361" font-size="1.0rem" font-weight="100" text-anchor="middle" >200</text>
  <text x="220.000000" y="21.177180" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,220.000000,21.177180)">1</text>
</svg>
            </td>
        </tr>
        </table>

    </details>
</div>

            <div style="">
    <svg width="24" height="24" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="position: absolute;">

        <circle cx="16" cy="16" r="14" style="stroke: var(--jp-ui-font-color2, #1D1D1D); fill: var(--jp-layout-color1, #F2F2F2);" stroke-width="2" />

    </svg>

    <details style="margin-left: 32px;">
        <summary style="margin-bottom: 10px; margin-top: 10px;">
            <h4 style="display: inline;">Layer6: add</h4>
        </summary>
        <p style="color: var(--jp-ui-font-color2, #5D5851); margin: -0.25em 0px 0px 0px;">
            add-e33b1625077ca0abd8576c8becc0d497
        </p>

        <table>
        <tr>
            <td>
                <table>

                    <tr>
                        <th style="text-align: left; width: 150px;">layer_type</th>
                        <td style="text-align: left;">Blockwise</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">is_materialized</th>
                        <td style="text-align: left;">False</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">number of outputs</th>
                        <td style="text-align: left;">2</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">shape</th>
                        <td style="text-align: left;">(200,)</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">dtype</th>
                        <td style="text-align: left;">int64</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">chunksize</th>
                        <td style="text-align: left;">(100,)</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">type</th>
                        <td style="text-align: left;">dask.array.core.Array</td>
                    </tr>

                    <tr>
                        <th style="text-align: left; width: 150px;">chunk_type</th>
                        <td style="text-align: left;">numpy.ndarray</td>
                    </tr>

                </table>
            </td>
            <td>
                <svg width="250" height="92" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="200" y2="0" style="stroke-width:2" />
  <line x1="0" y1="42" x2="200" y2="42" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="42" style="stroke-width:2" />
  <line x1="100" y1="0" x2="100" y2="42" />
  <line x1="200" y1="0" x2="200" y2="42" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 200.0,0.0 200.0,42.354360857637474 0.0,42.354360857637474" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="100.000000" y="62.354361" font-size="1.0rem" font-weight="100" text-anchor="middle" >200</text>
  <text x="220.000000" y="21.177180" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,220.000000,21.177180)">1</text>
</svg>
            </td>
        </tr>
        </table>

    </details>
</div>

        </div>
    </div>
</div>





```python
c.visualize()
```




    
![png](ch06-morethan_numpy_files/ch06-morethan_numpy_288_0.png)
    



### 基本API

 `delayed` デコレータを使用して、関数呼び出しを遅延で構成されたタスクグラフにラッピングします：



```python
import dask
```



```python
@dask.delayed
def inc(x):
    return x+1
@dask.delayed
def add(x, y):
    return x+y
```



```python
a = inc(1)
b = inc(2)
c = add(a, b)
```



```python
a, b, c
```




    (Delayed('inc-171697d8-9c8e-42a6-b8f2-3a1c5ff36faa'),
     Delayed('inc-5a5d5200-1e59-4d9d-b0e5-049cad501cf4'),Delayed('add-3d37a606-f83d-401e-bacb-c16f9bec7c9a'))





```python
c = c.compute()
c
```




    5



### 手配する

タスクグラフが生成された後、スケジューラはデフォルトでコンピュータのスレッドプールを使用して計算を実行します。

スレッドスケジューリングは、ローカルの `concurrent.futures.ThreadPoolExecutor` を使用して計算を実行し、Dask Array、Dask DataFrame、およびDask Delayedのデフォルト選択です。

Pythonのグローバルインタプリタロック（GIL）のため、このスケジューラは、非Pythonコードが計算に支配されている場合にのみ並列性を提供します。主にNumPy配列、Pandas DataFrames内のデジタルデータを操作したり、任意のエコシステム内の他のC/C++/Cythonベースのプロジェクトを使用したりします。



```python
# 对scheduler进行配置
dask.config.set(scheduler="threads")
```




    <dask.config.set at 0x104bfdbe0>





```python
dask.config.get("scheduler")
```




    'threads'



プロセススケジューリングは、ローカル `concurrent.futures.ProcessPoolExecutor` を使用して計算を実行し、Dask Bagのデフォルト選択です。

各タスクとそのすべての依存関系は実行のためにローカルプロセスに転送され、その結果はメインプロセスに戻ります。PythonのGIL問題を回避することができます。ただし、プロセスにデータを移動すると、特にプロセス間で大量のデータを転送する場合、パフォーマンスが低下する可能性があります。タスク間のデータ転送が含まれていない場合、出力と入力の両方が小さい場合は良い選択です。



```python
dask.config.set(scheduler='processes')
```




    <dask.config.set at 0x1046037c0>





```python
dask.config.get("scheduler")
```




    'processes'



シングルスレッド同期スケジューラは、すべての計算を1つのローカルスレッドで実行し、並列なしに実行します。一般的にデバッグや分析に使用されます。例えば、Jupyter Notebookの魔法メソッド `%debug`、 `%pdb`、 `%prun` などは、パラレルDaskスケジューリングを使用しても正常に動作しません。



```python
dask.config.set(scheduler='synchronous') 
```




    <dask.config.set at 0x104bfd9d0>





```python
dask.config.get("scheduler")
```




    'synchronous'



Daskは、単一または複数のマシンで動作する分散スケジューラを使用したより多くの制御をサポートしており、高度なスケジューラと見なすことができます。これも今のところ推奨されている方法です。

単一のマシンでも推奨される理由は、次のとおりです：

- 非同期APIアクセス、特にFutures
- パフォーマンスと進捗状況に関する意見を提供する診断ダッシュボードを提供
- データのローカル性をより復雑な方法で処理し、復数のプロセスを必要とするワークロードではマルチプロセッサスケジューラよりも効率的



```python
from dask.distributed import Client
```



```python
client = Client()
```



```python
client
```




<div>
    <div style="width: 24px; height: 24px; background-color: #e1e1e1; border: 3px solid #9D9D9D; border-radius: 5px; position: absolute;"> </div>
    <div style="margin-left: 48px;">
        <h3 style="margin-bottom: 0px;">Client</h3>
        <p style="color: #9D9D9D; margin-bottom: 0px;">Client-81d68614-e4ad-11ec-9f7c-acde48001122</p>
        <table style="width: 100%; text-align: left;">

        <tr>

            <td style="text-align: left;"><strong>Connection method:</strong> Cluster object</td>
            <td style="text-align: left;"><strong>Cluster type:</strong> distributed.LocalCluster</td>

        </tr>


            <tr>
                <td style="text-align: left;">
                    <strong>Dashboard: </strong> <a href="http://127.0.0.1:8787/status" target="_blank">http://127.0.0.1:8787/status</a>
                </td>
                <td style="text-align: left;"></td>
            </tr>


        </table>


            <details>
            <summary style="margin-bottom: 20px;"><h3 style="display: inline;">Cluster Info</h3></summary>
            <div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-mod-trusted jp-OutputArea-output">
    <div style="width: 24px; height: 24px; background-color: #e1e1e1; border: 3px solid #9D9D9D; border-radius: 5px; position: absolute;">
    </div>
    <div style="margin-left: 48px;">
        <h3 style="margin-bottom: 0px; margin-top: 0px;">LocalCluster</h3>
        <p style="color: #9D9D9D; margin-bottom: 0px;">0a242719</p>
        <table style="width: 100%; text-align: left;">
            <tr>
                <td style="text-align: left;">
                    <strong>Dashboard:</strong> <a href="http://127.0.0.1:8787/status" target="_blank">http://127.0.0.1:8787/status</a>
                </td>
                <td style="text-align: left;">
                    <strong>Workers:</strong> 4
                </td>
            </tr>
            <tr>
                <td style="text-align: left;">
                    <strong>Total threads:</strong> 4
                </td>
                <td style="text-align: left;">
                    <strong>Total memory:</strong> 8.00 GiB
                </td>
            </tr>

            <tr>
    <td style="text-align: left;"><strong>Status:</strong> running</td>
    <td style="text-align: left;"><strong>Using processes:</strong> True</td>
</tr>


        </table>

        <details>
            <summary style="margin-bottom: 20px;">
                <h3 style="display: inline;">Scheduler Info</h3>
            </summary>

            <div style="">
    <div>
        <div style="width: 24px; height: 24px; background-color: #FFF7E5; border: 3px solid #FF6132; border-radius: 5px; position: absolute;"> </div>
        <div style="margin-left: 48px;">
            <h3 style="margin-bottom: 0px;">Scheduler</h3>
            <p style="color: #9D9D9D; margin-bottom: 0px;">Scheduler-e756169f-3237-46e2-8b27-0cb87c44a843</p>
            <table style="width: 100%; text-align: left;">
                <tr>
                    <td style="text-align: left;">
                        <strong>Comm:</strong> tcp://127.0.0.1:63933
                    </td>
                    <td style="text-align: left;">
                        <strong>Workers:</strong> 4
                    </td>
                </tr>
                <tr>
                    <td style="text-align: left;">
                        <strong>Dashboard:</strong> <a href="http://127.0.0.1:8787/status" target="_blank">http://127.0.0.1:8787/status</a>
                    </td>
                    <td style="text-align: left;">
                        <strong>Total threads:</strong> 4
                    </td>
                </tr>
                <tr>
                    <td style="text-align: left;">
                        <strong>Started:</strong> Just now
                    </td>
                    <td style="text-align: left;">
                        <strong>Total memory:</strong> 8.00 GiB
                    </td>
                </tr>
            </table>
        </div>
    </div>

    <details style="margin-left: 48px;">
        <summary style="margin-bottom: 20px;">
            <h3 style="display: inline;">Workers</h3>
        </summary>


        <div style="margin-bottom: 20px;">
            <div style="width: 24px; height: 24px; background-color: #DBF5FF; border: 3px solid #4CC9FF; border-radius: 5px; position: absolute;"> </div>
            <div style="margin-left: 48px;">
            <details>
                <summary>
                    <h4 style="margin-bottom: 0px; display: inline;">Worker: 0</h4>
                </summary>
                <table style="width: 100%; text-align: left;">
                    <tr>
                        <td style="text-align: left;">
                            <strong>Comm: </strong> tcp://127.0.0.1:63957
                        </td>
                        <td style="text-align: left;">
                            <strong>Total threads: </strong> 1
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Dashboard: </strong> <a href="http://127.0.0.1:63958/status" target="_blank">http://127.0.0.1:63958/status</a>
                        </td>
                        <td style="text-align: left;">
                            <strong>Memory: </strong> 2.00 GiB
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Nanny: </strong> tcp://127.0.0.1:63939
                        </td>
                        <td style="text-align: left;"></td>
                    </tr>
                    <tr>
                        <td colspan="2" style="text-align: left;">
                            <strong>Local directory: </strong> /Users/Yam/Yam/powerful-numpy/src/skilled/dask-worker-space/worker-leteru63
                        </td>
                    </tr>





                </table>
            </details>
            </div>
        </div>

        <div style="margin-bottom: 20px;">
            <div style="width: 24px; height: 24px; background-color: #DBF5FF; border: 3px solid #4CC9FF; border-radius: 5px; position: absolute;"> </div>
            <div style="margin-left: 48px;">
            <details>
                <summary>
                    <h4 style="margin-bottom: 0px; display: inline;">Worker: 1</h4>
                </summary>
                <table style="width: 100%; text-align: left;">
                    <tr>
                        <td style="text-align: left;">
                            <strong>Comm: </strong> tcp://127.0.0.1:63948
                        </td>
                        <td style="text-align: left;">
                            <strong>Total threads: </strong> 1
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Dashboard: </strong> <a href="http://127.0.0.1:63949/status" target="_blank">http://127.0.0.1:63949/status</a>
                        </td>
                        <td style="text-align: left;">
                            <strong>Memory: </strong> 2.00 GiB
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Nanny: </strong> tcp://127.0.0.1:63936
                        </td>
                        <td style="text-align: left;"></td>
                    </tr>
                    <tr>
                        <td colspan="2" style="text-align: left;">
                            <strong>Local directory: </strong> /Users/Yam/Yam/powerful-numpy/src/skilled/dask-worker-space/worker-yl82zun0
                        </td>
                    </tr>





                </table>
            </details>
            </div>
        </div>

        <div style="margin-bottom: 20px;">
            <div style="width: 24px; height: 24px; background-color: #DBF5FF; border: 3px solid #4CC9FF; border-radius: 5px; position: absolute;"> </div>
            <div style="margin-left: 48px;">
            <details>
                <summary>
                    <h4 style="margin-bottom: 0px; display: inline;">Worker: 2</h4>
                </summary>
                <table style="width: 100%; text-align: left;">
                    <tr>
                        <td style="text-align: left;">
                            <strong>Comm: </strong> tcp://127.0.0.1:63951
                        </td>
                        <td style="text-align: left;">
                            <strong>Total threads: </strong> 1
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Dashboard: </strong> <a href="http://127.0.0.1:63952/status" target="_blank">http://127.0.0.1:63952/status</a>
                        </td>
                        <td style="text-align: left;">
                            <strong>Memory: </strong> 2.00 GiB
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Nanny: </strong> tcp://127.0.0.1:63937
                        </td>
                        <td style="text-align: left;"></td>
                    </tr>
                    <tr>
                        <td colspan="2" style="text-align: left;">
                            <strong>Local directory: </strong> /Users/Yam/Yam/powerful-numpy/src/skilled/dask-worker-space/worker-f1anlbjj
                        </td>
                    </tr>





                </table>
            </details>
            </div>
        </div>

        <div style="margin-bottom: 20px;">
            <div style="width: 24px; height: 24px; background-color: #DBF5FF; border: 3px solid #4CC9FF; border-radius: 5px; position: absolute;"> </div>
            <div style="margin-left: 48px;">
            <details>
                <summary>
                    <h4 style="margin-bottom: 0px; display: inline;">Worker: 3</h4>
                </summary>
                <table style="width: 100%; text-align: left;">
                    <tr>
                        <td style="text-align: left;">
                            <strong>Comm: </strong> tcp://127.0.0.1:63954
                        </td>
                        <td style="text-align: left;">
                            <strong>Total threads: </strong> 1
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Dashboard: </strong> <a href="http://127.0.0.1:63955/status" target="_blank">http://127.0.0.1:63955/status</a>
                        </td>
                        <td style="text-align: left;">
                            <strong>Memory: </strong> 2.00 GiB
                        </td>
                    </tr>
                    <tr>
                        <td style="text-align: left;">
                            <strong>Nanny: </strong> tcp://127.0.0.1:63938
                        </td>
                        <td style="text-align: left;"></td>
                    </tr>
                    <tr>
                        <td colspan="2" style="text-align: left;">
                            <strong>Local directory: </strong> /Users/Yam/Yam/powerful-numpy/src/skilled/dask-worker-space/worker-q3gnww72
                        </td>
                    </tr>





                </table>
            </details>
            </div>
        </div>


    </details>
</div>

        </details>
    </div>
</div>
            </details>


    </div>
</div>





```python
dask.config.get("scheduler")
```




    'dask.distributed'



上記 `Dashboard` のアドレスは、直接アクセスできます。

もちろん、上記のグローバル設定方法に加えて、コンテキストマネージャを使用したり、 `compute` を実行するときにパラメータを渡したりすることもできます。



```python
x = da.array([1,2])
x
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
with dask.config.set(scheduler="threads"):
    xo = x.compute()
```



```python
xo
```




    array([1, 2])





```python
dask.config.get("scheduler")
```




    'dask.distributed'





```python
x.compute(sheduler="threads")
```




    array([1, 2])



分散スケジューラの使用に関する詳細は、次のドキュメントを参照してください：

 [Deploy Dask Clusters—Dask documentation](https://docs.dask.org/en/stable/deploying.html)

### パフォーマンス比較



```python
%%time
rng = np.random.default_rng(42)
x = rng.normal(10, 0.1, size=(20000, 20000))
y = x.mean(axis=0)[::100]
```

    CPU times: user 8.14 s, sys: 894 ms, total: 9.03 sWall time: 9.48 s

    


```python
%%time
x = da.random.normal(
    10, 0.1, size=(20000, 20000), chunks=(1000, 1000)
)
y = x.mean(axis=0)[::100]
o = y.compute()
```

    CPU times: user 26.1 s, sys: 596 ms, total: 26.7 sWall time: 8.75 s

    


```python
x
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
                        <td> 2.98 GiB </td>
                        <td> 7.63 MiB </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (20000, 20000) </td>
                        <td> (1000, 1000) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 400 Tasks </td>
                        <td> 400 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="170" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="6" x2="120" y2="6" />
  <line x1="0" y1="12" x2="120" y2="12" />
  <line x1="0" y1="18" x2="120" y2="18" />
  <line x1="0" y1="24" x2="120" y2="24" />
  <line x1="0" y1="30" x2="120" y2="30" />
  <line x1="0" y1="36" x2="120" y2="36" />
  <line x1="0" y1="42" x2="120" y2="42" />
  <line x1="0" y1="48" x2="120" y2="48" />
  <line x1="0" y1="54" x2="120" y2="54" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="66" x2="120" y2="66" />
  <line x1="0" y1="72" x2="120" y2="72" />
  <line x1="0" y1="78" x2="120" y2="78" />
  <line x1="0" y1="84" x2="120" y2="84" />
  <line x1="0" y1="90" x2="120" y2="90" />
  <line x1="0" y1="96" x2="120" y2="96" />
  <line x1="0" y1="102" x2="120" y2="102" />
  <line x1="0" y1="108" x2="120" y2="108" />
  <line x1="0" y1="120" x2="120" y2="120" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="120" style="stroke-width:2" />
  <line x1="6" y1="0" x2="6" y2="120" />
  <line x1="12" y1="0" x2="12" y2="120" />
  <line x1="18" y1="0" x2="18" y2="120" />
  <line x1="24" y1="0" x2="24" y2="120" />
  <line x1="30" y1="0" x2="30" y2="120" />
  <line x1="36" y1="0" x2="36" y2="120" />
  <line x1="42" y1="0" x2="42" y2="120" />
  <line x1="48" y1="0" x2="48" y2="120" />
  <line x1="54" y1="0" x2="54" y2="120" />
  <line x1="60" y1="0" x2="60" y2="120" />
  <line x1="66" y1="0" x2="66" y2="120" />
  <line x1="72" y1="0" x2="72" y2="120" />
  <line x1="78" y1="0" x2="78" y2="120" />
  <line x1="84" y1="0" x2="84" y2="120" />
  <line x1="90" y1="0" x2="90" y2="120" />
  <line x1="96" y1="0" x2="96" y2="120" />
  <line x1="102" y1="0" x2="102" y2="120" />
  <line x1="108" y1="0" x2="108" y2="120" />
  <line x1="120" y1="0" x2="120" y2="120" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,120.0 0.0,120.0" style="fill:#8B4903A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="140.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20000</text>
  <text x="140.000000" y="60.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,140.000000,60.000000)">20000</text>
</svg>
        </td>
    </tr>
</table>



Daskはより速く完了しますが、総CPU時間はより多く使用されます。



```python
%%time
x = da.random.normal(
    10, 0.1, size=(20000, 20000), chunks=(20000, 20000)
)
y = x.mean(axis=0)[::100]
o = y.compute()
```

    CPU times: user 39.3 s, sys: 8.38 s, total: 47.7 sWall time: 59.3 s

    


```python
x
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
                        <td> 2.98 GiB </td>
                        <td> 2.98 GiB </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (20000, 20000) </td>
                        <td> (20000, 20000) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 1 Tasks </td>
                        <td> 1 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="170" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="120" x2="120" y2="120" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="120" style="stroke-width:2" />
  <line x1="120" y1="0" x2="120" y2="120" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,120.0 0.0,120.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="140.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20000</text>
  <text x="140.000000" y="60.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,140.000000,60.000000)">20000</text>
</svg>
        </td>
    </tr>
</table>





```python
%%time
x = da.random.normal(
    10, 0.1, size=(20000, 20000), chunks=(25, 25)
)
y = x.mean(axis=0)[::100]
o = y.compute()
```

    CPU times: user 3min 30s, sys: 36.4 s, total: 4min 7sWall time: 4min 16s

    


```python
x
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
                        <td> 2.98 GiB </td>
                        <td> 4.88 kiB </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (20000, 20000) </td>
                        <td> (25, 25) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 640000 Tasks </td>
                        <td> 640000 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="170" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="6" x2="120" y2="6" />
  <line x1="0" y1="12" x2="120" y2="12" />
  <line x1="0" y1="18" x2="120" y2="18" />
  <line x1="0" y1="25" x2="120" y2="25" />
  <line x1="0" y1="31" x2="120" y2="31" />
  <line x1="0" y1="37" x2="120" y2="37" />
  <line x1="0" y1="44" x2="120" y2="44" />
  <line x1="0" y1="50" x2="120" y2="50" />
  <line x1="0" y1="56" x2="120" y2="56" />
  <line x1="0" y1="63" x2="120" y2="63" />
  <line x1="0" y1="69" x2="120" y2="69" />
  <line x1="0" y1="75" x2="120" y2="75" />
  <line x1="0" y1="82" x2="120" y2="82" />
  <line x1="0" y1="88" x2="120" y2="88" />
  <line x1="0" y1="94" x2="120" y2="94" />
  <line x1="0" y1="100" x2="120" y2="100" />
  <line x1="0" y1="107" x2="120" y2="107" />
  <line x1="0" y1="113" x2="120" y2="113" />
  <line x1="0" y1="120" x2="120" y2="120" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="120" style="stroke-width:2" />
  <line x1="6" y1="0" x2="6" y2="120" />
  <line x1="12" y1="0" x2="12" y2="120" />
  <line x1="18" y1="0" x2="18" y2="120" />
  <line x1="25" y1="0" x2="25" y2="120" />
  <line x1="31" y1="0" x2="31" y2="120" />
  <line x1="37" y1="0" x2="37" y2="120" />
  <line x1="44" y1="0" x2="44" y2="120" />
  <line x1="50" y1="0" x2="50" y2="120" />
  <line x1="56" y1="0" x2="56" y2="120" />
  <line x1="63" y1="0" x2="63" y2="120" />
  <line x1="69" y1="0" x2="69" y2="120" />
  <line x1="75" y1="0" x2="75" y2="120" />
  <line x1="82" y1="0" x2="82" y2="120" />
  <line x1="88" y1="0" x2="88" y2="120" />
  <line x1="94" y1="0" x2="94" y2="120" />
  <line x1="100" y1="0" x2="100" y2="120" />
  <line x1="107" y1="0" x2="107" y2="120" />
  <line x1="113" y1="0" x2="113" y2="120" />
  <line x1="120" y1="0" x2="120" y2="120" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,120.0 0.0,120.0" style="fill:#8B4903A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="140.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20000</text>
  <text x="140.000000" y="60.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,140.000000,60.000000)">20000</text>
</svg>
        </td>
    </tr>
</table>



## Xarray

文書：[xarray: N-D labeled arrays and datasets in Python](https://xarray.pydata.org/en/stable/index.html)

xarrayは、元のNumPyのような配列の上に次元、座標、属性の形式のラベルを導入し、より直感的で簡潔でエラーの発生しにくい開発体験を提供します。

関連するツールキットは以下を参照してください：

[Installation](https://xarray.pydata.org/en/stable/getting-started-guide/installing.html)

xarrayには、NumPyとPandasの上に構築され、拡張された2つのコアデータ構造があり、いずれも多次元である。

- DataArray：ラベル付きN次元配列
- Dataset：多次元メモリ配列データベース



```python
import xarray as xr
xr.__version__
```




    '2022.3.0'



### 作成



```python
rng = np.random.default_rng(42)
a = rng.normal(size=(2,3))
a
```




    array([[ 0.30471708, -1.03998411,  0.7504512 ],
           [ 0.94056472, -1.95103519, -1.30217951]])





```python
data = xr.DataArray(a, dims=("x", "y"), coords={"x": [10, 20]})
```



```python
data
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (x: 2, y: 3)&gt;array([[ 0.30471708, -1.03998411,  0.7504512 ],
       [ 0.94056472, -1.95103519, -1.30217951]])
Coordinates:
  * x        (x) int64 10 20
Dimensions without coordinates: y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 2</li><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-be8db020-4ffe-4da5-aa04-900affd6f720' class='xr-array-in' type='checkbox' checked><label for='section-be8db020-4ffe-4da5-aa04-900affd6f720' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.3047 -1.04 0.7505 0.9406 -1.951 -1.302</span></div><div class='xr-array-data'><pre>array([[ 0.30471708, -1.03998411,  0.7504512 ],
       [ 0.94056472, -1.95103519, -1.30217951]])</pre></div></div></li><li class='xr-section-item'><input id='section-7ac6f09f-3504-4ac6-98f1-cac2b6f47b9a' class='xr-section-summary-in' type='checkbox'  checked><label for='section-7ac6f09f-3504-4ac6-98f1-cac2b6f47b9a' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 20</div><input id='attrs-e9d38272-8463-4d90-9f95-fe41956470b7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e9d38272-8463-4d90-9f95-fe41956470b7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d3240eb3-bf06-4dcf-a993-3cec57437a45' class='xr-var-data-in' type='checkbox'><label for='data-d3240eb3-bf06-4dcf-a993-3cec57437a45' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([10, 20])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f4da0e7e-7818-4592-99f8-5a91d57de1d2' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-f4da0e7e-7818-4592-99f8-5a91d57de1d2' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
data.x
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;x&#x27; (x: 2)&gt;array([10, 20])Coordinates:
  * x        (x) int64 10 20</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'x'</div><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 2</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-7817c841-aa1e-4bad-b2eb-99244e80c087' class='xr-array-in' type='checkbox' checked><label for='section-7817c841-aa1e-4bad-b2eb-99244e80c087' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>10 20</span></div><div class='xr-array-data'><pre>array([10, 20])</pre></div></div></li><li class='xr-section-item'><input id='section-7841a6c5-3c93-4213-85cb-3798d5e03009' class='xr-section-summary-in' type='checkbox'  checked><label for='section-7841a6c5-3c93-4213-85cb-3798d5e03009' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 20</div><input id='attrs-c25b2dc7-0e8d-47a4-8c56-81b9be6c7d6f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c25b2dc7-0e8d-47a4-8c56-81b9be6c7d6f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7e07ee95-5aa6-4498-af12-5ca138f557bd' class='xr-var-data-in' type='checkbox'><label for='data-7e07ee95-5aa6-4498-af12-5ca138f557bd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([10, 20])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-21c64bef-a8f7-4f8c-8a90-deb1470d2c48' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-21c64bef-a8f7-4f8c-8a90-deb1470d2c48' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
data.y
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;y&#x27; (y: 3)&gt;array([0, 1, 2])
Dimensions without coordinates: y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'y'</div><ul class='xr-dim-list'><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-2e46fb59-da66-4735-a06e-5271827d8c8a' class='xr-array-in' type='checkbox' checked><label for='section-2e46fb59-da66-4735-a06e-5271827d8c8a' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0 1 2</span></div><div class='xr-array-data'><pre>array([0, 1, 2])</pre></div></div></li><li class='xr-section-item'><input id='section-696fdb34-d9e9-4523-bf8d-99258e2db274' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-696fdb34-d9e9-4523-bf8d-99258e2db274' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-12c41a1a-a16a-48e8-b83b-30923ce73bf4' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-12c41a1a-a16a-48e8-b83b-30923ce73bf4' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
data.values
```




    array([[ 0.30471708, -1.03998411,  0.7504512 ],
           [ 0.94056472, -1.95103519, -1.30217951]])





```python
data.dims
```




    ('x', 'y')





```python
data.coords
```




    Coordinates:
      * x        (x) int64 10 20





```python
data.attrs
```




    {}





```python
data.x.values
```




    array([10, 20])





```python
data.y.values
```




    array([0, 1, 2])



### 索引



```python
data[0,:]
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (y: 3)&gt;array([ 0.30471708, -1.03998411,  0.7504512 ])Coordinates:
    x        int64 10
Dimensions without coordinates: y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-691730d9-b0e6-48ac-a6c8-55aa3a2ef292' class='xr-array-in' type='checkbox' checked><label for='section-691730d9-b0e6-48ac-a6c8-55aa3a2ef292' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.3047 -1.04 0.7505</span></div><div class='xr-array-data'><pre>array([ 0.30471708, -1.03998411,  0.7504512 ])</pre></div></div></li><li class='xr-section-item'><input id='section-65157f7b-fc05-464e-adce-45d1032e0a85' class='xr-section-summary-in' type='checkbox'  checked><label for='section-65157f7b-fc05-464e-adce-45d1032e0a85' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>x</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10</div><input id='attrs-2a7c40d2-ddc5-4fa4-afbe-850c88fbf1ad' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2a7c40d2-ddc5-4fa4-afbe-850c88fbf1ad' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-39f14374-3a9f-46cf-96ca-111e62bad774' class='xr-var-data-in' type='checkbox'><label for='data-39f14374-3a9f-46cf-96ca-111e62bad774' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(10)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-42149307-8b70-4a81-93d3-6e9a9d818034' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-42149307-8b70-4a81-93d3-6e9a9d818034' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
data.loc[10]
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (y: 3)&gt;array([ 0.30471708, -1.03998411,  0.7504512 ])Coordinates:
    x        int64 10
Dimensions without coordinates: y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-424b1407-eeda-4a50-ba58-73c4a00b6165' class='xr-array-in' type='checkbox' checked><label for='section-424b1407-eeda-4a50-ba58-73c4a00b6165' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.3047 -1.04 0.7505</span></div><div class='xr-array-data'><pre>array([ 0.30471708, -1.03998411,  0.7504512 ])</pre></div></div></li><li class='xr-section-item'><input id='section-7738c6f7-aced-413b-b2b2-022a52b49c98' class='xr-section-summary-in' type='checkbox'  checked><label for='section-7738c6f7-aced-413b-b2b2-022a52b49c98' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>x</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10</div><input id='attrs-429873b0-1c89-462b-b652-ab4a2ce6511d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-429873b0-1c89-462b-b652-ab4a2ce6511d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3e9e314e-bbc0-4286-894e-f012bcea1356' class='xr-var-data-in' type='checkbox'><label for='data-3e9e314e-bbc0-4286-894e-f012bcea1356' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(10)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d150451b-08b7-4620-a24a-050c73cd92f2' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-d150451b-08b7-4620-a24a-050c73cd92f2' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
data.loc[0]
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /usr/local/lib/python3.8/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       3620             try:
    -> 3621                 return self._engine.get_loc(casted_key)
       3622             except KeyError as err:
    

    /usr/local/lib/python3.8/site-packages/pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    /usr/local/lib/python3.8/site-packages/pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()
    

    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()
    

    KeyError: 0

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    <ipython-input-68-0dc99c936dd2> in <module>
    ----> 1 data.loc[0]
    

    /usr/local/lib/python3.8/site-packages/xarray/core/dataarray.py in __getitem__(self, key)
        197             labels = indexing.expanded_indexer(key, self.data_array.ndim)198             key = dict(zip(self.data_array.dims, labels))
    --> 199         return self.data_array.sel(key)
        200 201     def __setitem__(self, key, value) -> None:
    

    /usr/local/lib/python3.8/site-packages/xarray/core/dataarray.py in sel(self, indexers, method, tolerance, drop, **indexers_kwargs)
       1327         Dimensions without coordinates: points1328         """
    -> 1329         ds = self._to_temp_dataset().sel(
       1330             indexers=indexers,1331             drop=drop,
    

    /usr/local/lib/python3.8/site-packages/xarray/core/dataset.py in sel(self, indexers, method, tolerance, drop, **indexers_kwargs)
       2499         """2500         indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
    -> 2501         pos_indexers, new_indexes = remap_label_indexers(
       2502             self, indexers=indexers, method=method, tolerance=tolerance2503         )
    

    /usr/local/lib/python3.8/site-packages/xarray/core/coordinates.py in remap_label_indexers(obj, indexers, method, tolerance, **indexers_kwargs)
        419     }420 
    --> 421     pos_indexers, new_indexes = indexing.remap_label_indexers(
        422         obj, v_indexers, method=method, tolerance=tolerance423     )
    

    /usr/local/lib/python3.8/site-packages/xarray/core/indexing.py in remap_label_indexers(data_obj, indexers, method, tolerance)
        119     for dim, index in indexes.items():120         labels = grouped_indexers[dim]
    --> 121         idxr, new_idx = index.query(labels, method=method, tolerance=tolerance)
        122         pos_indexers[dim] = idxr123         if new_idx is not None:
    

    /usr/local/lib/python3.8/site-packages/xarray/core/indexes.py in query(self, labels, method, tolerance)
        239                             )240                     else:
    --> 241                         indexer = self.index.get_loc(label_value)
        242             elif label.dtype.kind == "b":243                 indexer = label
    

    /usr/local/lib/python3.8/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       3621                 return self._engine.get_loc(casted_key)3622             except KeyError as err:
    -> 3623                 raise KeyError(key) from err
       3624             except TypeError:3625                 # If we have a listlike key, _check_indexing_error will raise
    

    KeyError: 0




```python
# integer select
data.isel(x=0)
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (y: 3)&gt;array([ 0.30471708, -1.03998411,  0.7504512 ])Coordinates:
    x        int64 10
Dimensions without coordinates: y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-f2427962-bd91-4784-bc1f-6f9dba4e8688' class='xr-array-in' type='checkbox' checked><label for='section-f2427962-bd91-4784-bc1f-6f9dba4e8688' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.3047 -1.04 0.7505</span></div><div class='xr-array-data'><pre>array([ 0.30471708, -1.03998411,  0.7504512 ])</pre></div></div></li><li class='xr-section-item'><input id='section-4c1eb76e-9579-454d-8b32-5ba0e093f6ae' class='xr-section-summary-in' type='checkbox'  checked><label for='section-4c1eb76e-9579-454d-8b32-5ba0e093f6ae' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>x</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10</div><input id='attrs-27dd325f-1be4-448e-9783-8424d415088e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-27dd325f-1be4-448e-9783-8424d415088e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e762e94e-e70f-4f2e-8716-54940fc9d3e4' class='xr-var-data-in' type='checkbox'><label for='data-e762e94e-e70f-4f2e-8716-54940fc9d3e4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(10)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-334e461b-bbe9-4452-a2fe-759df4c4aa3e' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-334e461b-bbe9-4452-a2fe-759df4c4aa3e' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
data.isel(y=0)
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (x: 2)&gt;array([0.30471708, 0.94056472])Coordinates:
  * x        (x) int64 10 20</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 2</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-827c3059-9b7c-48c8-83d0-4bf62da7eb05' class='xr-array-in' type='checkbox' checked><label for='section-827c3059-9b7c-48c8-83d0-4bf62da7eb05' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.3047 0.9406</span></div><div class='xr-array-data'><pre>array([0.30471708, 0.94056472])</pre></div></div></li><li class='xr-section-item'><input id='section-ba70c275-a644-4750-8d01-273bc3d08af7' class='xr-section-summary-in' type='checkbox'  checked><label for='section-ba70c275-a644-4750-8d01-273bc3d08af7' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 20</div><input id='attrs-e9bfaaee-ac32-484e-8dd9-6d3afc5c0d2b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e9bfaaee-ac32-484e-8dd9-6d3afc5c0d2b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-186c6f79-81f2-4e43-8a59-f7d8b3bce432' class='xr-var-data-in' type='checkbox'><label for='data-186c6f79-81f2-4e43-8a59-f7d8b3bce432' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([10, 20])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-a7d18cff-4d03-4ccf-a88b-93b154552a55' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-a7d18cff-4d03-4ccf-a88b-93b154552a55' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
# 直接select
data.sel(x=10)
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (y: 3)&gt;array([ 0.30471708, -1.03998411,  0.7504512 ])Coordinates:
    x        int64 10
Dimensions without coordinates: y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-a2294a8c-f809-4af2-9209-306faf18ae2b' class='xr-array-in' type='checkbox' checked><label for='section-a2294a8c-f809-4af2-9209-306faf18ae2b' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.3047 -1.04 0.7505</span></div><div class='xr-array-data'><pre>array([ 0.30471708, -1.03998411,  0.7504512 ])</pre></div></div></li><li class='xr-section-item'><input id='section-49cf798e-06b3-4dbf-ac12-965ba626600b' class='xr-section-summary-in' type='checkbox'  checked><label for='section-49cf798e-06b3-4dbf-ac12-965ba626600b' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>x</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10</div><input id='attrs-d97a938a-5540-4370-b366-d7235bf836f6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d97a938a-5540-4370-b366-d7235bf836f6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f2357231-f86c-4dba-9336-0f70a06645ad' class='xr-var-data-in' type='checkbox'><label for='data-f2357231-f86c-4dba-9336-0f70a06645ad' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(10)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e3fde19e-63c3-4184-bdd1-3fa35d61fa41' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-e3fde19e-63c3-4184-bdd1-3fa35d61fa41' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



詳細は次のとおりです：

[Indexing and selecting data](https://xarray.pydata.org/en/stable/user-guide/indexing.html#indexing)

### プロパティ

DataArrayを設定するとき、メタデータプロパティを設定するのは通常良い実践です。一般的な属性には、 `long_name`、 `units` などがあります。



```python
data.attrs["long_name"] = "random velocity"
data.attrs["units"] = "metres/sec"
data.attrs["description"] = "A random variable created as an example."
```



```python
data
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (x: 2, y: 3)&gt;array([[ 0.30471708, -1.03998411,  0.7504512 ],
       [ 0.94056472, -1.95103519, -1.30217951]])
Coordinates:
  * x        (x) int64 10 20
Dimensions without coordinates: yAttributes:
    long_name:    random velocityunits:        metres/secdescription:  A random variable created as an example.</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 2</li><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-668f4a8d-9b44-4100-a7da-a8c647738082' class='xr-array-in' type='checkbox' checked><label for='section-668f4a8d-9b44-4100-a7da-a8c647738082' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.3047 -1.04 0.7505 0.9406 -1.951 -1.302</span></div><div class='xr-array-data'><pre>array([[ 0.30471708, -1.03998411,  0.7504512 ],
       [ 0.94056472, -1.95103519, -1.30217951]])</pre></div></div></li><li class='xr-section-item'><input id='section-537f2a98-dc3c-4286-aee1-3970780ca6b0' class='xr-section-summary-in' type='checkbox'  checked><label for='section-537f2a98-dc3c-4286-aee1-3970780ca6b0' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 20</div><input id='attrs-6b502d43-88c4-4c1d-a147-26d477ed315c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6b502d43-88c4-4c1d-a147-26d477ed315c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-52fe6657-5648-460b-a97a-d403017fc27a' class='xr-var-data-in' type='checkbox'><label for='data-52fe6657-5648-460b-a97a-d403017fc27a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([10, 20])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-312328c5-35e5-4e7a-b873-32153b71ca0f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-312328c5-35e5-4e7a-b873-32153b71ca0f' class='xr-section-summary' >Attributes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>long_name :</span></dt><dd>random velocity</dd><dt><span>units :</span></dt><dd>metres/sec</dd><dt><span>description :</span></dt><dd>A random variable created as an example.</dd></dl></div></li></ul></div></div>





```python
data.attrs
```




    {'long_name': 'random velocity',
     'units': 'metres/sec','description': 'A random variable created as an example.'}





```python
# 给坐标设置属性
data.x.attrs["units"] = "x units"
```



```python
data.x
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;x&#x27; (x: 2)&gt;array([10, 20])Coordinates:
  * x        (x) int64 10 20
Attributes:
    units:    x units</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'x'</div><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 2</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-d7b7e6f1-cbf1-4cde-b846-bb9952027abe' class='xr-array-in' type='checkbox' checked><label for='section-d7b7e6f1-cbf1-4cde-b846-bb9952027abe' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>10 20</span></div><div class='xr-array-data'><pre>array([10, 20])</pre></div></div></li><li class='xr-section-item'><input id='section-a8931639-2928-449f-b43a-6d1653911a3b' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a8931639-2928-449f-b43a-6d1653911a3b' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 20</div><input id='attrs-f7a64b2f-9f22-4ee7-a6e3-6ed8ee8ea5f7' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-f7a64b2f-9f22-4ee7-a6e3-6ed8ee8ea5f7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4c87ca56-df72-46d8-a4cc-72e5a58b1de5' class='xr-var-data-in' type='checkbox'><label for='data-4c87ca56-df72-46d8-a4cc-72e5a58b1de5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>units :</span></dt><dd>x units</dd></dl></div><div class='xr-var-data'><pre>array([10, 20])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e768e532-344f-4fbc-b651-d46ceb159990' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e768e532-344f-4fbc-b651-d46ceb159990' class='xr-section-summary' >Attributes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>units :</span></dt><dd>x units</dd></dl></div></li></ul></div></div>



### 計算



```python
data + 10
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (x: 2, y: 3)&gt;array([[10.30471708,  8.96001589, 10.7504512 ],
       [10.94056472,  8.04896481,  8.69782049]])
Coordinates:
  * x        (x) int64 10 20
Dimensions without coordinates: y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 2</li><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-33f27b23-b4c8-4fb2-bd06-085384720b7c' class='xr-array-in' type='checkbox' checked><label for='section-33f27b23-b4c8-4fb2-bd06-085384720b7c' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>10.3 8.96 10.75 10.94 8.049 8.698</span></div><div class='xr-array-data'><pre>array([[10.30471708,  8.96001589, 10.7504512 ],
       [10.94056472,  8.04896481,  8.69782049]])</pre></div></div></li><li class='xr-section-item'><input id='section-52bfa77e-6549-494d-be84-290b8ee6f258' class='xr-section-summary-in' type='checkbox'  checked><label for='section-52bfa77e-6549-494d-be84-290b8ee6f258' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 20</div><input id='attrs-15d8e6d1-3a30-45f3-8a4f-b1848ff027ca' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-15d8e6d1-3a30-45f3-8a4f-b1848ff027ca' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f2e767e4-5215-4035-913e-bcb38bfd17f6' class='xr-var-data-in' type='checkbox'><label for='data-f2e767e4-5215-4035-913e-bcb38bfd17f6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>units :</span></dt><dd>x units</dd></dl></div><div class='xr-var-data'><pre>array([10, 20])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-38807bab-79e6-4244-8c4c-2b7cfcacb875' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-38807bab-79e6-4244-8c4c-2b7cfcacb875' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
np.sin(data)
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (x: 2, y: 3)&gt;array([[ 0.3000233 , -0.86239618,  0.68196883],
       [ 0.80789103, -0.92857601, -0.96413891]])
Coordinates:
  * x        (x) int64 10 20
Dimensions without coordinates: yAttributes:
    long_name:    random velocityunits:        metres/secdescription:  A random variable created as an example.</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 2</li><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-aa69332c-27b1-48c6-b171-399d28b15465' class='xr-array-in' type='checkbox' checked><label for='section-aa69332c-27b1-48c6-b171-399d28b15465' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.3 -0.8624 0.682 0.8079 -0.9286 -0.9641</span></div><div class='xr-array-data'><pre>array([[ 0.3000233 , -0.86239618,  0.68196883],
       [ 0.80789103, -0.92857601, -0.96413891]])</pre></div></div></li><li class='xr-section-item'><input id='section-3504e098-9b19-4bbb-9546-9e2b0f4c6ecb' class='xr-section-summary-in' type='checkbox'  checked><label for='section-3504e098-9b19-4bbb-9546-9e2b0f4c6ecb' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 20</div><input id='attrs-fc8e498d-6d1b-445b-b025-15d728deff83' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-fc8e498d-6d1b-445b-b025-15d728deff83' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-614b2cca-423b-4e49-80c7-66ae5ec0d69b' class='xr-var-data-in' type='checkbox'><label for='data-614b2cca-423b-4e49-80c7-66ae5ec0d69b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>units :</span></dt><dd>x units</dd></dl></div><div class='xr-var-data'><pre>array([10, 20])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-effe754f-b1e4-4d57-9e4d-7f946ebdee9a' class='xr-section-summary-in' type='checkbox'  checked><label for='section-effe754f-b1e4-4d57-9e4d-7f946ebdee9a' class='xr-section-summary' >Attributes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>long_name :</span></dt><dd>random velocity</dd><dt><span>units :</span></dt><dd>metres/sec</dd><dt><span>description :</span></dt><dd>A random variable created as an example.</dd></dl></div></li></ul></div></div>





```python
data.T
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (y: 3, x: 2)&gt;array([[ 0.30471708,  0.94056472],
       [-1.03998411, -1.95103519],[ 0.7504512 , -1.30217951]])
Coordinates:
  * x        (x) int64 10 20
Dimensions without coordinates: yAttributes:
    long_name:    random velocityunits:        metres/secdescription:  A random variable created as an example.</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span>y</span>: 3</li><li><span class='xr-has-index'>x</span>: 2</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-00dc2fae-bf8e-4b98-b8e3-00d47144cf02' class='xr-array-in' type='checkbox' checked><label for='section-00dc2fae-bf8e-4b98-b8e3-00d47144cf02' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.3047 0.9406 -1.04 -1.951 0.7505 -1.302</span></div><div class='xr-array-data'><pre>array([[ 0.30471708,  0.94056472],
       [-1.03998411, -1.95103519],
       [ 0.7504512 , -1.30217951]])</pre></div></div></li><li class='xr-section-item'><input id='section-4ae42005-8eaf-4390-b33c-6402e8427bce' class='xr-section-summary-in' type='checkbox'  checked><label for='section-4ae42005-8eaf-4390-b33c-6402e8427bce' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 20</div><input id='attrs-bacb713d-82fb-48d0-932b-cc2dd5638c5a' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-bacb713d-82fb-48d0-932b-cc2dd5638c5a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-52eb9c4a-b5b9-40ed-b453-5042f3570b58' class='xr-var-data-in' type='checkbox'><label for='data-52eb9c4a-b5b9-40ed-b453-5042f3570b58' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>units :</span></dt><dd>x units</dd></dl></div><div class='xr-var-data'><pre>array([10, 20])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f5b189ad-4d63-4f38-ba3d-61b5cbe97259' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f5b189ad-4d63-4f38-ba3d-61b5cbe97259' class='xr-section-summary' >Attributes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>long_name :</span></dt><dd>random velocity</dd><dt><span>units :</span></dt><dd>metres/sec</dd><dt><span>description :</span></dt><dd>A random variable created as an example.</dd></dl></div></li></ul></div></div>





```python
data.sum()
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray ()&gt;
array(-2.29746581)</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-2f1cb6ac-0bcf-47ef-94b3-733fb62988e0' class='xr-array-in' type='checkbox' checked><label for='section-2f1cb6ac-0bcf-47ef-94b3-733fb62988e0' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>-2.297</span></div><div class='xr-array-data'><pre>array(-2.29746581)</pre></div></div></li><li class='xr-section-item'><input id='section-e021311a-84f8-4287-9581-b75426b77fb2' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-e021311a-84f8-4287-9581-b75426b77fb2' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-cb5da52c-f78b-4536-93a8-6fb1d691b9a7' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-cb5da52c-f78b-4536-93a8-6fb1d691b9a7' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
data.mean(dim="x")
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (y: 3)&gt;array([ 0.6226409 , -1.49550965, -0.27586416])
Dimensions without coordinates: y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-835b8899-4d86-4ece-b6d2-a2b343071b72' class='xr-array-in' type='checkbox' checked><label for='section-835b8899-4d86-4ece-b6d2-a2b343071b72' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.6226 -1.496 -0.2759</span></div><div class='xr-array-data'><pre>array([ 0.6226409 , -1.49550965, -0.27586416])</pre></div></div></li><li class='xr-section-item'><input id='section-c62c9340-2b23-403d-9c19-dbfac08d1d4b' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-c62c9340-2b23-403d-9c19-dbfac08d1d4b' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-6c2adc47-95f4-4b02-ae12-241b13bfbae0' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-6c2adc47-95f4-4b02-ae12-241b13bfbae0' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



NumPyと比較してみましょう：



```python
data.mean(axis=0)
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (y: 3)&gt;array([ 0.6226409 , -1.49550965, -0.27586416])
Dimensions without coordinates: y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-cdc21b7b-d491-4fae-9688-ae0f4d0de4bf' class='xr-array-in' type='checkbox' checked><label for='section-cdc21b7b-d491-4fae-9688-ae0f4d0de4bf' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.6226 -1.496 -0.2759</span></div><div class='xr-array-data'><pre>array([ 0.6226409 , -1.49550965, -0.27586416])</pre></div></div></li><li class='xr-section-item'><input id='section-473efec1-a264-4d34-be73-01e37ea94b55' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-473efec1-a264-4d34-be73-01e37ea94b55' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-302f49a3-64f6-417c-a881-4e5b87700108' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-302f49a3-64f6-417c-a881-4e5b87700108' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
data.mean(dim="y")
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (x: 2)&gt;array([ 0.00506139, -0.77088333])Coordinates:
  * x        (x) int64 10 20</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 2</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-086007f0-3b2f-4667-b771-9994e106dfd9' class='xr-array-in' type='checkbox' checked><label for='section-086007f0-3b2f-4667-b771-9994e106dfd9' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.005061 -0.7709</span></div><div class='xr-array-data'><pre>array([ 0.00506139, -0.77088333])</pre></div></div></li><li class='xr-section-item'><input id='section-f5318ceb-c894-4db4-9665-36c7bce2f66c' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f5318ceb-c894-4db4-9665-36c7bce2f66c' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 20</div><input id='attrs-a0d34ee6-906d-45eb-9c1c-d0ce147f096f' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-a0d34ee6-906d-45eb-9c1c-d0ce147f096f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f23e798b-2a73-4d4e-ade7-6e06f93488aa' class='xr-var-data-in' type='checkbox'><label for='data-f23e798b-2a73-4d4e-ade7-6e06f93488aa' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>units :</span></dt><dd>x units</dd></dl></div><div class='xr-var-data'><pre>array([10, 20])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-3aa93d24-0cd5-45d1-ad14-18acb83b2a58' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-3aa93d24-0cd5-45d1-ad14-18acb83b2a58' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



次元名に基づくブロードキャスト（位置合わせに仮想寸法を挿入する必要はありません）：



```python
[data.coords["y"]]
```




    [<xarray.DataArray 'y' (y: 3)>
     array([0, 1, 2])Dimensions without coordinates: y]





```python
rng = np.random.default_rng(42)
a = xr.DataArray(rng.integers(0, 10, 3), [data.coords["y"]])
b = xr.DataArray(rng.integers(0, 10, 4), dims="z")
```



```python
a
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (y: 3)&gt;array([0, 7, 6])Coordinates:
  * y        (y) int64 0 1 2</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-5ee65a23-6d8e-4d5c-a63a-de559fff130c' class='xr-array-in' type='checkbox' checked><label for='section-5ee65a23-6d8e-4d5c-a63a-de559fff130c' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0 7 6</span></div><div class='xr-array-data'><pre>array([0, 7, 6])</pre></div></div></li><li class='xr-section-item'><input id='section-8f1fef8e-7426-4405-bf1b-99d842e075de' class='xr-section-summary-in' type='checkbox'  checked><label for='section-8f1fef8e-7426-4405-bf1b-99d842e075de' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>y</span></div><div class='xr-var-dims'>(y)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2</div><input id='attrs-a95fddcc-2cb4-49f3-abb9-c6cbe5d5136d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a95fddcc-2cb4-49f3-abb9-c6cbe5d5136d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-faa489f5-ec5e-4c7e-841a-925bdae200eb' class='xr-var-data-in' type='checkbox'><label for='data-faa489f5-ec5e-4c7e-841a-925bdae200eb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-80e57178-f9e8-45be-b0b2-46ca8cfd55dc' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-80e57178-f9e8-45be-b0b2-46ca8cfd55dc' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
b
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (z: 4)&gt;array([4, 4, 8, 0])
Dimensions without coordinates: z</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span>z</span>: 4</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-16b602e8-eb77-43f7-b6cf-fdab39c51c98' class='xr-array-in' type='checkbox' checked><label for='section-16b602e8-eb77-43f7-b6cf-fdab39c51c98' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>4 4 8 0</span></div><div class='xr-array-data'><pre>array([4, 4, 8, 0])</pre></div></div></li><li class='xr-section-item'><input id='section-3ca67bb6-9b42-4968-8d57-9a035508511d' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-3ca67bb6-9b42-4968-8d57-9a035508511d' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-82951e98-a4f2-4cae-84fa-082775bbc474' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-82951e98-a4f2-4cae-84fa-082775bbc474' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
a + b
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (y: 3, z: 4)&gt;array([[ 4,  4,  8,  0],
       [11, 11, 15,  7],[10, 10, 14,  6]])
Coordinates:
  * y        (y) int64 0 1 2
Dimensions without coordinates: z</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>y</span>: 3</li><li><span>z</span>: 4</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-63ff8c0b-9839-4b6e-af90-acd023312869' class='xr-array-in' type='checkbox' checked><label for='section-63ff8c0b-9839-4b6e-af90-acd023312869' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>4 4 8 0 11 11 15 7 10 10 14 6</span></div><div class='xr-array-data'><pre>array([[ 4,  4,  8,  0],
       [11, 11, 15,  7],
       [10, 10, 14,  6]])</pre></div></div></li><li class='xr-section-item'><input id='section-a3166a27-2bc4-4a6c-960b-5a1f336954bc' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a3166a27-2bc4-4a6c-960b-5a1f336954bc' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>y</span></div><div class='xr-var-dims'>(y)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2</div><input id='attrs-2f1bd325-e09f-40ac-9bda-02d18aab80ad' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2f1bd325-e09f-40ac-9bda-02d18aab80ad' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-54f5b0ce-79d7-4fbe-8238-1c19788ef362' class='xr-var-data-in' type='checkbox'><label for='data-54f5b0ce-79d7-4fbe-8238-1c19788ef362' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-5df669c2-07da-48f5-9f0e-60c985207a7d' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-5df669c2-07da-48f5-9f0e-60c985207a7d' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
a.values + b.values
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-32-db9ccfe54ec4> in <module>
    ----> 1 a.values + b.values
    

    ValueError: operands could not be broadcast together with shapes (3,) (4,) 


ほとんどの場合、次元順序について心配する必要はありません：



```python
data.shape
```




    (2, 3)





```python
data - data.T
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (x: 2, y: 3)&gt;array([[0., 0., 0.],
       [0., 0., 0.]])
Coordinates:
  * x        (x) int64 10 20
Dimensions without coordinates: y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 2</li><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-c671cfe3-170b-4349-80f7-406acb155733' class='xr-array-in' type='checkbox' checked><label for='section-c671cfe3-170b-4349-80f7-406acb155733' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.0 0.0 0.0 0.0 0.0 0.0</span></div><div class='xr-array-data'><pre>array([[0., 0., 0.],
       [0., 0., 0.]])</pre></div></div></li><li class='xr-section-item'><input id='section-e3c84388-08c7-48e7-bce0-cb6b9806b817' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e3c84388-08c7-48e7-bce0-cb6b9806b817' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 20</div><input id='attrs-cf4f87e0-3b4c-45b6-a032-12bc8a4d7062' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cf4f87e0-3b4c-45b6-a032-12bc8a4d7062' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7124c363-fd8c-4eb2-8040-e07e126b3a5c' class='xr-var-data-in' type='checkbox'><label for='data-7124c363-fd8c-4eb2-8040-e07e126b3a5c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([10, 20])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-469dea3d-e833-475a-bdaa-3907dad76cdf' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-469dea3d-e833-475a-bdaa-3907dad76cdf' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



詳細は次のとおりです：

[Computation](https://xarray.pydata.org/en/stable/user-guide/computation.html#comput)

### GroupBy



```python
rng = np.random.default_rng(42)
a = xr.DataArray(rng.integers(1, 10, (2, 3)), dims=("x", "y"))
a
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (x: 2, y: 3)&gt;array([[1, 7, 6],
       [4, 4, 8]])
Dimensions without coordinates: x, y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span>x</span>: 2</li><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-38036ed6-adfa-4c69-b46a-7c9430ebdd92' class='xr-array-in' type='checkbox' checked><label for='section-38036ed6-adfa-4c69-b46a-7c9430ebdd92' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>1 7 6 4 4 8</span></div><div class='xr-array-data'><pre>array([[1, 7, 6],
       [4, 4, 8]])</pre></div></div></li><li class='xr-section-item'><input id='section-a4739d40-644e-480d-80d6-4e1ef184c6ba' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-a4739d40-644e-480d-80d6-4e1ef184c6ba' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-aa200c80-c541-4e47-a591-a5c811412fc4' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-aa200c80-c541-4e47-a591-a5c811412fc4' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
labels = xr.DataArray(["E", "F", "E"], dims="y", name="labels")
labels
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;labels&#x27; (y: 3)&gt;array([&#x27;E&#x27;, &#x27;F&#x27;, &#x27;E&#x27;], dtype=&#x27;&lt;U1&#x27;)
Dimensions without coordinates: y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'labels'</div><ul class='xr-dim-list'><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-d96f3b55-0cb2-41c5-8b41-f50f99e757a2' class='xr-array-in' type='checkbox' checked><label for='section-d96f3b55-0cb2-41c5-8b41-f50f99e757a2' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>&#x27;E&#x27; &#x27;F&#x27; &#x27;E&#x27;</span></div><div class='xr-array-data'><pre>array([&#x27;E&#x27;, &#x27;F&#x27;, &#x27;E&#x27;], dtype=&#x27;&lt;U1&#x27;)</pre></div></div></li><li class='xr-section-item'><input id='section-fa32a9a7-229f-4939-8b5f-a244f28cb87f' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-fa32a9a7-229f-4939-8b5f-a244f28cb87f' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-9ac0bab2-4378-47ef-9dd4-a1d18f94c7ee' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-9ac0bab2-4378-47ef-9dd4-a1d18f94c7ee' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
a.groupby(labels).mean("y")
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (x: 2, labels: 2)&gt;array([[3.5, 7. ],
       [6. , 4. ]])
Coordinates:
  * labels   (labels) object &#x27;E&#x27; &#x27;F&#x27;
Dimensions without coordinates: x</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span>x</span>: 2</li><li><span class='xr-has-index'>labels</span>: 2</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-ea06c1cd-be1c-4dae-b8c5-277db6d3faad' class='xr-array-in' type='checkbox' checked><label for='section-ea06c1cd-be1c-4dae-b8c5-277db6d3faad' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>3.5 7.0 6.0 4.0</span></div><div class='xr-array-data'><pre>array([[3.5, 7. ],
       [6. , 4. ]])</pre></div></div></li><li class='xr-section-item'><input id='section-cfd3d752-fbec-49a0-9954-602b07fa3c93' class='xr-section-summary-in' type='checkbox'  checked><label for='section-cfd3d752-fbec-49a0-9954-602b07fa3c93' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>labels</span></div><div class='xr-var-dims'>(labels)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;E&#x27; &#x27;F&#x27;</div><input id='attrs-705a87ff-b63c-44e3-a1c1-1cba04270710' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-705a87ff-b63c-44e3-a1c1-1cba04270710' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4206e963-90ed-4808-a8fd-f2c93e932f43' class='xr-var-data-in' type='checkbox'><label for='data-4206e963-90ed-4808-a8fd-f2c93e932f43' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;E&#x27;, &#x27;F&#x27;], dtype=object)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-cfaa6a03-e61b-409a-ac11-f56fc5f0fa23' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-cfaa6a03-e61b-409a-ac11-f56fc5f0fa23' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
a.groupby(labels).mean("x")
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (y: 3)&gt;array([2.5, 5.5, 7. ])
Dimensions without coordinates: y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-4243007f-7119-4624-9339-31eff5bfe865' class='xr-array-in' type='checkbox' checked><label for='section-4243007f-7119-4624-9339-31eff5bfe865' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>2.5 5.5 7.0</span></div><div class='xr-array-data'><pre>array([2.5, 5.5, 7. ])</pre></div></div></li><li class='xr-section-item'><input id='section-ea1e5d9a-ee53-4db1-ab47-c4cc2241df08' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-ea1e5d9a-ee53-4db1-ab47-c4cc2241df08' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-95765719-a8c1-41ac-ad74-f3f573c100ce' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-95765719-a8c1-41ac-ad74-f3f573c100ce' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>





```python
# 1 4 6 8 一组
# 7 4 一组
a.groupby(labels).map(lambda x: x - x.min())
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (x: 2, y: 3)&gt;array([[0, 3, 5],
       [3, 0, 7]])
Dimensions without coordinates: x, y</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span>x</span>: 2</li><li><span>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-0981d96a-b148-4724-8202-7a4f1dc5f308' class='xr-array-in' type='checkbox' checked><label for='section-0981d96a-b148-4724-8202-7a4f1dc5f308' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0 3 5 3 0 7</span></div><div class='xr-array-data'><pre>array([[0, 3, 5],
       [3, 0, 7]])</pre></div></div></li><li class='xr-section-item'><input id='section-4fec27cd-283b-4d0f-a138-f2d043a471ad' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-4fec27cd-283b-4d0f-a138-f2d043a471ad' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-a73cdce7-62f7-4cf0-9aa7-50a26d8de772' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-a73cdce7-62f7-4cf0-9aa7-50a26d8de772' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



### ビジュアル化



```python
rng = np.random.default_rng(42)
a = xr.DataArray(
    rng.integers(1, 10, (2, 3)), 
    dims=("x", "y"),
    coords={"x": [10, 20], "y": [10, 20, 30]},
    attrs={
        "long_name": "random integers",
        "units": "null",
        "description": "demo random"
    }
)
```



```python
a.x.attrs["units"] = "x 10-20"
```



```python
a
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));--xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));--xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));--xr-border-color: var(--jp-border-color2, #e0e0e0);--xr-disabled-color: var(--jp-layout-color3, #bdbdbd);--xr-background-color: var(--jp-layout-color0, white);--xr-background-color-row-even: var(--jp-layout-color1, white);--xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);--xr-font-color2: rgba(255, 255, 255, 0.54);--xr-font-color3: rgba(255, 255, 255, 0.38);--xr-border-color: #1F1F1F;--xr-disabled-color: #515151;--xr-background-color: #111111;--xr-background-color-row-even: #111111;--xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;min-width: 300px;max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */display: none;
}

.xr-header {
  padding-top: 6px;padding-bottom: 6px;margin-bottom: 4px;border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,.xr-header > ul {
  display: inline;margin-top: 0;margin-bottom: 0;
}

.xr-obj-type,.xr-array-name {
  margin-left: 2px;margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;display: grid;grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;color: var(--xr-font-color2);font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display：inline-block;content：'►';font-size：11px;width：15px;text-align：center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content：'▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,.xr-section-inline-details {
  padding-top: 4px;padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;grid-column: 1 / -1;margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;display: grid;grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,.xr-array-data {
  padding: 0 5px !important;grid-column: 2;
}

.xr-array-data,.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;list-style: none;padding: 0 !important;margin: 0;
}

.xr-dim-list li {
  display: inline-block;padding: 0;margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,.xr-var-item {
  display: contents;
}

.xr-var-item > div,.xr-var-item label,.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,.xr-var-list > li:nth-child(odd) > label,.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;text-align: right;color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt {
  white-space: nowrap;overflow: hidden;text-overflow: ellipsis;padding-right: 10px;
}

.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover {
  overflow: visible;width: auto;z-index: 1;
}

.xr-var-attrs,.xr-var-data {
  display: none;background-color: var(--xr-background-color) !important;padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,.xr-var-data,.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,.xr-var-attrs,.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;margin: 0;display: grid;grid-template-columns: 125px auto;
}

.xr-attrs dt,.xr-attrs dd {
  padding: 0;margin: 0;float: left;padding-right: 10px;width: auto;
}

.xr-attrs dt {
  font-weight: normal;grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;background: var(--xr-background-color);padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;white-space: pre-wrap;word-break: break-all;
}

.xr-icon-database,.xr-icon-file-text2 {
  display: inline-block;vertical-align: middle;width: 1em;height: 1.5em !important;stroke-width: 0;stroke: currentColor;fill: currentColor;
}</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (x: 2, y: 3)&gt;array([[1, 7, 6],
       [4, 4, 8]])
Coordinates:
  * x        (x) int64 10 20
  * y        (y) int64 10 20 30
Attributes:
    long_name:    random integersunits:        nulldescription:  demo random</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 2</li><li><span class='xr-has-index'>y</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-e65e95b9-2875-477b-822d-5ea97dfb74bd' class='xr-array-in' type='checkbox' checked><label for='section-e65e95b9-2875-477b-822d-5ea97dfb74bd' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>1 7 6 4 4 8</span></div><div class='xr-array-data'><pre>array([[1, 7, 6],
       [4, 4, 8]])</pre></div></div></li><li class='xr-section-item'><input id='section-54a2aa86-9d60-4cb0-91c0-cf579e05a577' class='xr-section-summary-in' type='checkbox'  checked><label for='section-54a2aa86-9d60-4cb0-91c0-cf579e05a577' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 20</div><input id='attrs-acf35efe-5da1-4d6c-ae12-41d5f897e1dc' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-acf35efe-5da1-4d6c-ae12-41d5f897e1dc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f5f9c602-7ba2-435e-a142-75b08bacc3a2' class='xr-var-data-in' type='checkbox'><label for='data-f5f9c602-7ba2-435e-a142-75b08bacc3a2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>units :</span></dt><dd>x 10-20</dd></dl></div><div class='xr-var-data'><pre>array([10, 20])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>y</span></div><div class='xr-var-dims'>(y)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 20 30</div><input id='attrs-36c18929-32f6-4d19-83f3-88fb492babae' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-36c18929-32f6-4d19-83f3-88fb492babae' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f2d3fbb1-6d29-40f2-9e7c-189c314bb82e' class='xr-var-data-in' type='checkbox'><label for='data-f2d3fbb1-6d29-40f2-9e7c-189c314bb82e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([10, 20, 30])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-cb9b691c-a99c-4bea-ae54-78cf39ffd2ce' class='xr-section-summary-in' type='checkbox'  checked><label for='section-cb9b691c-a99c-4bea-ae54-78cf39ffd2ce' class='xr-section-summary' >Attributes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>long_name :</span></dt><dd>random integers</dd><dt><span>units :</span></dt><dd>null</dd><dt><span>description :</span></dt><dd>demo random</dd></dl></div></li></ul></div></div>





```python
a.plot()
```




    <matplotlib.collections.QuadMesh at 0x11b0604f0>




    
![png](ch06-morethan_numpy_files/ch06-morethan_numpy_390_1.png)
    


## まとめ



```python

```

## 参考

- [Beyond Numpy Arrays in Python](http://matthewrocklin.com/blog/work/2018/05/27/beyond-numpy)



```python

```
