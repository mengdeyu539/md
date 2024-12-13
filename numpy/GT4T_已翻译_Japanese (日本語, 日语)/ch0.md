<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"> <ul class="toc-item"><li><span><a href="#创建和生成" data-toc-modified-id="创建和生成-1">作成と生成 </a></span><ul class="toc-item"><li><span><a href="#从-python-列表或元组创建" data-toc-modified-id="从-python-列表或元组创建-1.1">pythonリストまたはタプルから作成 </a></span><span><a href="#使用-arange-生成" data-toc-modified-id="使用-arange-生成-1.2">arangeを使用して生成 </span><a href="#使用-linspace/logspace-生成" data-toc-modified-id="使用-linspace/logspace-生成-1.3">linspace/logspaceを使用して生成 <ul class="toc-item"><li><li><code></li>ファイルからの読み取り </a></span></li></li><li><span><a href="#统计和属性" data-toc-modified-id="统计和属性-2">統計と属性 </a></span><li><span><a href="#尺寸相关" data-toc-modified-id="尺寸相关-2.1">寸法関連 </a></span></li><span><code>np.shape </code></a></li></span><span><a href="#np.max/min" data-toc-modified-id="np.max/min-2.2.1"><code><a href="#np.max/min" data-toc-modified-id="np.max/min-2.2.1">np.max/min </code><平均和標準偏差</a></span><ul class="toc-item"><li><a href="#np.average" data-toc-modified-id="np.average-2.3.1"><code>np.average </code></span><code></code></ul></ul><li><a href="#形状和转换" data-toc-modified-id="形状和转换-3">形状と変換 </a></span><ul class="toc-item"><li><span><a href="#改变形状" data-toc-modified-id="改变形状-3.1">形状の変更 </a></span><li><a href="#np.expand_dims" data-toc-modified-id="np.expand_dims-3.1.1"><code>np.expand_dims </code></span></li><code>np.squeeze </code><li><span><a href="#np.reshape/arr.reshape" data-toc-modified-id="np.reshape/arr.reshape-3.1.3"><gt r="分解と組み合わせ</a></span><ul class="toc-item"><li><span><a href="#切片和索引" data-toc-modified-id="切片和索引-4.1">スライスとインデックス </a></span><li><span><code>index/slice </code></span><li></span></span><code><code></code></span>フィルターとフィルター </a></span><ul class="toc-item"><li><span><a href="#条件筛选" data-toc-modified-id="条件筛选-5.1">条件フィルター </a></span></li><span><a href="#提取" data-toc-modified-id="提取-5.2">抽出 </a></span><li><a href="#抽样" data-toc-modified-id="抽样-5.3">サンプリング </a><li><span></span><li><span><a href="#np.argmax/argmin" data-toc-modified-id="np.argmax/argmin-5.4.1"><code>np.argmax/argmin <gt r="516"/行列和演算</a></span><li><span><a href="#算术" data-toc-modified-id="算术-6.1">演算 </a></li><li></span><li><a href="#arr.dot" data-toc-modified-id="arr.dot-6.2.1"><code>arr.dot </code><li><gt r="561"概要と経験</a></span><ul class="toc-item"><li><span><a href="#内容小结" data-toc-modified-id="内容小结-7.1">内容概要 <gt</div>

このチュートリアルの内容は、基礎のない学生が迅速にマスターするのに役立つように `numpy` の一般的な機能は、日常のほとんどのシーンで使用することを保証します。機械学習やディープラーニングの先修コースとしても、クイック準備マニュアルとしても利用できます。

特筆すべきは、ディープラーニングの各フレームワークの多くのAPIと `numpy` も脈々と受け継がれていることで、 `numpy` 遊びが熟していると言えるでしょう。いくつかのディープラーニングフレームワークの多くのAPIも同時に学びました。


チュートリアルの原則は次の

- 実用的な高周波API
- 実際の使い方を示す
- シンプルでストレート

使用のための指示は次のとおりです。

- 各小節に重要度を示す⭐（1~5個）があり、多ければ多いほど重要になる
- 各サブセクションの下で重要なものは、ディレクトリから直接アクセスできるより小さなサブセクションに個別にリストされます。分割線の下のものは補足的に理解できます
- ⚠️特に注意が必要だと言っている


 **特に注意しなければならないのは、**、APIのさまざまなパラメータの詳細にあまり注意する必要はありません。チュートリアルでは、ほとんどのシナリオに対応するのに十分な使用方法が提供されています。より深い使用方法は、必要に応じてその後の「基本チュートリアル」を探索または学習することができます。



```python
# 导入 library
import numpy as np
# 画图工具
import matplotlib.pyplot as plt
```

## 作成と生成

本節では主に紹介しているarrayの作成と生成。なぜこれを最前面に置いたのでしょうか？主に次の2つの理由があります：

- 実際の作業の過程では、時々array関連のAPIや相互運用を検証したり確認したりする必要があります。
- sklearn、matplotlib、PyTorch、Tensorflowなどのツールを使用するときに、実験のために簡単なデータを必要とすることもあります。

それで、まず素早く手に入れる方法を学びましょうarrayにはいろいろな利点があります。このセクションでは、次の一般的な作成方法について説明します：

- リストまたはタプルの使用
- arangeの使用
- linspace/logspaceの使用
- ones/zerosの使用
- randomを使う
- ファイルからの読み取り

このうち、最も一般的に使用されるのはlinspace/logspaceとrandomであり、前者は座標軸を描く上でよく使用され、後者はシミュレーションデータを生成するために使用されます。例えば、関数の画像を描く必要がある場合、Xはlinspaceを使用して生成され、関数式を使用してYを求め、plotを求めます。randomは、いくつかの入力（例えばX）や中間入力（例えばEmbedding、hidden state）を構築する必要があるときに非常に便利です。

### pythonリストまたはタプルから作成

⭐⭐はリストを入力してarrayを作成することに重点を置いています： `np.array(list)`


⚠️注意すべき点：データ型。十分に注意してみると、以下の2番目のコードセットの2番目の数字は小数であることがわかります（注：Pythonでは1.==1.0）、arrayはすべての要素の型が同じであることを保証するため、arrayをfloat型に変換するのに役立ちます。



```python
# 一个 list
np.array([1,2,3])
```




    array([1, 2, 3])





```python
# 二维（多维类似）
# 注意，有一个小数哦
np.array([[1, 2., 3], [4, 5, 6]])
```




    array([[1., 2., 3.],
           [4., 5., 6.]])





```python
# 您也可以指定数据类型
np.array([1, 2, 3], dtype=np.float16)
```




    array([1., 2., 3.], dtype=float16)





```python
# 如果指定了 dtype，输入的值都会被转为对应的类型，而且不会四舍五入
lst = [
    [1, 2, 3],
    [4, 5, 6.8]
]
np.array(lst, dtype=np.int32)
```




    array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)



---



```python
# 一个 tuple
np.array((1.1, 2.2))
```




    array([1.1, 2.2])





```python
# tuple，一般用 list 就好，不需要使用 tuple
np.array([(1.1, 2.2, 3.3), (4.4, 5.5, 6.6)])
```




    array([[1.1, 2.2, 3.3],
           [4.4, 5.5, 6.6]])





```python
# 转换而不是上面的创建，其实是类似的，无须过于纠结
np.asarray((1,2,3))
```




    array([1, 2, 3])





```python
np.asarray(([1., 2., 3.], (4., 5., 6.)))
```




    array([[1., 2., 3.],
           [4., 5., 6.]])



### arangeを使用して生成する

⭐⭐

rangeはPythonに組み込まれた整数系列生成器で、arangeはnumpyのもので、同様に効果があり、1次元のベクトルを生成します。たまに、このような方法でarrayを構築する必要があります。例えば：

- 入力として連続した1次元ベクトルを作成する必要があります（例えば、位置をエンコードするときに使用できます）
- スクリーニング、サンプリングの結果を観察する必要がある場合、秩序あるarrayの方が一般的に観察しやすい

⚠️注意してください： `reshape` の場合、ターゲットのshapeに必要な要素の数は元の要素の数と同じでなければなりません。



```python
np.arange(12).reshape(3, 4)
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],[ 8,  9, 10, 11]])





```python
# 注意，是小数哦
np.arange(12.0).reshape(4, 3)
```




    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],[ 6.,  7.,  8.],[ 9., 10., 11.]])





```python
np.arange(100, 124, 2).reshape(3, 2, 2)
```




    array([[[100, 102],
            [104, 106]],
    
           [[108, 110],
            [112, 114]],
    
           [[116, 118],
            [120, 122]]])





```python
# shape size 相乘要和生成的元素数量一致
np.arange(100., 124., 2).reshape(2,3,4)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-20-fc850bf3c646> in <module>
    ----> 1 np.arange(100., 124., 2).reshape(2,3,4)
    

    ValueError: cannot reshape array of size 12 into shape (2,3,4)


### linspace/logspaceを使用して生成

⭐⭐⭐

OK、これは私たちが出会った最初の比較的重要なAPIです。前者は3つのパラメータを入力する必要があります：始まり、終わり、数量、後者は追加のbaseを受信する必要があり、デフォルトは10です。

⚠️注意が必要なのは、3番目の引数は**じゃない**ステップサイズであることです。

#### `np.linspace`



```python
# 线性
np.linspace(0, 9, 10).reshape(2, 5)
```




    array([[0., 1., 2., 3., 4.],
           [5., 6., 7., 8., 9.]])





```python
np.linspace(0, 9, 6).reshape(2, 3)
```




    array([[0. , 1.8, 3.6],
           [5.4, 7.2, 9. ]])



---



```python
# 指数 base 默认为 10
np.logspace(0, 9, 6, base=np.e).reshape(2, 3)
```




    array([[1.00000000e+00, 6.04964746e+00, 3.65982344e+01],
           [2.21406416e+02, 1.33943076e+03, 8.10308393e+03]])





```python
# _ 表示上（最近）一个输出
# logspace 结果 log 后就是上面 linspace 的结果
np.log(_)
```




    array([[0. , 1.8, 3.6],
           [5.4, 7.2, 9. ]])



ここでさらに見てみましょう：



```python
N = 20
x = np.arange(N)
y1 = np.linspace(0, 10, N) * 100
y2 = np.logspace(0, 10, N, base=2)

plt.plot(x, y2, '*');
plt.plot(x, y1, 'o');
```


    
![png](ch0_files/ch0_27_0.png)

    



```python
# 检查每个元素是否为 True
# base 的 指数为 linspace 得到的就是 logspace
np.alltrue(2 ** np.linspace(0, 10, N)  == y2)
```




    True



> ⚠️追加：についてarrayの条件判断



```python
# 不能直接用 if 判断 array 是否符合某个条件
arr = np.array([1, 2, 3])
cond1 = arr > 2
cond1
```




    array([False, False,  True])





```python
if cond1:
    print("这不行")
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-184-6bd8dc445309> in <module>
    ----> 1 if cond1:
          2     print("这不行")
    

    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()




```python
# 即便你全是 True 它也不行
arr = np.array([1, 2, 3])
cond2 = arr > 0
cond2
```




    array([ True,  True,  True])





```python
if cond2:
    print("这还不行")
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-187-7fedc8ba71a0> in <module>
    ----> 1 if cond2:
          2     print("这还不行")
    

    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()




```python
# 咱们只能用 any 或 all，这个很容易犯错，请务必注意。
if cond1.any():
    print("只要有一个为True就可以，所以——我可以")
```

    只要有一个为True就可以，所以——我可以

    


```python
if cond2.all():
    print("所有值为True才可以，我正好这样")
```

    所有值为True才可以，我正好这样
    

### ones/zerosで作成

⭐

全1/0アレイへのショートカットを作成します。注意してください `np.zeros_like` または `np.ones_like` は、指定されたarrayの同じshapeの0または1ベクトルを素早く生成することができます。これは、Maskの特定の位置が必要な場合に使用される可能性があります。

⚠️注意してください：作成されたarrayはデフォルトでfloat型です。



```python
np.ones(3)
```




    array([1., 1., 1.])





```python
np.ones((2, 3))
```




    array([[1., 1., 1.],
           [1., 1., 1.]])





```python
np.zeros((2,3,4))
```




    array([[[0., 0., 0., 0.],
            [0., 0., 0., 0.],[0., 0., 0., 0.]],
    
           [[0., 0., 0., 0.],
            [0., 0., 0., 0.],[0., 0., 0., 0.]]])





```python
# 像给定向量那样的 0 向量（ones_like 是 1 向量）
np.zeros_like(np.ones((2,3,3)))
```




    array([[[0., 0., 0.],
            [0., 0., 0.],[0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.],[0., 0., 0.]]])



### randomを使って生成する

⭐⭐⭐⭐⭐

このセクションで最も重要なAPIを選ぶなら、それは間違いなく `random` に違いありません。ここでは、本番データ関連の一般的なAPIをいくつか紹介します。トレーニングデータやテストデータのランダム生成、ニューラルネットワークの初期化などによく使用されます。

⚠️注意しなければならないのは、ここでは新しいAPI方式を使用して作成することを統一的に推奨しています。つまり、 `np.random.default_rng()` を使用して `Generator` を作成し、それに基づいてさまざまな分布のデータを生成します（記憶がより簡単で明確になります）。しかし、私たちは依然として古いAPIの使用法を紹介します。多くのコードではまだ古いものが使用されているので、あなたは見覚えがあることができます。



```python
# 0-1 连续均匀分布
np.random.rand(2, 3)
```




    array([[0.42508994, 0.5842191 , 0.09248675],
           [0.656858  , 0.88171822, 0.81744539]])





```python
# 单个数
np.random.rand()
```




    0.29322641374172986





```python
# 0-1 连续均匀分布
np.random.random((3, 2))
```




    array([[0.17586271, 0.5061715 ],
           [0.14594537, 0.34365713],[0.28714656, 0.40508807]])





```python
# 指定上下界的连续均匀分布
np.random.uniform(-1, 1, (2, 3))
```




    array([[ 0.66638982, -0.65327069, -0.21787878],
           [-0.63552782,  0.51072282, -0.14968825]])





```python
# 上面两个的区别是 shape 的输入方式不同，无伤大雅了
# 不过从 1.17 版本后推荐这样使用（以后大家可以用新的方法）
# rng 是个 Generator，可用于生成各种分布
rng = np.random.default_rng(42)
rng
```




    Generator(PCG64) at 0x111B5C5E0





```python
# 推荐的连续均匀分布用法
rng.random((2, 3))
```




    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235]])





```python
# 可以指定上下界，所以更加推荐这种用法
rng.uniform(0, 1, (2, 3))
```




    array([[0.47673156, 0.59702442, 0.63523558],
           [0.68631534, 0.77560864, 0.05803685]])





```python
# 随机整数（离散均匀分布），不超过给定的值（10）
np.random.randint(10, size=2)
```




    array([6, 3])





```python
# 随机整数（离散均匀分布），指定上下界和 shape
np.random.randint(0, 10, (2, 3))
```




    array([[8, 6, 1],
           [3, 8, 1]])





```python
# 上面推荐的方法，指定大小和上界
rng.integers(10, size=2)
```




    array([9, 7])





```python
# 上面推荐的方法，指定上下界
rng.integers(0, 10, (2, 3))
```




    array([[5, 9, 1],
           [8, 5, 7]])





```python
# 标准正态分布
np.random.randn(2, 4)
```




    array([[-0.61241167, -0.55218849, -0.50470617, -1.35613877],
           [-1.34665975, -0.74064846, -2.5181665 ,  0.66866357]])





```python
# 上面推荐的标准正态分布用法
rng.standard_normal((2, 4))
```




    array([[ 0.09130331,  1.06124845, -0.79376776, -0.7004211 ],
           [ 0.71545457,  1.24926923, -1.22117522,  1.23336317]])





```python
# 高斯分布
np.random.normal(0, 1, (3, 5))
```




    array([[ 0.30037773, -0.17462372,  0.23898533,  1.23235421,  0.90514996],
           [ 0.90269753, -0.5679421 ,  0.8769029 ,  0.81726869, -0.59442623],[ 0.31453468, -0.18190156, -2.95932929, -0.07164822, -0.23622439]])





```python
# 上面推荐的高斯分布用法
rng.normal(0, 1, (3, 5))
```




    array([[ 2.20602146, -2.17590933,  0.80605092, -1.75363919,  0.08712213],
           [ 0.33164095,  0.33921626,  0.45251278, -0.03281331, -0.74066207],[-0.61835785, -0.56459129,  0.37724436, -0.81295739,  0.12044035]])



要するに、一般的に使われるのは、均一分布と正規（ガウス）分布の2つの分布である。また、 `size` shapeを指定することもできます。



```python
rng = np.random.default_rng(42)
```



```python
# 离散均匀分布
rng.integers(low=0, high=10, size=5)
```




    array([0, 7, 6, 4, 4])





```python
# 连续均匀分布
rng.uniform(low=0, high=10, size=5)
```




    array([6.97368029, 0.94177348, 9.75622352, 7.61139702, 7.86064305])





```python
# 正态（高斯）分布
rng.normal(loc=0.0, scale=1.0, size=(2, 3))
```




    array([[-0.01680116, -0.85304393,  0.87939797],
           [ 0.77779194,  0.0660307 ,  1.12724121]])



### ファイルからの読み取り

⭐

このセクションは主に記憶されたウェイトパラメータや前処理されたデータセットをロードして実現するために使用されており、訓練されたモデルパラメータをメモリにロードして推論サービスを提供したり、長い時間をかけた前処理データを直接保存したりして、複数回の実験時に再処理する必要がないなど、便利であることがあります。

⚠️注意点：保存時にファイル名接尾辞を書く必要はなく、自動的に追加されます。



```python
# 直接将给定矩阵存为 a.npy
np.save('./data/a', np.array([[1, 2, 3], [4, 5, 6]]))
```



```python
# 可以将多个矩阵存在一起，名为 `b.npz`
np.savez("./data/b", a=np.arange(12).reshape(3, 4), b=np.arange(12.).reshape(4, 3))
```



```python
# 和上一个一样，只是压缩了
np.savez_compressed("./data/c", a=np.arange(12).reshape(3, 4), b=np.arange(12.).reshape(4, 3))
```



```python
# 加载单个 array
np.load("data/a.npy")
```




    array([[1, 2, 3],
           [4, 5, 6]])





```python
# 加载多个，可以像字典那样取出对应的 array
arr = np.load("data/b.npz")
```



```python
arr["a"]
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],[ 8,  9, 10, 11]])





```python
arr["b"]
```




    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],[ 6.,  7.,  8.],[ 9., 10., 11.]])





```python
# 后缀都一样，你干脆当它和上面的没区别即可
arr = np.load("data/c.npz")
```



```python
arr["b"]
```




    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],[ 6.,  7.,  8.],[ 9., 10., 11.]])



## 統計と属性

このセクションでは、arrayの基本的な統計属性から始めて、作成したばかりのarrayについてもっと理解します。主に以下の側面が含まれています：

- サイズ依存
- 最大、最小、中央、分位値
- 平均、和、標準偏差など

いずれも記述統計に関連する指標であり、arrayを全体的に理解するのに役立ちます。その中で最も多く使われているのは、サイズ関連の `shape`、最大値、最小値、平均値、和値などです。

このセクションの内容は非常にシンプルで、2つの重要な特徴に特に注目する必要があります（覚えておく）：

- 次元別に結果を求めます (axisを指定します)。一般的に0は列を表し、1は行を表しますが、**行/列に沿った操作**と理解することができますが、不確かな場合は例を挙げてみてください。
- 計算後の次元の保持 ( `keepdims=True`)


また、操作を容易にするために、ランダムに生成されたarrayを操作オブジェクトとして使用します。同時に、実行するたびに、誰もが同じ結果を見るように、seedを指定しました。一般的に、我々はモデルを訓練する際に、seedを指定する必要があり、同じ条件で調整することができる。



```python
#  先创建一个 Generator
rng = np.random.default_rng(seed=42)
#  再生成一个均匀分布
arr = rng.uniform(0, 1, (3, 4))
arr
```




    array([[0.77395605, 0.43887844, 0.85859792, 0.69736803],
           [0.09417735, 0.97562235, 0.7611397 , 0.78606431],[0.12811363, 0.45038594, 0.37079802, 0.92676499]])



### サイズ依存

⭐⭐

このセクションには、次元、形状、データ量が含まれています。このうち、形状 `shape` が最も多く使用されています。

⚠️注意が必要なのは：sizeはshapeではなく、ndimはいくつかの次元があることを示しています。



```python
# 维度，array 是二维的（两个维度）
arr.ndim
```




    2



#### `np.shape`



```python
# 形状，返回一个 Tuple
arr.shape
```




    (3, 4)





```python
# 数据量
arr.size
```




    12



### 最大分位

⭐⭐⭐

この小節には主に：最大値、最小値、中央値、その他の分位数が含まれており、その中で『**最大値と最小値**』は私たちが普段最も多く使っている。

⚠️注意しなければならないのは、分位数は0~1の任意の小数（対応分位を表す）であり、分位数は必ずしも元のarrayにあるとは限らないことです。



```python
arr
```




    array([[0.77395605, 0.43887844, 0.85859792, 0.69736803],
           [0.09417735, 0.97562235, 0.7611397 , 0.78606431],[0.12811363, 0.45038594, 0.37079802, 0.92676499]])





```python
# 所有元素中最大的
arr.max()
```




    0.9756223516367559



#### `np.max/min`



```python
# 按维度（列）最大值
arr.max(axis=0)
```




    array([0.77395605, 0.97562235, 0.85859792, 0.92676499])





```python
# 同理，按行
arr.max(axis=1)
```




    array([0.85859792, 0.97562235, 0.92676499])





```python
# 是否保持原来的维度
# 这个需要特别注意下，很多深度学习模型中都需要保持原有的维度进行后续计算
# shape 是 (3,1)，array 的 shape 是 (3,4)，按行，同时保持了行的维度
arr.min(axis=1, keepdims=True)
```




    array([[0.43887844],
           [0.09417735],[0.12811363]])





```python
# 保持维度：（1，4），原始array是（3，4）
arr.min(axis=0, keepdims=True)
```




    array([[0.09417735, 0.43887844, 0.37079802, 0.69736803]])





```python
# 一维了
arr.min(axis=0, keepdims=False)
```




    array([0.09417735, 0.43887844, 0.37079802, 0.69736803])



---



```python
# 另一种用法，不过我们一般习惯使用上面的用法，其实两者一回事
np.amax(arr, axis=0)
```




    array([0.77395605, 0.97562235, 0.85859792, 0.92676499])





```python
# 同 amax
np.amin(arr, axis=1)
```




    array([0.43887844, 0.09417735, 0.12811363])





```python
# 中位数
# 其他用法和 max，min 是一样的
np.median(arr)
```




    0.7292538655248584





```python
# 分位数，按列取1/4数
np.quantile(arr, q=0.25, axis=0)
```




    array([0.11114549, 0.44463219, 0.56596886, 0.74171617])





```python
# 分位数，按行取 3/4，同时保持维度
np.quantile(arr, q=0.75, axis=1, keepdims=True)
```




    array([[0.79511652],
           [0.83345382],[0.5694807 ]])





```python
# 分位数，注意，分位数可以是 0-1 之间的任何数字（分位）
# 如果是 1/2 分位，那正好是中位数
np.quantile(arr, q=1/2, axis=1)
```




    array([0.73566204, 0.773602  , 0.41059198])



### 平均和標準偏差

⭐⭐⭐

この小節には、主に平均値、累積和、分散、標準偏差などのさらなる統計指標が含まれている。その中で最も多く使われているのが平均値です。



```python
arr
```




    array([[0.77395605, 0.43887844, 0.85859792, 0.69736803],
           [0.09417735, 0.97562235, 0.7611397 , 0.78606431],[0.12811363, 0.45038594, 0.37079802, 0.92676499]])



#### `np.average`



```python
# 平均值
np.average(arr)
```




    0.6051555606435642





```python
# 按维度平均（列）
np.average(arr, axis=0)
```




    array([0.33208234, 0.62162891, 0.66351188, 0.80339911])



---



```python
# 另一个计算平均值的 API
# 它与 average 的主要区别是，np.average 可以指定权重，即可以用于计算加权平均
# 一般建议使用 average，忘掉 mean 吧！
np.mean(arr, axis=0)
```




    array([0.33208234, 0.62162891, 0.66351188, 0.80339911])



#### `np.sum`



```python
# 求和，不多说了，类似
np.sum(arr, axis=1)
```




    array([2.76880044, 2.61700371, 1.87606258])





```python
np.sum(arr, axis=1, keepdims=True)
```




    array([[2.76880044],
           [2.61700371],[1.87606258]])



---



```python
# 按列累计求和
np.cumsum(arr, axis=0)
```




    array([[0.77395605, 0.43887844, 0.85859792, 0.69736803],
           [0.8681334 , 1.41450079, 1.61973762, 1.48343233],[0.99624703, 1.86488673, 1.99053565, 2.41019732]])





```python
# 按行累计求和
np.cumsum(arr, axis=1)
```




    array([[0.77395605, 1.21283449, 2.07143241, 2.76880044],
           [0.09417735, 1.0697997 , 1.8309394 , 2.61700371],[0.12811363, 0.57849957, 0.94929759, 1.87606258]])





```python
# 标准差，用法类似
np.std(arr)
```




    0.28783096517727075





```python
# 按列求标准差
np.std(arr, axis=0)
```




    array([0.3127589 , 0.25035525, 0.21076935, 0.09444968])





```python
# 方差
np.var(arr, axis=1)
```




    array([0.02464271, 0.1114405 , 0.0839356 ])



## シェイプと変換

arrayはほとんどの場合多次元の形で現れ、一般的に2次元を超える多次元arrayをテンソル、2次元行列、1次元ベクトルと呼ぶ。多次元なので、自然に形状の変化や変換に関わり、テンソルの最も基礎的な操作と言える。

このセクションでは、主に以下の3つの側面について説明します：

- 形状を変える
- 逆順序
- トランスポーツ

その中でも形状変更や転置はとてもよく使われていますので、熟練しておくことをおすすめします。

### 形状を変える

⭐⭐⭐⭐⭐

このセクションのAPIは非常に頻繁に使用されています。特に1次元を拡張する `expand_dims` と1次元を除去する `squeeze` は、将来多くのニューラルネットワークアーキテクチャでこの2つの商品の姿を見ることができます。

⚠️注意すべきことは、拡張するか縮小するかにかかわらず、shapeの多いか少ないかは1であり、 `squeeze` の場合、次元を指定した場合、その次元のshapeは1でなければなりません。



```python
# 换个整数的随机 array
rng = np.random.default_rng(seed=42)
arr = rng.integers(1, 100, (3, 4))
arr
```




    array([[ 9, 77, 65, 44],
           [43, 86,  9, 70],[20, 10, 53, 97]])





```python
# 有时候您可能需要将多维 array 打平
arr.ravel()
```




    array([ 9, 77, 65, 44, 43, 86,  9, 70, 20, 10, 53, 97])





```python
arr.shape
```




    (3, 4)



#### `np.expand_dims`



```python
#### 扩展 1 个维度，需要（必须）指定维度
# 其实就是多嵌套了一下
np.expand_dims(arr, 1).shape
```




    (3, 1, 4)





```python
# 扩充维度
expanded = np.expand_dims(arr, axis=(1, 3, 4))
expanded.shape
```




    (3, 1, 4, 1, 1)





```python
# 扩充维度不能跳跃
expanded = np.expand_dims(arr, axis=(1, 3, 8))
```


    ---------------------------------------------------------------------------

    AxisError                                 Traceback (most recent call last)

    <ipython-input-344-2c2510eb807f> in <module>
          1 # 扩充维度不能跳跃
    ----> 2 expanded = np.expand_dims(arr, axis=(1, 3, 8))
    

    <__array_function__ internals> in expand_dims(*args, **kwargs)
    

    /usr/local/lib/python3.8/site-packages/numpy/lib/shape_base.py in expand_dims(a, axis)
        595 596     out_ndim = len(axis) + a.ndim
    --> 597     axis = normalize_axis_tuple(axis, out_ndim)
        598 599     shape_it = iter(a.shape)
    

    /usr/local/lib/python3.8/site-packages/numpy/core/numeric.py in normalize_axis_tuple(axis, ndim, argname, allow_duplicate)
       1356             pass1357     # Going via an iterator directly is slower than via list comprehension.
    -> 1358     axis = tuple([normalize_axis_index(ax, ndim, argname) for ax in axis])
       1359     if not allow_duplicate and len(set(axis)) != len(axis):1360         if argname:
    

    /usr/local/lib/python3.8/site-packages/numpy/core/numeric.py in <listcomp>(.0)
       1356             pass1357     # Going via an iterator directly is slower than via list comprehension.
    -> 1358     axis = tuple([normalize_axis_index(ax, ndim, argname) for ax in axis])
       1359     if not allow_duplicate and len(set(axis)) != len(axis):1360         if argname:
    

    AxisError: axis 8 is out of bounds for array of dimension 5


#### `np.squeeze`



```python
# squeeze 指定 axis 的shape必须为1
np.squeeze(expanded, axis=0)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-345-8a712eef1189> in <module>
          1 # squeeze 指定 axis 的shape必须为1
    ----> 2 np.squeeze(expanded, axis=0)
    

    <__array_function__ internals> in squeeze(*args, **kwargs)
    

    /usr/local/lib/python3.8/site-packages/numpy/core/fromnumeric.py in squeeze(a, axis)
       1493         return squeeze()1494     else:
    -> 1495         return squeeze(axis=axis)
       1496 1497 
    

    ValueError: cannot select an axis to squeeze out which has size not equal to one




```python
# 如果指定了维度，那就只会去除该维度，指定的维度必须为 1
np.squeeze(expanded, axis=1).shape
```




    (3, 3, 1, 1)





```python
# 去除所有维度为 1 的
np.squeeze(expanded).shape
```




    (3, 3)



#### `np.reshape/arr.reshape`



```python
# reshape 成另一个形状
# 也可以直接变为一维向量
arr.reshape(2, 2, 3)
```




    array([[[ 9, 77, 65],
            [44, 43, 86]],
    
           [[ 9, 70, 20],
            [10, 53, 97]]])





```python
# 可以偷懒，使用 -1 表示其他维度（此处 -1 为 3），注意，reshape 参数可以是 tuple 或连续整数
arr1 = arr.reshape((4, -1))
arr1
```




    array([[ 9, 77, 65],
           [44, 43, 86],[ 9, 70, 20],[10, 53, 97]])





```python
# 元素数量必须与原array一致
arr.reshape(3, 3)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-348-67f1d69569ea> in <module>
          1 # 元素数量必须与原array一致
    ----> 2 arr.reshape(3, 3)
    

    ValueError: cannot reshape array of size 12 into shape (3,3)


---



```python
# 另一种变换形状的方式 —— 原地变换
# 不过不能用-1
# 另外 resize 不一定和原来的元素数量一样多
arr2 = arr.resize((4, 3))
# 注意：上面的 reshape 会生成一个新的 array，但 resize 不会，所以我们需要用原变量名将它显示出来
# arr2 没有值
arr2
```



```python
arr
```




    array([[ 9, 77, 65],
           [44, 43, 86],[ 9, 70, 20],[10, 53, 97]])





```python
# 直接 resize，如果元素数量多时会提示错误
arr.resize((2, 3))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-351-a23c63ab1d76> in <module>
          1 # 直接 resize，如果元素数量多时会提示错误
    ----> 2 arr.resize((2, 3))
    

    ValueError: cannot resize an array that references or is referencedby another array in this way.Use the np.resize function or refcheck=False




```python
# 可以copy一份
arrcopy = np.copy(arr)
arrcopy.resize((2, 3))
arrcopy
```




    array([[ 9, 77, 65],
           [44, 43, 86]])





```python
# arr 保持不变
arr
```




    array([[ 9, 77, 65],
           [44, 43, 86],[ 9, 70, 20],[10, 53, 97]])





```python
# 也可以将 refcheck 设为 False
# 此时 arr 会发生变化
# 元素数量超出时，截断；元素数量不够时，0填充
arr.resize((2,3), refcheck=False)
arr
```




    array([[ 9, 77, 65],
           [44, 43, 86]])





```python
arr.resize((3, 3), refcheck=False)
arr
```




    array([[ 9, 77, 65],
           [44, 43, 86],[ 0,  0,  0]])





```python
arr
```




    array([[ 9, 77, 65],
           [44, 43, 86],[ 0,  0,  0]])





```python
# 如果用 np.resize 会略有不同
# 元素数量不够时，会自动复制
np.resize(arr, (5, 3))
```




    array([[ 9, 77, 65],
           [44, 43, 86],[ 0,  0,  0],[ 9, 77, 65],[44, 43, 86]])





```python
# 元素数量多出来时，会自动截断
np.resize(arr, (2, 2))
```




    array([[ 9, 77],
           [65, 44]])



### 逆順序

⭐

元の配列への変換とも見ることができ、あまり使われていないので、次のインデックスとスライスのためのウォーミングアップを理解することができます。

文字列や配列を逆順序にすると、いろいろな方法が考えられるかもしれません。例えば、 `reversed`、メソッドを書いたり、Python listのインデックス机能を使ったりします。これは `numpy` でarrayを逆順序にする方法でもあります。



```python
# 字符串
s = "uevol"
s[::-1]
```




    'loveu'





```python
# 数组
lst = [1, "1", 5.2]
lst[::-1]
```




    [5.2, '1', 1]





```python
arr
```




    array([[ 9, 77, 65],
           [44, 43, 86],[ 9, 70, 20],[10, 53, 97]])





```python
arr
```


    array([[ 9, 77, 65],
           [44, 43, 86],[ 9, 70, 20],[10, 53, 97]])





```python
# 我们按上面的套路：默认列反序
arr[::-1]
```




    array([[10, 53, 97],
           [ 9, 70, 20],[44, 43, 86],[ 9, 77, 65]])





```python
# 列不变行反序
arr[::-1, :]
```




    array([[10, 53, 97],
           [ 9, 70, 20],[44, 43, 86],[ 9, 77, 65]])





```python
# 在不同维度上操作：行不变列反序
arr[:, ::-1]
```




    array([[65, 77,  9],
           [86, 43, 44],[20, 70,  9],[97, 53, 10]])





```python
# 行变列也变
arr[::-1, ::-1]
```




    array([[97, 53, 10],
           [20, 70,  9],[86, 43, 44],[65, 77,  9]])



### トランスポーツ

⭐⭐⭐

転置は線形代数の基本的な操作であり、二次元行列を例にとると、一般的に理解されているのはそれを倒し、shapeが反転し、行が列になり、列が行になります。もちろん、多次元についても同様です。2次元行列には `arr.T` を使用することをお勧めします（はるかに速くなります）、2次元以上のテンソルには `np.transpose` を使用することをお勧めします。

⚠️注意しなければならないのは、1次元配列転置か自分であることです。



```python
# 一维
np.array([1,2]).T.shape
```




    (2,)





```python
arr
```




    array([[ 9, 77, 65],
           [44, 43, 86],[ 9, 70, 20],[10, 53, 97]])



#### `arr.T`



```python
# 简便用法，把所有维度顺序都给倒过来
arr.T
```




    array([[ 9, 44,  9, 10],
           [77, 43, 70, 53],[65, 86, 20, 97]])





```python
# 将 shape=(1,1,3,4) 的转置后得到 shape=(4,3,1,1)
arr.reshape(1, 1, 3, 4).T.shape
```




    (4, 3, 1, 1)





```python
# 同上
arr.reshape(1, 2, 2, 1, 3, 1).T.shape
```




    (1, 3, 1, 2, 2, 1)



#### `np.transpose`



```python
# 这种转置方式可以指定 axes
np.transpose(arr)
```




    array([[ 9, 44,  9, 10],
           [77, 43, 70, 53],[65, 86, 20, 97]])





```python
# 不指定 axes 时和 .T 是一样的
np.transpose(arr.reshape(1, 2, 2, 1, 3, 1)).shape
```




    (1, 3, 1, 2, 2, 1)





```python
# 指定 axes，不过 axes 数量必须包含所有维度的序列
# 比如两个维度就是 (0, 1)，四个就是 (0, 1, 2, 3)
# 当然，顺序可以改变，比如下面就是只转置第 2 个和第 3 个维度
# 注意，只有超过 2 维时，这样才有意义
# 下面的结果中，中间2个维度被调换顺序了，顺序就在axes中指定的
np.transpose(arr.reshape(1, 1, 3, 4), axes=(0, 2, 1, 3)).shape
```




    (1, 3, 1, 4)



## 分解と組み合わせ

このセクションでは、アレイの分解と組み合わせを主に学びます。このセクションはすべての章の中で最も重要なセクションです。このセクションを通じて、 `numpy`（そしてPython言語）の強さを十分に理解することができます。この操作上の優雅さは未来とは言えませんが、少なくとも前代未聞です。

内容は大まかに以下のサブセクションで構成されています

- スライスとインデックス
- スプリッシング
- リピート
- スピンオフ

その中で、最も重要なのはスライスとインデックスであり、それは基礎的で、それは高周波で、それはどこにでもあります。マスターすることを強くお勧めします。他の3つは比較的簡単で、それぞれ1つのAPIを覚えておくだけです。

### スライスとインデックス

⭐⭐⭐⭐⭐

ポイントを取ってください！スライスとインデックスは、既存のarrayを操作することによって、所望の部分要素を得る行為プロセスである。その中心的なアクションは、次元ごとに `start:stop:step` に基づいてarrayを操作すると要約できます。

この部分の内容の核心は処理を次元ごとに分け、処理しない次元を統一的に `:` または `...` に置き換えることである。操作を見るときも、まず `,` がどこにあるかに注目してください。処理する次元は以前の `arange` `linspace` などのインタフェースの使用方法と同じです。

⚠️注意しなければならないのは、引用は負の数をサポートしていること、すなわち後から前にインデックスしていることです。



```python
rng = np.random.default_rng(42)
arr = rng.integers(0, 20, (5, 4))
arr
```




    array([[ 1, 15, 13,  8],
           [ 8, 17,  1, 13],[ 4,  1, 10, 19],[14, 15, 14, 15],[10,  2, 16,  9]])



#### `index/slice`



```python
# 取第 0 行
arr[0]
```




    array([ 1, 15, 13,  8])





```python
# 取第 0 行第 1 个元素
arr[0, 1]
```




    15





```python
# 然后带点范围 第 1-2 行
arr[0:3]
```




    array([[ 1, 15, 13,  8],
           [ 8, 17,  1, 13],[ 4,  1, 10, 19]])





```python
# 离散也可以：第 1，3 行
arr[[0, 3]]
```




    array([[ 1, 15, 13,  8],
           [14, 15, 14, 15]])





```python
arr
```




    array([[ 1, 15, 13,  8],
           [ 8, 17,  1, 13],[ 4,  1, 10, 19],[14, 15, 14, 15],[10,  2, 16,  9]])





```python
# 再来加上维度：第 1-2 行，第 1 列
arr[1:3, 1]
```




    array([17,  1])





```python
# 离散也是一样：第 1，3 行，第 0 列
arr[[1,3], [0]]
```




    array([ 8, 14])





```python
# 还可以有简写：到最后或到开始。如第 3 行到最后一行
arr[3:]
```




    array([[14, 15, 14, 15],
           [10,  2, 16,  9]])





```python
# 开始到第 3 行，第 1-3 列
arr[:3, 1:3]
```




    array([[15, 13],
           [17,  1],[ 1, 10]])





```python
# 还可以来点跳跃，步长：start:stop:step，第 1 行到第 4 行，间隔为 2，即第 1、3 行
arr[1: 4: 2]
```




    array([[ 8, 17,  1, 13],
           [14, 15, 14, 15]])





```python
# 加上列也可以哦，第 1、3 行，第 0、2 列
arr[1:4:2, 0:3:2]
```




    array([[ 8,  1],
           [14, 14]])





```python
arr
```




    array([[ 1, 15, 13,  8],
           [ 8, 17,  1, 13],[ 4,  1, 10, 19],[14, 15, 14, 15],[10,  2, 16,  9]])





```python
# 第一列的值，其实是所有其他维度第 1 维的值
arr[...,1]
```




    array([15, 17,  1, 15,  2])





```python
# 与上面类似，但用的更多
arr[:,1]
```




    array([15, 17,  1, 15,  2])



### スプリッシング

⭐⭐⭐⭐

時には、既存のいくつかのアレイをスプライスして大きなアレイを形成する必要があります（一般的な例としては、異なるタイプのフィーチャーのスプライスなど）。このセクションには厳密に言えば、2つのAPIしかありません。前者はスプライスであり、後者はスタッキングであり（次元を追加することができます）、どちらも次元を指定できます。覚えておいてください、それら2つで十分です。

⚠️注意が必要なのは、 `hstack` と `vstack` と `stack` は関係なく、むしろ `concatenate` です。



```python
rng = np.random.default_rng(42)

arr1 = rng.random((2, 3))
arr2 = rng.random((2, 3))
arr1, arr2
```




    (array([[0.77395605, 0.43887844, 0.85859792],
            [0.69736803, 0.09417735, 0.97562235]]),
     array([[0.7611397 , 0.78606431, 0.12811363],
            [0.45038594, 0.37079802, 0.92676499]]))



#### `np.concatenate`



```python
# 默认沿axis=0（列）连接
np.concatenate((arr1, arr2))
```




    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235],[0.7611397 , 0.78606431, 0.12811363],[0.45038594, 0.37079802, 0.92676499]])





```python
# 沿 axis=1（行）连接
np.concatenate((arr1, arr2), axis=1)
```




    array([[0.77395605, 0.43887844, 0.85859792, 0.7611397 , 0.78606431,
            0.12811363],
           [0.69736803, 0.09417735, 0.97562235, 0.45038594, 0.37079802,
            0.92676499]])



---



```python
# 竖直按行顺序拼接
# 注意：vstack 虽然看起来是 stack，但其实它是 concatenate，建议您只使用 `np.concatenate`
np.vstack((arr1, arr2))
```




    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235],[0.7611397 , 0.78606431, 0.12811363],[0.45038594, 0.37079802, 0.92676499]])





```python
# 水平按列顺序拼接
# 道理和 vstack 一样，建议使用 `np.concatenate` axis=1
np.hstack((arr1, arr2))
```




    array([[0.77395605, 0.43887844, 0.85859792, 0.7611397 , 0.78606431,
            0.12811363],
           [0.69736803, 0.09417735, 0.97562235, 0.45038594, 0.37079802,
            0.92676499]])



#### `np.stack`



```python
# 堆叠，默认根据 axis=0 进行
np.stack((arr1, arr2))
```




    array([[[0.77395605, 0.43887844, 0.85859792],
            [0.69736803, 0.09417735, 0.97562235]],
    
           [[0.7611397 , 0.78606431, 0.12811363],
            [0.45038594, 0.37079802, 0.92676499]]])





```python
_.shape
```




    (2, 2, 3)





```python
# 堆叠，根据 axis=2
np.stack((arr1, arr2), axis=2)
```




    array([[[0.77395605, 0.7611397 ],
            [0.43887844, 0.78606431],[0.85859792, 0.12811363]],
    
           [[0.69736803, 0.45038594],
            [0.09417735, 0.37079802],[0.97562235, 0.92676499]]])





```python
# _ 表示上一个 Cell 的输出结果
_.shape
```




    (2, 3, 2)



---



```python
# 纵深按 axis=2 堆叠，不管它就行，我们认准 `stack`
np.dstack((arr1, arr2))
```




    array([[[0.77395605, 0.7611397 ],
            [0.43887844, 0.78606431],[0.85859792, 0.12811363]],
    
           [[0.69736803, 0.45038594],
            [0.09417735, 0.37079802],[0.97562235, 0.92676499]]])





```python
_.shape
```




    (2, 3, 2)



### リピート

⭐⭐⭐

繰り返しは実際には別の接続方法であり、繰り返す次元を指定することもできます。いくつかのディープラーニングモデルのデータ構築に非常に有用（便利）です。

⚠️注意しなければならないのは、アレイ全体ではなく、次元ごとに次元ごとに繰り返されるということです。



```python
rng = np.random.default_rng(42)
arr = rng.integers(0, 10, (3, 4))
arr
```




    array([[0, 7, 6, 4],
           [4, 8, 0, 6],[2, 0, 5, 9]])





```python
# 在 axis=0（沿着列）上重复 2 次
np.repeat(arr, 2, axis=0)
```




    array([[0, 7, 6, 4],
           [0, 7, 6, 4],[4, 8, 0, 6],[4, 8, 0, 6],[2, 0, 5, 9],[2, 0, 5, 9]])





```python
# 在 axis=1（沿着行）上重复 3 次
np.repeat(arr, 3, axis=1)
```




    array([[0, 0, 0, 7, 7, 7, 6, 6, 6, 4, 4, 4],
           [4, 4, 4, 8, 8, 8, 0, 0, 0, 6, 6, 6],[2, 2, 2, 0, 0, 0, 5, 5, 5, 9, 9, 9]])



### スピンオフ

⭐⭐⭐

スプライススタックがあれば、当然分割があります。これはスライスとインデックスではなく、arrayを希望の数部に分割することに注意してください。あまり使われているわけではありません。APIは `np.split` を覚えておくだけで、その他はショートカットです。

⚠️注意が必要なのは：分割されたaxisはその次元を分割することです。



```python
rng = np.random.default_rng(42)
arr = rng.integers(1, 100, (6, 4))
arr
```




    array([[ 9, 77, 65, 44],
           [43, 86,  9, 70],[20, 10, 53, 97],[73, 76, 72, 78],[51, 13, 84, 45],[50, 37, 19, 92]])



#### `np.split`



```python
# 默认切分列（axis=0），切成 3 份
np.split(arr, 3)
```




    [array([[ 9, 77, 65, 44],
            [43, 86,  9, 70]]),
     array([[20, 10, 53, 97],
            [73, 76, 72, 78]]),
     array([[51, 13, 84, 45],
            [50, 37, 19, 92]])]





```python
# （axis=1）切分行
np.split(arr, 2, axis=1)
```




    [array([[ 9, 77],
            [43, 86],[20, 10],[73, 76],[51, 13],[50, 37]]),
     array([[65, 44],
            [ 9, 70],[53, 97],[72, 78],[84, 45],[19, 92]])]



---



```python
# 和上面的一个效果
np.vsplit(arr, 3)
```




    [array([[ 9, 77, 65, 44],
            [43, 86,  9, 70]]),
     array([[20, 10, 53, 97],
            [73, 76, 72, 78]]),
     array([[51, 13, 84, 45],
            [50, 37, 19, 92]])]





```python
# 等价的用法
np.hsplit(arr, 2)
```




    [array([[ 9, 77],
            [43, 86],[20, 10],[73, 76],[51, 13],[50, 37]]),
     array([[65, 44],
            [ 9, 70],[53, 97],[72, 78],[84, 45],[19, 92]])]



## フィルタリングとフィルタ

このセクションはインデックスとスライスと似ているが、条件に合致するコンテンツを全体から統一的にフィルタリングする傾向があり、インデックスとスライスは何らかの方法でコンテンツを切り出すことが多い。この小節の内容も同様に非常に重要であり、2番目に最も重要な小節と言える。主に以下の内容が含まれます：

- 条件フィルタリング
- 抽出（条件別）
- サンプリング（分布による）
- 最大最小インデックス (特殊値)

これらのいくつかのコンテンツはいずれも重要で、非常に頻繁に使われています。条件フィルタリングはMaskまたは異常値処理によく使用され、抽出は結果フィルタリングによく使用され、サンプリングはデータ生成（負サンプルサンプリングなど）によく使用され、最大最小indexは機械学習モデルの予測結果判定によく使用されます（最大確率があるindexに基づいて結果がどの種類に属するかを決定する）。



```python
rng = np.random.default_rng(42)
arr = rng.integers(1, 100, (3, 4))
arr
```




    array([[ 9, 77, 65, 44],
           [43, 86,  9, 70],[20, 10, 53, 97]])



### 条件フィルタリング

⭐⭐⭐

その名の通り、一定の条件によってarrayはフィルタリング（タグ付け）を行い、その後の処理を行います。コアAPIは `np.where` です。

⚠️注意すべきは、whereは各次元のindexをそれぞれ返し、値を付与したものは条件を満たしていないことである。



```python
# 条件筛选，可以直接在整个 array 上使用条件
arr > 50
```




    array([[False,  True,  True, False],
           [False,  True, False,  True],[False, False,  True,  True]])





```python
# 返回满足条件的索引，因为是两个维度，所以会返回两组结果
np.where(arr > 50)
```




    (array([0, 0, 1, 1, 2, 2]), array([1, 2, 1, 3, 2, 3]))





```python
# 不满足条件的赋值，将 <=50 的替换为 -1
np.where(arr > 50, arr, -1)
```




    array([[-1, 77, 65, -1],
           [-1, 86, -1, 70],[-1, -1, 53, 97]])



### 取り出す

⭐

指定した条件の値をarray内で抽出します。

⚠️注意が必要なのは、抽出とユニークな値はいずれも1次元ベクトルを返します。



```python
# 提取指定条件的值
np.extract(arr > 50, arr)
```




    array([77, 65, 86, 70, 53, 97])





```python
# 唯一值，是另一种形式的提取
np.unique(arr)
```




    array([ 9, 10, 20, 43, 44, 53, 65, 70, 77, 86, 97])



### 抜き取り

⭐⭐⭐⭐⭐

モデルを実行するときは、データの一部を使用してプロセス全体を迅速に検証する必要があることがよくあります。もちろん、シミュレーションデータを生成するには `np.random` を使用することができます。しかし、実際のデータがある場合は、実際のデータからランダムにサンプリングした方が良いでしょう。



```python
rng = np.random.default_rng(42)
# 第一个参数是要抽样的集合，如果是一个整数，则表示从 0 到该值
# 第二个参数是样本大小
# 第三个参数表示结果是否可以重复
# 第四个参数表示出现的概率，长度和第一个参数一样

# 由于（0 1 2 3）中 2 和 3 的概率比较高，自然就选择了 2 和 3
rng.choice(4, 2, replace=False, p=[0.1, 0.2, 0.3, 0.4])
```




    array([3, 2])





```python
# 旧的 API
# 如果是抽样语料的 index，更多的方法是这样：
data_size = 10000
np.random.choice(data_size, 50, replace=False)
```




    array([6339, 4894, 1531, 7814,  224, 9538, 9619, 3801, 3359, 3617, 2795,
           6627, 8501, 1681, 4212, 5085, 2439,  744, 9123, 6733, 5688, 5480,6983, 7058,  310, 1838, 5072,  746, 5873, 9372, 5953, 4944, 1780,
            464, 1247,  845, 1807, 7354, 4925,  547, 2996, 3909, 7344, 9617,
           8642,  661, 2453, 5475,  228, 2427])



### 最大値Index

⭐⭐⭐⭐⭐

このセクションは主に2つのAPI： `np.argmax(min)` と `np.argsort` ですが、もちろん最もよく使われるのは最初のAPIです。言うまでもなくaxisを指定することができます（必要です）。



```python
rng = np.random.default_rng(42)
arr = rng.uniform(1, 100, (3, 4))
arr
```




    array([[77.62164881, 44.44896554, 86.00119407, 70.03943488],
           [10.32355744, 97.58661281, 76.3528305 , 78.82036622],[13.68324963, 45.58820785, 37.7090044 , 92.7497339 ]])



#### `np.argmax/argmin`



```python
# 所有值中最大值的 Index，基本不这么用
np.argmax(arr)
```




    5





```python
# 按列（axis=0）最大值的 Index
np.argmax(arr, axis=0)
```




    array([0, 1, 0, 2])





```python
# 按行（axis=1）最小值的 Index
np.argmin(arr, axis=1)
```




    array([1, 0, 0])



#### `np.argsort`



```python
arr
```




    array([[77.62164881, 44.44896554, 86.00119407, 70.03943488],
           [10.32355744, 97.58661281, 76.3528305 , 78.82036622],[13.68324963, 45.58820785, 37.7090044 , 92.7497339 ]])





```python
# 默认按行（axis=1）排序的索引
np.argsort(arr)
```




    array([[1, 3, 0, 2],
           [0, 2, 3, 1],[0, 2, 1, 3]])





```python
# 数据按行（axis=1）排序的索引，同上
np.argsort(arr, axis=1)
```




    array([[1, 3, 0, 2],
           [0, 2, 3, 1],[0, 2, 1, 3]])





```python
# 数据按列（axis=0）排序索引
np.argsort(arr, axis=0)
```




    array([[1, 0, 2, 0],
           [2, 2, 1, 1],[0, 1, 0, 2]])



## 行列和演算

このセクションでは、行列と関連する演算に焦点を当てます。主に次のとおりです：

- 算術（四則演算及びその他の基本的な算術）
- 放送
- 行列相関

これらのコンテンツは実際には非常に一般的に使われていて、私たちは自分が使っていることにさえ気づかないほど一般的で、しかも非常に簡単です。もちろん、高緯度の計算はここでは触れていませんが、論理は一致していますが、ただもっと複雑です。

### 算術

⭐⭐⭐⭐

すべての算術関数はarrayに直接適用できます。

⚠️注意すべきは、 `mod` 演算は複数の被除数を指定することができます。



```python
rng = np.random.default_rng(42)
arr = rng.integers(1, 20, (3, 4))
arr
```




    array([[ 2, 15, 13,  9],
           [ 9, 17,  2, 14],[ 4,  2, 11, 19]])





```python
# +-*/ 四则运算，就跟两个数字计算一样
arr * 2
```




    array([[ 4, 30, 26, 18],
           [18, 34,  4, 28],[ 8,  4, 22, 38]])





```python
# 平方也可以
arr ** 2
```




    array([[  4, 225, 169,  81],
           [ 81, 289,   4, 196],[ 16,   4, 121, 361]])





```python
# 开方
np.sqrt(arr)
```




    array([[1.41421356, 3.87298335, 3.60555128, 3.        ],
           [3.        , 4.12310563, 1.41421356, 3.74165739],[2.        , 1.41421356, 3.31662479, 4.35889894]])





```python
# log
np.log(arr)
```




    array([[0.69314718, 2.7080502 , 2.56494936, 2.19722458],
           [2.19722458, 2.83321334, 0.69314718, 2.63905733],[1.38629436, 0.69314718, 2.39789527, 2.94443898]])





```python
# 超过5的都换成5
np.minimum(arr, 5)
```




    array([[2, 5, 5, 5],
           [5, 5, 2, 5],[4, 2, 5, 5]])





```python
# 低于5的都换成5
np.maximum(arr, 5)
```




    array([[ 5, 15, 13,  9],
           [ 9, 17,  5, 14],[ 5,  5, 11, 19]])





```python
# 四舍五入
np.round(np.sqrt(arr), 2)
```




    array([[1.41, 3.87, 3.61, 3.  ],
           [3.  , 4.12, 1.41, 3.74],[2.  , 1.41, 3.32, 4.36]])





```python
# floor/ceil
np.floor(np.sqrt(arr))
```




    array([[1., 3., 3., 3.],
           [3., 4., 1., 3.],[2., 1., 3., 4.]])





```python
np.ceil(np.sqrt(arr))
```




    array([[2., 4., 4., 3.],
           [3., 5., 2., 4.],[2., 2., 4., 5.]])





```python
arr
```




    array([[ 2, 15, 13,  9],
           [ 9, 17,  2, 14],[ 4,  2, 11, 19]])





```python
# mod <=> x % 3
np.mod(arr, 3)
```




    array([[2, 0, 1, 0],
           [0, 2, 2, 2],[1, 2, 2, 1]])





```python
arr-5
```




    array([[-3, 10,  8,  4],
           [ 4, 12, -3,  9],[-1, -3,  6, 14]])





```python
# 还可以使用多个被除数
np.mod(arr, arr-5)
```




    array([[-1,  5,  5,  1],
           [ 1,  5, -1,  5],[ 0, -1,  5,  5]])



> 放送について

 `numpy` 異なる形状のアレイを扱う際に使用する手段は、ユーザーの利便性を極めて高めている。計算中、小さい配列は、相手の形状に合わせるように、大きい配列上でブロードキャストされます。

⚠️注意が必要なのは、放送は対応する形状を満たす必要があることです。



```python
rng = np.random.default_rng(42)
a = rng.integers(1, 100, (3, 4))
a
```




    array([[ 9, 77, 65, 44],
           [43, 86,  9, 70],[20, 10, 53, 97]])





```python
# 广播，后面的被当做 1 行 4 列
a + [1,2,3,4]
```




    array([[ 10,  79,  68,  48],
           [ 44,  88,  12,  74],[ 21,  12,  56, 101]])





```python
# 或者这样广播，后面的被当做 3 行 1 列
a + [[1], [2], [3]]
```




    array([[ 10,  78,  66,  45],
           [ 45,  88,  11,  72],[ 23,  13,  56, 100]])





```python
# 之前的取余也是可以的
np.mod(a, [1,2,3,4])
```




    array([[0, 1, 2, 0],
           [0, 0, 0, 2],[0, 0, 2, 1]])



### マトリクス

⭐⭐⭐⭐⭐

このセクションでは、線形代数における行列の処理について説明しており、行列に関する一般的なAPIをいくつか紹介します。

⚠️注意が必要なのは、 `dot` と `matmul` は高次元では異なります。



```python
rng = np.random.default_rng(42)
a = rng.integers(1, 10, (3, 2))
b = rng.integers(1, 10, (2, 4))
c = rng.integers(1, 10, (3, 3))
a, b, c
```




    (array([[1, 7],
            [6, 4],[4, 8]]),
     array([[1, 7, 2, 1],
            [5, 9, 7, 7]]),
     array([[7, 8, 5],
            [2, 8, 5],[5, 4, 2]]))



#### `arr.dot`



```python
# array 乘法
np.dot(a, b)
```




    array([[ 36,  70,  51,  50],
           [ 26,  78,  40,  34],[ 44, 100,  64,  60]])





```python
# 或者这样乘
a.dot(b)
```




    array([[ 36,  70,  51,  50],
           [ 26,  78,  40,  34],[ 44, 100,  64,  60]])





```python
# 我们看下高维度下 dot 和 matmul 的区别
# ijk, lkm -> ijlm
np.dot(np.ones((5, 2, 3)), np.ones((4, 3, 6))).shape
```




    (5, 2, 4, 6)



#### `np.matmul`



```python
# 矩阵乘法
# 与 dot 的主要区别是：matmul 矩阵（好像元素一样）堆叠在一起广播
np.matmul(a, b)
```




    array([[ 36,  70,  51,  50],
           [ 26,  78,  40,  34],[ 44, 100,  64,  60]])





```python
# 同上，写起来比较好看的方法
a @ b
```




    array([[ 36,  70,  51,  50],
           [ 26,  78,  40,  34],[ 44, 100,  64,  60]])





```python
# ijk, ikl -> ijl
np.matmul(np.ones((5, 2, 3)), np.ones((4, 3, 6))).shape
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-95-a5c509c4e8f1> in <module>
          1 # ijk, ikl -> ijl
    ----> 2 np.matmul(np.ones((5, 2, 3)), np.ones((4, 3, 6))).shape
    

    ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (5,2,3)->(5,newaxis,newaxis) (4,3,6)->(4,newaxis,newaxis) and requested shape (2,6)




```python
# ijk, ikl -> ijl
np.matmul(np.ones((4, 2, 3)), np.ones((4, 3, 6))).shape
```




    (4, 2, 6)



---



```python
# 点积
np.vdot(a, a)
```




    182





```python
# 对，就是点积
np.sum(a*a)
```




    182





```python
# 内积
np.inner(a, a)
```




    array([[50, 34, 60],
           [34, 52, 56],[60, 56, 80]])





```python
# 对，就是内积
a.dot(a.T)
```




    array([[50, 34, 60],
           [34, 52, 56],[60, 56, 80]])





```python
# 行列式
np.linalg.det(c)
```




    -19.999999999999996





```python
# 逆矩阵（方阵）
np.linalg.inv(c)
```




    array([[ 0.2 , -0.2 ,  0.  ],
           [-1.05,  0.55,  1.25],[ 1.6 , -0.6 , -2.  ]])



## まとめと心得

Numpyの『小白から入門まで』のチュートリアルを終えておめでとうございます。この基礎があれば、今後ほとんどの使用シーンに対応できると信じています。もしあなたがNumpyを深く学びたいなら、私たちのフォローアップコース「入門からマスターまで」に注目してください。そこで、私たちは `numpy` 内部の詳細を深く掘り下げ、その原理を体系的かつ包括的に紹介します。後でお会いできることを楽しみにしています。


### 内容のまとめ

このチュートリアルは6つの部分で構成されています。最も重要な（3つの⭐以上の）APIをリストして、あなたの記憶と印象を深めることができます：

- 作成と生成
  - `np.linspace(start, end, nums)`
  - `rng.integers/uniform(low, high, size)`
  - `rng.normal(loc, scale, size)`
- 統計と属性
  - `arr.shape`
  - `arr.sum/max/min(axis, keepdims)`
  - `np.average(arr, axis)`
- シェイプと変換
  - `arr.reshpae/np.reshape`
  - `np.expand_dims(arr, axis)`
  - `np.squeeze(arr axis)`
  - `np.transpose(arr, axis)`
  - `arr.T`
- 分解と組み合わせ
  - `arr[start:stop:step, ...]`
  - `np.concatenate((arr1, arr2), axis)`
  - `np.stack((arr1, arr2), axis)`
  - `np.repeat(arr, repeat_num, axis)`
  - `np.split(arr, part_num, axis)`
- フィルタリングとフィルタ
  - `np.where(condition, arr, replaced_val)`
  - `rng.choice(a, size, replace=False, p=probs_size_equals_a)`
  - `rng.argmax/argmin/argsort(arr, axis)`
- 行列と計算
  - `+-*/`
  - `np.dot(a, b) == a.dot(b)`
  - `np.matmul(a, b) == a @ b`


### 心得テクニック

- アレイを生成/表示するときは、具体的なデータ型に注意してください
- 多くのAPIにaxisがあり、それを特定の次元に沿ってまたはそれを操作すると理解しやすく理解できます

## 強化と練習

次に、学んだ成果をチェックする時期ですので、そのためにはおかず2品とマルチメニューを用意しましたので、どうぞお楽しみください。

背景說明：irisデータセットを使用して、targetは0と1を選択して（setosa、versicolor）、2を舍てて（私たちは二分類のみを使ってよい）

### 基本テーマ1

データを分割するには、次の要件があります：

- それぞれのtargetごとに80%/20%でデータを訓練データとテストデータに分割した
- 分割する際には5つのサンプルごとに、中間の1つをテストデータ、残りを訓練データとする必要がある

### 基本テーマ2

結果を予測するには、次のような要件があります：

- トレーニングデータにあらかじめトレーニングされたウェイトパラメータをロードする（パラメータを格納する方法： `np.savez("data/weight", weight=weight)`）
- モデル `sigmoid(W·X)` を使用したテストデータの予測
- 出力予測の精度


### 高度なテーマ

すでに機械学習の基礎を持っている場合は、この練習を試してみてはいかがでしょうか。 `numpy` より多くのAPIを使用します。

Numpyを使用して単純なニューラルネットワークを実装するには、以下の要件が必要です。

- フィーチャーセレクション2番目と4番目 (sepal width,petal width)
- 逆伝播と勾配降下を用いた学習



ヒント：


```python
# 数据集
from sklearn.datasets import load_iris

iris = load_iris()
data = iris["data"]
target = iris["target"]

# Sigmoid
def sigmoid(x: np.array, derive: bool = False) -> np.array:
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
```

## 解答と参考

以下は練習問題の参考解答です。参考である以上、基準ではないに違いありません。最終的な目的を達成するには任意の方法で答えを得ることができます。あなたが望むならforループを使ってもいいですが、最終的な答えが間違っている場合は、やはり再確認する必要があります。私たちはプロセスを特に重視し、注目していますが、結果はもっと重要です。すべての道はローマに通じて、私たちは最後にローマを手に入れるのではないでしょうか？

最後に、『Unixプログラミングの芸術』の一つの観点を引用します。まずプログラムを正しく実行させることです。完成> 完璧で、健康で幸せになります。



```python
def sigmoid(x: np.array, derive: bool = False) -> np.array:
    # 求导数（用于反向传播）
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
```



```python
from sklearn.datasets import load_iris

# 加载 iris 数据集
iris = load_iris()
# 提取 target ≤ 1 的 target，并将维度从 (length, ) 拓展为 (length, 1)
y = np.expand_dims(np.extract(iris["target"] <= 1, iris["target"]), 1)
# 根据 y 的 size 拿到对应的 x
x = iris["data"][:y.shape[0]]
```

### 基本テーマ1



```python
# 有多少条数据
data_size = x.shape[0]
# 20% 作为测试数据
test_size = int(data_size * 0.2)
# 找到测试数据的 index
test_idx = [2 + i*5 for i in range(test_size)]
# 过滤掉测试数据的 index，剩下的就是训练数据的 index
train_idx = [i for i in range(data_size) if i not in test_idx]
```



```python
# 使用训练数据的 index 得到训练数据
x_train = x[train_idx]
y_train = y[train_idx]
# 直接使用索引切片得到测试数据，当然也可以使用测试数据 index
x_test = x[2: data_size : 5]
y_test = y[2: data_size : 5]
```

### 基本テーマ2



```python
# 预测函数
def predict(x, w):
    return sigmoid(x.dot(w))
```



```python
# 加载权重，注意 key="weight"
w = np.load("./data/weight.npz")["weight"]
```



```python
# 拿到 sigmoid 的结果，即 probability
prob = predict(x_test, w)
```



```python
# >（或 ≥）0.5 作为 1 或 0 的标准
sum((prob > 0.5) == y_test) / len(y_test)
```




    array([1.])



### 高度なテーマ



```python
# 只取第2和第4个特征
x_train_part = x_train[:, [1, 3]]
x_test_part = x_test[:, [1, 3]]
```



```python
def test(x, y, wh, wo, use_hidden: bool = False):
    """
    预测函数，如果使用隐层的话，需要增加对应的计算
    """
    if use_hidden:
        ypred = sigmoid(sigmoid(x.dot(wh)).dot(wo))
    else:
        ypred = sigmoid(x.dot(wo))
    print("Acc: %.2f" % (sum((ypred >= 0.5 ) == y) / len(y))[0])
```



```python
def train(x, y, use_hidden: bool = False, lr: float = 0.01):
    """
    训练函数，可以支持单隐层或无隐层
    """
    hidden_size = 8 if use_hidden else x.shape[1]
    # 初始化，可以有多种初始化方式，我们这里使用 uniform
    wh = np.random.uniform(-np.sqrt(1/hidden_size), np.sqrt(1/hidden_size), (x.shape[1], hidden_size))
    wo = np.random.uniform(-1, 1, (hidden_size, 1))
    # 1000 个 Epoch
    for epoch in range(1, 1001):
        # 使用隐层
        if use_hidden:
            hidden = np.dot(x, wh)
            hidden = sigmoid(hidden)
        else:
            hidden = x
        logits = np.dot(hidden, wo)
        pred = sigmoid(logits)
        
        # 输出层的 error
        pred_err = y - pred
        # 反向传播，输出层的更新权重
        pred_delta = pred_err * sigmoid(pred, True)
        if use_hidden:
            # 隐层的 error 和更新权重
            hidd_err = pred_delta.dot(wo.T)
            hidd_delta = hidd_err * sigmoid(hidden, True)
        
        # 更新参数
        wo += lr * hidden.T.dot(pred_delta)
        if use_hidden:
            wh += lr * x.T.dot(hidd_delta)

        # 计算损失
        loss = np.mean(np.abs(pred_err))
        if epoch % 200 == 0:
            # 如果有验证集，可以在这里对验证集进行验证，结果有提升（如 loss下降或 acc 提高）时进行存储
            # 还可以根据结果提升情况设计自动提前终止的方案（比如连续 3 次不再提升）
            print("Error:" + str(loss))
    
    # 得到训练集的 Acc
    test(x, y, wh, wo, use_hidden)
    # 返回权重（参数）
    return wh, wo
```



```python
wh, wo = train(x_train_part, y_train)
```

    Error:0.12291230663724437Error:0.08282596392702782Error:0.06586770282341464Error:0.056016204876575326Error:0.049407934168643926Acc: 1.00

    


```python
test(x_test_part, y_test, wh, wo)
```

    Acc: 1.00

    


```python
wh, wo = train(x_train_part, y_train, True)
```

    Error:0.16416415380726673Error:0.08236885703184059Error:0.058163838923958065Error:0.04626498346103437Error:0.03903864391270989Acc: 1.00

    


```python
test(x_test_part, y_test, wh, wo, True)
```

    Acc: 1.00
    

## 文献と資料

-  [NumPyチュートリアル初心者チュートリアル](https://www.runoob.com/numpy/numpy-tutorial.html)
-  [NumPy 中文](https://www.numpy.org.cn/)



```python

```
