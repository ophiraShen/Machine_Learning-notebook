

# Self-attention 自注意力机制

Self - attention 要解决的问题：

"输入是一排向量，且输入向量的数目不一定"

输出：每个输入向量对应一个标签；全部输入对应一个标签；输出个数由机器决定(Seq2seq)；

<img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228182216295.png" alt="image-20231228182216295" style="zoom:20%;"/>   <img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228182659844.png" alt="image-20231228182659844" style="zoom:20%;" />

例如：文字处理，语音处理，图网处理（结点和边），分子；

## 模型架构

### 发展过程（Sequence Labeling）

1）把每个向量分别输入到 Fully-connected 里，然后给出每个向量对应的输出。

​	**问题：**同样的输入，一定会给出相同的输出，但是以图中为例，第一个 "saw" 和第二个 "saw" 的词性不一样，需要不同的输出。

​	**解决思路：**让模型能够看到上下文;

<img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228183514590.png" alt="image-20231228183514590" style="zoom:30%;"/>  



2）将前后向量都传给 Fully-connected，设定一个 window。

​	**问题：**window 多大才合适？如果任务需要考虑整个 sequence 才可以；

<img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228183808487.png" alt="image-20231228183808487" style="zoom:30%;"/>  





### Self-attention 模型架构

**思想：**模型会考虑整个 sequence 的信息；输入个数 == 输出个数； 

<img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228185735097.png" alt="image-20231228185735097" style="zoom:30%;" />  





#### 如何做到考虑整个 sequence？

​	输入向量组：$a^1, a^2, a^3, a^4$；

​	输出向量组：$b^1, b^2, b^3, b^4$；

​	其中，$b^1$ 就是 $a^1$ 考虑了整个向量组之后输出的结果；计算架构入下图：

​	<img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228221036575.png" alt="image-20231228221036575" style="zoom:30%;" /> 

​	其中：

​	**$q^i$：**query，当前询问的向量的信息，$q^i = W^q a^i$；

​	**$k^i$：** key，sequence 中其它要考虑的向量的信息，  $k^i = W^k a^i$；

​	**$v^i$：** value，从 $a^i$ 中进一步提取的信息，$v^i = W^v a^i$；

​	**$\alpha_{i,j}$：** 以 $a^i$ 为 query，$a^j$ 为 key，求得的两者间的相关性；

​	**$\alpha'_{i,j}$：** 经一步处理之后的 attention score； 

​	**1）$\alpha$ 的计算方式；**                                                           **2）对 $\alpha$ 进行处理，生成 $\alpha'$；**

​	<img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228185909260.png" alt="image-20231228185909260" style="zoom:20%;" /> <img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228192738698.png" alt="image-20231228192738698" style="zoom:20%;" />   







#### Self - attention 并行计算

Self-Attention是如何实现并行化计算的呢？上述Self-Attention的计算其实都是一些矩阵运算，因此可以使用GPU加速。



1. 输入：$I$；

2. 求 $Q, K, V,$ 矩阵

   1. $Q = W^q I$
   2. $K = W^k I$
   3. $V = W^v I$

3. $A = K^T Q$

4. $A' = softmax(A)$

5. 输出：$O = VA'$

   

   

   <img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228223158950.png" alt="image-20231228223158950" style="zoom:20%;"/>  <img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228223454158.png" alt="image-20231228223454158" style="zoom:20%;"/>   



<img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228223541248.png" alt="image-20231228223541248" style="zoom:20%;"/>  <img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228223606536.png" alt="image-20231228223606536" style="zoom:20%;"/> 







## Multi-head Self-attention

Self-attention有一种变形是Multi-head Self-attention，现以2个head的情况为例介绍Multi-head Self-attention。

Multi-head的作用在于不同head关注的东西可能不一样。

<img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228224046570.png" alt="image-20231228224046570" style="zoom:30%;"/>              <img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228224119061.png" alt="image-20231228224119061" style="zoom:30%;"/>







## Position Encoding

paper: "Learning to Encode Position for Transformer with Continuous Dynamical Model"  https://arxiv.org/abs/2003.09229

上面讲的Self-Attention并没有考虑sequence中元素之间的顺序，所以需要Positional Encoding。

**方法：**

1. 为 sequence 中每一个位置设置一个专属的代表位置的 vector，用 $e^i$ 表示；是人设定，不是机器学习的；

2. $e^i + a^i$；

   <img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228224942628.png" alt="image-20231228224942628" style="zoom:25%;"/> 







## Self-attention v.s. CNN

paper: "On the Relationship between Self-Attention and Convolutional Layers"  https://arxiv.org/abs/1911.03584

CNN 可以看做是一种简化版的 Self-attetion；

因为，在 CNN 中，只考虑 receptive field 里的信息，而 Self-attention 会考虑整张图片的信息；以某一个 pixel 为 query，其余的 pixel为 key，也可以人为，Self-attention 中的 receptive field 是机器自己学出来的，而 CNN 中是人为设定的；



 <img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228231200664.png" alt="image-20231228231200664" style="zoom:23%;"/>   <img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228231420453.png" alt="image-20231228231420453" style="zoom:20%;"/> 







## Self-attention v.s. RNN

paper: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"  https://arxiv.org/abs/2006.16236

1. 当最后一个 vector 需要考虑第一个 vector 的信息时，RNN 需要将第一个 vector 的信息一直存储，直到执行到最后一个 vector为止；而 Self-attention 不需要，在 Self-attention 中，任何两个 vector 关系都会同时 被考虑到；

2. Self-attention 可以进行并行计算，output 中的 vector 是同时被计算出的，不需要互相等待；而 RNN 无法进行并行计算，output 中的 vector 是依次被计算出的；

   

   <img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228232931474.png" alt="image-20231228232931474" style="zoom:25%;"/>    







## Self-attention for Graph

在 Graph 中，只需要计算 edge 间两个 nodes 的关联性；



<img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228233223870.png" alt="image-20231228233223870" style="zoom:25%;"/>   



## More

paper: "Long Range Arena: A Benchmark for Efficient Transformers"  https://arxiv.org/abs/2011.04006

paper: "Efficient Transformers: A Survey"  https://arxiv.org/abs/2009.06732

<img src="/Users/shenyige/Library/Application Support/typora-user-images/image-20231228233436824.png" alt="image-20231228233436824" style="zoom:20%;" /> 



------



