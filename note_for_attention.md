# Attention

面向文本处理，基于Seq2seq模型，我先讲讲我对于上下文向量的理解。

Seq2seq的encoder会编码出一个凝练序列信息的向量c，seq2seq的解码器会以上一个时间步的输出和隐藏状态作为输入，那么c的作用就是引导序列的继续输出，如果没有c，那么解码器的内容就会困在了第一个词里。

seq2seq的问题是太多的信息凝聚在一个向量中，无法有效聚焦有用的信息，所以引入attention机制。

seq2seq中：c = g (h<sub>1</sub> + h<sub>2</sub> + ......)   g是一个函数，可能是求和等

引入attention: c<sub>i</sub> = $\Sigma$($\alpha$<sub>i</sub> * $h_i$)  其中$\alpha_i$是注意力权重，$\alpha_i$通过计算解码器状态和编码器所有隐藏状态点积再softmax得到（相似度，相似点积大）

这样就有了动态的上下文向量，效果会更好。

## Attention机制

对于seq2seq模型，attention机制是这样起作用的 ：
![seq2seq](./pic/seq2seq.png)

上图是原始的seq2seq，编码器encoder通过RNN编码出一个上下文向量c，在decoder去输出。

![seq2seq_with_attention](./pic/seq2seq_with_attention.png)

上图是融入注意力机制的seq2seq，encoder编码时保留每一个时间步的隐藏状态$h_t-h_{t-1}$，decoder进行解码时，每次都用上一个时间步的隐藏状态$s_{t-1}$，然后去计算跟编码器各个隐藏状态的相似度，一般可以用点积的方式去求，即$s_{t-1}*h_i^T$，然后进行一层softmax获得各个隐藏状态的权重$\alpha_1-\alpha_n$，通过权重得到上下文向量$c_i=\Sigma\alpha_i*h_i$，利用$c_i$继续解码。

从seq2seq的优化中抽象出注意力机制，Q、K、V。

Q是Query，查询

K是Key，键值

V是Value，实际值

总的流程是，利用Q和各个K匹配计算权重，利用权重对V加权求和，得到注意力值。

## Transformer

![transformer](./pic/Transformer.png)

一步步剖析一下Transformer

首先第一步是**词嵌入**，这里transformer是初始化随机生成词嵌入(embedding)矩阵，在训练过程在进行参数更新。

然后是**位置编码**，因为transformer完全基于注意力机制，所以不会保留它的位置信息，所以要进行位置编码。位置编码需要：每个位置唯一且确定，有界稳定，可以泛化到更长序列。transformer的解决方案是正余弦编码，序列的第pos个位置，维度半索引是i，位置编码：

$PE(pos,2i)=sin(pos/(10000^{2i/d\_model})$

$PE(pos,2i+1)=cos(pos/(10000^{2i/d\_model}))$

- **`pos`**：词在序列中的位置（第0个，第1个，第2个...）

- **`i`**：编码向量的维度半索引（从0到255，如果`d_model=512`）

- **`2i` 和 `2i+1`**：对维度进行分组，相邻的两个维度使用相同的频率但不同的相位（正弦和余弦）

- **`10000^(2i/d_model)`**：这是频率项，决定了正弦波的波长（这里10000是一个经验值）

精妙之处在于，相对位置可以用线性表达

$sin((pos + k) * w_i) = sin(pos * w_i)cos(k * w_i) + cos(pos * w_i)sin(k * w_i)$
$cos((pos + k) * w_i) = cos(pos * w_i)cos(k * w_i) - sin(pos * w_i)sin(k * w_i)$

用矩阵形式表示：

[PE(pos+k, 2i)  ]       [cos(k*w_i)   sin(k*w_i)]    [PE(pos, 2i)    ]
[PE(pos+k, 2i+1)] = [-sin(k*w_i)  cos(k*w_i)] * [PE(pos, 2i+1)]

这意味着，PE(pos+k) 可以通过PE(pos) 乘以一个固定的旋转矩阵得到。

**位置编码如何与词嵌入结合：** 最终输入 = 词嵌入向量 + 位置编码向量

接下来是重要的**自注意力机制**
