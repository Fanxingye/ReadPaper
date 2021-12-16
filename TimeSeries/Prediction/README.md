# **Informer: Beyond Efficient Transformer for Long SequenceTime-Series Forecastin**g

## 1. Motivation

&emsp; (1) LSTM对于长时间序列（48个点以上）的预测能力较弱，transfomer在长序列上的效果更好，但是transformer计算量太大。  

&emsp; (2) transformer处理长时间序列的三个显著限制：  

&emsp;&emsp; a) self-attention $L^2$级别时间复杂度；

&emsp;&emsp; b) 堆叠多个encoding-decoding layer对内存消耗大； 

&emsp;&emsp; c) 预测长输出的速度低。

&emsp; (3) 本文的目标是改进transformer模型，提高计算、内存使用的效率，改进模型结构的有效性，提高预测精度。

## 2. Contributions
&emsp; (1) 提出transformer结构的Informer模型，捕捉到时间序列的长期依赖  
&emsp; (2) 提出*ProbSparse* self-attention机制，实现了$O(LlogL)$的时间复杂度和$O(LlogL)$的内存使用  
&emsp; (3) 提出self-attention蒸馏算子，减小网络大小  
&emsp; (4) 提出生成式的decoder，只需一次前向传播就能获取长序列的输出

## 3. Content
### 3.1 self-attention的表达形式
&emsp; 经典的self-attention可表示为：
$$
\mathcal{A}\rm(Q, K,V) = Softmax(\frac{QK^T}{\sqrt{d}})V
$$
&emsp; the $i$-th query's attention is defined as a kernel smoother in a probability:
$$
\mathcal{A}\rm(q_{\mathcal{i}},K,V) = \sum_\mathcal{j}(\frac{\mathcal{k}(q_\mathcal{i}, k_\mathcal{j})}{\sum_\mathcal{i}(\mathcal{k}(q_\mathcal{i}, k_\mathcal{j}))})V_\mathcal{i}
$$
&emsp; 其中，$\mathcal k \rm(q_{\mathcal i}, k_{\mathcal j}) = e^{\frac{q_{\mathcal i}k_{\mathcal j}^{T}}{\sqrt{d}}}$

###3.2 *ProbSparse* Self-attention
&emsp; (1) 启发： The “sparsity” self-attention score forms a long tail distribution (see Appendix C for details), i.e., a few dot-product pairs contribute to the major attention, andothers generate trivia attention.
&emsp; (2) Define  the $i$-th query's sparsity measurement as:
$$
\mathcal M \rm(q_{\mathcal i}, K) = ln \sum_{\mathcal j=1}^{L} e^{\frac{q_{\mathcal i}k_{\mathcal j}^{T}}{\sqrt{d}}} - \frac{1}{L} \sum_{\mathcal j=1}^{L}\frac{q_{\mathcal i}k_{\mathcal j}^{T}}{\sqrt{d}}
$$

&emsp; (3) 为避免$ln$中的数趋于0，用$\mathcal M$的上确界$\mathcal{\bar M} $替代稀疏性的度量：
$$
\mathcal{\bar M} (\rm q_{\mathcal i}, K) = \max_{\mathcal j}\frac{q_{\mathcal i}k_{\mathcal j}^{T}}{\sqrt{d}} - \frac{1}{L} \sum_{\mathcal j=1}^{L}\frac{q_{\mathcal i}k_{\mathcal j}^{T}}{\sqrt{d}}
$$

&emsp; (4) 随机选取$U = LlnL$个dot-product pairs，计算得到$\mathcal{\bar M}$；选取Top-$u$组成$\bar Q$，得到*ProbSparse* Self-attention 的计算公式：
$$
\mathcal{A}\rm(Q, K,V) = Softmax(\frac{\bar QK^T}{\sqrt{d}})V
$$
&emsp; &emsp; 其中$\bar Q$是稀疏矩阵, $u = c \cdot lnL$。

### 3.3 Model
&emsp; Informer 遵循transformer模型的Encoder-Decoder结构，可以一次性输出预测长序列的全部结果。  

<img src="./figures/Informer_01.png" style="zoom:50%;" align="center">;

#### 3.3.1 Encoder
<img src="./figures/Informer_02.png" style="zoom:60%;" align="center">   

&emsp; (1) 使用DataLoader读取数据(batch_x, batch_y, batch_x_mark, batch_y_mark)，其中batch_x 和batch_y $\in \mathcal R^{B, L, c\_in}$，batch_x_mark和batch_y_mark $\in \mathcal R^{B, L, c\_timedim}$。batch_x, batch_y中均包含Feature和Label，只是batch_y是未来数据，batch_x_mark和batch_y_mark是基于时间衍生的时间特征（如year、month、day等）。 

&emsp; (2) 分别将batch_x和batch_x_mark进行Embedding，维度变为[B, L, d_model]，并且对序列书记的位置编码，将三者相加。源码如下：
```python
class DataEmbedding(nn.Module):
    def __init__(self,
                 c_in,
                 d_model,
                 embed_type='fixed',
                 freq='h',
                 dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, embed_type=embed_type,
            freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(
            x) + self.temporal_embedding(x_mark)

        return self.dropout(x)
```
&emsp; (3) Encoder中包含ProbAttention、EncoderLayer、ConvLayer等几个模块，是这几个模块的堆叠，接下来分别介绍。

#### 3.3.1.1 ProbAttention
&emsp; ProbAttention是对FullAttention的改进，主要思想是：矩阵Q中的每个向量只和矩阵K中的$factor*lnL\_K$个向量进行相似度计算，取最大值，再选择top_u个相似度最高的query，记录这些query在Q中的index。最后在计算score时只用这些query计算，矩阵K仍用所有向量。部分源码如下：
```python
M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
M_top = M.topk(n_top, sorted=False)[1] # M_top: [B, H, n_top]


# use the reduced Q to calculate Q_K
Q_reduce = Q[torch.arange(B)[:, None, None],
                torch.arange(H)[None, :, None],
                M_top, :]  # factor*ln(L_q)
# Q_reduce: [B, H, n_top, E]
Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
# Q_K: [B, H, n_top, L_K]
```
#### 3.3.1.2 EncoderLayer
&emsp; EncoderLayer包含attention、LayerNorm、Dropout等操作
```python
def forward(self, x, attn_mask=None):
    # x [B, L, D]
    # x = x + self.dropout(self.attention(
    #     x, x, x,
    #     attn_mask = attn_mask
    # ))
    new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
    x = x + self.dropout(new_x)

    y = x = self.norm1(x)
    y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1)))) # [B, 4D, L]
    y = self.dropout(self.conv2(y).transpose(-1, 1)) # [B, 4D, L]

    return self.norm2(x + y), attn
```

#### 3.3.1.3 ConvLayer
&emsp; ConvLayer利用maxPool对序列进行降维
```python
def forward(self, x):
    x = self.downConv(x.permute(0, 2, 1))
    x = self.norm(x)
    x = self.activation(x)
    x = self.maxPool(x)
    x = x.transpose(1, 2)
    return x
```

### 3.3.2 Decoder
&emsp; Decoder包含ProbAttention和FullAttention，并且FullAttention中的K和V来自Encoder的输出。
```python
def forward(self, x, cross, x_mask=None, cross_mask=None):
    x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
    x = self.norm1(x)

    x = x + self.dropout(
        self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])

    y = x = self.norm2(x)
    y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
    y = self.dropout(self.conv2(y).transpose(-1, 1))

    return self.norm3(x + y)
```
### 3.3.3 Model structure
&emsp; 整个Informer网络包含Encoder、Decoder和输出层，源码如下：
```python
def forward(self,
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            enc_self_mask=None,
            dec_self_mask=None,
            dec_enc_mask=None):
    enc_out = self.enc_embedding(x_enc, x_mark_enc)
    enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

    dec_out = self.dec_embedding(x_dec, x_mark_dec)
    dec_out = self.decoder(dec_out,
                            enc_out,
                            x_mask=dec_self_mask,
                            cross_mask=dec_enc_mask)
    dec_out = self.projection(dec_out)

    # dec_out = self.end_conv1(dec_out)
    # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
    if self.output_attention:
        return dec_out[:, -self.pred_len:, :], attns
    else:
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
```
