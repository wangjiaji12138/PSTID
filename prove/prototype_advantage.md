# 当前原型分配相对 Linear–Softmax–Linear 的优势：条件、定理与证明

## 1. 研究对象及结论边界

本文参考 [TOPROVE.md](../TOPROVE.md)，但以当前源码为准：

- [空间分支](../models/PROPHET/spatial.py)：余弦分配与键值共享，对比 Linear–Softmax–Linear。
- [时间分支](../models/PROPHET/temporal.py)：GELU MLP 分配，对比 Linear–Softmax–Linear。
- [融合模块](../models/PROPHET/prototype.py) 和 [完整模型](../models/PROPHET/model.py)：两种配置都保留 Query 残差、输出投影和主干。
- [训练入口](../training/runner.py)：实际使用 MAE 训练。下面涉及 MSE 的结论不自动成为 MAE 的结论；适用 MAE 的定理会单独说明。

分析对象为 `proto_ablation="linear_softmax_linear"`。早期单 Linear、删除整个分支、硬分配和无旁路结构的结果不能混入此对照。

**不能无条件证明当前模型必胜。** 例如令两种模型的最终回归权重和偏置都为零，则它们都输出零；在零目标分布上两者风险相同。更强的“所有数据上最优风险严格更低”也被常数目标反例排除。本文件证明结构性质、条件风险优势和明确构造上的严格优势，不把实验观察直接提升为普遍定理。

## 2. 与代码逐项对应的数学表示

统一使用列向量。令 `d=proto_emb_dim`，空间与时间的原型数量分别为 `K_s,K_t`；下文在单个分支中简写为 `K`。用 `s(l)=softmax(l)` 表示沿分配维度的 Softmax。

### 2.1 空间分支

空间输入为图嵌入或节点嵌入，与可选历史均值的拼接。记为 `x_s`，其 Query 为

\[
q=C_sx_s+c_s\in\mathbb R^d.
\]

原型矩阵采用列存储记法，\(P=[p_1,\ldots,p_K]\in\mathbb R^{d\times K}\)，与源码中的 `prototypes` 互为转置。令 \(\bar q=q/\|q\|\)、\(\bar P=[p_k/\|p_k\|]_k\)。在非零且未触发数值截断时，原型增强为

\[
z_{s,P}(q)=P\,s(\bar P^T\bar q/\tau_s). \tag{1}
\]

同一个 `P` 同时决定分配用的键和输出用的值。空间温度是固定超参数。

当前对照为

\[
z_{s,L}(q)=B_s\,s(A_sq+a_s)+b_s,\quad
A_s\in\mathbb R^{K\times d},\ B_s\in\mathbb R^{d\times K}. \tag{2}
\]

空间对照的输入是已投影的 `q`，不是直接使用原始 `x_s`。两次 Linear 都带偏置。

### 2.2 时间分支

令 \(x_t\in\mathbb R^m\) 为最后时刻的时间身份嵌入及可选历史嵌入的拼接。当前分配器隐藏宽度为 `2m`：

\[
z_{t,P}(x)=P_t\,s\left(\frac{W_2g(W_1x+c_1)+c_2}{\tau_t}\right),
\quad W_1\in\mathbb R^{2m\times m},\ W_2\in\mathbb R^{K\times2m}, \tag{3}
\]

其中 `g` 为逐坐标 GELU。时间温度是可学习标量；源码没有强制其为正或远离零。需要正温度的构造均明确取 \(\tau_t>0\)，不声称训练自动满足该条件。

对照为

\[
z_{t,L}(x)=B_t\,s(A_tx+a_t)+b_t. \tag{4}
\]

两种配置都有独立且结构相同的时间 Query MLP \(q_t(x)\)。两种配置的参数在分别训练后不要求相等。

### 2.3 完整预测包含旁路

忽略广播和张量排列，用 `D` 表示后续编码及回归，`H` 表示主干拼接嵌入，则

\[
\widehat y=D\bigl(H+W_o(z_s+z_t+q_s+q_t)+b_o\bigr). \tag{5}
\]

因此 `z` 的性质不能未经证明转移给 \(\widehat y\)。例如 \(W_o=0\) 时，原型分配的任何差异都不影响预测。

## 3. 共同性质：当前对照也具有“原型凸组合”结构

### 命题 1：输出偏置可被吸收到字典中

对任意 \(\alpha=s(Ax+a)\)，有 \(\mathbf1^T\alpha=1\)，故

\[
B\alpha+b=(B+b\mathbf1^T)\alpha. \tag{6}
\]

因此式 (2)、(4) 都是对 \(K\) 个可学习向量的凸组合。若 \(\widetilde B=B+b\mathbf1^T\)，则输出属于 \(\operatorname{conv}(\widetilde B)\)。这不是对 `x` 的普通仿射映射。

对固定参数，两种结构的输出仿射包维数均不超过

\[
r\le\min(d,K-1). \tag{7}
\]

**证明。** 对任意两组分配 \(\alpha,\beta\)，有 \(\mathbf1^T(\alpha-\beta)=0\)。因此输出差属于 \(B\{v:\mathbf1^Tv=0\}\)，其维数至多 `K−1` 且至多 `d`。原型分支同理。有限 logits 给出严格正分配，但原型存在冗余时不保证分配可由输出唯一恢复。证毕。

**含义。** “原型版用 K 维结构，而对照要在 d 维完全自由估计”不符合当前对照。两者都有 `K` 个共享输出向量，且完整表示还包含式 (5) 中不受凸包限制的旁路。

## 4. 空间分支：可证明的是尺度不变性及条件去干扰优势

### 定理 2：对 Query 的正比例缩放不变

固定 `P` 和 \(\tau_s>0\)，对于 \(q\ne0\)、\(r>0\)，理想归一化下

\[
z_{s,P}(rq)=z_{s,P}(q). \tag{8}
\]

**证明。** \(\overline{rq}=rq/(r\|q\|)=\bar q\)，代入式 (1)。证毕。

实现用 \(N_\epsilon(q)=q/\max(\|q\|,\epsilon)\)，默认 `eps=1e-12`。因此式 (8) 对实际实现要求 \(\|q\|\ge\epsilon\) 且 \(r\|q\|\ge\epsilon\)；不能声称在数值截断区间也严格成立。该定义已核对本地接口及 [PyTorch normalize 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.normalize.html)。

还需区分 `q` 与原始输入：若 \(q=C_sx_s+c_s\)，通常 \(q(rx_s)\ne rq(x_s)\)。所以本定理只消除 **Query 空间中的径向变化**，不能直接宣称原始历史数据的乘性噪声全部被消除。

### 命题 3：径向导数为零，并有局部敏感度界

令 \(\alpha=s(\bar P^T\bar q/\tau_s)\)，\(S(\alpha)=\operatorname{diag}(\alpha)-\alpha\alpha^T\)。在 \(\|q\|>\epsilon\) 时

\[
J_P(q)=\frac{1}{\tau_s\|q\|}
P S(\alpha)\bar P^T(I-\bar q\bar q^T),\qquad J_P(q)q=0. \tag{9}
\]

此外

\[
\|J_P(q)\|_2\le\frac{\|P\|_2\|\bar P\|_2}{2\tau_s\|q\|}. \tag{10}
\]

**证明。** 直接求导得 \(D(q/\|q\|)=(I-\bar q\bar q^T)/\|q\|\)，Softmax 的 Jacobian 为 `S`，链式法则给出式 (9)。由于 \((I-\bar q\bar q^T)q=0\)，径向导数为零。`S` 为对称半正定矩阵，其第 `i` 行绝对值和为 \(2\alpha_i(1-\alpha_i)\le1/2\)，从而 \(\|S\|_2\le1/2\)。结合矩阵范数次乘性得到式 (10)。证毕。

对照的 Jacobian 为

\[
J_L(q)=B_s S(s(A_sq+a_s))A_s, \tag{11}
\]

一般不满足 \(J_L(q)q=0\)。式 (10) 不是“原型版对所有方向都更稳定”的证明：其上界依赖原型范数、温度及 Query 范数，且没有给出与式 (11) 的全局大小关系。

### 定理 4：径向干扰下的精确风险分解

设随机 Query \(Q=RU\)，其中 \(\|U\|=1\)、\(R>0\)。不要求 `R` 与 `U` 独立，但要求相关随机变量二阶可积。目标满足

\[
Y=m(U)+\varepsilon,\qquad \mathbb E[\varepsilon\mid U,R]=0.
\]

对任意确定性预测分支 `f`，定义 \(\mu_f(U)=\mathbb E[f(RU)\mid U]\)，则平方风险为

\[
\mathcal R_2(f)=\mathbb E\|\varepsilon\|^2
+\mathbb E\|\mu_f(U)-m(U)\|^2
+\mathbb E\|f(RU)-\mu_f(U)\|^2. \tag{12}
\]

若一个原型分支 \(f_P\) 满足 \(f_P(U)=m(U)\)，且尺度变化均处于定理 2 的有效区间，则

\[
\mathcal R_2(f_L)-\mathcal R_2(f_P)
=\mathbb E\|\mu_{f_L}(U)-m(U)\|^2
+\mathbb E\|f_L(RU)-\mu_{f_L}(U)\|^2\ge0. \tag{13}
\]

右侧任一项为正时严格大于零。

**证明。** 先利用条件零均值去掉 \(f(Q)-m(U)\) 与噪声的交叉项，再把 \(f(RU)-m(U)\) 写成 \(f(RU)-\mu_f(U)+\mu_f(U)-m(U)\)。第一项给定 `U` 的条件均值为零，第二个交叉项也消失。定理 2 使 \(f_P(RU)=m(U)\)，故原型风险只剩噪声项。证毕。

**条件不可省略。** 原型分支必须足够准确地表示 `m`；若存在偏差，比较式应额外减去其偏差平方。若幅值携带真实预测信息，丢弃径向信息可能反而有害。例如同一方向上令 \(Y=R\)，任何尺度不变的单独分支都无法精确拟合两个不同幅值的目标。式 (12) 是 MSE 分解，不能照搬为 MAE 分解。

### 定理 5：空间分支上严格优势的四点构造

取 `d=K=2`、\(P=I_2\)、\(\tau_s>0\)。Query 在

\[
\{e_1,2e_1,e_2,2e_2\}
\]

上均匀分布，目标为原型增强的第一坐标：

\[
Y(e_1)=Y(2e_1)=a,\quad
Y(e_2)=Y(2e_2)=1-a,\quad a=\operatorname{sigmoid}(1/\tau_s)>1/2. \tag{14}
\]

则原型分支风险为零。任意**有限参数**的 `K=2` Linear–Softmax–Linear 分支的该坐标，其 MSE 和 MAE 均严格为正。

**证明。** 任意这种对照的标量输出都可写成

\[
f_L(q)=c+v\,\operatorname{sigmoid}(w^Tq+t).
\]

若在四个点上精确拟合，由两种目标不同知 \(v\ne0\)。在 `e1,2e1` 上输出相等，结合 sigmoid 严格单调性得 \(w_1=2w_1\)，故 \(w_1=0\)。同理 \(w_2=0\)。此时所有点输出相等，与 \(a\ne1-a\) 矛盾。有限支持每个点都有正概率，至少一处非零误差意味着 MSE 和 MAE 都严格为正。证毕。

这里不能交换“对每个参数风险为正”和“风险下确界为正”。事实上令 \(w=(M,-M)\)、\(t=0\)、\(c=1-a\)、\(v=2a-1\)，当 \(M\to\infty\) 时，对照风险趋于零。因此无界参数类的风险下确界仍为零。

若将全部有效参数 \((w,t,c,v)\) 限制在任意固定的非空紧集内，则风险连续、最小值能够取到；结合不存在零风险参数，得到严格正的最小风险 \(\delta>0\)。这给出了**有界参数假设下的严格函数类风险差距**，但有限数值的 `weight_decay` 不自动等于这样的硬约束。

### 命题 6：归一化输入后，空间原型是受约束的 LSL 特例

若对照接收的是 \(\bar q\) 而非 `q`，取

\[
A_s=\bar P^T/\tau_s,\quad a_s=0,\quad B_s=P,\quad b_s=0,
\]

即可精确实现式 (1)。因此空间原型的特殊性是 **Query 归一化及键值绑定**。这可能构成适合数据的约束，但不意味着该分支比所有自由键值分配器都更有表达能力。当前对照没有这一步归一化，所以两者不能直接按此构造判定等价。

## 5. 时间分支：严格的函数类包含与交互表达优势

GELU 为 \(g(t)=t\Phi(t)\)，其中 \(\Phi\) 为标准正态分布函数；本地代码为 `nn.GELU(approximate='none')`。该定义见 [GELU 原始论文](https://arxiv.org/abs/1606.08415)。以下恒等式为直接推导，不依赖通用逼近定理。

### 引理 7：GELU 可以精确重建线性映射

由 \(\Phi(-t)=1-\Phi(t)\)，有

\[
g(t)-g(-t)=t. \tag{15}
\]

故隐藏宽度 `2m` 足以精确构造任意从 \(\mathbb R^m\) 到 \(\mathbb R^K\) 的仿射映射：先计算 \([g(x);g(-x)]\)，再作差并进行线性组合。

### 定理 8：时间原型增强的函数类包含当前对照

固定任意 `m,d,K`，不施加额外参数范数限制。设 \(\mathcal H_{t,P}\)、\(\mathcal H_{t,L}\) 分别是式 (3)、(4) 的增强特征函数类，则

\[
\mathcal H_{t,L}\subseteq\mathcal H_{t,P}. \tag{16}
\]

**证明。** 给定任意 \(f_L(x)=B s(Ax+a)+b\)，任取 \(\tau_t>0\)，设置

\[
W_1=\begin{bmatrix}I_m\\-I_m\end{bmatrix},\quad c_1=0,
\quad W_2=\tau_t[A,-A],\quad c_2=\tau_t a,
\quad P_t=B+b\mathbf1^T.
\]

由式 (15)，MLP 输出为 \(\tau_t(Ax+a)\)，除以温度后与对照 logits 相同。再由式 (6) 吸收输出偏置，得到 \(f_P(x)=f_L(x)\) 对所有 `x` 成立。证毕。

因此，对任何给定数据分布和任何定义良好的损失，两个**增强分支的最优风险**满足

\[
\inf_{f\in\mathcal H_{t,P}}\mathcal R(f)
\le\inf_{f\in\mathcal H_{t,L}}\mathcal R(f). \tag{17}
\]

式 (17) 不涉及优化器能否找到这些参数，也不保证严格小于。若外部网络和其他分支完全固定且只比较时间分支，包含关系可以经相同外部映射组合后保留。当前 `use_proto` 同时改变空间分支，因此不能仅由式 (16) 得到完整模型的包含关系。

### 定理 9：相同 K 下可存在严格的 MSE 和 MAE 差距

取 `m=2,d=1,K=2`，\(X=(X_1,X_2)\) 在 \(\{-1,+1\}^2\) 上均匀分布。令

\[
Y=\mathbf1\{X_1=X_2\}. \tag{18}
\]

这表示一个纯交互任务：同号与异号需要不同编码。对于单独的时间增强分支，

\[
\inf_{f\in\mathcal H_{t,P}}\mathbb E(f(X)-Y)^2=0,
\qquad
\inf_{f\in\mathcal H_{t,L}}\mathbb E(f(X)-Y)^2\ge\frac1{16}, \tag{19}
\]

以及

\[
\inf_{f\in\mathcal H_{t,P}}\mathbb E|f(X)-Y|=0,
\qquad
\inf_{f\in\mathcal H_{t,L}}\mathbb E|f(X)-Y|\ge\frac18. \tag{20}
\]

**原型分支的构造。** 令 \(G(t)=g(t)+g(-t)\)，\(C=G(2)=2(2\Phi(2)-1)>0\)。设置

\[
h(x)=\frac{G(x_1+x_2)-G(x_1-x_2)}C.
\]

在同号两点有 `h=1`，异号两点有 `h=−1`。计算 `h` 恰好需要四个 GELU 单元，其第一层权重行分别为

\[
(1,1),\ (-1,-1),\ (1,-1),\ (-1,1),
\]

与代码中的 `2m=4` 一致。取 \(\tau_t=1\)，两维 logits 为 \(((\log3)h(x),0)\)，则第一分配权重在同号点为 `3/4`、异号点为 `1/4`。取两个标量原型 \(p_1=3/2,p_2=-1/2\)，输出为

\[
f_P(x)=\tfrac32\alpha_1-\tfrac12(1-\alpha_1)=2\alpha_1-\tfrac12=Y.
\]

全部参数有限，故两种损失都精确为零。

**对照的下界。** 任意 `K=2` 对照都是

\[
f_L(x)=c+v\,\operatorname{sigmoid}(w_1x_1+w_2x_2+t).
\]

若四个点绝对误差都小于 `1/2`，则同号两点输出均大于 `1/2`，异号两点均小于 `1/2`。`v=0` 时不可能；`v≠0` 时，函数关于仿射得分严格单调，因此存在一个阈值把两组得分严格分开。

但同号两点得分平均为

\[
\tfrac12[(w_1+w_2+t)+(-w_1-w_2+t)]=t,
\]

异号两点得分平均也为

\[
\tfrac12[(w_1-w_2+t)+(-w_1+w_2+t)]=t.
\]

两组不可能严格位于同一阈值两侧，矛盾。因此至少一点绝对误差不小于 `1/2`；该点概率为 `1/4`，分别得到 MSE 下界 \((1/4)(1/2)^2=1/16\) 和 MAE 下界 \((1/4)(1/2)=1/8\)。证毕。

**适用范围。** 此例证明的是 `K=2` 的严格包含与分支风险差距。它不直接证明默认 `K_t=8` 也有相同下界；增加 `K` 后对照可能表示这一四点任务。它也不适用于允许自由训练 Query MLP 和主干的完整网络，因为这些旁路也可以学习交互。其意义是证明当前时间分配器确实具有线性 logits 无法一概替代的非线性交互能力。

## 6. 参数化、容量与训练后风险：不能跳过的环节

### 6.1 参数计数不支持“两条原型分支都更小”

以下只计入被替换部分，排除两边共有的 Query、融合和主干。

| 分支 | 原型版 | Linear–Softmax–Linear |
| --- | ---: | ---: |
| 空间 | \(Kd\) | \(2Kd+K+d\) |
| 时间 | \(2m^2+2m+2mK+K+Kd+1\) | \(mK+K+Kd+d\) |

时间原型版的 `+1` 来自可学习温度；所有 Linear 的 bias 均已计入。取 `m=d=64,K_t=8` 时，时间部分分别为 `9865` 与 `1096`；空间 `d=64,K_s=16` 时分别为 `1024` 与 `2128`。

空间键值共享减少了自由参数，时间 MLP 则增加了表达容量。参数计数不是泛化误差定理，不能仅凭参数更少或更多推出测试指标排序。

### 定理 10：训练后优势的一个充分条件

令 \(\mathcal H_P,\mathcal H_L\) 为被比较的预测函数类，定义总体风险 \(\mathcal R\) 及经验风险 \(\widehat{\mathcal R}\)。假设在某事件上

\[
\sup_{f\in\mathcal H_P}|\mathcal R(f)-\widehat{\mathcal R}(f)|\le\epsilon_P,
\]

且训练得到的 \(\widehat f_P\) 满足经验优化误差界

\[
\widehat{\mathcal R}(\widehat f_P)
\le\inf_{f\in\mathcal H_P}\widehat{\mathcal R}(f)+\eta_P.
\]

若最优风险差距

\[
\Delta=\inf_{f\in\mathcal H_L}\mathcal R(f)
-\inf_{f\in\mathcal H_P}\mathcal R(f)
>2\epsilon_P+\eta_P, \tag{21}
\]

则在同一事件上，对于任意 \(\widehat f_L\in\mathcal H_L\)，都有

\[
\mathcal R(\widehat f_P)<\mathcal R(\widehat f_L).
\]

**证明。** 依次使用总体—经验偏差界、经验优化界，再次使用总体—经验偏差界：

\[
\mathcal R(\widehat f_P)
\le\widehat{\mathcal R}(\widehat f_P)+\epsilon_P
\le\inf_{f\in\mathcal H_P}\mathcal R(f)+2\epsilon_P+\eta_P
<\inf_{f\in\mathcal H_L}\mathcal R(f)
\le\mathcal R(\widehat f_L).
\]

若下确界不取到，可对任意逼近下确界的函数写此链，再取极限。证毕。

这给出了“表达优势大于统计及优化代价”的精确含义。它不是无条件保证：本项目尚未证明完整网络的 `Δ`、`ε_P` 或 `η_P` 满足式 (21)。训练集与测试集存在时空相关，不能不加说明地套用独立样本集中不等式。

本定理对 MAE 和 MSE 均可用，只要相应风险及统一偏差界成立。MAE 无界输出时不自动满足有界损失假设；训练后观测到较低的经验损失也不等于已知 `η_P` 很小。

## 7. 对 TOPROVE 低秩方差思路的严格修正版

以下是辅助统计模型，展示原草稿试图表达的偏差—方差机制。它不替代前面对当前代码的分析。

### 定理 11：已知仿射子空间的精确风险

考虑固定设计线性观测

\[
y=X\theta^*+\varepsilon,\quad X^TX=nI_d,
\quad \mathbb E\varepsilon=0,\quad \operatorname{Cov}(\varepsilon)=\sigma^2I_n.
\]

令 \(\widehat\theta=X^Ty/n\)，已知仿射空间为 \(\mathcal A=a+S\)，其中 `S` 是固定的 `r` 维子空间，正交投影矩阵为 \(\Pi_S\)。定义受约束估计

\[
\widehat\theta_{\mathcal A}=a+\Pi_S(\widehat\theta-a).
\]

则

\[
\mathbb E\|\widehat\theta-\theta^*\|^2=\frac{\sigma^2d}{n},\qquad
\mathbb E\|\widehat\theta_{\mathcal A}-\theta^*\|^2
=\operatorname{dist}(\theta^*,\mathcal A)^2+\frac{\sigma^2r}{n}. \tag{22}
\]

因此受约束估计严格更好当且仅当

\[
\operatorname{dist}(\theta^*,\mathcal A)^2<\frac{\sigma^2(d-r)}n. \tag{23}
\]

**证明。** 记 \(\xi=X^T\varepsilon/n\)，则 \(\mathbb E\xi=0\)、\(\operatorname{Cov}(\xi)=\sigma^2I_d/n\)。无约束风险为协方差迹。受约束误差为

\[
\Pi_S\xi-(I-\Pi_S)(\theta^*-a).
\]

两项正交，故平方范数直接相加，第一项期望为 \(\sigma^2\operatorname{tr}(\Pi_S)/n=\sigma^2r/n\)，第二项为距离平方。证毕。

### 推论 12：已知原型凸包中的投影估计

若 \(\mathcal C=\operatorname{conv}(P)\) 是固定的非空凸包，仿射包维数为 \(r\le K-1\)，且 \(\theta^*\in\mathcal C\)，令 \(\widehat\theta_{\mathcal C}\) 为 \(\widehat\theta\) 到该凸包的欧氏投影，则

\[
\mathbb E\|\widehat\theta_{\mathcal C}-\theta^*\|^2\le\frac{\sigma^2r}{n}. \tag{24}
\]

**证明。** 先投影到仿射包，记为 `u`，再投影到凸包得到 `p`，与直接投影等价，因为垂直于仿射包的距离是常数。最优性条件给出 \(\langle u-p,\theta^*-p\rangle\le0\)，故

\[
\|u-\theta^*\|^2
=\|u-p\|^2+\|p-\theta^*\|^2+2\langle u-p,p-\theta^*\rangle
\ge\|p-\theta^*\|^2.
\]

再由定理 11、\(\theta^*\in\mathcal A\)，得到式 (24)。证毕。

**为何不能直接用它解释当前实验。** 当前两边都有 `K` 个共享输出向量；字典由同一训练数据端到端学习，分配不是求解凸包投影，训练损失是 MAE，主干也不是上述固定设计线性模型。若字典或子空间由独立数据估计，可以条件化后分析子空间误差；若由同一批数据估计，则还需要处理选择依赖。不能忽略这些条件后直接宣称 `K/n` 优于 `d/n`。

## 8. 从机制证明到当前实验解释

现有严格结论支持两个候选机制：

1. 空间分支通过 Query 归一化消除径向变化，并通过键值共享限制分配结构；当方向承载目标、幅值主要是干扰时，这可能降低风险。定理 4、5 给出条件化的严格形式。
2. 时间分支的 GELU 分配器包含线性分配器，并能表达后者在某些相同 `K` 设置下不能表达的交互。定理 8、9 给出构造证明；是否有足够训练数据和优化能力实现收益，需要另外验证。

不能将两条机制写成“原型有凸包约束、对照没有”，也不能写成“原型一定稀疏、一定更少参数、一定更容易优化”。

若要将用户观察到的预测优势归因于这些机制，应进一步对当前版本做以下有针对性的实验，而不是把旧版本诊断当作本版本证明：

| 待验证解释 | 可区分机制的实验 | 需要观察的量 |
| --- | --- | --- |
| 空间归一化有帮助 | 对照只增加 Query 归一化，其余尽量固定 | 同种子预测指标、Query 径向扰动下的分配与最终预测变化 |
| 键值共享有帮助 | 在都归一化的前提下比较共享键值与独立键值 | 泛化差距、原型/权重范数、训练与验证误差 |
| 时间非线性分配有帮助 | 只替换时间 allocator，固定空间选择；另做容量匹配 | 同版本多种子 MAE、RMSE、参数量与优化曲线 |
| 增强分支影响了最终预测 | 冻结同一检查点，对 `z_s`、`z_t` 分别干预 | 预测误差变化；同时记录分布偏移，不能把干预等同于从头训练消融 |

### 可用于论文的审慎表述

> 对于原型数量相同的 Linear–Softmax–Linear 对照，两种增强分支都具有共享字典和凸组合结构，因此其差异不能仅由输出低秩性解释。我们证明：空间余弦分配在归一化有效区间内具有 Query 尺度不变性；在径向变化为干扰且方向决定目标的明确条件下，这种不变性可以产生严格风险优势。时间分支的 GELU 分配器精确包含线性分配器，并在两原型交互任务上产生严格的 MAE 与 MSE 表达差距。这些结论为模型在适配数据分布上的优势提供机制依据，但不保证端到端训练在任意数据上必然优于对照。

## 9. 复核说明

本文解析证明在实数精确运算下成立；[verify_math.py](verify_math.py) 用 CPU float64 对实际模块做数值交叉检查。它不是自动定理证明器，也不会通过有限随机样本代替全称证明。源码指纹及结果见 [verification.json](verification.json)。

外部资料仅用于核对基础定义：GELU 的定义来自 [Hendrycks 与 Gimpel 的原始论文](https://arxiv.org/abs/1606.08415)，归一化截断来自 [PyTorch normalize 官方文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.normalize.html)。本文的风险构造、包含关系和误差界均在正文给出完整推导，没有将外部经验结果当作本项目的证明。
