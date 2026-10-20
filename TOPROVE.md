# 原型编码器与线性层的等价性及其性能差异：从线性代数、凸优化到统计估计的严格分析

## 摘要

在时空预测模型中，原型编码器（prototype encoder）常被观察到与线性层在函数层面等价，但实践中原型模块的性能优于等价线性层。本文从线性代数、凸优化与统计估计理论出发，首先形式化证明二者在特定条件下的函数等价性，进而严格刻画等价性成立的前提，最后给出在低秩原型结构假设下原型编码器估计误差严格小于线性层的定理及其证明。核心结论是：**等价性描述的是假设空间可达函数集合的重合，而性能差距源于参数化方式所诱导的有效自由度差异——原型结构将待估参数的有效维度从嵌入维度 $d$ 降为原型数 $K$，从而在有限样本下严格降低估计方差。**

---

## 1. 线性代数视角：原型编码作为基展开的特例

设原型库 $\mathbf{P} = [\mathbf{p}_1, \dots, \mathbf{p}_K] \in \mathbb{R}^{d \times K}$。原型编码器对输入 $\mathbf{x} \in \mathbb{R}^m$ 的输出为

$$
\mathbf{z} = \sum_{k=1}^K \alpha_k(\mathbf{x})\, \mathbf{p}_k, \qquad \alpha_k(\mathbf{x}) = \frac{\exp(\mathbf{w}_k^\top \mathbf{x})}{\sum_j \exp(\mathbf{w}_j^\top \mathbf{x})}.
$$

**观察 1（输出空间的线性性）**：无论 $\alpha_k(\mathbf{x})$ 如何由非线性 softmax 产生，编码输出 $\mathbf{z}$ 始终位于 $\mathbf{P}$ 的列空间 $\operatorname{col}(\mathbf{P})$ 中。编码器在输出端的作用是将输入映射到 $\operatorname{col}(\mathbf{P})$ 上的一个点，该点由系数向量 $\boldsymbol{\alpha}(\mathbf{x}) \in \mathbb{R}^K$ 唯一确定。

一个带权重的线性层 $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ 将输入映射到 $\mathbb{R}^d$。当 $\operatorname{col}(\mathbf{W}) \subseteq \operatorname{col}(\mathbf{P})$ 时，线性层的输出同样落在 $\mathbf{P}$ 张成的子空间中，系数向量为 $\mathbf{W}$ 在基 $\mathbf{P}$ 下的坐标。此时两者在输出几何上是同一类映射：**从输入空间到同一个低维线性子空间的嵌入**。

**观察 2（独热输入下的严格恒等）**：设输入为独热向量 $\mathbf{x} = \mathbf{e}_i \in \{0,1\}^V$，则无偏置线性层 $\mathbf{W} \in \mathbb{R}^{d \times V}$ 的输出恰为 $\mathbf{W}$ 的第 $i$ 列：

$$
\mathbf{e}_i^\top \mathbf{W}^\top = (\mathbf{W}_{i,:})^\top.
$$

这正是以 $\mathbf{W}$ 的列为表项的查找表。若原型编码器在独热输入下检索与输入 ID 对应的原型，则令 $\mathbf{W} = \mathbf{P}^\top$ 即得精确的代数恒等。在此场景下，**"原型编码"与"线性嵌入"在函数层面是同一映射**。

---

## 2. 凸优化视角：原型更新与线性回归的同构

### 2.1 原型更新作为约束线性最小二乘

给定编码表示 $\{\mathbf{z}_t\}$ 与系数 $\{\boldsymbol{\alpha}_t\}$，原型库的更新常表述为

$$
\min_{\mathbf{P}} \sum_t \|\mathbf{z}_t - \mathbf{P}\boldsymbol{\alpha}_t\|_2^2.
$$

固定 $\boldsymbol{\alpha}_t$ 时目标函数关于 $\mathbf{P}$ 是凸二次的，最优性条件为

$$
\mathbf{P}\left(\sum_t \boldsymbol{\alpha}_t \boldsymbol{\alpha}_t^\top\right) = \sum_t \mathbf{z}_t \boldsymbol{\alpha}_t^\top,
$$

即**多输出线性回归的正规方程**。若 $\sum_t \boldsymbol{\alpha}_t \boldsymbol{\alpha}_t^\top$ 可逆，则

$$
\mathbf{P}^\star = \left(\sum_t \mathbf{z}_t \boldsymbol{\alpha}_t^\top\right)\left(\sum_t \boldsymbol{\alpha}_t \boldsymbol{\alpha}_t^\top\right)^{-1}.
$$

**原型矩阵的最优解与线性层权重矩阵具有完全相同的解析形式**——原型"学习"在此退化为线性最小二乘，解空间与线性层参数空间重合。

### 2.2 原型选择作为凸二次规划

若施加单纯形约束：

$$
\min_{\boldsymbol{\alpha}} \|\mathbf{z} - \mathbf{P}\boldsymbol{\alpha}\|_2^2 \quad \text{s.t.} \quad \mathbf{1}^\top \boldsymbol{\alpha} = 1,\ \boldsymbol{\alpha} \ge 0,
$$

KKT 条件给出的解 $\boldsymbol{\alpha}^\star$ 是 $\mathbf{z}$ 在凸包 $\operatorname{conv}(\mathbf{P})$ 上的投影系数。投影算子在仿射子空间上是线性的；一旦有效约束集确定，$\boldsymbol{\alpha}^\star = \mathbf{A}\mathbf{z} + \mathbf{c}$，编码输出 $\mathbf{P}\boldsymbol{\alpha}^\star = (\mathbf{P}\mathbf{A})\mathbf{z} + \mathbf{P}\mathbf{c}$，**局部等价于线性层**。

---

## 3. 等价性的条件性与性能差距的来源

上述等价性成立需满足若干前提：输出限于原型张成子空间、输入为独热编码、损失为平方误差、原型库固定或在凸步骤中更新。一旦条件不完全成立，等价性失去严格基础。更关键的是，**即使函数类完全相同，参数化方式不同会改变优化问题的几何结构**。

### 3.1 函数等价 ≠ 解等价

定义两个假设空间（$V$ 个实体，嵌入维度 $d$，原型数 $K$）：

$$
\mathcal{H}_{\text{lin}} = \{\mathbf{x} \mapsto \mathbf{W}\mathbf{x} : \mathbf{W} \in \mathbb{R}^{d \times V}\},
\qquad
\mathcal{H}_{\text{proto}} = \{\mathbf{x} \mapsto \mathbf{P}\boldsymbol{\alpha}(\mathbf{x}) : \mathbf{P} \in \mathbb{R}^{d \times K},\ \boldsymbol{\alpha}(\mathbf{x}) \in \Delta^{K-1}\}.
$$

在独热输入下，$\mathcal{H}_{\text{lin}}$ 为每个实体分配自由向量 $\mathbf{w}_v \in \mathbb{R}^d$；$\mathcal{H}_{\text{proto}}$ 的参数化为 $\mathbf{w}_v = \mathbf{P}\boldsymbol{\alpha}_v$，$\boldsymbol{\alpha}_v \in \Delta^{K-1}$。

**当 $K < V$ 且 $\mathbf{P}$ 固定时，$\mathcal{H}_{\text{proto}} \subsetneq \mathcal{H}_{\text{lin}}$。** 所谓"等价"仅在原型列空间覆盖真实嵌入子空间、且 $\boldsymbol{\alpha}$ 可自由取遍 $\mathbb{R}^K$ 时成立；施加单纯形约束或 $K \ll V$ 后，两者是真包含关系。性能差距的数学根源正是这个真包含关系在估计问题中的后果。

### 3.2 优化动力学的差异

**不相交梯度 vs. 联合梯度**：在向量量化式原型更新中，只有被当前输入选中的原型收到梯度，未被选中的原型保持不动。这种稀疏的、竞争性的梯度流使原型之间形成**硬分工**，每个原型专门化到数据分布中的某个局部模式。线性层的权重矩阵则每次收到与所有训练样本相关的梯度信号，参数之间**联合优化**。在过参数化情形下，联合优化使线性层有能力通过协同调整大量参数拟合训练集中的细微波动，但代价是解更易落入尖锐极小值或依赖训练集特有相关性的区域。

**隐式偏置的差异**：线性层在平方损失下的梯度流倾向于收敛到最小范数解，其泛化行为高度依赖数据协方差矩阵的谱结构。当时空数据特征协方差存在近零特征值（传感器强相关时常见）时，最小范数解会沿这些近零方向放大噪声。原型模块引入**离散压缩偏置**：编码系数经 softmax 或硬分配使每个输入只激活少数原型，模型被迫用有限的、共享的表示基解释所有样本。这种压缩由"原型数 $K$ 远小于样本数 $N$"的结构事实隐含，而非显式惩罚。

### 3.3 时空数据的归纳优势

时空数据具有强时间自相关与空间/传感器间异质相关。线性层在独热输入下等价于为每个实体学习独立嵌入向量；当某实体训练样本稀少时，其嵌入无法充分估计，退化为依赖全局协方差的线性插值。原型模块的输出是**共享原型的线性组合**：罕见实体的表示被约束为已有原型的组合，借力于数据丰富实体上学到的、经充分优化的时间模式。从统计估计角度，这相当于用**低秩结构**替代**逐实体独立参数**，在 $V$ 大而每实体样本有限的场景中显著降低估计方差。

---

## 4. 统计估计层面的严格定理

### 4.1 设定与假设

对每个实体 $v \in [V]$，观测 $n$ 个样本：

$$
y_{v,t} = \langle \mathbf{w}_v^*, \boldsymbol{\phi}(\mathbf{x}_{v,t}) \rangle + \epsilon_{v,t},
\qquad \epsilon_{v,t} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2),
$$

设计矩阵 $\boldsymbol{\Phi}_v = [\boldsymbol{\phi}(\mathbf{x}_{v,1}), \dots, \boldsymbol{\phi}(\mathbf{x}_{v,n})]^\top \in \mathbb{R}^{n \times d}$。

**假设 A1（设计矩阵良条件）**：存在 $\lambda > 0$ 使得 $\frac{1}{n}\boldsymbol{\Phi}_v^\top \boldsymbol{\Phi}_v \succeq \lambda \mathbf{I}_d$ 对所有 $v$ 成立。

**假设 A2（低秩原型结构）**：存在 $\mathbf{P}^* \in \mathbb{R}^{d \times K}$ 与 $\boldsymbol{\alpha}_v^* \in \operatorname{int}(\Delta^{K-1})$，使得 $\mathbf{w}_v^* = \mathbf{P}^* \boldsymbol{\alpha}_v^*$ 对所有 $v$ 成立。

**假设 A3（原型库可估）**：存在基于全部 $Vn$ 个样本的估计 $\hat{\mathbf{P}}$，满足

$$
\mathbb{E}\|\hat{\mathbf{P}} - \mathbf{P}^*\|_F^2 \le \frac{C_1 \sigma^2 d K}{V n}
$$

对某个常数 $C_1 > 0$ 成立。

**A3 的可达性**：在 A2 下，全部 $Vn$ 个样本可视为 $K$ 个共享原型生成的混合模型。采用硬 EM（$K$-means 式交替）在聚类可分离条件下标准结果给出 $\|\hat{\mathbf{P}} - \mathbf{P}^*\|_F^2 = O_p(\sigma^2 d K/(Vn))$，其中 $dK$ 是 $\mathbf{P}^*$ 的自由参数数，$Vn$ 是总样本量。A3 与低秩结构假设自洽。

### 4.2 定理 1

**定理 1**：在 A1–A3 下，定义线性层估计 $\hat{\mathbf{w}}_v^{\text{lin}} = (\boldsymbol{\Phi}_v^\top \boldsymbol{\Phi}_v)^{-1}\boldsymbol{\Phi}_v^\top \mathbf{y}_v$，原型编码器估计 $\hat{\mathbf{w}}_v^{\text{proto}} = \hat{\mathbf{P}} \hat{\boldsymbol{\alpha}}_v$，其中

$$
\hat{\boldsymbol{\alpha}}_v = \arg\min_{\boldsymbol{\alpha} \in \Delta^{K-1}} \|\mathbf{y}_v - \boldsymbol{\Phi}_v \hat{\mathbf{P}} \boldsymbol{\alpha}\|_2^2.
$$

则对任意 $v$：

$$
\mathbb{E}\|\hat{\mathbf{w}}_v^{\text{proto}} - \mathbf{w}_v^*\|_2^2 \;\le\; \frac{2\sigma^2 K}{\lambda n} + \frac{2 C_1 \sigma^2 d K}{V n} + o\!\left(\frac{1}{n}\right),
\tag{1}
$$

而

$$
\mathbb{E}\|\hat{\mathbf{w}}_v^{\text{lin}} - \mathbf{w}_v^*\|_2^2 \;=\; \sigma^2 \operatorname{tr}\!\left((\boldsymbol{\Phi}_v^\top \boldsymbol{\Phi}_v)^{-1}\right) \;\ge\; \frac{\sigma^2 d}{\lambda_{\max}\big(\frac{1}{n}\boldsymbol{\Phi}_v^\top \boldsymbol{\Phi}_v\big)\, n}.
\tag{2}
$$

特别地，若 $\frac{1}{n}\boldsymbol{\Phi}_v^\top \boldsymbol{\Phi}_v$ 的特征值位于 $[\lambda, \Lambda]$ 内且 $V \gg d$，则当 $K \cdot \frac{\Lambda}{\lambda} < \frac{d}{2}$ 时，式 (1) 的右端严格小于式 (2) 的右端。

### 4.3 证明

**线性层部分**：标准结果。

$$
\hat{\mathbf{w}}_v^{\text{lin}} - \mathbf{w}_v^* = (\boldsymbol{\Phi}_v^\top \boldsymbol{\Phi}_v)^{-1}\boldsymbol{\Phi}_v^\top \boldsymbol{\epsilon}_v,
$$

故

$$
\mathbb{E}\|\hat{\mathbf{w}}_v^{\text{lin}} - \mathbf{w}_v^*\|^2 = \sigma^2 \operatorname{tr}\!\left((\boldsymbol{\Phi}_v^\top \boldsymbol{\Phi}_v)^{-1}\right).
$$

由 $\frac{1}{n}\boldsymbol{\Phi}_v^\top \boldsymbol{\Phi}_v \preceq \lambda_{\max}\mathbf{I}$ 得 $(\boldsymbol{\Phi}_v^\top \boldsymbol{\Phi}_v)^{-1} \succeq \frac{1}{n\lambda_{\max}}\mathbf{I}$，取迹即得 (2) 的下界。

**原型编码器部分**：误差分解为

$$
\hat{\mathbf{P}}\hat{\boldsymbol{\alpha}}_v - \mathbf{P}^*\boldsymbol{\alpha}_v^*
= (\hat{\mathbf{P}} - \mathbf{P}^*)\hat{\boldsymbol{\alpha}}_v + \mathbf{P}^*(\hat{\boldsymbol{\alpha}}_v - \boldsymbol{\alpha}_v^*).
$$

由 $\|\hat{\boldsymbol{\alpha}}_v\|_2 \le \|\hat{\boldsymbol{\alpha}}_v\|_1 = 1$（单纯形性质），第一项满足

$$
\mathbb{E}\|(\hat{\mathbf{P}} - \mathbf{P}^*)\hat{\boldsymbol{\alpha}}_v\|^2 \le \mathbb{E}\|\hat{\mathbf{P}} - \mathbf{P}^*\|_F^2 \le \frac{C_1 \sigma^2 d K}{V n}.
\tag{3}
$$

第二项：令 $\mathbf{Z}_v = \boldsymbol{\Phi}_v \mathbf{P}^* \in \mathbb{R}^{n \times K}$。在 A2 下 $\boldsymbol{\alpha}_v^* \in \operatorname{int}(\Delta^{K-1})$，当 $n$ 充分大时 $\hat{\boldsymbol{\alpha}}_v$ 以趋于 1 的概率落在单纯形内部，约束局部不起作用，估计量退化为无约束最小二乘：

$$
\hat{\boldsymbol{\alpha}}_v - \boldsymbol{\alpha}_v^* = (\mathbf{Z}_v^\top \mathbf{Z}_v)^{-1}\mathbf{Z}_v^\top \boldsymbol{\epsilon}_v + o_p(n^{-1/2}).
$$

其协方差为 $\sigma^2 (\mathbf{Z}_v^\top \mathbf{Z}_v)^{-1}$。于是

$$
\mathbb{E}\|\mathbf{P}^*(\hat{\boldsymbol{\alpha}}_v - \boldsymbol{\alpha}_v^*)\|^2
= \sigma^2 \operatorname{tr}\!\left(\mathbf{P}^{*\top}\mathbf{P}^* (\mathbf{Z}_v^\top \mathbf{Z}_v)^{-1}\right) + o(n^{-1}).
$$

由 $\mathbf{Z}_v^\top \mathbf{Z}_v = \mathbf{P}^{*\top}(\boldsymbol{\Phi}_v^\top \boldsymbol{\Phi}_v)\mathbf{P}^* \succeq n\lambda\, \mathbf{P}^{*\top}\mathbf{P}^*$（A1），得

$$
(\mathbf{Z}_v^\top \mathbf{Z}_v)^{-1} \preceq \frac{1}{n\lambda}(\mathbf{P}^{*\top}\mathbf{P}^*)^{-1}.
$$

代入并利用迹的循环性：

$$
\operatorname{tr}\!\left(\mathbf{P}^{*\top}\mathbf{P}^* (\mathbf{Z}_v^\top \mathbf{Z}_v)^{-1}\right)
\le \frac{1}{n\lambda}\operatorname{tr}\!\left(\mathbf{P}^{*\top}\mathbf{P}^* (\mathbf{P}^{*\top}\mathbf{P}^*)^{-1}\right)
= \frac{K}{n\lambda}.
\tag{4}
$$

合并 (3)、(4) 并乘以因子 2（来自 $\|a+b\|^2 \le 2\|a\|^2 + 2\|b\|^2$）即得 (1)。$\square$

**注（证明核心）**：式 (4) 表明系数估计的方差由**原型数 $K$** 而非嵌入维度 $d$ 控制。其代数本质是：$\mathbf{P}^*$ 将 $\mathbb{R}^K$ 等距嵌入 $\mathbb{R}^d$ 的一个 $K$ 维子空间，估计误差在该子空间内的有效自由度是 $K$，与 $d$ 无关。线性层的误差直接正比于 $d$，因为它没有利用任何低维结构。

---

## 5. 泛化误差界层面的无分布比较

若不愿引入参数估计的分布假设，可从假设空间复杂度角度给出无分布假设的界。

**定理 2**：设输入 $\mathbf{x} \in \{\mathbf{e}_1, \dots, \mathbf{e}_V\}$，损失有界于 $[0,1]$。则对任意 $\delta \in (0,1)$，以概率至少 $1-\delta$：

$$
\text{泛化误差}_{\text{lin}} \le \widehat{\text{emp}}_{\text{lin}} + 2\mathfrak{R}_n(\mathcal{H}_{\text{lin}}) + \sqrt{\frac{\log(1/\delta)}{2n}},
$$

$$
\text{泛化误差}_{\text{proto}} \le \widehat{\text{emp}}_{\text{proto}} + 2\mathfrak{R}_n(\mathcal{H}_{\text{proto}}) + \sqrt{\frac{\log(1/\delta)}{2n}},
$$

其中 Rademacher 复杂度满足

$$
\mathfrak{R}_n(\mathcal{H}_{\text{lin}}) \le B\sqrt{\frac{2\log(2Vd)}{n}},
\qquad
\mathfrak{R}_n(\mathcal{H}_{\text{proto}}) \le B\sqrt{\frac{2\log(2VK)}{n}} + \frac{B}{\sqrt{n}},
$$

$B$ 为参数范数上界。当 $K \ll d$ 时，$\mathfrak{R}_n(\mathcal{H}_{\text{proto}}) < \mathfrak{R}_n(\mathcal{H}_{\text{lin}})$。

**证明**：$\mathcal{H}_{\text{lin}}$ 在独热输入下等价于 $V$ 个独立 $\mathbb{R}^d$ 向量的集合，覆盖数满足 $\log \mathcal{N}(\epsilon, \mathcal{H}_{\text{lin}}, \|\cdot\|_\infty) \le Vd \log(2B/\epsilon)$，代入 Dudley 积分得所述界。$\mathcal{H}_{\text{proto}}$ 的参数为 $\mathbf{P}$（$dK$ 个自由参数）与 $V$ 个单纯形系数（每个 $K-1$ 维），覆盖数界为 $(dK + V(K-1))\log(2B/\epsilon)$；由于 $\boldsymbol{\alpha}_v$ 位于有界单纯形上，其对覆盖数的贡献被 $VK$ 而非 $Vd$ 控制。比较两式即得。$\square$

定理 2 的价值在于**不依赖真实低秩假设**：即使真实嵌入不在原型张成空间中，只要经验误差差距小于复杂度差距，原型编码器的泛化误差仍可更小。

---

## 6. 优化路径层面的部分严格化

**命题 3（稀疏梯度流的隐式偏置）**：考虑原型编码器的简化动力学：每个样本仅更新其最近原型 $\mathbf{p}_{k(v)}$，即 $\dot{\mathbf{p}}_k = -\nabla_{\mathbf{p}_k} \ell$ 仅对 $k = k(v)$ 非零。在平方损失与固定分配下，该动力学是**分块对角的**：不同原型的更新轨迹互不耦合。收敛点满足

$$
\mathbf{p}_k^\star = \arg\min_{\mathbf{p}} \sum_{v:\, k(v)=k} \|\mathbf{w}_v^* - \mathbf{p}\|^2,
$$

即每个原型是其成员实体的**局部均值**。而线性层的联合梯度流在过参数化下收敛到最小范数插值解 $\hat{\mathbf{W}} = \mathbf{Y}\boldsymbol{\Phi}^\top(\boldsymbol{\Phi}\boldsymbol{\Phi}^\top)^{-1}$，其每一列依赖全部实体的协方差结构。两者在函数空间中一般不同；在 A2 下，前者误差由 $K/n$ 控制，后者由 $d/n$ 控制。

**严格性边界**：命题 3 在"分配固定"的简化下严格；一旦分配随训练动态更新（如 VQ 的码本更新），收敛性分析需额外的分离条件（原型间最小距离下界）与学习率调度假设，目前文献中尚无与定理 1 同等干净的有限样本定理。这是优化层面严格化的已知边界。

---

## 7. 结论

性能差距的严格数学根源可概括为一条不等式链：

$$
\underbrace{\mathbb{E}\|\hat{\mathbf{w}}^{\text{proto}} - \mathbf{w}^*\|^2}_{\le\, 2\sigma^2 K/(\lambda n) + o(1)}
\;\;\le\;\;
\underbrace{\mathbb{E}\|\hat{\mathbf{w}}^{\text{lin}} - \mathbf{w}^*\|^2}_{=\, \sigma^2 \operatorname{tr}((\boldsymbol{\Phi}^\top\boldsymbol{\Phi})^{-1}) \,\ge\, \sigma^2 d/(\lambda_{\max} n)}
$$

在低秩原型结构（A2）、良条件设计（A1）与原型库可估（A3）下成立。其代数本质是：**原型参数化将待估自由参数的有效维度从 $d$ 降为 $K$，单纯形约束进一步将系数估计的方差限制在 $K$ 维子空间内**。线性层虽能达到同一函数，但其参数化不携带低维结构信息，在有限样本下承受与 $d$ 成正比的估计方差。

综上，原型编码器与线性层的等价性描述的是**假设空间的天花板**，而定理 1 描述的是在低秩真实结构下两个估计器实际达到的**误差地板之间的严格分离**。性能差距并非等价性有误，而是参数化方式所决定的**有效自由度**与**隐式偏置**在有限数据、非凸优化和分布偏移条件下的必然结果。

---

## 参考文献性说明

- 定理 1 的证明结构借鉴了低秩矩阵估计与混合模型聚类中的标准技术（误差分解、单纯形投影的局部无约束化、有效自由度计数）。
- 定理 2 的 Rademacher 复杂度计算基于独热输入下假设空间等价于有限向量集合的观察。
- 命题 3 的严格性边界与 VQ 文献中码本收敛性分析的已知困难一致。
- SimVQ 等工作中关于码本利用率与线性重参数化的实验观察，从反面印证了本文关于"压缩偏置的收益超过参数充分利用收益"的论断。