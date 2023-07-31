## 核心思想
将一群线性相关的向量正交变换为一群线性无关的向量
1. 原始特征空间的重构
2. 最大投影方差，最小重构距离

## 记号
$$
\bold{X}=(\bold{x_1},\cdots,\bold{x_N})^T=\left(\begin{matrix}
x_{11}&x_{12}&x_{13}&\cdots&x_{1p}\\
&&\vdots\\
x_{N1}&x_{N2}&x_{N3}&\cdots&x_{Np}
\end{matrix}\right)_{N\times p}
$$
样本均值
$$
\bold{\bar x}_{p\times 1}=\dfrac{1}{N}\sum_1^N\bold{x_i}=\dfrac{1}{N}\bold{(x_1,\dots,x_N)}\left(\begin{matrix}
1\\
1\\
\vdots\\
1
\end{matrix}\right)=\dfrac{1}{N}\bold{X^T1_N}\\
$$
样本方差矩阵
$$
\begin{aligned}
\bold{S}&=\dfrac{1}{N}\sum_1^N\bold{(x_i-\bar x)(x_i-\bar x)^T}\\
&=\dfrac{1}{N}\bold{X^THH^TX}\\
&=\dfrac{1}{N}\bold{X^THX}
\end{aligned}
$$
## PCA过程
1. 中心化
   对数据进行以样本均值为中心的中心化过程
   $$
    \bold{x_i-\bar x},i=1,2,\dots,N
   $$
2. 寻找一个单位方向向量$\bold{u_e}$，使得中心化后的向量$\bold{x_i-\bar x}$在$\bold{u_e}$上的**投影的方差最大化**
    $$
    Proj_{\bold{u_e}}(\bold{x_i-\bar x})=\bold{(x_i-\bar x)^Tu_e}
    $$
    投影方差即
    $$
    \begin{aligned}
    &\dfrac{1}{N}\sum_1^N\{Proj_{\bold{u_e}}(\bold{x_i-\bar x})-0\}^2\\
    &=\dfrac{1}{N}\sum_1^N\bold{(x_i-\bar x)^Tu_e}\bold{(x_i-\bar x)^Tu_e}\\
    &=\dfrac{1}{N}\sum_1^N\underbrace{\{\bold{(x_i-\bar x)^Tu_e}\}^T}_{实数的转置就是他本身}\bold{(x_i-\bar x)^Tu_e}\\
    &=\dfrac{1}{N}\sum_1^N\bold{u_e^T(x_i-\bar x)(x_i-\bar x)^Tu_e}\\
    &=\dfrac{1}{N}\bold{u_e^T}\sum_1^N\bold{(x_i-\bar x)(x_i-\bar x)^T}\bold{u_e}\\
    &=\bold{u^T_eSu_e}
    \end{aligned}
    $$
    所以得到了一个带有约束的拉格朗日数乘优化问题
    $$
    \begin{cases}
    \bold{\hat u_e}=argmax_{\bold{u_e}}\bold{u^T_eSu_e}\\
    \bold{u^T_eu_e}=1
    \end{cases}
    $$
    得到拉格朗日方程
    $$
    \clubsuit(\bold{u_e},\lambda)=\bold{u^T_eSu_e}+\lambda(\bold{1-u^T_eu_e})\\
    \dfrac{\partial \clubsuit}{\partial \bold{u_e}}=2\bold{Su_e}-\lambda2\bold{u_e}=\bold{0}\\
    \dfrac{\partial \clubsuit}{\partial \lambda}=\bold{1-u^T_eu_e}=0
    $$
    即
    $$
    \bold{Su_e}=\lambda\bold{u_e}
    $$
    所以$\bold{\hat u_e}$就是$S$的特征向量
    **即中心化后的点在其方差矩阵的特征向量上的投影的方差是最大的**

3. 重构
   初始的中心化$\bold{x_i}$
   $$
    \bold{x_j}=\sum_{i=1}^p\bold{(x_j^Te_i)e_i}=x_1\left(\begin{matrix}
    1\\
    0\\
    \vdots\\
    0
    \end{matrix}\right)+x_2\left(\begin{matrix}
    0\\
    1\\
    \vdots\\
    0
    \end{matrix}\right)+\dots+x_p\left(\begin{matrix}
    0\\
    0\\
    \vdots\\
    1
    \end{matrix}\right)_{p\times 1}
   $$
   重构后的中心化$\bold{\hat x_i}$
    $$
    \bold{\hat x_j}=\sum_{i=1}^{q(q\leq p)}\bold{(x_j^Tu_{ei})e_i}=\bold{x_j^Tu_{e1}}\left(\begin{matrix}
    1\\
    0\\
    \vdots\\
    0
    \end{matrix}\right)+\bold{x_j^Tu_{e2}}\left(\begin{matrix}
    0\\
    1\\
    \vdots\\
    0
    \end{matrix}\right)+\dots+\bold{x_j^Tu_{eq}}\left(\begin{matrix}
    0\\
    \vdots\\
    1_q\\
    \vdots\\
    0
    \end{matrix}\right)_{p\times 1}
    $$
    不足的用0补全，达到重构或者降维的目的

4. 最小重构代价
   但是一般很难一步就让$q<p$达成降维的目的
   所以这一步要考察这$p$个线性无关的一组基中那一些基的向量的组合能够最大程度接近重构后的点$\bold{\hat x_i}$
   重构代价可以被定义为
   $$
    J=\sum_1^N\Vert \bold{x_i-\hat x_i}\Vert^2
   $$
   **因此要尽可能扔掉这p个线性无关的基内的向量(主要目标是降维)** 
   **又要尽可能地使得丢失的信息量变小**   
___
## 奇异值分解
任何矩阵都可以进行奇异值分解（Singular Value Decomposition, SVD）。奇异值分解是一种将矩阵分解为三个矩阵乘积的方法，即 $\bold{A = UΣV^T}$，其中 $\bold{A}$ 是一个 m×n 的矩阵，$\bold{U}$ 是一个 m×m 的正交矩阵，$\bold{Σ}$ 是一个 m×n 的对角矩阵，$\bold{V}$ 是一个 n×n 的正交矩阵。奇异值分解在很多应用中非常有用，例如在数据降维、矩阵逆运算、图像压缩等领域都有广泛的应用。
___
?