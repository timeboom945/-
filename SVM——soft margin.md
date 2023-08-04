$$
\{(\bold{x_i},y_i)\}_{i=1}^N，\bold{x_i}\in\R^p，y_i\in \{-1,1\}\\
y取-1或者是1
$$
对于优化问题
$$
(\bold{w},b)=min_{\bold{w},b}\Vert \bold{w}\Vert\\
y_i(\bold{w^Tx_i}+b)\geq 1,\forall i=1,2,\dots,N
$$
只能适用于严格可分的，但是由于噪音，往往并不是这么回事
## soft
soft：**允许一点点错误**——即允许有分类错误的点
所以优化问题变成了
$$
\min \{\dfrac{1}{2}\bold{w^Tw}+loss\}
$$
loss表示在这N个样本中违反$y_i(\bold{w^Tx_i}+b)\geq 1$的点的个数
因此对$\bold{w}$的选择也更加的宽松，使得严格线性不可分问题可能变成soft可分
比如选择$\bold{w_1}$有$loss_1=0$，有$\bold{w_2}$有$loss_2=2$，但是有$\dfrac{1}{2}\bold{w_1^Tw_1}+loss_1>\dfrac{1}{2}\bold{w_2^Tw_2}+loss_2$，即通过允许增加$loss$而显著地让$\bold{w^Tw}$下降
![](SVM-soft.png)
但是他的数学性质不连续，因此不可导
## 连续性定义hinge loss
loss：距离
$$
\begin{cases}
y_i(\bold{w^Tx_i+b})\geq 1，loss=0\\
y_i(\bold{w^Tx_i+b})<1，loss=1-y_i(\bold{w^Tx_i+b})
\end{cases}
$$
hinge loss就被定义为
$$
hinge loss=\sum_1^Nmax\{0,1-y_i(\bold{w^Tx_i+b})\}
$$
![](hinge%20loss.png)
一般令$\xi_i=max\{0,1-y_i(\bold{w^Tx_i+b})\}$，于是
$$
hingeloss=\sum_1^N\xi_i
$$
所以soft优化问题最终的KKT条件形式就是
$$
\begin{cases}
min_{\bold{w},b}\{\dfrac{1}{2}\bold{w^Tw}+\overbrace{C}^{超参数}\sum_1^N\xi_i\}\\
y_i(\bold{w^Tx_i+b})\geq 1-\xi_i\\
\xi_i\geq0
\end{cases}\tag 1
$$