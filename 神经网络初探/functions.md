# 项目所有公式
## 一、神经网络公式
### 前向传递
$$
z^{l+1} = w^l a^l + b^l \tag{1}
$$

$$
a^l = r(z^l) \tag{2}
$$

$$
a^L = s(z^L) \tag{3}
$$
### 代价函数
$$
C = H(y, a^L) = -\ln(a^L) \odot y \tag{4}
$$
### 误差定义
$$
\delta^{l} \equiv \frac{\partial C}{\partial z^{l}} \tag{5}
$$
### 反向传递
$$
\nabla C\equiv\left(\frac{\partial C}{\partial v_1},\ldots,\frac{\partial C}{\partial v_m}\right)^T     \tag{6}
$$

$$
\delta^{L} = \nabla_{a^{L}} C \odot \sigma^{'}(z^{L}) \tag{7}
$$

$$
\delta^{l} = ((w^{l})^{T} \delta^{l+1}) \odot \sigma^{'}(z^{L}) \tag{8}
$$

$$
\frac{\partial C}{\partial b^{l}} = \delta^{l+1} \tag{9}
$$

$$
\frac{\partial C}{\partial w^{l}} = \delta^{l+1} (a^{l})^{T} \tag{10}
$$
$$\frac{\sum_{j=1}^m\nabla C_{X_j}}m\approx\frac{\sum_x\nabla C_x}n=\nabla C$$
$$\Delta v=-\eta\nabla C \tag{11}$$
#### 最外层使用了softmax作为激活函数，其余使用relu函数
#### 代价函数为交叉熵
## 二、杂项公式
### 1.柯西-施瓦茨不等式
$$
(a,b)^2 \leq (a,a)(b,b) \tag{12}
$$
### 2.SoftMax激活函数
$$D=max(z)\\
$$
$$
\begin{aligned}
&softmax(z_i)\\
&=\frac{e^{z_i}}{\sum_{c}e^{z_c}}\\
&=\frac{e^{z_i-D}}{\sum_{c}e^{z_c-D}}\end{aligned}\tag{13}$$ 
#### 其偏导数为
$$a_i=\frac{e^{z_i}}{\sum_{t}{z_t}}=\frac{e^{z_i}}{sum}$$
$$\left.\frac{\partial a_i}{\partial z_j}=\left\{\begin{array}{rc}-a_i^2+a_i,&i=j\\-a_ia_j,&i\neq j\end{array}\right.\right. \tag{14}
$$
#### 结合独热编码交叉熵的导数
$$
-\ln a_k \nabla_z a_k \tag{15}
$$
其中$k$为标签
### 3.he初始化方法
$$
w\sim N(0,\sigma^{2})\\
\sigma = \sqrt{\frac2{a^{l}}} 
\tag{16} 
$$ 
