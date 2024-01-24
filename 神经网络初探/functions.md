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
\delta^{l} \equiv \frac{\partial C}{\partial b^{l}} \tag{5}
$$
### 反向传递
$$
\delta^{L} = \nabla_{a^{L}} C \odot \sigma^{'}(z^{L}) \tag{6}
$$

$$
\delta^{l} = ((w^{l})^{T} \delta^{l+1}) \odot \sigma^{'}(z^{L}) \tag{7}
$$

$$
\frac{\partial C}{\partial b^{l}} = \delta^{l+1} \tag{8}
$$

$$
\frac{\partial C}{\partial w^{l}} = \delta^{l+1} (a^{l})^{T} \tag{9}
$$
#### 最外层使用了softmax作为激活函数，其余使用relu函数
#### 代价函数为交叉熵
## 二、杂项公式
### 柯西-施瓦茨不等式
$$
(a,b)^2 \leq (a,a)(b,b) \tag{10}
$$
