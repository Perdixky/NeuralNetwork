#include <Eigen/Dense>

#include "NeuralNetwork.h"

/*
 * \brief 激活函数relu及其导数，在隐藏层使用
 * \note 我们的输入使用了0-255的区间，使用sigmoid函数会导致梯度消失，所以使用relu函数
 */
inline Eigen::VectorXd relu(const Eigen::VectorXd& vector)
{
	/*return vector.unaryExpr([](const double elem) { return std::max(0.0, elem); });*/
	return vector.cwiseMax(0.0);
}
Eigen::VectorXd relu_derivative(const Eigen::VectorXd& vector)
{
	return vector.unaryExpr([](const double elem) {
		if (elem > 0)
			return 1.0;
		else
			return 0.0;
		});
}

/*
 * \brief 激活函数softmax及其导数，在输出层使用，计算偏导时只需要考虑i=j的情况
 */
Eigen::VectorXd soft_max(const Eigen::VectorXd& vector)
{
	const double vector_mean = vector.mean();
	//为了防止指数爆炸，先减去平均值
	const Eigen::VectorXd exp_vector = vector.unaryExpr([vector_mean](const double elem) { return std::exp(elem - vector_mean); });
	return exp_vector / exp_vector.sum();
}
Eigen::VectorXd soft_max_derivative(const Eigen::VectorXd& vector)
{
	const Eigen::VectorXd soft_max_vector = soft_max(vector);
	return soft_max_vector.unaryExpr([](const double elem) { return elem * (1 - elem); });
}

/*
 * \brief 交叉熵损失函数及其导数
 * \note 由于我们的输出是经过softmax函数的，所以这里的损失函数是交叉熵损失函数
 * \note 由于标签是独热编码，索引交叉熵很简单，即为-log(y_i)，其中y_i为正确数值
 */
out_put_vector cross_entropy_cost(const out_put_vector& vector, const Eigen::VectorXd& label)
{
	return -label.array() * vector.array().log();
}
out_put_vector cross_entropy_cost_derivative(const out_put_vector& vector, const Eigen::VectorXd& label)
{
	return -label.array() / vector.array();
}
