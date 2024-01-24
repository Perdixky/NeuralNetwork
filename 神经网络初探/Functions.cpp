#include <Eigen/Dense>

#include "NeuralNetwork.h"

/*
 * \brief �����relu���䵼���������ز�ʹ��
 * \note ���ǵ�����ʹ����0-255�����䣬ʹ��sigmoid�����ᵼ���ݶ���ʧ������ʹ��relu����
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
 * \brief �����softmax���䵼�����������ʹ�ã�����ƫ��ʱֻ��Ҫ����i=j�����
 */
Eigen::VectorXd soft_max(const Eigen::VectorXd& vector)
{
	const double vector_mean = vector.mean();
	//Ϊ�˷�ָֹ����ը���ȼ�ȥƽ��ֵ
	const Eigen::VectorXd exp_vector = vector.unaryExpr([vector_mean](const double elem) { return std::exp(elem - vector_mean); });
	return exp_vector / exp_vector.sum();
}
Eigen::VectorXd soft_max_derivative(const Eigen::VectorXd& vector)
{
	const Eigen::VectorXd soft_max_vector = soft_max(vector);
	return soft_max_vector.unaryExpr([](const double elem) { return elem * (1 - elem); });
}

/*
 * \brief ��������ʧ�������䵼��
 * \note �������ǵ�����Ǿ���softmax�����ģ������������ʧ�����ǽ�������ʧ����
 * \note ���ڱ�ǩ�Ƕ��ȱ��룬���������غܼ򵥣���Ϊ-log(y_i)������y_iΪ��ȷ��ֵ
 */
out_put_vector cross_entropy_cost(const out_put_vector& vector, const Eigen::VectorXd& label)
{
	return -label.array() * vector.array().log();
}
out_put_vector cross_entropy_cost_derivative(const out_put_vector& vector, const Eigen::VectorXd& label)
{
	return -label.array() / vector.array();
}
