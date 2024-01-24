#pragma once
#include <vector>
#include <Eigen/Dense>
/*
 *On the topic of performance,
 *all what matters is that you give Eigen as much information as possible at compile time.
 */

/*
 * 第一层为输入层，没有权重，偏置，而c++中计数从0开始，因而在这个程序中，公式为
 * z ^ (l + 1) = w ^ l * z ^ l + b ^ l
 */

using image_vector = Eigen::Vector<unsigned char, 784>;
using out_put_vector = Eigen::Vector<unsigned char, 10>;
class NeuralNetwork
{
public:
	explicit NeuralNetwork(const Eigen::VectorXi& sizes, const double  learning_speed);//使用explicit关键字，防止隐式转换

	void LoadData(const int data_index);  //获取输入的向量和输出的向量
	void Feedforward();  //前向传播
	void CalculateCost();  //获取各层的误差
	void PerformBackpropagation();  //反向传播
	void train();  //训练
private:
	//这里是私有成员变量，储存了每一次训练时的各层神经元数值
	//不同层的信息通过向量来储存，同层的信息通过矩阵来储存，这样可以方便地进行矩阵运算
	unsigned char layers_num_{ 0 };  //层数
	double learning_speed_{ 0 };  //学习速率
	std::vector<Eigen::VectorXd> neuron_values_per_layer_;  //各层的神经元数值，储存的是z ^ l值，即w * a ^ (l - 1) + b，而不是a ^ l
	std::vector<Eigen::MatrixXd> weight_between_layers_; //各层之间的权重方阵
	std::vector<Eigen::VectorXd> bias_per_layer_;  //各层之间的偏置向量
	out_put_vector label_data_;  //标签
	std::vector<Eigen::VectorXd> layer_costs_; //各层的误差，每一层的误差值与代价函数关于⽹络中任意偏置的改变率相同
	std::vector<Eigen::VectorXd> bias_gradients_sum_;  //各层偏置的梯度向量和
	std::vector<Eigen::MatrixXd> weight_gradients_sum_; //各层权重的梯度矩阵和
};

