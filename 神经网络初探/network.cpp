#include "NeuralNetwork.h"
#include "Functions.h"
#include <iostream>
#include "Read_Image.h"


extern std::vector<unsigned char*> images;
extern std::vector<unsigned char> labels;




/*
 * \brief 初始化神经网络，包括生成各层之间的权重矩阵和偏置向量，同时读取了训练数据
 */
NeuralNetwork::NeuralNetwork(const Eigen::VectorXi& sizes, const double learning_speed)
{
	layers_num_ = static_cast<unsigned char>(sizes.size()); //原本使用long long类型，但是用unsigned char就够了
	for (int i{ 1 }; i < layers_num_; i++)
	{
		const Eigen::MatrixXd weight = Eigen::MatrixXd::Random(sizes[i], sizes[i - 1]);
		weight_between_layers_.emplace_back(weight);
		//emplace_back比push_back效率高，c++11后引入，不会在复制进容器前创建临时对象
		const Eigen::VectorXd bias = Eigen::VectorXd::Random(sizes[i]);
		bias_per_layer_.emplace_back(bias);

		bias_gradients_sum_.emplace_back(Eigen::VectorXd::Zero(sizes[i]));
		weight_between_layers_.emplace_back(Eigen::MatrixXd::Zero(sizes[i], sizes[i - 1]));
		//这里的权重矩阵和偏置向量的梯度矩阵和初始化为0，方便之后相加
	}
	read_image_data();
	read_labels();
	images.pop_back();
	labels.pop_back(); //最后一个元素是错误数据，需要弹出
}

/*
 * \brief 读取每次训练数据
 */
void NeuralNetwork::LoadData(const int data_index)
{
	image_vector input;
	for (unsigned short pixel{ 0 }; pixel < 784; pixel++)
	{
		input(pixel) = images[data_index][pixel];
	}
	neuron_values_per_layer_.emplace_back(input);
	label_data_ = out_put_vector::Zero();
	label_data_(labels[data_index]) = 1;
}

/*
 * \brief 向前传播，计算各层的输出
 * \note  每一层储存的是z ^ l值，即w * a ^ (l - 1) + b，而不是a ^ l
 * \note  第一层恒正，所以加上relu函数不影响
 */
void NeuralNetwork::Feedforward()
{
	for (int i{ 1 }; i < layers_num_; i++)
	{
		neuron_values_per_layer_.emplace_back(relu(weight_between_layers_[i - 1]) * neuron_values_per_layer_[i - 1] + bias_per_layer_[i - 1]);
	}
}

/*
 * \brief 反向传播，计算各层的梯度向量
 * \note 公式详见
 */
void NeuralNetwork::CalculateCost()
{
	layer_costs_[layers_num_ - 1] = cross_entropy_cost_derivative(soft_max(neuron_values_per_layer_.back()), label_data_).array() * soft_max_derivative(neuron_values_per_layer_.back()).array();//计算最后一层的误差向量

	/*for(unsigned char i{ 0 }; i < layers_num_ - 1; i++)
	{
		layer_costs_[layers_num_ - 2 - i] = ((weight_between_layers_[layers_num_ - 2 - i].transpose() * layer_costs_[layers_num_ - 1 - i]).array() * relu_derivative(neuron_values_per_layer_[layers_num_ - 2 - i]).array()).matrix();
	}*/
	for (unsigned char i = layers_num_ - 2; i != 255; --i)
		//由于 i 是无符号字符类型，直接使用 i >= 0 将导致无限循环，因为无符号类型在减到0以下时会回绕到最大值。因此，我们使用 i != static_cast<unsigned char>(-1) = 255 来检测回绕。
		//当 i 为 0 时，再减一会变成无符号类型的最大值，即 static_cast<unsigned char>(-1)。
	{
		layer_costs_[i] = ((weight_between_layers_[i].transpose() * layer_costs_[i + 1]).array() * relu_derivative(neuron_values_per_layer_[i]).array()).matrix();
		bias_gradients_sum_[i] += layer_costs_[i + 1];
		weight_gradients_sum_[i] += layer_costs_[i + 1] * neuron_values_per_layer_[i].transpose();
	}
}

/*
 * \brief 训练网络，包括数据获取，向前传播和反向传播
 * \note 由于训练数据是60000个，所以每次训练时，每个batch有100个数据，一共有600个batch
 */
void NeuralNetwork::train()
{
	//60000个数据，分成600个batch，每个batch100个数据
	for (unsigned short batch_iteration{ 0 }; batch_iteration < 600; batch_iteration++)
	{
		for (unsigned char inner_iteration{ 0 }; inner_iteration < 100; inner_iteration++)
		{
			LoadData(inner_iteration * batch_iteration);
			Feedforward();
			CalculateCost();
		}
		PerformBackpropagation();
	}
}

int main()
{
	
}