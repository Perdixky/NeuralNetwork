#include "NeuralNetwork.h"
#include "Functions.h"
#include <iostream>
#include "Read_Image.h"


extern std::vector<unsigned char*> images;
extern std::vector<unsigned char> labels;




/*
 * \brief ��ʼ�������磬�������ɸ���֮���Ȩ�ؾ����ƫ��������ͬʱ��ȡ��ѵ������
 */
NeuralNetwork::NeuralNetwork(const Eigen::VectorXi& sizes, const double learning_speed)
{
	layers_num_ = static_cast<unsigned char>(sizes.size()); //ԭ��ʹ��long long���ͣ�������unsigned char�͹���
	for (int i{ 1 }; i < layers_num_; i++)
	{
		const Eigen::MatrixXd weight = Eigen::MatrixXd::Random(sizes[i], sizes[i - 1]);
		weight_between_layers_.emplace_back(weight);
		//emplace_back��push_backЧ�ʸߣ�c++11�����룬�����ڸ��ƽ�����ǰ������ʱ����
		const Eigen::VectorXd bias = Eigen::VectorXd::Random(sizes[i]);
		bias_per_layer_.emplace_back(bias);

		bias_gradients_sum_.emplace_back(Eigen::VectorXd::Zero(sizes[i]));
		weight_between_layers_.emplace_back(Eigen::MatrixXd::Zero(sizes[i], sizes[i - 1]));
		//�����Ȩ�ؾ����ƫ���������ݶȾ���ͳ�ʼ��Ϊ0������֮�����
	}
	read_image_data();
	read_labels();
	images.pop_back();
	labels.pop_back(); //���һ��Ԫ���Ǵ������ݣ���Ҫ����
}

/*
 * \brief ��ȡÿ��ѵ������
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
 * \brief ��ǰ�����������������
 * \note  ÿһ�㴢�����z ^ lֵ����w * a ^ (l - 1) + b��������a ^ l
 * \note  ��һ����������Լ���relu������Ӱ��
 */
void NeuralNetwork::Feedforward()
{
	for (int i{ 1 }; i < layers_num_; i++)
	{
		neuron_values_per_layer_.emplace_back(relu(weight_between_layers_[i - 1]) * neuron_values_per_layer_[i - 1] + bias_per_layer_[i - 1]);
	}
}

/*
 * \brief ���򴫲������������ݶ�����
 * \note ��ʽ���
 */
void NeuralNetwork::CalculateCost()
{
	layer_costs_[layers_num_ - 1] = cross_entropy_cost_derivative(soft_max(neuron_values_per_layer_.back()), label_data_).array() * soft_max_derivative(neuron_values_per_layer_.back()).array();//�������һ����������

	/*for(unsigned char i{ 0 }; i < layers_num_ - 1; i++)
	{
		layer_costs_[layers_num_ - 2 - i] = ((weight_between_layers_[layers_num_ - 2 - i].transpose() * layer_costs_[layers_num_ - 1 - i]).array() * relu_derivative(neuron_values_per_layer_[layers_num_ - 2 - i]).array()).matrix();
	}*/
	for (unsigned char i = layers_num_ - 2; i != 255; --i)
		//���� i ���޷����ַ����ͣ�ֱ��ʹ�� i >= 0 ����������ѭ������Ϊ�޷��������ڼ���0����ʱ����Ƶ����ֵ����ˣ�����ʹ�� i != static_cast<unsigned char>(-1) = 255 �������ơ�
		//�� i Ϊ 0 ʱ���ټ�һ�����޷������͵����ֵ���� static_cast<unsigned char>(-1)��
	{
		layer_costs_[i] = ((weight_between_layers_[i].transpose() * layer_costs_[i + 1]).array() * relu_derivative(neuron_values_per_layer_[i]).array()).matrix();
		bias_gradients_sum_[i] += layer_costs_[i + 1];
		weight_gradients_sum_[i] += layer_costs_[i + 1] * neuron_values_per_layer_[i].transpose();
	}
}

/*
 * \brief ѵ�����磬�������ݻ�ȡ����ǰ�����ͷ��򴫲�
 * \note ����ѵ��������60000��������ÿ��ѵ��ʱ��ÿ��batch��100�����ݣ�һ����600��batch
 */
void NeuralNetwork::train()
{
	//60000�����ݣ��ֳ�600��batch��ÿ��batch100������
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