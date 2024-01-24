#pragma once
#include <Eigen/Dense>

#include "NeuralNetwork.h"

inline Eigen::VectorXd relu(const Eigen::VectorXd& vector);
Eigen::VectorXd relu_derivative(const Eigen::VectorXd& vector);

Eigen::VectorXd soft_max(const Eigen::VectorXd& vector);
Eigen::VectorXd soft_max_derivative(const Eigen::VectorXd& vector);

out_put_vector cross_entropy_cost(const out_put_vector& vector, const Eigen::VectorXd& label);
out_put_vector cross_entropy_cost_derivative(const out_put_vector& vector, const Eigen::VectorXd& label);