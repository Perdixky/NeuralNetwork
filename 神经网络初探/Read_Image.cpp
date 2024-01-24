#include <iostream>
#include <fstream>
#include <vector>

std::vector<unsigned char*> images;
std::vector<unsigned char> labels;//似乎在这个实践中，使用全局变量更优
void read_image_data()
{
	std::ifstream stream(R"(C:\Users\k\Downloads\Compressed\MNIST_data\MNIST_data\train-images-idx3-ubyte\train-images.idx3-ubyte)", std::ios::binary);
	if (stream.is_open())
	{
		std::cout << "image data open success" << '\n';
	}
	else
	{
		std::cout << "image data open failed" << '\n';
	}

	unsigned char* pre_image_data = new unsigned char[16];

	stream.read(reinterpret_cast<char*>(pre_image_data), 16);
	for (int i{ 0 }; i < 16; i++)
		std::cout << std::hex << static_cast<int>(pre_image_data[i]) << std::dec << ' ';
	std::cout << '\n';
	while (!stream.eof())
	{
		unsigned char* image_data = new unsigned char[784];
		stream.read(reinterpret_cast<char*>(image_data), 784);
		/*reinterpret_cast 是 C++ 中的一种类型转换运算符，用于进行低级别的强制类型转换。
		 *它基本上告诉编译器，将某个类型的指针或引用当作另一个完全不同类型的指针或引用来处理。
		 *使用 reinterpret_cast 时，程序员需要非常确信这种转换是安全和合理的，因为编译器无法检查其有效性或安全性。*/
		images.push_back(image_data);
	} 
}

void read_labels()
{
	std::ifstream stream(R"(C:\Users\k\Downloads\Compressed\MNIST_data\MNIST_data\train-labels-idx1-ubyte\train-labels.idx1-ubyte)", std::ios::binary);
	if (stream.is_open())
	{
		std::cout << "labels open success" << '\n';
	}
	else
	{
		std::cout << "labels open failed" << '\n';
	}

	unsigned char* pre_label_data = new unsigned char[8];
	stream.read(reinterpret_cast<char*>(pre_label_data), 8);
	for (int i{ 0 }; i < 8; i++)
	{
		std::cout << std::hex << static_cast<int>(pre_label_data[i]) << std::dec << ' ';
	}
	std::cout << '\n';
	while (!stream.eof())
	{
		unsigned char* label_data = new unsigned char[1];
		stream.read(reinterpret_cast<char*>(label_data), 1);
		labels.push_back(*label_data);
	}
}

int main()
{
	read_image_data();
	read_labels();
	images.pop_back();
	labels.pop_back();//最后一个元素是错误数据，需要弹出
	return 0;
}