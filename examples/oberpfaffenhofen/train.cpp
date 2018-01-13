/*
	Copyright (c) 2013, Taiga Nomi and the respective contributors
	All rights reserved.

	Use of this source code is governed by a BSD-style license that can be found
	in the LICENSE file.
*/
#include <cstdlib>
#include <iostream>
#include <vector>
#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/algorithm/string.hpp"
#include "tiny_dnn/tiny_dnn.h"

template <typename N>
void construct_net(N &nn, tiny_dnn::core::backend_t backend_type) {

	using conv = tiny_dnn::convolutional_layer;
	using pool = tiny_dnn::max_pooling_layer;
	using fc = tiny_dnn::complex_fully_connected_layer;
	using relu = tiny_dnn::relu_layer;
	using softmax = tiny_dnn::softmax_layer;

	//nn << conv(27, 27, 5, 3, 10, tiny_dnn::padding::valid, true, 1, 1,
	   // backend_type)                      // C1
	   // << pool(23, 23, 10, 2, backend_type)  // P2
	   // << relu()          
	   // << conv(11, 11, 3, 10, 6, tiny_dnn::padding::valid, true, 1, 1,
		  //  backend_type)                      // C1
	   // << relu() 
	   // << fc(9 * 9 * 6, 6, true, backend_type)    // FC7
	   // << relu()                                            // activation
	   // << fc(6, 5, true, backend_type)
	   // << softmax(5);  // FC10

	nn << fc(2 * 2, 2, true, tiny_dnn::core::backend_t::internal)
		<< softmax(2);  // FC10

}

void train_OBER(std::string data_dir_path,
	double learning_rate,
	const int n_train_epochs,
	const int n_minibatch,
	tiny_dnn::core::backend_t backend_type,
	std::ostream &log) {
	// specify loss-function and learning strategy
	tiny_dnn::network<tiny_dnn::sequential> nn;
	tiny_dnn::adam optimizer;

	construct_net(nn, backend_type);

	std::cout << "load models..." << std::endl;

	// load cifar dataset
	std::vector<tiny_dnn::label_t> train_labels, test_labels, test_labels_loaded, train_labels_loaded;
	std::vector<tiny_dnn::vec_t> train_images, test_images, test_images_loaded, train_images_loaded;

	tiny_dnn::vec_c complex_img;
	std::vector<tiny_dnn::vec_c> complex_images;
	
	complex_img.push_back(tiny_dnn::complex_t(1,0));
	complex_img.push_back(tiny_dnn::complex_t(-1, 0));
	complex_img.push_back(tiny_dnn::complex_t(-1, 0));
	complex_img.push_back(tiny_dnn::complex_t(-1, 0));
	complex_images.push_back(complex_img);
	//parse_rgb_db(100, 1, data_dir_path+ "/train", &train_images_loaded, &train_labels_loaded, -1.0, 1.0);
	//parse_rgb_db(1, 1, data_dir_path + "/test", &test_images_loaded, &test_labels_loaded, -1.0, 1.0);

	// ++ TEST ONLY
	/*train_images.push_back(train_images_loaded.at(0));
	train_labels.push_back(train_labels_loaded.at(0));
	train_images.push_back(train_images_loaded.at(1));
	train_labels.push_back(train_labels_loaded.at(1));
	test_images.push_back(train_images_loaded.at(0));
	test_labels.push_back(test_labels_loaded.at(0));*/
	// -- TEST ONLY

	//parse_rgb_db(100, 1, data_dir_path + "/train", &train_images, &train_labels, -1.0, 1.0);
	//parse_rgb_db(10, 1, data_dir_path + "/test", &test_images, &test_labels, -1.0, 1.0);

	// ++ TEST ONLY
	int IMAGE_SIZE = 2 * 2;
	std::vector<unsigned char> buf(IMAGE_SIZE);
	tiny_dnn::vec_t img;

	buf[0] = 255;
	buf[1] = 0;
	buf[2] = 0;
	buf[3] = 0;

	//buf[4] = 0;
	//buf[5] = 0;
	//buf[6] = 0;
	//buf[7] = 0;

	//buf[8] = 0;
	//buf[9] = 0;
	//buf[10] = 0;
	//buf[11] = 0;

	std::transform(buf.begin(), buf.end(), std::back_inserter(img),
		[=](unsigned char c) {
		return -1 + (1 + 1) * c / 255;
	});
	train_images.push_back(img);
	train_labels.push_back(0);

	img.clear();
	buf[0] = 0;
	buf[1] = 0;
	buf[2] = 0;
	buf[3] = 0;

	/*buf[4] = 255;
	buf[5] = 0;
	buf[6] = 0;
	buf[7] = 0;

	buf[8] = 0;
	buf[9] = 0;
	buf[10] = 0;
	buf[11] = 0;*/

	std::transform(buf.begin(), buf.end(), std::back_inserter(img),
		[=](unsigned char c) {
		return -1 + (1 + 1) * c / 255;
	});
	/*train_images.push_back(img);
	train_labels.push_back(1);

	img.clear();
	buf[0] = 0;
	buf[1] = 255;
	buf[2] = 0;
	buf[3] = 0;

	buf[4] = 0;
	buf[5] = 0;
	buf[6] = 0;
	buf[7] = 0;

	buf[8] = 0;
	buf[9] = 0;
	buf[10] = 0;
	buf[11] = 0;

	std::transform(buf.begin(), buf.end(), std::back_inserter(img),
		[=](unsigned char c) {
		return -1 + (1 + 1) * c / 255;
	});*/

	test_images.push_back(img);
	test_labels.push_back(0);

	// -- TEST ONLY

	std::cout << "Start learning" << std::endl;

	tiny_dnn::progress_display disp(complex_images.size());
	tiny_dnn::timer t;

	optimizer.alpha *=
		static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate);

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;
		++epoch;
		//tiny_dnn::result res = nn.test(test_images, test_labels);
		//log << res.num_success << "/" << res.num_total << std::endl;

		//disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

	// training
	nn.train<tiny_dnn::mse>(optimizer, complex_images, train_labels,
		n_minibatch, n_train_epochs,
		on_enumerate_minibatch, on_enumerate_epoch);

	std::cout << std::endl<<"End training." << std::endl;

	// test and show results
	auto results = nn.test(test_images, test_labels);
	std::cout << std::endl;
	results.print_detail(std::cout);
	// save networks
	std::ofstream ofs("cifar-weights");
	ofs << nn;
}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
	const std::array<const std::string, 5> names = {
	  "internal", "nnpack", "libdnn", "avx", "opencl",
	};
	for (size_t i = 0; i < names.size(); ++i) {
		if (name.compare(names[i]) == 0) {
			return static_cast<tiny_dnn::core::backend_t>(i);
		}
	}
	return tiny_dnn::core::default_engine();
}

static void usage(const char *argv0) {
	std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
		<< " --learning_rate 0.01"
		<< " --epochs 30"
		<< " --minibatch_size 10"
		<< " --backend_type internal" << std::endl;
}

int main(int argc, char **argv) {
	double learning_rate = 0.01;
	int epochs = 1;
	//tnn_rgb_test_images //tnn_complex_ober
	std::string data_path = "C:/Data/tnn_rgb_test_images";
	int minibatch_size = 1;
	tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();


	if (argc == 2) {
		std::string argname(argv[1]);
		if (argname == "--help" || argname == "-h") {
			usage(argv[0]);
			return 0;
		}
	}
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--learning_rate") {
			learning_rate = atof(argv[count + 1]);
		}
		else if (argname == "--epochs") {
			epochs = atoi(argv[count + 1]);
		}
		else if (argname == "--minibatch_size") {
			minibatch_size = atoi(argv[count + 1]);
		}
		else if (argname == "--backend_type") {
			backend_type = parse_backend_name(argv[count + 1]);
		}
		else if (argname == "--data_path") {
			data_path = std::string(argv[count + 1]);
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			usage(argv[0]);
			return -1;
		}
	}
	if (data_path == "") {
		std::cerr << "Data path not specified." << std::endl;
		usage(argv[0]);
		return -1;
	}
	if (learning_rate <= 0) {
		std::cerr
			<< "Invalid learning rate. The learning rate must be greater than 0."
			<< std::endl;
		return -1;
	}
	if (epochs <= 0) {
		std::cerr << "Invalid number of epochs. The number of epochs must be "
			"greater than 0."
			<< std::endl;
		return -1;
	}
	if (minibatch_size <= 0 || minibatch_size > 50000) {
		std::cerr
			<< "Invalid minibatch size. The minibatch size must be greater than 0"
			" and less than dataset size (50000)."
			<< std::endl;
		return -1;
	}

	boost::posix_time::ptime now(boost::posix_time::microsec_clock::local_time());
	const std::string str_time = to_iso_string(now);
	std::ofstream out("C:\\Data\\" + str_time + ".txt");
	std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!
	std::cout << "Train net" << std::endl;

	std::cout << "Running with the following parameters:" << std::endl
		<< "Data path: " << data_path << std::endl
		<< "Learning rate: " << learning_rate << std::endl
		<< "Minibatch size: " << minibatch_size << std::endl
		<< "Number of epochs: " << epochs << std::endl
		<< "Backend type: " << backend_type << std::endl
		<< std::endl;
	try {
		train_OBER(data_path, learning_rate, epochs, minibatch_size,
			backend_type, std::cout);
	}
	catch (tiny_dnn::nn_error &err) {
		std::cerr << "Exception: " << err.what() << std::endl;
	}
}
