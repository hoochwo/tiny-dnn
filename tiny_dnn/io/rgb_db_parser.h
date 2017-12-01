/*
	Copyright (c) 2013, Taiga Nomi and the respective contributors
	All rights reserved.

	Use of this source code is governed by a BSD-style license that can be found
	in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <vector>
#include "tiny_dnn/util/util.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <random>

namespace tiny_dnn {

	template<typename T>
	std::vector<T> splitCustomString(const T & str, const T & delimiters) {
		std::vector<T> v;
		typename T::size_type start = 0;
		auto pos = str.find_first_of(delimiters, start);
		while (pos != T::npos) {
			if (pos != start) // ignore empty tokens
				v.emplace_back(str, start, pos - start);
			start = pos + 1;
			pos = str.find_first_of(delimiters, start);
		}
		if (start < str.length()) // ignore trailing delimiter
			v.emplace_back(str, start, str.length() - start); // add what's left of the string
		return v;
	}


	/*cv::Mat*/ void getImageByIndex(std::vector<unsigned char>& buf, cv::Mat& image, int row, int column, int radius) {

		//std::cout << "--------------- Get image " << std::endl;

		int indexColumnCopyFrom;
		int indexRowCopyFrom;
		int indexColumn;
		int indexRow;
		int countWl;
		int countHl;
		int countWr;
		int countHr;

		cv::Mat entry = cv::Mat::zeros(2 * radius + 1, 2 * radius + 1, image.type());

		if ((column - radius) < 0) {
			indexColumnCopyFrom = 0;
			indexColumn = radius - column;
			countWl = column;
		}
		else {
			indexColumnCopyFrom = column - radius;
			indexColumn = 0;
			countWl = radius;
		}

		if ((row - radius) < 0) {
			indexRowCopyFrom = 0;
			indexRow = radius - row;
			countHl = row;
		}
		else {
			indexRowCopyFrom = row - radius;
			indexRow = 0;
			countHl = radius;
		}

		if ((column + radius) >= image.cols)
		{
			countWr = (image.cols - 1) - column;
		}
		else {
			countWr = radius;
		}

		if ((row + radius) >= image.rows)
		{
			countHr = image.rows - 1 - row;
		}
		else {
			countHr = radius;
		}

		cv::Mat tmp = image(cv::Rect(indexColumnCopyFrom, indexRowCopyFrom, countWl + countWr + 1, countHr + countHl + 1));

		//std::cout << "tmp.type()" << tmp.type() <<"  image.type() "<< image.type()<< std::endl;

		for (int i = indexColumn, i2 = 0; i2 < tmp.cols; i++, i2++) {
			for (int j = indexRow, j2 = 0; j2 < tmp.rows; j++, j2++) {
		
				if (image.type() == CV_8UC3)
				{
					entry.at<cv::Vec3b>(j, i)[0] = tmp.at<cv::Vec3b>(j2, i2)[0];
					entry.at<cv::Vec3b>(j, i)[1] = tmp.at<cv::Vec3b>(j2, i2)[1];
					entry.at<cv::Vec3b>(j, i)[2] = tmp.at<cv::Vec3b>(j2, i2)[2];
				}  

				if (image.type() == CV_32FC3)
				{
					entry.at<cv::Vec3f>(j, i)[0] = tmp.at<cv::Vec3f>(j2, i2)[0];
					entry.at<cv::Vec3f>(j, i)[1] = tmp.at<cv::Vec3f>(j2, i2)[1];
					entry.at<cv::Vec3f>(j, i)[2] = tmp.at<cv::Vec3f>(j2, i2)[2];
				}

				if (image.type() == CV_32FC(6))
				{
					entry.at<cv::Vec6f>(j, i)[0] = tmp.at<cv::Vec6f>(j2, i2)[0];
					entry.at<cv::Vec6f>(j, i)[1] = tmp.at<cv::Vec6f>(j2, i2)[1];
					entry.at<cv::Vec6f>(j, i)[2] = tmp.at<cv::Vec6f>(j2, i2)[2];
					entry.at<cv::Vec6f>(j, i)[3] = tmp.at<cv::Vec6f>(j2, i2)[3];
					entry.at<cv::Vec6f>(j, i)[4] = tmp.at<cv::Vec6f>(j2, i2)[4];
					entry.at<cv::Vec6f>(j, i)[5] = tmp.at<cv::Vec6f>(j2, i2)[5];
				}
			}
		}

		for (int i = 0; i < entry.rows; i++) {
			for (int j = 0; j < entry.cols; j++) {
				int offset = i*tmp.rows + j;
				buf[offset] = entry.at<cv::Vec3b>(i, j)[0];
				buf[tmp.cols*tmp.rows + offset] = entry.at<cv::Vec3b>(i, j)[1];
				buf[2 * tmp.cols*tmp.rows + offset] = entry.at<cv::Vec3b>(i, j)[2];
				//std::cout << offset << " , " << tmp.cols*tmp.rows + offset << " , " << 2 * tmp.cols*tmp.rows + offset << std::endl << std::endl;
			}
		}

		return;// entry;
	}



	std::vector<std::string> getLabelsPathes(std::string path) {
		std::vector<std::string> labelsPath;
		std::ifstream infile(path);
		std::string line;
		while (std::getline(infile, line)) {
			labelsPath.push_back(line);
		}

		return labelsPath;
	}


	cv::Mat getLableMat(std::string path) {

		std::vector<std::string> labelsPath = getLabelsPathes(path);
		cv::Mat lableInt;

		cv::Mat labelBinary = cv::imread(labelsPath.at(0));

		cv::Mat outPut = cv::Mat::zeros(labelBinary.size(), CV_8UC1);

		//std::cout <<"-------------------"<<std::endl<< " Creat label MAT: " << path<< std::endl << "-------------------" << std::endl;

		for (int i = 0; i < labelsPath.size(); i++) {

			labelBinary = cv::imread(labelsPath.at(i));

			cv::cvtColor(labelBinary, labelBinary, CV_BGR2GRAY, 1);
			//std::cout << "lableInt " << labelBinary.size() << std::endl << labelBinary << std::endl;

			cv::threshold(labelBinary, lableInt, 0, i + 1, CV_THRESH_BINARY);
			//std::cout << labelsPath.at(i) <<"  :  " << i<< std::endl << lableInt << std::endl;
			outPut = outPut + lableInt;
			//std::cout << "outPut after " << "  :  " << i << std::endl << outPut << std::endl;
			//std::cout << " ------- " << i << " ------- " <<std::endl;

		}
		//std::cout << "Final label mat:" << std::endl<< outPut << std::endl<<std::endl;

		/*cv::Mat show = outPut.clone();

		cv::normalize(show, show, 255, 0, cv::NORM_MINMAX);
		cv::cvtColor(show, show, CV_GRAY2BGR, 3);
		cv::namedWindow("Labels", cv::WINDOW_NORMAL);
		cv::imshow("Labels", show);
		cv::waitKey();*/

		//std::cout << " -------- Labels mat created ---------- " << std::endl;
		/*	std::cout << (int)outPut.at<uchar>(44 , 101) << std::endl;
		std::cout << (int)outPut.at<uchar>(108, 226) << std::endl;
		std::cout << (int)outPut.at<uchar>(102, 390) << std::endl;
		std::cout << (int)outPut.at<uchar>(6080, 30) << std::endl;
		std::cout << (int)outPut.at<uchar>(96, 84) << std::endl;
		std::cout << " ------------------------ " << std::endl;*/


		//cv::Mat colored = showInPseudoColorFirst5Classes(outPut);

		//cv::namedWindow("LabelsColored", cv::WINDOW_NORMAL);
		//cv::imshow("LabelsColored", colored);
		//cv::waitKey(10);

		return outPut;

	}

	void generateLinesAndLabels(int numOfPoints, cv::Mat& image, int width, int height, std::string root_folder, std::vector<std::pair<std::string, int>>& lines) {

		std::random_device rdRows;     // only used once to initialise (seed) engine
		std::mt19937 rngRows(rdRows());    // random-number engine used (Mersenne-Twister in this case)
		std::uniform_int_distribution<int> uniRows(0, image.rows - 1); // guaranteed unbiased
		std::random_device rdCols;     // only used once to initialise (seed) engine
		std::mt19937 rngCols(rdCols());    // random-number engine used (Mersenne-Twister in this case)
		std::uniform_int_distribution<int> uniCols(0, image.cols - 1); // guaranteed unbiased
		cv::Mat labels = getLableMat(root_folder + "/labels_list.txt");
		std::vector<std::string> labelsPath = getLabelsPathes(root_folder + "/labels_list.txt");

		int numOfClasses = labelsPath.size();
		std::vector<std::map<std::string, int>> points(numOfClasses);
		std::vector<int> totalPoints(numOfClasses);
		std::vector<bool> fullPoints(numOfClasses);
		std::vector<bool> maxPointsPerClass(numOfClasses);
		std::map<std::string, int>::iterator it;
		bool bInsert = false;
		cv::Mat labelBinary;

		int labeledPoints = 0;
		int numOfAllointsTogether = 0;
		for (int i = 0; i < labelsPath.size(); i++) {
			labelBinary = cv::imread(labelsPath.at(i));
			cv::cvtColor(labelBinary, labelBinary, CV_BGR2GRAY, 1);
			labeledPoints = cv::countNonZero(labelBinary);
			if (labeledPoints > numOfPoints) {
				maxPointsPerClass.at(i) = numOfPoints;
				numOfAllointsTogether += numOfPoints;
			}
			else {
				maxPointsPerClass.at(i) = labeledPoints;
				numOfAllointsTogether += labeledPoints;
			}
		}

		int totalSum = 0;
		double min, max;
		cv::minMaxLoc(labels, &min, &max);
		std::cout << "Labels mat generated : contains labels from " << min << "  to " << max << std::endl;
		while (totalSum < numOfAllointsTogether) {

			auto row = uniRows(rngRows);
			auto col = uniCols(rngCols);

			std::string indexes = std::to_string(row) + "," + std::to_string(col);
			int label = (int)labels.at<uchar>(row, col);

			bInsert = false;

			if (label < 1) {
				continue;
			}

			it = points.at(label - 1).find(indexes);

			//This pixel is firstly seen
			if (it == points.at(label - 1).end()) {
				points.at(label - 1).insert(std::pair<std::string, int>(indexes, label - 1));
				lines.push_back(std::make_pair(indexes, (int)labels.at<uchar>(row, col) - 1));
				totalSum++;
			}
		}
	}


	/**
	 * parse database format images
	 *
	 * @param filename [in] filename of database(binary version)
	 * @param train_images [out] parsed images
	 * @param train_labels [out] parsed labels
	 * @param scale_min  [in]  min-value of output
	 * @param scale_max  [in]  max-value of output
	 * @param x_padding  [in]  adding border width (left,right)
	 * @param y_padding  [in]  adding border width (top,bottom)
	 **/
	void parse_rgb_db(int num_of_data_points, int radiusAroundMainPx, const std::string &dirname,
		std::vector<vec_t> *train_images,
		std::vector<label_t> *train_labels,
		float_t scale_min,
		float_t scale_max) {

		int IMAGE_RADIUS = radiusAroundMainPx;
		int IMAGE_DEPTH = 3;
		int IMAGE_WIDTH = IMAGE_RADIUS*2+1;
		int IMAGE_HEIGHT = IMAGE_RADIUS * 2 + 1;
		int IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT;
		int IMAGE_SIZE = IMAGE_AREA * IMAGE_DEPTH;
	
		if (scale_min >= scale_max)
			throw nn_error("scale_max must be greater than scale_min");


		cv::Mat  image = cv::imread(dirname+"/magnitude_image.png", CV_LOAD_IMAGE_COLOR);	
		std::vector<std::pair<std::string, int>> lines;

		generateLinesAndLabels(num_of_data_points, image, image.cols, image.rows, dirname, lines);
		std::vector<unsigned char> buf(IMAGE_SIZE);
		
		for (int i = 0; i < lines.size(); i++) {
			std::string indexes = lines[i].first;
			std::vector<std::string> strIndexes = splitCustomString<std::string>(indexes, ",");
			//cv::Mat cv_img = 
			getImageByIndex(buf, image, std::stoi(strIndexes.at(0)), std::stoi(strIndexes.at(1)), IMAGE_RADIUS);
			vec_t img;		
			std::transform(buf.begin(), buf.end(), std::back_inserter(img),
				[=](unsigned char c) {
				return scale_min + (scale_max - scale_min) * c / 255;
			});
			train_images->push_back(img);
			train_labels->push_back((uint8_t)lines[i].second);
		}
	}
}  // namespace tiny_dnn
