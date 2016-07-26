#include "MNISTReader.h"

#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>

int MNISTReader::reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


void MNISTReader::read_mnist(std::string path, bool training)
{
  std::ifstream file (path);
  std::vector<std::vector<std::vector< double> > > pixels;
  if (file.is_open())
  {
    int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;

    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);

    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);

    if (training == true) fNumber_of_training_inputs = number_of_images;
    if (training == false) fNumber_of_testing_inputs = number_of_images;
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
    fNrows = n_rows;
    file.read((char*)&n_cols,sizeof(n_cols));
    n_cols= reverseInt(n_cols);
    fNcols = n_cols;

    pixels.resize(number_of_images);
    for (unsigned int j = 0; j < pixels.size(); ++j){
      pixels.at(j).resize(n_rows);
      for (unsigned int k = 0; k < pixels.at(j).size(); ++k){
        pixels.at(j).at(k).resize(n_cols);
      }
    }
    // pixels.resize(number_of_images);

    for(int i=0;i<number_of_images;++i)
    {
      for(int r=0;r<n_rows;++r)
      {
        for(int c=0;c<n_cols;++c)
        {
          unsigned char temp=0;
          file.read((char*)&temp,sizeof(temp));
          // std::cout << static_cast<double>(temp) << std::endl;
          if (static_cast<double>(temp) == 0) pixels.at(i).at(r).at(c) = 0.;
          else pixels.at(i).at(r).at(c) = static_cast<double>(temp) / 255.;
        }
      }
    }
  }
  if (training == true) fTrainingsdata = pixels;
  else fTestingdata = pixels;
}

void MNISTReader::read_mnist_labels(std::string full_path, bool training)
{
  auto reverseInt = [](int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  };
  int number_of_labels;
  if (training == true) number_of_labels = fNumber_of_training_inputs;
  if (training == false) number_of_labels = fNumber_of_testing_inputs;
  // std::cout << "number_of_labels = " << number_of_labels << std::endl;
  std::ifstream file(full_path);

  std::vector<double> dataset;
  if(file.is_open()) {
    int magic_number = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

    file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

    // uchar* _dataset = new uchar[number_of_labels];

    dataset.resize(number_of_labels);

    for(int i = 0; i < number_of_labels; i++) {
      unsigned char temp;
      file.read((char*)&temp, 1);
      dataset.at(i) = static_cast<double>(temp);
    }
  } else {
      throw std::runtime_error("Unable to open file `" + full_path + "`!");
  }
  // std::cout << "dataset.size() = " << dataset.size() << std::endl;
  if (training == true) fTrainingslabel = dataset;
  else fTestinglabel = dataset;
}

int MNISTReader::GetLabel(const unsigned int NumberOfLabel, bool training){
  if (training == true) {
    // std::cout << "Training" << std::endl;
    // std::cout << NumberOfLabel << std::endl;
    // std::cout << fTrainingslabel.size() << std::endl;
    return fTrainingslabel.at(NumberOfLabel);
  }
  else{
    // std::cout << "Testing" << std::endl;
    // std::cout << NumberOfLabel << std::endl;
    // std::cout << fTestinglabel.size() << std::endl;
    return fTestinglabel.at(NumberOfLabel);
  }
}

std::vector<double> MNISTReader::Get1DVector(const unsigned int NumberOfLabel, bool training){
  std::vector<std::vector<std::vector< double> > >* dataset;
  if (training == true) dataset = &fTrainingsdata;
  else dataset = &fTestingdata;

  std::vector<double> vec;
  vec.resize(fNcols * fNrows);
  // std::cout << dataset->at(NumberOfLabel).size() << std::endl;

  // for (unsigned int i = 0; i < dataset->at(NumberOfLabel).size(); ++i){
  //   for (unsigned int j = 0; j < dataset->at(NumberOfLabel)[i].size(); ++j){
  for (unsigned int i = 0; i < 28; ++i){
    for (unsigned int j = 0; j < 28; ++j){

      // double temp = dataset->at(NumberOfLabel).at(i).at(j);
      // std::cout << temp;
      // vec.at(i * dataset->at(NumberOfLabel).at(i).size() + j) = dataset->at(NumberOfLabel).at(i).at(j);;
      vec[i * dataset->at(NumberOfLabel)[i].size() + j] = dataset->at(NumberOfLabel)[i][j];
    }
  }
  // for (unsigned int i = 0; i < vec.size(); ++i) std::cout << vec.at(i);
  // std::cout << std::endl;
  return vec;
}


// Prints label of picture with number NumberOfLabel
void MNISTReader::PrintLabel(const unsigned int NumberOfLabel, bool training){
  std::vector<std::vector<std::vector< double> > > dataset;
  if (training == true) dataset = fTrainingsdata;
  else dataset = fTestingdata;

  if (NumberOfLabel > dataset.size()) return;
  if (dataset.size() == 0) {std::cout << "No trainingsdata loaded" << std::endl; return;}
  std::cout << "Label of picture " << NumberOfLabel << " = " << GetLabel(NumberOfLabel, training) << std::endl;
}

void MNISTReader::PrintPicture(const unsigned int NumberOfPicture, bool training){
  std::vector<std::vector<std::vector< double> > > dataset;
  if (training == true) dataset = fTrainingsdata;
  else dataset = fTestingdata;

  if (NumberOfPicture > dataset.size()) return;
  if (dataset.size() == 0) {std::cout << "No trainingsdata loaded" << std::endl; return;}
  for (unsigned int i = 0; i < dataset.at(NumberOfPicture).size(); ++i){
    for (unsigned int j = 0; j < dataset.at(NumberOfPicture).at(i).size(); ++j){
      if (dataset.at(NumberOfPicture).at(i).at(j) > 0) std::cout << "1";
      else std::cout << "0";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;

}

std::vector<std::vector<double> > MNISTReader::GetTrainingsDataSet() {
  std::vector<std::vector<double> > vec;
  vec.resize(fNumber_of_training_inputs);
  for (unsigned int k = 0; k < fNumber_of_training_inputs; ++k){
    vec[k].resize(28*28);
    for (unsigned int i = 0; i < 28; ++i){
      for (unsigned int j = 0; j < 28; ++j){
        // vec[i * fTrainingsdata.at(k)[i].size() + j] = fTrainingsdata.at(k)[i][j];
        vec[k][j + i * 28] = fTrainingsdata.at(k)[i][j];
      }
    }
  }
  return vec;
}
std::vector<std::vector<double> > MNISTReader::GetTestingDataSet() {
  std::vector<std::vector<double> > vec;
  vec.resize(fNumber_of_testing_inputs);
  for (unsigned int k = 0; k < fNumber_of_testing_inputs; ++k){
    vec[k].resize(28*28);
    for (unsigned int i = 0; i < 28; ++i){
      for (unsigned int j = 0; j < 28; ++j){
        // vec[i * fTrainingsdata.at(k)[i].size() + j] = fTrainingsdata.at(k)[i][j];
        vec[k][j + i * 28] = fTestingdata.at(k)[i][j];
      }
    }
  }
  return vec;
}

std::vector<int> MNISTReader::GetTrainingsLabel(){
  std::vector<int> vec;
  vec.resize(fNumber_of_training_inputs);
  for (unsigned int i = 0; i < fNumber_of_training_inputs; ++i){
    vec[i] = std::round(fTrainingslabel[i]);
  }
  return vec;
}
std::vector<int> MNISTReader::GetTestingLabel(){
  std::vector<int> vec;
  vec.resize(fNumber_of_testing_inputs);
  for (unsigned int i = 0; i < fNumber_of_testing_inputs; ++i){
    vec[i] = std::round(fTestinglabel[i]);
  }
  return vec;
}
