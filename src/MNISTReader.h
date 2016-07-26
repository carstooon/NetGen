#ifndef MNISTREADER_H
#define MNISTREADER_H

#include <string>
#include <vector>

class MNISTReader{
private:
  static int reverseInt (int i);
  unsigned int fNumber_of_training_inputs;
  unsigned int fNumber_of_testing_inputs;
  unsigned int fNrows;
  unsigned int fNcols;
  std::vector<std::vector<std::vector< double> > > fTrainingsdata;
  std::vector<double> fTrainingslabel;
  std::vector<std::vector<std::vector< double> > > fTestingdata;
  std::vector<double> fTestinglabel;

public:
  MNISTReader(){};
  ~MNISTReader(){};
  void read_mnist(std::string path, bool training = true);
  void read_mnist_labels(std::string path, bool training = true);

  int GetLabel(const unsigned int NumberOfLabel, bool training = true);
  std::vector<double> Get1DVector(const unsigned int NumberOfLabel, bool training = true);

  void PrintLabel(const unsigned int NumberOfLabel, bool training = true);
  void PrintPicture(const unsigned int NumberOfPicture, bool training = true);

  std::vector<std::vector<double> > GetTrainingsDataSet();
  std::vector<std::vector<double> > GetTestingDataSet();
  std::vector<int> GetTrainingsLabel();
  std::vector<int> GetTestingLabel();
};

#endif
