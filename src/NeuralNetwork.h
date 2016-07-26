#ifndef NeuralNetwork_H
#define NeuralNetwork_H

#include <vector>
#include <utility>
#include <armadillo>
#include <string>

class NeuralNetwork {
private:
  std::vector<int> neurons;
  std::vector<arma::vec > vecBiases;
  std::vector<arma::mat > matWeights;
  std::vector<std::pair<arma::vec, int> > vecMiniBatch;
  unsigned int fMiniBatchSize;
  int fCostFunction = kMSE;
  int fTrainingsSize = 0;
  int fTestingSize   = 0;
  double fLearningRate = 3;
  double fDecayRate = 1;
  const double fConstPrefactor;
  std::vector<std::vector<double> > fTrainingsdata;
  std::vector<int> fTrainingslabel;
  std::vector<std::vector<double> > fTestingdata;
  std::vector<int> fTestinglabel;


  arma::vec FeedForward(std::vector<double> input);
  int SoftMax(arma::vec vec);
  arma::vec ActivationFunction(arma::vec vec);
  arma::vec ActivationFunctionDerivative(arma::vec vec);
  arma::vec CostFunctionDerivative(arma::vec outputNN, arma::vec truevec, arma::vec zValues, int type);
  void SetInputVectorInMiniBatch(std::vector<double> input, int label);
  void Backprop(std::vector<arma::vec>& delta_nabla_b, std::vector<arma::mat>& delta_nabla_w,
                const arma::vec& input, int output);
  void LearnMiniBatches();


public:
  NeuralNetwork(std::vector<int> neurons, int CostFunction, int nTrainingsSize, double LearningRate = 0.5, double DecayRate = 0.0);
  virtual ~NeuralNetwork(){};

  // Setting data set, training it and using the trained network
  void SetTrainingsdata (std::vector<std::vector<double> > Trainingsdata);
  void SetTrainingslabel(std::vector<int> Trainingslabel) {fTrainingslabel = Trainingslabel;}
  void SetTestingdata   (std::vector<std::vector<double> > Testingdata);
  void SetTestinglabel  (std::vector<int> Testinglabel) {fTestinglabel = Testinglabel;}
  double LearnGivenData(int MiniBatchSize, int epochs, int TrainingsSize = -1, bool TestOnTrainingsData = false, bool TestOnTestingData = false);
  bool Evaluate(std::vector<double> input, int label);

  // File Handling
  bool WriteWeightsToFile(std::string);
  bool ReadWeightsFromFile(std::string);


  enum CostFunction{ kMSE, kCE};
};


#endif
