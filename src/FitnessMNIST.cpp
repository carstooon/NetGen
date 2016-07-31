#include "FitnessMNIST.h"
#include "NeuralNetwork.h"
#include "MNISTReader.h"

std::string FitnessMNIST::BinToDec(const std::string& sseq){ // Binary to decimal
  // int seq = std::stoi(sseq);
  int result = 0;
  int pow = 1;

  for (int i = sseq.size() - 1; i >= 0; --i, pow <<= 1){
    result += (sseq[i] - '0') * pow;
  }
  return std::to_string(result);
}

void FitnessMNIST::PrintGen(std::pair<std::string, double> i){
  std::string sLearningRate = BinToDec(i.first.substr(0,6));
  double LearningRate = 0.1 * std::stod(sLearningRate);
  std::string sDecayRate = BinToDec(i.first.substr(6,6));
  double DecayRate = 0.1 * std::stod(sDecayRate);
  std::string sNeuronsPerLayer = BinToDec(i.first.substr(12,6));
  int NeuronsPerLayer = std::stoi(sNeuronsPerLayer);

  std::cout << "Genom Parameter:\t" <<
  "LR = " << LearningRate << "\t" <<
  "DR = " << DecayRate << "\t" <<
  "NNeurons = " << NeuronsPerLayer << "\t" <<
  " F: " << i.second << std::endl;

}



double FitnessMNIST::CalculateFitness(std::string genome){
  double fitness = -1;


  int MaxSizeOfTrainingsSample = 60000;
  std::string sLearningRate = BinToDec(genome.substr(0,6));
  double LearningRate = 0.1 * std::stod(sLearningRate);
  std::string sDecayRate = BinToDec(genome.substr(6,6));
  double DecayRate = 0.1 * std::stod(sDecayRate);
  std::string sNeuronsPerLayer = BinToDec(genome.substr(12,6));
  int NeuronsPerLayer = std::stoi(sNeuronsPerLayer);
  std::vector<int> neuronsPerLayer = {784, NeuronsPerLayer, 10};


  NeuralNetwork net(neuronsPerLayer,
                    NeuralNetwork::CostFunction::kCE,
                    MaxSizeOfTrainingsSample,
                    LearningRate,
                    DecayRate);

  MNISTReader reader;
  reader.read_mnist("data/train-images.idx3-ubyte", true);
  reader.read_mnist_labels("data/train-labels.idx1-ubyte", true);
  reader.read_mnist("data/t10k-images.idx3-ubyte", false);
  reader.read_mnist_labels("data/t10k-labels.idx1-ubyte", false);

  net.SetTrainingsdata(reader.GetTrainingsDataSet());
  net.SetTrainingslabel(reader.GetTrainingsLabel());

  net.SetTestingdata(reader.GetTestingDataSet());
  net.SetTestinglabel(reader.GetTestingLabel());


  int mini_batch_size = 10;
  int epochs = 1;
  int size_of_trainings_sample = 600;
  bool Evaluate_performance_on_trainingsdata = true;
  bool Evaluate_performance_on_testdata = true;

  fitness = net.LearnGivenData(mini_batch_size, epochs, size_of_trainings_sample, Evaluate_performance_on_trainingsdata, Evaluate_performance_on_testdata);


  return fitness;
}
