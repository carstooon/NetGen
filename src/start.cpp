#include <armadillo>
#include <iostream>
#include <vector>
#include <utility>
#include <string>

#include "MNISTReader.h"
#include "NeuralNetwork.h"
#include "GeneticAlgorithm.h"

void MNIST_NN();
void GeneticTest();



int main(){
  // MNIST_NN();
  GeneticTest();
  return 0;
}

void GeneticTest(){
  GeneticAlgorithm gen(5, 6, 3);
  gen.PrintPopulation();
  gen.MakeNewGeneration();
  gen.PrintPopulation();
  // gen.MakeNewChild();
  // std::cout << gen.CreateRandomChromosom(6, 4) << std::endl;
}


void MNIST_NN(){
  // // Number of layers 3
  // // fashion 784 (input) * 20 (hidden) * 10 (output)
  // // To be set by user
  MNISTReader reader;
  reader.read_mnist("data/train-images.idx3-ubyte", true);
  reader.read_mnist_labels("data/train-labels.idx1-ubyte", true);
  reader.read_mnist("data/t10k-images.idx3-ubyte", false);
  reader.read_mnist_labels("data/t10k-labels.idx1-ubyte", false);



  std::vector<int> neuronsPerLayer = {784, 100, 100, 10};
  int MaxSizeOfTrainingsSample = 60000;
  double LearningRate = 1.0;
  double DecayRate = 2.0;


  int epochs = 10;
  int size_of_trainings_sample = 60000;
  bool Evaluate_performance_on_trainingsdata = true;
  bool Evaluate_performance_on_testdata = true;


  NeuralNetwork net(neuronsPerLayer,
                    NeuralNetwork::CostFunction::kCE,
                    MaxSizeOfTrainingsSample,
                    LearningRate,
                    DecayRate);


  net.SetTrainingsdata(reader.GetTrainingsDataSet());
  net.SetTrainingslabel(reader.GetTrainingsLabel());

  net.SetTestingdata(reader.GetTestingDataSet());
  net.SetTestinglabel(reader.GetTestingLabel());

  net.LearnGivenData(10, epochs, size_of_trainings_sample, Evaluate_performance_on_trainingsdata, Evaluate_performance_on_testdata);



  // SIMPLE LOGIC TRAINER !!!
  // std::vector<int> neuronsPerLayer = {2, 10, 2};
  // NeuralNetwork net(neuronsPerLayer);
  //
  // // net.SetInputVectorInMiniBatch(std::vector<double>{0,0}, 0);
  // // net.SetInputVectorInMiniBatch(std::vector<double>{1,0}, 0);
  // // net.SetInputVectorInMiniBatch(std::vector<double>{0,1}, 0);
  // // net.SetInputVectorInMiniBatch(std::vector<double>{1,1}, 1);
  //
  // for (int i = 0; i < 100000; ++i){
  //   // std::cout << i << "-te Minibatch Learning" << std::endl;
  //
  //   net.SetInputVectorInMiniBatch(std::vector<double>{0,0}, 0);
  //   net.SetInputVectorInMiniBatch(std::vector<double>{1,0}, 0);
  //   net.SetInputVectorInMiniBatch(std::vector<double>{0,1}, 0);
  //   net.SetInputVectorInMiniBatch(std::vector<double>{1,1}, 1);
  //   net.LearnMiniBatches();
  // }
  // int sum = 0;
  // if (net.Evaluate(std::vector<double>{0,0},  0) == true) sum++;
  // if (net.Evaluate(std::vector<double>{1,0}, 0) == true) sum++;
  // if (net.Evaluate(std::vector<double>{0,1}, 0) == true) sum++;
  // if (net.Evaluate(std::vector<double>{1,1}, 1) == true) sum++;
  // std::cout << "sum = " << sum << std::endl;
  // END SIMPLE LOGIC TRAINER

  // std::string filename = "weights.dat";
  // net.WriteWeightsToFile(filename);
  // net.ReadWeightsFromFile("weights.dat");
  // std::cout << std::endl;

}
