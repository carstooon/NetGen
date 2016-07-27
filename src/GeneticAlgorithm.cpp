#include <iostream>
#include <utility>
#include <limits>

#include "MNISTReader.h"
#include "NeuralNetwork.h"

#include "GeneticAlgorithm.h"

GeneticAlgorithm::GeneticAlgorithm(const unsigned int SizeOfPopulation, unsigned int ChromosomLength, unsigned int NumberOfGenes) : fSizeOfPopulation(SizeOfPopulation){
  random1.seed(seed);
  for(unsigned int i = 0; i < SizeOfPopulation; ++i){
    std::string sTemp = CreateRandomChromosom(ChromosomLength, NumberOfGenes);
    double fitness = EvaluateFitness(sTemp);
    population.emplace_back(std::make_pair(sTemp, fitness));
  }
}

void GeneticAlgorithm::PrintPopulation(){
  double meanOfFitness = 0;
  for (auto& i: population){
    // std::cout << i.first << " " << i.second << "\n";
    meanOfFitness += i.second;

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

  std::cout << "Mean of Fitness in Population: " << meanOfFitness/fSizeOfPopulation << std::endl;
}

// Create random Chromosom with given length
std::string GeneticAlgorithm::CreateRandomChromosom(unsigned int ChromosomLength, unsigned int NumberOfGenes){
  std::string chromosom;
  for (unsigned int i = 0; i < ChromosomLength * NumberOfGenes; ++i){
    chromosom += std::to_string(distributionBinary(random1));
  }
  // std::cout << chromosom << std::endl;
  return chromosom;
}

std::string GeneticAlgorithm::Crossover(std::string partner1, std::string partner2){
  std::string child("");
  child.resize(partner1.size());
  const unsigned int CutPosition = 6;
  for (unsigned int i = 0; i < partner1.size(); ++i){
    if (i < CutPosition) child[i] = partner1[i];
    else child[i] = partner2[i];
  }
  return child;
}

void GeneticAlgorithm::Mutation(std::string& chromosome){
  double MutationRate = 0.1;
  for (unsigned int i = 0; i < chromosome.size(); ++i){
    if (MutationRate > distributionFlat(random1)){
      if (chromosome[i] == '1') chromosome[i] = '0';
      else if (chromosome[i] == '0') chromosome[i] = '1';
      else std::cout << "You fucked up" << std::endl;
    }
  }

}

int GeneticAlgorithm::RouletteSelection(std::vector<std::pair<std::string, double> > population){
  std::vector<double> weights;
  for (auto i: population){
    // std::cout << i.second << std::endl;
    weights.push_back(i.second);
  }

  std::discrete_distribution<int> distribution {weights.begin(), weights.end()};
  return distribution(random1);
}

void GeneticAlgorithm::MakeNewChild(){

  int x = RouletteSelection(population);
  int y = x;
  while (y == x) y = RouletteSelection(population);

  std::string partner1 = population[x].first;
  std::string partner2 = population[y].first;

  std::string child = Crossover(partner1, partner2);
  // Mutation(child);
  double fitness = EvaluateFitness(child);

  std::string sLearningRate = BinToDec(child.substr(0,6));
  double LearningRate = 0.1 * std::stod(sLearningRate);
  std::string sDecayRate = BinToDec(child.substr(6,6));
  double DecayRate = 0.1 * std::stod(sDecayRate);
  std::string sNeuronsPerLayer = BinToDec(child.substr(12,6));
  int NeuronsPerLayer = std::stoi(sNeuronsPerLayer);

  std::cout << "Genom Parameter:\n" <<
               "LearningRate = " << LearningRate << "\t" <<
               "DecayRate = " << DecayRate << "\t" <<
               "NumberOfNeurons = " << NeuronsPerLayer << "\n" <<
               " Fitness: " << fitness << std::endl;

  int IndexOfWeakest = GetWeakling(population);
  population[IndexOfWeakest] = std::make_pair(child, fitness);

  return;
}

void GeneticAlgorithm::MakeNewGeneration(){
  for (unsigned int i = 0; i < fSizeOfPopulation; ++i){
    MakeNewChild();
  }
  return;
}

int GeneticAlgorithm::GetWeakling(std::vector<std::pair<std::string, double> > temp){
  int index = -2;
  double value = std::numeric_limits<double>::infinity();
  for (unsigned int i = 0; i < temp.size(); ++i){
    if (temp[i].second < value) {
      value = temp[i].second;
      index = i;
    }
  }
  return index;
}

std::string GeneticAlgorithm::BinToDec(const std::string& sseq){ // Binary to decimal
  // int seq = std::stoi(sseq);
  int result = 0;
  int pow = 1;

  for (int i = sseq.size() - 1; i >= 0; --i, pow <<= 1){
    result += (sseq[i] - '0') * pow;
  }
  return std::to_string(result);
}

double GeneticAlgorithm::EvaluateFitness(std::string newChild){
  double fitness = -1;

  std::string sLearningRate = BinToDec(newChild.substr(0,6));
  double LearningRate = 0.1 * std::stod(sLearningRate);
  std::string sDecayRate = BinToDec(newChild.substr(6,6));
  double DecayRate = 0.1 * std::stod(sDecayRate);

  std::string sNeuronsPerLayer = BinToDec(newChild.substr(12,6));
  int NeuronsPerLayer = std::stoi(sNeuronsPerLayer);

  MNISTReader reader;
  reader.read_mnist("data/train-images.idx3-ubyte", true);
  reader.read_mnist_labels("data/train-labels.idx1-ubyte", true);
  reader.read_mnist("data/t10k-images.idx3-ubyte", false);
  reader.read_mnist_labels("data/t10k-labels.idx1-ubyte", false);


  std::vector<int> neuronsPerLayer = {784, NeuronsPerLayer, 10};
  int MaxSizeOfTrainingsSample = 60000;


  int epochs = 1;
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

  fitness = net.LearnGivenData(10, epochs, size_of_trainings_sample, Evaluate_performance_on_trainingsdata, Evaluate_performance_on_testdata);


  return fitness;
}


GeneticAlgorithm::~GeneticAlgorithm(){};

// 000000       000000          000000
// DecayRate    LearningRate    NumberOfNeurons
//
// DecayRate
// LearningRate
// NeuronsPerLayer {12}
