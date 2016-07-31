#include <iostream>
#include <utility>
#include <limits>

#include "MNISTReader.h"
#include "NeuralNetwork.h"
#include "FitnessClass.h"
#include "GeneticAlgorithm.h"

GeneticAlgorithm::GeneticAlgorithm(const unsigned int SizeOfPopulation,
                                   unsigned int ChromosomLength,
                                   unsigned int NumberOfGenes,
                                   int MaxNumberOfGenerations,
                                   FitnessClass* fitness)
                                   :
                                   fSizeOfPopulation(SizeOfPopulation),
                                   fMaxNumberOfGenerations(MaxNumberOfGenerations),
                                   fCanvas{"Canvas","", 800, 600},
                                   fFitnessPerGeneration{"fitness", "fitness", fMaxNumberOfGenerations, 0., 100.},
                                   fFitnessClass(fitness)
{
  std::cout << "First generation:" << std::endl;
  random1.seed(seed);
  double SumOfFitness = 0;
  for(unsigned int i = 0; i < SizeOfPopulation; ++i){
    std::string sTemp = CreateRandomChromosom(ChromosomLength, NumberOfGenes);
    double fitness = fFitnessClass->CalculateFitness(sTemp);

    SumOfFitness += fitness;

    population.emplace_back(std::make_pair(sTemp, fitness));
    fFitnessClass->PrintGen(population[i]);
  }
  double MeanOfFitness = SumOfFitness/SizeOfPopulation;
  fFitnessPerGeneration.SetBinContent(fCurrentGeneration, MeanOfFitness);
}

double GeneticAlgorithm::PrintPopulation(){
  std::cout << "Generation Nr." << fCurrentGeneration << std::endl;
  double meanOfFitness = 0;
  for (auto& i: population){
    meanOfFitness += i.second;

    fFitnessClass->PrintGen(i);
  }

  std::cout << "Mean of Fitness in Population: " << meanOfFitness/fSizeOfPopulation << std::endl;
  return meanOfFitness/fSizeOfPopulation;
  std::cout << "===================================" << std::endl;
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
  Mutation(child);

  double fitness = fFitnessClass->CalculateFitness(child);

  int IndexOfWeakest = GetWeakling(population);
  population[IndexOfWeakest] = std::make_pair(child, fitness);

  // fFitnessClass->PrintGen(population[IndexOfWeakest]);
  return;
}


void GeneticAlgorithm::MakeNewGeneration(){
  fCurrentGeneration++;
  for (unsigned int i = 0; i < fSizeOfPopulation; ++i){
    MakeNewChild();
  }
  double MeanOfFitness = PrintPopulation();
  fFitnessPerGeneration.SetBinContent(fCurrentGeneration, MeanOfFitness);
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


double GeneticAlgorithm::EvaluateFitness(std::string newChild){
  return fFitnessClass->CalculateFitness(newChild);
}


void GeneticAlgorithm::SaveFitnessHistogram(std::string filename){
  fFitnessPerGeneration.SetAxisRange(0.5, 1., "Y");
  fFitnessPerGeneration.SetTitle("Fitness per Generation;#Generation;Fitness");
  fFitnessPerGeneration.SetStats(false);
  fFitnessPerGeneration.SetMarkerStyle(20);
  fFitnessPerGeneration.SetMarkerColor(kAzure-2);
  fFitnessPerGeneration.SetMarkerSize(2.);
  fFitnessPerGeneration.Draw("p");

  fCanvas.SetGrid();
  fCanvas.SaveAs(filename.c_str());
}



GeneticAlgorithm::~GeneticAlgorithm(){};

// 000000       000000          000000
// DecayRate    LearningRate    NumberOfNeurons
//
// DecayRate
// LearningRate
// NeuronsPerLayer {12}
