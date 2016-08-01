#include <iostream>
#include <utility>
#include <limits>

#include "MNISTReader.h"
#include "NeuralNetwork.h"
#include "FitnessClass.h"
#include "GeneticAlgorithm.h"
#include "TLegend.h"

GeneticAlgorithm::GeneticAlgorithm(const unsigned int SizeOfPopulation,
                                   unsigned int ChromosomLength,
                                   unsigned int NumberOfGenes,
                                   int MaxNumberOfGenerations,
                                   FitnessClass* fitness)
                                   :
                                   fSizeOfPopulation(SizeOfPopulation),
                                   fMaxNumberOfGenerations(MaxNumberOfGenerations),
                                   fCanvas{"Canvas","", 800, 600},
                                   fFitnessPerGeneration{"fitness", "fitness",
                                                         fMaxNumberOfGenerations + 1, 0.5,
                                                         static_cast<double>(fMaxNumberOfGenerations+1.5)},
                                   fFitnessClass(fitness)
{
  std::cout << "First generation:" << std::endl;
  random1.seed(seed);
  double SumOfFitness = 0;
  for(unsigned int i = 0; i < SizeOfPopulation; ++i){
    std::string sTemp = CreateRandomChromosom(ChromosomLength, NumberOfGenes);
    double fitness = fFitnessClass->CalculateFitness(sTemp, fCurrentGeneration, i);

    SumOfFitness += fitness;

    population.emplace_back(std::make_pair(sTemp, fitness));
    fFitnessClass->PrintGen(population[i]);

  }
  double MeanOfFitness = SumOfFitness/SizeOfPopulation;
  fFitnessPerGeneration.SetBinContent(fCurrentGeneration, MeanOfFitness);
  std::cout << "Mean of Fitness in Population: " << MeanOfFitness << std::endl;
}

double GeneticAlgorithm::PrintPopulation(){
  std::cout << "Generation Nr." << fCurrentGeneration << std::endl;
  double meanOfFitness = 0;
  for (auto& i: population){
    meanOfFitness += i.second;

    fFitnessClass->PrintGen(i);
  }

  std::cout << "Mean of Fitness in Population: " << meanOfFitness/fSizeOfPopulation << std::endl;
  std::cout << "===================================" << std::endl;
  return meanOfFitness/fSizeOfPopulation;
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
  for (unsigned int i = 0; i < partner1.size(); ++i){
    if (i < fFitnessClass->CutPositionInGenom) child[i] = partner1[i];
    else child[i] = partner2[i];
  }
  return child;
}


void GeneticAlgorithm::Mutation(std::string& chromosome){
  for (unsigned int i = 0; i < chromosome.size(); ++i){
    if (fMutationRate > distributionFlat(random1)){
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

void GeneticAlgorithm::MakeNewChild(int genPerGeneration){

  int x = RouletteSelection(population);
  int y = x;
  while (y == x) y = RouletteSelection(population);

  std::string partner1 = population[x].first;
  std::string partner2 = population[y].first;

  std::string child = Crossover(partner1, partner2);
  Mutation(child);

  double fitness = fFitnessClass->CalculateFitness(child, fCurrentGeneration, genPerGeneration);

  int IndexOfWeakest = GetWeakling(population);
  population[IndexOfWeakest] = std::make_pair(child, fitness);

  return;
}


void GeneticAlgorithm::MakeNewGeneration(){
  fCurrentGeneration++;
  for (unsigned int i = 0; i < fSizeOfPopulation; ++i){
    MakeNewChild(i);
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
  fFitnessPerGeneration.SetAxisRange(0.0, 1., "Y");
  fFitnessPerGeneration.SetTitle("Fitness per Generation;#Generation;Fitness");
  fFitnessPerGeneration.SetStats(false);
  fFitnessPerGeneration.SetNdivisions(fMaxNumberOfGenerations+2);
  fFitnessPerGeneration.SetMarkerStyle(20);
  fFitnessPerGeneration.SetMarkerColor(kAzure-2);
  fFitnessPerGeneration.SetMarkerSize(2.);
  fCanvas.SetGrid();
  fFitnessPerGeneration.Draw("TEXT p");

  // double FitnessMaximum = fFitnessPerGeneration.GetMaximum();
  TLegend leg(0.6,0.1,0.9,0.2);
  // leg.SetHeader("");
  leg.AddEntry(&fFitnessPerGeneration,  Form("Fitness Maximum: %4.3f)", fFitnessPerGeneration.GetBinContent(fFitnessPerGeneration.GetMaximumBin())), "p");
  // leg.AddEntry(&fFitnessPerGeneration,  Form("Fitness Maximumbum: %d)", fFitnessPerGeneration.GetMaximumBin()), "p");
  leg.Draw("same");

  fCanvas.SaveAs(filename.c_str());
}



GeneticAlgorithm::~GeneticAlgorithm(){};

// 000000       000000          000000
// DecayRate    LearningRate    NumberOfNeurons
//
// DecayRate
// LearningRate
// NeuronsPerLayer {12}
