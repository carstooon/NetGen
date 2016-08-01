/*
Generic and generalized class for using genetic algorithms with strings as
genoms. Specific problems have to be formulated in a class which inherits
from FitnessClass.
*/


#ifndef GENETICALGORITHM_H

#include <string>
#include <vector>
#include <random>
#include <utility>
#include "TH1F.h"
#include "TCanvas.h"

class FitnessClass;

class GeneticAlgorithm {
private:
  unsigned int fSizeOfPopulation;
  // Random Number Generators
  unsigned int seed = 112;
  std::mt19937_64 random1;
  std::discrete_distribution<int> distributionBinary {1,1};
  std::uniform_real_distribution<double> distributionFlat {0., 1.};

  // Population
  std::string CreateRandomChromosom(unsigned int ChromosomLength, unsigned int NumberOfGenes);

  void        MakeNewChild(int genInGeneration);
  std::string Crossover(std::string partner1, std::string partner2);
  void        Mutation(std::string& chromosome);
  double      EvaluateFitness(std::string newChild);
  int         RouletteSelection(std::vector<std::pair<std::string, double> > weights);
  int         GetWeakling(std::vector<std::pair<std::string, double> >);

  double      PrintPopulation();

  // private member variables
  std::vector<std::pair<std::string, double> > population;
  double      fMutationRate = 0.1; // Chance to flip one bit in a genom
  int         fCurrentGeneration = 1;
  int         fMaxNumberOfGenerations;
  TCanvas     fCanvas;
  TH1F        fFitnessPerGeneration;
  FitnessClass* fFitnessClass; // Class which specifies Problem and inherits from FitnessClass.h
public:
  void        MakeNewGeneration();
  void        SaveFitnessHistogram(std::string filename = "fitness.pdf");
  // void        SetNumberOfGenerations(const int i){fMaxNumberOfGenerations = i;}


  GeneticAlgorithm (unsigned int SizeOfPopulation,
                    unsigned int ChromosomLength,
                    unsigned int NumberOfGenes,
                    int maxNumberofGenerations,
                    FitnessClass* fitness);
  virtual ~GeneticAlgorithm ();

};

#define GENETICALGORITHM_H
#endif
