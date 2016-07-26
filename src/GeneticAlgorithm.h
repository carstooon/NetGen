#ifndef GENETICALGORITHM_H

#include <string>
#include <vector>
#include <random>
#include <utility>

// unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();


class GeneticAlgorithm {
private:
  unsigned int fSizeOfPopulation;
  // Random Number Generators
  unsigned int seed = 112;
  std::mt19937_64 random1;
  std::discrete_distribution<int> distributionBinary {1,1};
  std::uniform_real_distribution<double> distributionFlat {0., 1.};

  // Population
  std::vector<std::pair<std::string, double> > population;
  std::string CreateRandomChromosom(unsigned int ChromosomLength, unsigned int NumberOfGenes);
  std::string Crossover(std::string partner1, std::string partner2);
  void        Mutation(std::string& chromosome);
  int         RouletteSelection(std::vector<std::pair<std::string, double> > weights);
  double      EvaluateFitness(std::string newChild);
  int         GetWeakling(std::vector<std::pair<std::string, double> >);
  std::string BinToDec(const std::string& sseq);
public:
  void        MakeNewChild();
  void        MakeNewGeneration();


  void PrintPopulation();
  GeneticAlgorithm (unsigned int SizeOfPopulation, unsigned int ChromosomLength, unsigned int NumberOfGenes);
  virtual ~GeneticAlgorithm ();

};

#define GENETICALGORITHM_H
#endif
