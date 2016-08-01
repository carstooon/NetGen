#ifndef FITNESSMNIST_H
#define FITNESSMNIST_H

#include <string>
#include "FitnessClass.h"

class FitnessMNIST : public FitnessClass {
private:
public:
  std::string BinToDec(const std::string& sseq);
  double CalculateFitness(std::string genome, int currentGeneration=0, int genNumber = 0);
  void PrintGen(std::pair<std::string, double>);
  FitnessMNIST (){};
  virtual ~FitnessMNIST (){};

  unsigned int CutPositionInGenom = 6;

};

#endif
