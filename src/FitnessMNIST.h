#ifndef FITNESSMNIST_H
#define FITNESSMNIST_H

#include <string>
#include "FitnessClass.h"

class FitnessMNIST : public FitnessClass {
private:
public:
  std::string BinToDec(const std::string& sseq);
  double CalculateFitness(std::string genome);
  void PrintGen(std::pair<std::string, double>);
  FitnessMNIST (){};
  virtual ~FitnessMNIST (){};

};

#endif
