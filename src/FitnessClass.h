#ifndef FITNESSCLASS_H
#define FITNESSCLASS_H

#include <string>

/*
This is a purely virtual class. All the methods have to be reimplemented.
Have a look at 'FitnessMNIST.h' as an example.
*/

class FitnessClass {
private:
public:
  virtual std::string BinToDec(const std::string& sseq) = 0;
  virtual double CalculateFitness(std::string genome, int currentGeneration = 0, int genNumber = 0) = 0;
  virtual void PrintGen(std::pair<std::string, double>) = 0;
  FitnessClass (){};
  virtual ~FitnessClass (){};

  unsigned int CutPositionInGenom;

};

#endif
