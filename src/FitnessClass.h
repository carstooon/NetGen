#ifndef FITNESSCLASS_H
#define FITNESSCLASS_H

#include <string>


class FitnessClass {
private:
public:
  virtual std::string BinToDec(const std::string& sseq) = 0;
  virtual double CalculateFitness(std::string genome) = 0;
  virtual void PrintGen(std::pair<std::string, double>) = 0;
  FitnessClass (){};
  virtual ~FitnessClass (){};

};

#endif
