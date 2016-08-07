#include <iostream>
#include <cmath>
#include <fstream>

#include "NeuralNetwork.h"
#include "MNISTReader.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TLegend.h"

NeuralNetwork::NeuralNetwork(std::vector<int> inputneurons, int type, int TrainingsSize,
                             double LearningRate, double DecayRate)
                            : fMiniBatchSize(10), fCostFunction(type), fTrainingsSize(TrainingsSize),
                              fLearningRate(LearningRate), fDecayRate(DecayRate),
                              fConstPrefactor(1 - fLearningRate * fDecayRate / fTrainingsSize){
  // Default initializing the vectors and matrices
  // vecActivations.resize(neurons.size());
  neurons = inputneurons;
  vecBiases.resize(neurons.size() - 1);
  // vecMiniBatch.resize(fMiniBatchSize);
  matWeights.resize(neurons.size() - 1); // 1 layer fewer weights
  for (unsigned int i = 0; i < neurons.size() - 1; ++i){
    vecBiases.at(i).randn(neurons.at(i + 1));
    matWeights.at(i).randn(neurons.at(i+1), neurons.at(i));
    // std::cout << "START: vecBiases.at(" << i << ") \n " << vecBiases.at(i) << std::endl;
    // std::cout << "START: matWeights.at(" << i << ") = \n" << matWeights.at(i) << std::endl;
  }
  // std::cout << fConstPrefactor << std::endl;
}


// ################################################
// Adds input and desired output to a mini batch vector
void NeuralNetwork::SetInputVectorInMiniBatch(std::vector<double> input, int label){
  vecMiniBatch.push_back(std::make_pair(arma::vec(input), label));
  return;
}


// ################################################
// Gives true when NN output is the same as label
// bool NeuralNetwork::Evaluate(int inputNr, MNISTReader reader){
bool NeuralNetwork::Evaluate(std::vector<double> input, int label){
  // std::cout << "Evaluation" << std::endl;
  arma::vec temp = FeedForward(input);
  // std::cout << temp;
  int counter = -2;
  double max = -1;
  for (unsigned int i = 0; i < temp.size(); ++i){
    if (temp.at(i) > max){
      max = temp.at(i);
      counter = i;
    }
  }
  // const int element = std::round(FeedForward(input).);
  if (counter == label) {
    // std::cout << "Correctly identified! Output element Nr " << counter << std::endl;
    return true;
  }
  else {
    // std::cout << "Wrongly identified! Output element Nr " << counter << std::endl;
    return false;
  }
}

// ################################################
// Feed forwards input and returns output of neural network
arma::vec NeuralNetwork::FeedForward(std::vector<double> input){
  std::vector<arma::vec > vecActivations;
  vecActivations.resize(neurons.size());
  vecActivations.at(0) = input;

  for (unsigned int i = 0; i < vecActivations.size()-1; ++i){
    vecActivations.at(i+1) = ActivationFunction((matWeights.at(i) * vecActivations.at(i)) + vecBiases.at(i));
  }
  return vecActivations.at(vecActivations.size()-1); // return last vector
}


// ################################################
arma::vec NeuralNetwork::ActivationFunction(arma::vec vec){
  vec.for_each( [](arma::mat::elem_type& val) {
                  val = 1. / (1. + exp(-val));
              });
  return vec;
}


// ################################################
arma::vec NeuralNetwork::ActivationFunctionDerivative(arma::vec vec){
  arma::vec ones;
  ones.ones(vec.n_rows);
  vec = ActivationFunction(vec) % (ones - ActivationFunction(vec));
  return vec;
}


// ################################################
arma::vec NeuralNetwork::CostFunctionDerivative(arma::vec outputNN, arma::vec truevec, arma::vec zValues){
  if (fCostFunction == kCE) return (outputNN - truevec);
  if (fCostFunction == kMSE) return (outputNN - truevec) % ActivationFunctionDerivative(zValues);
  else { std::cout <<  "NO COST FUNCTION SET; Using CE" << std::endl; return (outputNN - truevec);}
}


// ################################################
// Takes learned minibatch and does the learning
void NeuralNetwork::LearnMiniBatches(){
  std::vector<arma::vec > vecNablaBiases;
  std::vector<arma::mat > matNablaWeights;
  std::vector<arma::vec > delta_nabla_b;
  std::vector<arma::mat > delta_nabla_w;

  vecNablaBiases.resize(matWeights.size());
  matNablaWeights.resize(matWeights.size());
  delta_nabla_b.resize(matWeights.size());
  delta_nabla_w.resize(matWeights.size());


  for (unsigned int i = 0; i < matWeights.size(); ++i){
    vecNablaBiases[i] = arma::vec(vecBiases.at(i).n_rows, arma::fill::zeros);
    delta_nabla_b[i] = arma::vec(vecBiases.at(i).n_rows, arma::fill::zeros);
    matNablaWeights[i] = arma::mat(matWeights.at(i).n_rows, matWeights.at(i).n_cols, arma::fill::zeros);
    delta_nabla_w[i] = arma::mat(matWeights.at(i).n_rows, matWeights.at(i).n_cols, arma::fill::zeros);
  }
  for (unsigned int k = 0; k < vecMiniBatch.size(); ++k){
    // Calculate the delta b and w by minimizing the cost function
    Backprop(delta_nabla_b, delta_nabla_w, vecMiniBatch.at(k).first, vecMiniBatch.at(k).second);

    // Update the vectors according to the calculated deltas
    for (unsigned int i = 0; i < delta_nabla_b.size(); ++i){
      vecNablaBiases.at(i) += delta_nabla_b.at(i);
      matNablaWeights.at(i) += delta_nabla_w.at(i);
    }
  }
  for (unsigned int i = 0; i < matWeights.size(); ++i){
    matWeights.at(i) = fConstPrefactor * matWeights.at(i) - (fLearningRate / vecMiniBatch.size()) * matNablaWeights.at(i);
    vecBiases.at(i) = vecBiases.at(i) - (fLearningRate / vecMiniBatch.size()) * vecNablaBiases.at(i);
  }
  // std::cout << "Finish learning this batch" << std::endl;
  vecMiniBatch.clear();
  return;
}


// ################################################
void NeuralNetwork::Backprop(std::vector<arma::vec>& delta_nabla_b, // is to be changed
                             std::vector<arma::mat>& delta_nabla_w, // is to be changed
                             const arma::vec& input, int output){

  std::vector<arma::vec> activations;
  activations.push_back(input);
  std::vector<arma::vec> zValues;

  for (unsigned int i = 0; i < vecBiases.size(); ++i){
    arma::vec z = matWeights[i] * activations[i] + vecBiases[i];
    zValues.push_back(z);
    activations.push_back(ActivationFunction(z));
  }

  arma::vec trueoutput;
  trueoutput.zeros(neurons[neurons.size()-1]);
  trueoutput[output] = 1;

  arma::vec delta = CostFunctionDerivative(activations[activations.size()-1], trueoutput, zValues[zValues.size()-1]);

  delta_nabla_b[delta_nabla_b.size()-1] = delta;
  delta_nabla_w[delta_nabla_w.size()-1] = delta * activations[activations.size() - 2].t();

  for (unsigned int i = 2; i < neurons.size(); ++i){
    arma::vec z = zValues[zValues.size() - i];
    arma::vec sp = ActivationFunctionDerivative(z);
    delta = matWeights[matWeights.size()-i+1].t() * delta % sp;
    delta_nabla_b[delta_nabla_b.size()-i] = delta;
    delta_nabla_w[delta_nabla_w.size()-i] = delta * activations[activations.size()-i-1].t();
  }
  return;
}


// ################################################
void NeuralNetwork::SetTrainingsdata (std::vector<std::vector<double> > Trainingsdata) {
  fTrainingsdata = Trainingsdata;
  fTrainingsSize = fTrainingsdata.size();
}
void NeuralNetwork::SetTestingdata   (std::vector<std::vector<double> > Testingdata) {
  fTestingdata = Testingdata;
  fTestingSize = fTestingdata.size();
}


// ################################################
double NeuralNetwork::LearnGivenData(int MiniBatchSize, int epochs, int TrainingsSize, bool TestOnTrainingsData, bool TestOnTestingData, std::string SavePath){

  TCanvas c1("Accuracy", "Accuracy", 600, 500);
  TCanvas c2("Cost", "Cost", 600, 500);
  TCanvas c3("summary", "summary", 1200, 500);
  c3.SetGrid();
  c3.Divide(2);
  c1.cd();
  TH1F axis("axis", "Accuracy of Neural Network;epochs", epochs, 0.5, epochs+0.5);
  TH1F acc_test ("acc_test" , "Classification accuracy", epochs, 0.5, epochs+0.5);
  TH1F acc_train("acc_train", "Classification accuracy", epochs, 0.5, epochs+0.5);
  TH1F axis2("axis2", "Cost Function of Neural Network;epochs", epochs, 0.5, epochs+0.5);
  TH1F cost_test ("cost_test" , "Cost Function", epochs, 0.5, epochs+0.5);
  TH1F cost_train("cost_train", "Cost Function", epochs, 0.5, epochs+0.5);
  if (TrainingsSize == -1) TrainingsSize = fTrainingsSize;

  for (int k = 0; k < epochs; ++k){
    // if (k % 5 == 0) std::cout << "Epoch nr. " << k << " started!" << std::endl;
    for (int i = 0; i < TrainingsSize / MiniBatchSize; ++i){
      for (int j = 0; j < MiniBatchSize; ++j){
        SetInputVectorInMiniBatch(fTrainingsdata[i*10 + j], fTrainingslabel[i*10 + j]);
      }
      LearnMiniBatches();
    }
    if (TestOnTestingData){
      double correct = 0;
      double cost = 0;
      for (int i = 0; i < fTestingSize; ++i){
        if (Evaluate(fTestingdata[i], fTestinglabel[i]) == true) correct++;
        cost += EvaluateCost(fTestingdata[i], fTestinglabel[i]);
      }
      // std::cout << "Accuracy of test data: " << correct / fTestingSize << " %"<< std::endl;
      acc_test.SetBinContent(k+1, correct / fTestingSize);
      cost_test.SetBinContent(k+1, cost / fTestingSize);
    }
    if (TestOnTrainingsData){
      double correct = 0;
      double cost = 0;
      for (int i = 0; i < TrainingsSize; ++i){
        if (Evaluate(fTrainingsdata[i], fTrainingslabel[i]) == true) correct++;
        cost += EvaluateCost(fTrainingsdata[i], fTrainingslabel[i]);
      }
      // std::cout << "Accuracy of trainings data: " << correct / TrainingsSize << " %" << std::endl;
      acc_train.SetBinContent(k+1, correct / TrainingsSize);
      cost_train.SetBinContent(k+1, cost / fTrainingsSize);
    }
  }
  axis.SetStats(false);
  axis.GetYaxis()->SetNdivisions(524);
  axis.SetAxisRange(0.5, 1.1, "Y");
  axis.Draw("");

  double markersize = 1;
  if (epochs > 100) {
    markersize = 0.5;
    c1.SetLogx();
  }

  acc_test.SetStats(false);
  acc_test.SetMarkerStyle(20);
  acc_test.SetMarkerColor(kOrange-3);
  acc_test.SetMarkerSize(1.);
  acc_test.Draw("p same");
  acc_train.SetStats(false);
  acc_train.SetMarkerStyle(20);
  acc_train.SetMarkerColor(kAzure-2);
  acc_train.SetMarkerSize(1.);
  acc_train.Draw("p same");
  // c1.SetLogx();
  c1.SetGrid();

  // Calculate maxima
  double max_trainingsdata = acc_train.GetMaximum();
  double max_testdata      = acc_test.GetMaximum();

  TLegend leg(0.4,0.1,0.9,0.3);
  leg.SetHeader("Accuracy of");
  leg.AddEntry(&acc_test,  Form("Test data (Maximum: %4.3f)", max_testdata), "p");
  leg.AddEntry(&acc_train, Form("Trainings data (Maximum: %4.3f)", max_trainingsdata), "p");
  leg.Draw("same");

  c1.SaveAs(Form("%saccuracy.pdf", SavePath.c_str()));



  c2.cd();
  axis2.SetStats(false);
  axis2.GetYaxis()->SetNdivisions(524);
  axis2.SetAxisRange(cost_train.GetMinimum()*0.9, cost_test.GetMaximum()*1.1, "Y");
  axis2.Draw("");

  if (epochs > 100) {
    markersize = 0.5;
    c2.SetLogx();
  }

  cost_train.SetStats(false);
  cost_train.SetMarkerStyle(20);
  cost_train.SetMarkerColor(kAzure-2);
  cost_train.SetMarkerSize(1.);
  cost_train.Draw("p same");
  cost_test.SetStats(false);
  cost_test.SetMarkerStyle(20);
  cost_test.SetMarkerColor(kOrange-3);
  cost_test.SetMarkerSize(1.);
  cost_test.Draw("p same");
  // c1.SetLogx();
  c2.SetGrid();

  // Calculate maxima
  double min_trainingsdata = cost_train.GetMinimum();
  double min_testdata      = cost_test.GetMinimum();

  TLegend leg2(0.4,0.7,0.9,0.9);
  leg2.SetHeader("Cost Function");
  leg2.AddEntry(&acc_test,  Form("Test data (Minimum: %4.3f)", min_testdata), "p");
  leg2.AddEntry(&acc_train, Form("Trainings data (Minimum: %4.3f)", min_trainingsdata), "p");
  leg2.Draw("same");

  c2.SaveAs(Form("%scost.pdf", SavePath.c_str()));

  c3.cd(1);
  axis.Draw("");
  acc_test.Draw("p same");
  acc_train.Draw("p same");
  leg.Draw("same");
  c3.cd(2);
  axis2.Draw("");
  cost_test.Draw("p same");
  cost_train.Draw("p same");
  leg2.Draw("same");
  c3.SaveAs(Form("%ssummary.pdf", SavePath.c_str()));



  // std::cout << "Learning completed" << std::endl;
  return max_testdata;
}

double NeuralNetwork::EvaluateCost(std::vector<double> input, int label){
  arma::vec outputNN = FeedForward(input);

  arma::vec trueoutput;
  trueoutput.zeros(neurons[neurons.size()-1]);
  trueoutput[label] = 1;

  double cost = 0;
  if (fCostFunction == kCE) cost = 0.5 * arma::norm(outputNN - trueoutput, 2);
  if (fCostFunction == kMSE) cost = arma::accu(-trueoutput % arma::log(outputNN) - (1-trueoutput) % arma::log(1-outputNN));


  return cost;
}



// ################################################
bool NeuralNetwork::WriteWeightsToFile(std::string filename){
  std::ofstream outfile;
  outfile.open(filename);
  int neuron_size = neurons.size();
  outfile << "Number of Layers\n";
  outfile << neuron_size << "\n";
  outfile << "Number of Neurons per Layer\n";
  for (int i = 0; i < neuron_size; ++i){
    outfile << neurons.at(i) << "\n";
  }
  for (int i = 0; i < neuron_size - 1; ++i){
    outfile << "Weight between layer " << i+1 << " and " << i+2 << "\n";
    outfile << matWeights.at(i);
    outfile << "Biases in layer " << i+2 << "\n";
    outfile << vecBiases.at(i);
  }
  outfile << "END OF FILE\n";
  outfile.close();
  return true;
}

std::vector<std::string> split(const char *str, char c = ' '){
  std::vector<std::string> result;
  do
  {
    const char *begin = str;
    while(*str != c && *str)
    str++;
    result.push_back(std::string(begin, str));
  } while (0 != *str++);
  return result;
}


// TODO GROSSE BAUSTELLE HIER
bool NeuralNetwork::ReadWeightsFromFile(std::string filename){
  std::ifstream infile;
  infile.open(filename);
  std::string temp;
  std::vector<std::vector<double> > input;
  input.clear();

  neurons.clear();
  matWeights.clear();
  vecBiases.clear();

  int layers = 0;


  int line = 1;
  while(std::getline(infile, temp)) {
    if (line == 2) { layers = std::stoi(temp); }

    if (line >= 4 && line < 4+layers) neurons.push_back(std::stoi(temp));
    line++;

  }
  for (auto i: neurons) std::cout << i << std::endl;

  infile.close();
  return true;
}
