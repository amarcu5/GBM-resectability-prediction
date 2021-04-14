/*
  main.cc
  gbm_prediction_ann

  Created by Adam Marcus on 21/08/2018.

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <fann.h>

#include <iostream>
#include <string>
#include <vector>

#include "config.h"
#include "crossvalidate.h"
#include "data.h"
#include "ensemble.h"
#include "evolve.h"
#include "fann_extension.h"
#include "fann_types.h"
#include "network.h"
#include "train.h"

/** Returns resection status (complete=1, incomplete=0) for a given sample. */
unsigned resectionStatusHelper(float *input, float *output) {
  return *output >= 0.5f ? 0 : 1;
}

int main(int argc, char **argv) {

  // Load and combine the data sets
  std::vector<std::string> file_paths(argv + 1, argv + argc);
  FannTrainData data_combined = LoadTrainData(file_paths);
  if (!data_combined) {
    std::cout << "Usage: train datafile1 datafile2 ..." << std::endl;
    return 0;
  }
  unsigned num_samples = fann_length_train_data(data_combined.get());
  std::cout << "Loaded " << num_samples << " samples" << std::endl;
  
  // Outer cross validation loop for network evaluation stratified by
  // resection status
  std::vector<FannTrainData> resection_data = StratifyTrainData(
      data_combined, 2, resectionStatusHelper);
  CrossValidation(resection_data, [](FannTrainData &training_data,
                                     FannTrainData &testing_data,
                                     int fold, int repeat) {
    
    // Network selection using evolution to find the best network design
    // (hides inner cross validation loop)
    std::vector<FannTrainData> resection_data = StratifyTrainData(
        training_data, 2, resectionStatusHelper);
    FannNetworkDescriptor best_descriptor = EvolutionaryOptimize(
        resection_data);
    
    // Generate stacked ensemble with the best network design found
    // Note: Cross validation is used as a convenience to create the ensemble
    Ensemble ensemble;
    auto network = best_descriptor.CreateNetwork();
    CrossValidation(resection_data, [&](FannTrainData &training_data_subsample,
                                        FannTrainData &validation_data) {
      best_descriptor.IntializeWeights(network, training_data_subsample);
      TrainNetwork(network, training_data_subsample, validation_data);
      ensemble.Add(FannNetwork(fann_copy(network.get())));
    }, 10, kEnsembleSize);
    
    // Make predictions with stacked ensemble on testing data
    std::vector<std::vector<float>> predictions_ann = ensemble.Predict(
        testing_data);
    
    // Save predictions and training data
    int run = fold * kCrossValidationOuterFolds + repeat;
    std::vector<std::string> train_header = {
      "input0", "input1", "input2", "input3", "input4",
      "output0"
    };
    WriteCsv("data/processed/train-" + std::to_string(run) + ".csv",
             GetTrainDataValues(training_data), train_header);
    WriteCsv("data/processed/test-" + std::to_string(run) + ".csv",
             GetTrainDataValues(testing_data), train_header);
    WriteCsv("models/predict-ann-" + std::to_string(run) + ".csv",
             predictions_ann, { "predict0" });
    
  }, kCrossValidationOuterFolds, kCrossValidationOuterRepeats);
  
  return 0;
}
