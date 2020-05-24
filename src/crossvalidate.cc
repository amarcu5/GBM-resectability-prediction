/*
  crossvalidate.cc
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
#include "crossvalidate.h"

#include <fann.h>

#include <algorithm>
#include <cmath>

#include "fann_extension.h"

void CrossValidation(std::vector<FannTrainData> &data,
                     const CVFuncExt& process_data, int folds, int repeats) {
  
  // Calculate number of samples needed per strata for a given fold
  unsigned total_samples = 0;
  std::vector<float> proportions;
  proportions.reserve(data.size());
  for (FannTrainData &data : data) {
    unsigned num_samples = fann_length_train_data(data.get());
    total_samples += num_samples;
    proportions.push_back(static_cast<float>(num_samples));
  }
  std::transform(proportions.begin(), proportions.end(), proportions.begin(),
                 std::bind(std::divides<float>(),
                           std::placeholders::_1,
                           static_cast<float>(folds)));
  
  // Allocate memory to hold samples for each folds
  unsigned num_input = fann_num_input_train_data(data.back().get());
  unsigned num_output = fann_num_output_train_data(data.back().get());
  float fold_size = std::floor(static_cast<float>(total_samples) / folds);
  unsigned max_fold_size = static_cast<unsigned>(fold_size) + folds;
  std::vector<FannTrainData> folds_data;
  folds_data.reserve(folds);
  for (int fold = 0; fold < folds; ++fold) {
    folds_data.emplace_back(FannTrainData(fann_create_train(max_fold_size,
                                                            num_input,
                                                            num_output)));
  }
  
  auto training_data = FannTrainData(fann_create_train(total_samples,
                                                       num_input,
                                                       num_output));
  
  for (int repeat = 0; repeat < repeats; ++repeat) {
    
    // Shuffle each strata
    for (FannTrainData &data_stratum : data) {
      fann_shuffle_train_data(data_stratum.get());
    }
    
    // Generate folds using shuffled samples for each strata
    std::vector<int> stratum_sample(data.size(), 0);
    std::vector<float> stratum_remainder(data.size(), 0);
    for (int fold = 0; fold < folds; ++fold) {
      unsigned fold_sample_position = 0;
      for (unsigned stratum = 0; stratum < data.size(); ++stratum) {
        float current_size = stratum_remainder[stratum] + proportions[stratum];
        current_size = std::round(current_size);
        stratum_remainder[stratum] += (proportions[stratum] - current_size);
        unsigned num_samples = static_cast<unsigned>(current_size);
        for (unsigned sample = 0; sample < num_samples; ++sample) {
          fann_set_train_data(
              folds_data[fold].get(),
              fold_sample_position,
              fann_get_train_input(data[stratum].get(),
                                   stratum_sample[stratum]),
              fann_get_train_output(data[stratum].get(),
                                    stratum_sample[stratum]));
          ++fold_sample_position;
          ++stratum_sample[stratum];
        }
      }
      folds_data[fold]->num_data = fold_sample_position;
      fann_shuffle_train_data(folds_data[fold].get());
    }
    
    // Merge folds and process
    for (int fold = 0; fold < folds; ++fold) {
      unsigned fold_samples = fann_length_train_data(folds_data[fold].get());
      unsigned training_data_size = total_samples - fold_samples;
      training_data->num_data = training_data_size;
      
      unsigned fold_sample_position = 0;
      for (int fold_to_copy = 0; fold_to_copy < folds; ++fold_to_copy) {
        if (fold_to_copy == fold) continue;
        unsigned fold_to_copy_samples = fann_length_train_data(
            folds_data[fold_to_copy].get());
        for (unsigned sample = 0; sample < fold_to_copy_samples; ++sample) {
          fann_set_train_data(
              training_data.get(),
              fold_sample_position,
              fann_get_train_input(folds_data[fold_to_copy].get(), sample),
              fann_get_train_output(folds_data[fold_to_copy].get(), sample));
          ++fold_sample_position;
        }
      }
      
      process_data(training_data, folds_data[fold], fold, repeat);
    }
    
  }
  
}

void CrossValidation(std::vector<FannTrainData> &data,
                     const CVFunc& process_data, int folds, int repeats) {
  
  CrossValidation(
      data,
      [&](FannTrainData &training_data, FannTrainData &validation_data,
          int fold, int repeat) {
        process_data(training_data, validation_data);
      },
      folds,
      repeats);
}
