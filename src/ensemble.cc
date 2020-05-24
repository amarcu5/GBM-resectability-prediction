/*
  ensemble.cc
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

#include "ensemble.h"

#include <fann.h>

#include <algorithm>
#include <functional>

Ensemble::Ensemble() {}
  
void Ensemble::Add(FannNetwork network) {
  networks_.push_back(std::move(network));
}

std::vector<float> Ensemble::Run(float *input) {
  std::vector<float> ensemble_output(fann_get_num_output(networks_[0].get()),
                                     0.0f);
  
  for (unsigned network = 0; network < networks_.size(); ++network) {
    float *network_output = fann_run(networks_[network].get(), input);
    for (unsigned output = 0; output < ensemble_output.size(); ++output) {
      ensemble_output[output] += network_output[output];
    }
  }
  
  // Calculate the mean output from the networks in the ensemble
  std::transform(ensemble_output.begin(), ensemble_output.end(),
                 ensemble_output.begin(),
                 std::bind(std::divides<float>(), std::placeholders::_1,
                           static_cast<float>(networks_.size())));
  
  return ensemble_output;
}

std::vector<std::vector<float>> Ensemble::Predict(FannTrainData &data) {
  std::vector<std::vector<float>> ensemble_predictions;
  
  unsigned num_samples = fann_length_train_data(data.get());
  for (unsigned sample = 0; sample < num_samples; ++sample) {
    ensemble_predictions.push_back(Run(data->input[sample]));
  }
  
  return ensemble_predictions;
}

void Ensemble::Reset() {
  networks_.clear();
}
