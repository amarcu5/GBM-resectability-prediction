/*
  train.cc
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

#include "train.h"

#include <fann.h>

#include <limits>
#include <memory>

#include "config.h"

float TrainNetwork(FannNetwork &network,
                   FannTrainData &training_data,
                   FannTrainData &validation_data) {
  
  unsigned num_connections = fann_get_total_connections(network.get());
  
  float best_validation_error = std::numeric_limits<float>::max();
  auto best_connections = std::make_unique<fann_connection[]>(num_connections);
  int epochs_since_best_error = 0;
  
  for (int epoch = 0; epoch < kTrainMaxEpochs; ++epoch) {
    fann_train_epoch(network.get(), training_data.get());
    float validation_error = fann_test_data(network.get(),
                                            validation_data.get());
    
    if (validation_error < best_validation_error) {
      fann_get_connection_array(network.get(), best_connections.get());
      best_validation_error = validation_error;
      epochs_since_best_error = 0;
    } else {
      // Early stopping
      if (++epochs_since_best_error >= kTrainEarlyStoppingCount) {
        break;
      }
    }
  }
  
  // Restore parameters of the network with the lowest validation error
  fann_set_weight_array(network.get(),
                        best_connections.get(),
                        num_connections);
  
  return best_validation_error;
}
