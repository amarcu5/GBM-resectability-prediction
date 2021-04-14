/*
  network.h
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

#ifndef NETWORK_H_
#define NETWORK_H_

#include <fann.h>

#include <vector>

#include "fann_types.h"

/** A descriptor used to create a ``FannNetwork`` object. */
class FannNetworkDescriptor {
 public:
  /** Create a descriptor with a default configuration. */
  FannNetworkDescriptor(unsigned input_size = 1, unsigned output_size = 1);
  
  /** Create a ``FannNetwork`` object using the descriptor. */
  FannNetwork CreateNetwork();
  
  /** Initialise weights of a ``FannNetwork`` using descriptor configuration. */
  void IntializeWeights(FannNetwork& ann, FannTrainData& train_data);
  
  /** Reset descriptor to default configuration. */
  void Reset();
  
  /** Merge two network descriptors. */
  void Merge(const FannNetworkDescriptor &descriptor);
  
  /** Mutate descriptor randomly. */
  void Mutate(float small_chance = 0.25f,
              float small_factor = 0.1f,
              float big_chance = 0.05f);
  
  /** Print the descriptor configuration. */
  void PrintDescription();
  
 private:
  unsigned num_input_;
  unsigned num_output_;
  
  float learning_momentum_;
  float learning_rate_;
  fann_train_enum training_algorithm_;
  
  std::vector<unsigned> layers_;
  std::vector<fann_activationfunc_enum> layer_activation_funcs_;
  std::vector<float> layer_activation_steepness_;
  
  float quickprop_decay_;
  float quickprop_mu_;
  
  float rprop_increase_factor_;
  float rprop_decrease_factor_;
  float rprop_delta_min_;
  float rprop_delta_max_;
  float rprop_delta_zero_;
  
  float sarprop_temperature_;
  float sarprop_weight_decay_shift_;
  float sarprop_step_error_shift_;
  float sarprop_step_error_threshold_factor_;
  
  bool wn_weight_init_;
  float min_weight_;
  float max_weight_;
};


#endif // NETWORK_H_
