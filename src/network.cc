/*
  network.cc
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

#include "network.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

thread_local static std::mt19937 rng{std::random_device{}()};

FannNetworkDescriptor::FannNetworkDescriptor(unsigned input_size,
                                             unsigned output_size)
    : num_input_(input_size), num_output_(output_size) {
  Reset();
}
  
FannNetwork FannNetworkDescriptor::CreateNetwork() {
  auto ann = FannNetwork(fann_create_standard_array(
      static_cast<unsigned>(layers_.size()), layers_.data()));
  
  for (unsigned layer = 1; layer < layers_.size(); ++layer) {
    fann_set_activation_steepness_layer(ann.get(),
                                        layer_activation_steepness_[layer - 1],
                                        layer);
    fann_set_activation_function_layer(ann.get(),
                                       layer_activation_funcs_[layer - 1],
                                       layer);
  }
  
  fann_set_training_algorithm(ann.get(), training_algorithm_);

  fann_set_learning_rate(ann.get(), learning_rate_);
  fann_set_learning_momentum(ann.get(), learning_momentum_);

  fann_set_quickprop_decay(ann.get(), quickprop_decay_);
  fann_set_quickprop_mu(ann.get(), quickprop_mu_);
  
  fann_set_rprop_increase_factor(ann.get(), rprop_increase_factor_);
  fann_set_rprop_decrease_factor(ann.get(), rprop_decrease_factor_);
  fann_set_rprop_delta_min(ann.get(), rprop_delta_min_);
  fann_set_rprop_delta_max(ann.get(), rprop_delta_max_);
  fann_set_rprop_delta_zero(ann.get(), rprop_delta_zero_);
  
  fann_set_sarprop_temperature(ann.get(), sarprop_temperature_);
  fann_set_sarprop_weight_decay_shift(ann.get(), sarprop_weight_decay_shift_);
  fann_set_sarprop_step_error_shift(ann.get(), sarprop_step_error_shift_);
  fann_set_sarprop_step_error_threshold_factor(
      ann.get(), sarprop_step_error_threshold_factor_);
  
  return ann;
}
  
void FannNetworkDescriptor::IntializeWeights(FannNetwork& ann,
                                             FannTrainData& train_data) {
  if (wn_weight_init_) {
    fann_init_weights(ann.get(), train_data.get());
  } else {
    fann_randomize_weights(ann.get(), min_weight_, max_weight_);
  }
}
  
void FannNetworkDescriptor::Reset() {
  learning_momentum_ = 0.f;
  learning_rate_ = 0.7f;
  training_algorithm_ = FANN_TRAIN_RPROP;
  
  layers_.clear();
  layer_activation_funcs_.clear();
  layer_activation_steepness_.clear();
  layers_.insert(layers_.end(), 
                 {num_input_, unsigned(sqrt(num_input_)), num_output_});
  layer_activation_funcs_.insert(layer_activation_funcs_.end(),
                                 {FANN_SIGMOID, FANN_SIGMOID});
  layer_activation_steepness_.insert(layer_activation_steepness_.end(),
                                     {0.5f, 0.5f});
  
  quickprop_decay_ = -0.0001f;
  quickprop_mu_ = 1.75f;
  
  rprop_increase_factor_ = 1.2f;
  rprop_decrease_factor_ = 0.5f;
  rprop_delta_min_ = 0.0f;
  rprop_delta_max_ = 50.0f;
  rprop_delta_zero_ = 0.1f;
  
  sarprop_weight_decay_shift_ = -6.644f;
  sarprop_step_error_threshold_factor_ = 0.1f;
  sarprop_step_error_shift_ = 1.385f;
  sarprop_temperature_ = 0.015f;
  
  wn_weight_init_ = false;
  min_weight_ = -0.1f;
  max_weight_ = 0.1f;
}
  
void FannNetworkDescriptor::Merge(const FannNetworkDescriptor &descriptor) {
  
  thread_local static std::uniform_int_distribution<> bool_dist(0, 1);

  // Where appropriate, use the mean hyperparameter value
  auto mean = [](float num1, float num2) { return (num1 + num2) * 0.5f; };
  
  learning_momentum_ = mean(learning_momentum_, descriptor.learning_momentum_);
  learning_rate_ = mean(learning_rate_, descriptor.learning_rate_);
  
  quickprop_decay_ = mean(quickprop_decay_, descriptor.quickprop_decay_);
  quickprop_mu_ = mean(quickprop_mu_, descriptor.quickprop_mu_);
  
  rprop_increase_factor_ = mean(rprop_increase_factor_,
                               descriptor.rprop_increase_factor_);
  rprop_decrease_factor_ = mean(rprop_decrease_factor_,
                               descriptor.rprop_decrease_factor_);
  rprop_delta_min_ = mean(rprop_delta_min_, descriptor.rprop_delta_min_);
  rprop_delta_max_ = mean(rprop_delta_max_, descriptor.rprop_delta_max_);
  rprop_delta_zero_ = mean(rprop_delta_zero_, descriptor.rprop_delta_zero_);
  
  sarprop_temperature_ = mean(sarprop_temperature_,
                             descriptor.sarprop_temperature_);
  sarprop_weight_decay_shift_ = mean(sarprop_weight_decay_shift_,
                                    descriptor.sarprop_weight_decay_shift_);
  sarprop_step_error_shift_ = mean(sarprop_step_error_shift_,
                                  descriptor.sarprop_step_error_shift_);
  sarprop_step_error_threshold_factor_ = mean(
      sarprop_step_error_threshold_factor_,
      descriptor.sarprop_step_error_threshold_factor_);

  min_weight_ = mean(min_weight_, descriptor.min_weight_);
  max_weight_ = mean(max_weight_, descriptor.max_weight_);
  training_algorithm_ = bool_dist(rng) == 0
      ? training_algorithm_ : descriptor.training_algorithm_;
  wn_weight_init_ = bool_dist(rng) == 0
      ? wn_weight_init_ : descriptor.wn_weight_init_;
  
  // For layers, select a random layer from each descriptor
  int min_layer_size = static_cast<int>(std::min(layers_.size(),
                                                 descriptor.layers_.size()));
  int max_layer_size = static_cast<int>(std::max(layers_.size(),
                                                 descriptor.layers_.size()));
  std::uniform_int_distribution<> layer_dist(min_layer_size, max_layer_size);
  unsigned number_of_layers_ = static_cast<unsigned>(layer_dist(rng));

  std::vector<unsigned> new_layers_;
  std::vector<fann_activationfunc_enum> new_layer_activation_funcs_;
  std::vector<float> new_layer_activation_steepness_;
  new_layers_.push_back(layers_[0]);  // Input layer is fixed
  for (unsigned layer = 1; layer < number_of_layers_; ++layer) {
    if (layer == (number_of_layers_ - 1)) {
      new_layers_.push_back(layers_[layers_.size() - 1]);
    } else if ((bool_dist(rng) == 0 && layer < layers_.size()) ||
               layer >= descriptor.layers_.size()) {
      new_layers_.push_back(layers_[layer]);
    } else {
      new_layers_.push_back(descriptor.layers_[layer]);
    }
    if ((bool_dist(rng) == 0 && layer < layers_.size()) ||
        layer >= descriptor.layers_.size()) {
      new_layer_activation_funcs_.push_back(layer_activation_funcs_[layer - 1]);
    } else {
      new_layer_activation_funcs_.push_back(
          descriptor.layer_activation_funcs_[layer - 1]);
    }
    if ((bool_dist(rng) == 0 && layer < layers_.size()) ||
        layer >= descriptor.layers_.size()) {
      new_layer_activation_steepness_.push_back(
          layer_activation_steepness_[layer - 1]);
    } else {
      new_layer_activation_steepness_.push_back(
          descriptor.layer_activation_steepness_[layer - 1]);
    }
  }
  layers_ = new_layers_;
  layer_activation_funcs_ = new_layer_activation_funcs_;
  layer_activation_steepness_ = new_layer_activation_steepness_;
}

void FannNetworkDescriptor::Mutate(float small_chance,
                                   float small_factor,
                                   float big_chance) {
  
  thread_local static std::uniform_int_distribution<> bool_dist(0, 1);
  thread_local static std::uniform_real_distribution<> real_dist(0, 1);
  thread_local static std::uniform_int_distribution<> hidden_dist(-4, 4);
  
  auto mutate_hyperparameter = [&](float& hyperparameter,
                                   const float& small,
                                   const float& large) {
    thread_local static std::uniform_real_distribution<> chance_dist(0.0, 1.0);
    thread_local static std::uniform_real_distribution<> adjust_dist(-1.0, 1.0);
    std::uniform_real_distribution<> value_dist(small, large);
    
    const float chance = chance_dist(rng);
    if (chance <= big_chance) {
      hyperparameter = value_dist(rng);
    } else if (chance <= small_chance) {
      hyperparameter += (hyperparameter * adjust_dist(rng) * small_factor);
      hyperparameter = std::max(std::min(hyperparameter, large), small);
    }
  };
  
  mutate_hyperparameter(learning_momentum_, 0.0, 1.0);
  mutate_hyperparameter(learning_rate_, 0.0, 1.0);
  
  mutate_hyperparameter(quickprop_decay_, -0.1, 0.0);
  mutate_hyperparameter(quickprop_mu_, 1.0, 10.0);
  
  mutate_hyperparameter(rprop_decrease_factor_, 0.0, 1.0);
  mutate_hyperparameter(rprop_increase_factor_, 1.0, 10.0);
  mutate_hyperparameter(rprop_delta_min_, 0.0, 0.1);
  mutate_hyperparameter(rprop_delta_max_, 0.0, 500.0);
  mutate_hyperparameter(rprop_delta_zero_, 0.0, 1.0);
  
  mutate_hyperparameter(sarprop_weight_decay_shift_, -50.0, 0.0);
  mutate_hyperparameter(sarprop_step_error_threshold_factor_, 0.0, 1.0);
  mutate_hyperparameter(sarprop_step_error_shift_, 0.0, 10.0);
  mutate_hyperparameter(sarprop_temperature_, 0.0, 1.0);
  
  mutate_hyperparameter(min_weight_, -1.0, 0.0);
  mutate_hyperparameter(max_weight_, 0.0, 1.0);
  
  if (real_dist(rng) <= big_chance) {
    wn_weight_init_ = bool_dist(rng) == 0 ? true : false;
  }

  if (real_dist(rng) <= big_chance) {
    static std::initializer_list<fann_train_enum> training_algorithm_s = {
      FANN_TRAIN_INCREMENTAL,
      FANN_TRAIN_BATCH,
      FANN_TRAIN_RPROP,
      FANN_TRAIN_QUICKPROP,
      FANN_TRAIN_SARPROP,
    };
    static std::uniform_int_distribution<std::size_t> training_dist(
        0, training_algorithm_s.size() - 1);
    training_algorithm_ = *(training_algorithm_s.begin() + training_dist(rng));
  }
  
  // Increase or decrease number of hidden layers
  if (real_dist(rng) <= big_chance) {
    
    // Calculate average number of hidden neurones per layer
    int hidden_neurones = 0;
    float steepness = 0.0f;
    for (unsigned layer = 1; layer < (layers_.size() - 1); ++layer) {
      hidden_neurones += layers_[layer];
      steepness += layer_activation_steepness_[layer];
    }
    int average_hidden_neurones_per_layer = static_cast<int>(
        static_cast<float>(hidden_neurones) / (layers_.size() - 2));
    float average_steepness = steepness / (layers_.size() - 2);
    
    // Calculate new layer count
    int new_layer_count = static_cast<int>(layers_.size()) + hidden_dist(rng);
    if (new_layer_count < 3) new_layer_count = 3;
    
    std::vector<unsigned> new_layers_;
    std::vector<fann_activationfunc_enum> new_layer_activation_funcs_;
    std::vector<float> new_layer_activation_steepness_;
    
    fann_activationfunc_enum last_func = layer_activation_funcs_.back();
    
    new_layers_.push_back(layers_[0]);
    for (int layer = 1; layer < new_layer_count - 1; ++layer) {
      if (layer < static_cast<int>(layers_.size() - 1)) {
        new_layers_.push_back(layers_[layer]);
        new_layer_activation_funcs_.push_back(
            layer_activation_funcs_[layer - 1]);
        new_layer_activation_steepness_.push_back(
            layer_activation_steepness_[layer - 1]);
        last_func = layer_activation_funcs_[layer - 1];
      } else {
        new_layers_.push_back(average_hidden_neurones_per_layer);
        new_layer_activation_steepness_.push_back(average_steepness);
        new_layer_activation_funcs_.push_back(last_func);
      }
    }
    new_layers_.push_back(layers_[layers_.size() - 1]);
    new_layer_activation_funcs_.push_back(
        layer_activation_funcs_[layer_activation_funcs_.size() - 1]);
    new_layer_activation_steepness_.push_back(
        layer_activation_steepness_[layer_activation_steepness_.size() - 1]);
    
    layers_ = new_layers_;
    layer_activation_funcs_ = new_layer_activation_funcs_;
    layer_activation_steepness_ = new_layer_activation_steepness_;
  }
  
  // Increase or decrease hidden neurone count/activation function/steepness
  for (unsigned layer = 0; layer < layer_activation_funcs_.size() - 1; ++layer) {
    if (layer < (layer_activation_funcs_.size() - 1) &&
        real_dist(rng) <= big_chance) {
      int current_layers_ = layers_[layer + 1];
      current_layers_ += hidden_dist(rng);
      layers_[layer + 1] = std::max(current_layers_, 1);
    }
    
    mutate_hyperparameter(layer_activation_steepness_[layer], 0.0, 1.0);
    
    if (real_dist(rng) <= big_chance) {
      static std::initializer_list<fann_activationfunc_enum> activations = {
        FANN_LINEAR,
        FANN_THRESHOLD_SYMMETRIC,
        FANN_SIGMOID,
        FANN_SIGMOID_STEPWISE,
        FANN_SIGMOID_SYMMETRIC,
        FANN_SIGMOID_SYMMETRIC_STEPWISE,
        FANN_GAUSSIAN,
        FANN_GAUSSIAN_SYMMETRIC,
        FANN_GAUSSIAN_STEPWISE,
        FANN_ELLIOT,
        FANN_ELLIOT_SYMMETRIC,
        FANN_LINEAR_PIECE,
        FANN_LINEAR_PIECE_SYMMETRIC,
        FANN_SIN_SYMMETRIC,
        FANN_COS_SYMMETRIC,
        FANN_SIN,
        FANN_COS,
      };
      static std::uniform_int_distribution<std::size_t> activations_dist(
          0, activations.size() - 1);
      layer_activation_funcs_[layer] = *(activations.begin() +
                                        activations_dist(rng));
    }

  }
}

void FannNetworkDescriptor::PrintDescription() {
  std::cout << "Network: ";
  for (unsigned layer = 0; layer < layers_.size(); ++layer) {
    std::cout << layers_[layer] << " ";
    if (layer > 0) {
      std::cout << "(" << layer_activation_funcs_[layer - 1] << ", "
                << layer_activation_steepness_[layer - 1] << ") ";
    }
  }
  std::cout << std::endl;
  std::cout << "Algorithm: ";
  switch (training_algorithm_) {
    case FANN_TRAIN_INCREMENTAL:
      std::cout << "Incremental";
      break;
    case FANN_TRAIN_BATCH:
      std::cout << "Batch";
      break;
    case FANN_TRAIN_RPROP:
      std::cout << "Rprop "
                << rprop_increase_factor_ << " "
                << rprop_decrease_factor_ << " "
                << rprop_delta_min_ << " "
                << rprop_delta_max_ << " "
                << rprop_delta_zero_;
      break;
    case FANN_TRAIN_QUICKPROP:
      std::cout << "Quickprop "
                << quickprop_decay_ << " "
                << quickprop_mu_;
      break;
    case FANN_TRAIN_SARPROP:
      std::cout << "Sarprop "
                << sarprop_weight_decay_shift_ << " "
                << sarprop_step_error_threshold_factor_ << " "
                << sarprop_step_error_shift_ << " "
                << sarprop_temperature_;
      break;
  }
  std::cout << std::endl;
  std::cout << "Learning: "
            << learning_rate_ << " "
            << learning_momentum_ << std::endl;
  std::cout << "Initialization: ";
  if (wn_weight_init_) {
    std::cout << "W&N algorithm";
  } else {
    std::cout << "Random " << min_weight_ << " " << max_weight_;
  }
  std::cout << std::endl;
}
