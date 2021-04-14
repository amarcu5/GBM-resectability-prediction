/*
  evolve.cc
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

#include "evolve.h"

#include <fann.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

#include "config.h"
#include "crossvalidate.h"
#include "fann_extension.h"
#include "train.h"

FannNetworkDescriptor EvolutionaryOptimize(
    std::vector<FannTrainData> &stratified_data) {
  
  // Breeds new descriptors using crossover and random mutation
  auto generate_descriptors = [](
      std::vector<std::pair<FannNetworkDescriptor, double>> descriptors,
      float small_chance,
      float small_factor,
      float big_chance) -> std::vector<FannNetworkDescriptor> {
    
    // Initialise random distribution to pick from population
    thread_local static std::mt19937 rng{std::random_device{}()};
    thread_local static std::uniform_int_distribution<> descriptor_dist(
        0, static_cast<int>(descriptors.size() - 1));
    
    std::vector<FannNetworkDescriptor> new_descriptors;
    for (int network = 0; network < kNetworksPerGeneration; ++network) {
      
      // Pick 2 random descriptors and mate them
      auto random_descriptor1 = descriptors[descriptor_dist(rng)].first;
      auto random_descriptor2 = descriptors[descriptor_dist(rng)].first;
      FannNetworkDescriptor new_descriptor = random_descriptor1;
      new_descriptor.Merge(random_descriptor2);  // Crossover
      new_descriptor.Mutate(small_chance, small_factor,
                            big_chance);  // Random mutation
      new_descriptors.push_back(new_descriptor);
    }
    
    return new_descriptors;
  };

  double best_ever_score = std::numeric_limits<float>::max();

  unsigned input_size = fann_num_input_train_data(stratified_data[0].get());
  unsigned output_size = fann_num_output_train_data(stratified_data[0].get());

  // Initialise descriptor population with 1 parent using default configuration
  std::vector<std::pair<FannNetworkDescriptor, double>> scored_descriptors;
  scored_descriptors.push_back(std::make_pair(
      FannNetworkDescriptor(input_size, output_size), 
      std::numeric_limits<double>::max()));

  for (int generation = 0; generation < kMaxGenerations; ++generation) {

    // Breed new descriptors with exponential annealling of mutation rate
    float big_mutation_chance = kBigMutationStartChance - kBigMutationEndChance;
    big_mutation_chance *= std::pow(kBigMutationCoefficient, generation);
    big_mutation_chance += kBigMutationEndChance;
    std::vector<FannNetworkDescriptor> descriptors = generate_descriptors(
        scored_descriptors, 0.25f, 0.1f, big_mutation_chance);

    // Evaluate fitness of each descriptor in the population
    scored_descriptors.clear();
#ifdef MULTITHREAD
    std::mutex scored_descriptors_mutex;
    std::mutex train_data_mutex;
    std::vector<std::future<void>> threads;
    
    const unsigned num_threads = std::thread::hardware_concurrency();
    const unsigned blocksize = static_cast<unsigned>(
        descriptors.size() / num_threads);
    
    auto perform = [&](unsigned start, unsigned end) {
      std::vector<FannTrainData> private_stratified_data;
      {
        std::lock_guard<std::mutex> lock(train_data_mutex);
        private_stratified_data.reserve(stratified_data.size());
        for (auto &data : stratified_data) {
          private_stratified_data.push_back(FannTrainData(
              fann_duplicate_train_data(data.get())));
        }
      }
      for (unsigned descriptor = start; descriptor < end; ++descriptor) {
        double error = 0;
        FannNetwork network = descriptors[descriptor].CreateNetwork();
        CrossValidation(private_stratified_data,
                        [&](FannTrainData &training_data,
                            FannTrainData &validation_data){
          descriptors[descriptor].IntializeWeights(network, training_data);
          error += static_cast<double>(TrainNetwork(
              network, training_data, validation_data));
        }, kCrossValidationInnerFolds, kCrossValidationInnerRepeats);
        
        std::lock_guard<std::mutex> lock(scored_descriptors_mutex);
        scored_descriptors.push_back(
            std::make_pair(descriptors[descriptor], error));
      }
    };
    
    for (unsigned thread = 1; thread < num_threads; ++thread) {
      threads.emplace_back(std::async(
          perform,
          static_cast<unsigned>((thread - 1) * blocksize),
          static_cast<unsigned>(thread * blocksize)));
    }
    perform((num_threads - 1) * blocksize,
            static_cast<unsigned>(descriptors.size()));
    
    for (auto &thread : threads) {
      thread.wait();
    }
    threads.clear();
#else
    for (auto descriptor : descriptors) {
      double error = 0;
      FannNetwork network = descriptor.CreateNetwork();
      CrossValidation(stratified_data, [&](FannTrainData &training_data,
                                           FannTrainData &validation_data) {
        descriptor.IntializeWeights(network, training_data);
        error += static_cast<double>(TrainNetwork(network,
                                                  training_data,
                                                  validation_data));
      }, kCrossValidationInnerFolds, kCrossValidationInnerRepeats);
      scored_descriptors.push_back(std::make_pair(descriptor, error));
    }
#endif
    
    // Sort descriptors by fitness
    std::sort(scored_descriptors.begin(), scored_descriptors.end(),
              [](auto &left, auto &right) {
      return left.second < right.second;
    });
    
    // Store fittest descriptor across all generations
    if (scored_descriptors[0].second < best_ever_score) {
      best_ever_score = scored_descriptors[0].second;
    }
    
#ifdef DEBUG
    // Print this generation results
    std::cout << "Generation " << (generation + 1)
              << "  (mutation rate " << big_mutation_chance << ")" << std::endl;
    std::cout << "Fittest network MSE: " << scored_descriptors[0].second
              << " (best MSE " << best_ever_score << ")" << std::endl;
    scored_descriptors[0].first.PrintDescription();
    std::cout << std::endl;
#endif
    
    // Remove least-fit decriptors from the population
    scored_descriptors.resize(kNetworksMatingPerGeneration);
  }
  
  // Get best network design descriptor
  return std::move(scored_descriptors[0].first);
}
