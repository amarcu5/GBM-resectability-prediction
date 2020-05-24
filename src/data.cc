/*
  data.cc
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

#include "data.h"

#include <fann.h>

#include <fstream>

#include "fann_extension.h"

FannTrainData LoadTrainData(std::vector<std::string> files) {
  FannTrainData data;
  
  if (!files.empty()) {
    data = FannTrainData(fann_read_train_from_file(files.front().c_str()));
    
    for (unsigned file_index = 1; file_index < files.size(); ++file_index) {
      auto data_to_merge = FannTrainData(fann_read_train_from_file(
          files[file_index].c_str()));
      data = FannTrainData(fann_merge_train_data(data.get(),
                                                 data_to_merge.get()));
    }
  }
  
  return data;
}

std::vector<FannTrainData> StratifyTrainData(
    FannTrainData &data,
    unsigned groups,
    const std::function <unsigned(float*, float*)>& stratify_func) {
  
  // Get combined data set properties
  unsigned input_size = fann_num_input_train_data(data.get());
  unsigned output_size = fann_num_output_train_data(data.get());
  unsigned num_samples = fann_length_train_data(data.get());
  
  // Allocate resection stratified data sets
  std::vector<unsigned> stratified_data_position(groups, 0);
  std::vector<FannTrainData> stratified_data;
  for (unsigned group = 0; group < groups; ++group) {
    stratified_data.push_back(FannTrainData(
        fann_create_train(num_samples, input_size, output_size)));
  }
  
  // Generate resection stratified data sets
  for (unsigned sample = 0; sample < num_samples; ++sample) {
    float *data_input = fann_get_train_input(data.get(), sample);
    float *data_output = fann_get_train_output(data.get(), sample);
    
    unsigned group = stratify_func(data_input, data_output);
    fann_set_train_data(stratified_data[group].get(),
                        stratified_data_position[group],
                        data_input,
                        data_output);
    ++stratified_data_position[group];
  }
  for (unsigned group = 0; group < groups; ++group) {
    stratified_data[group]->num_data = stratified_data_position[group];
  }
  
  return stratified_data;
}


void WriteCsv(std::string path,
              std::vector<std::vector<float>> data,
              std::vector<std::string> header) {

  std::ofstream csv_ostream(path, std::ofstream::out|std::ofstream::trunc);
  
  // Write the headers
  for (unsigned col = 0; col < header.size(); ++col) {
    csv_ostream << (col ? "," : "") << header[col];
  }
  csv_ostream << std::endl;
  
  // Write the data
  for (unsigned row = 0; row < data.size(); ++row) {
    for (unsigned col = 0; col < data[row].size(); ++col) {
      csv_ostream << (col ? "," : "") << std::to_string(data[row][col]);
    }
    csv_ostream << std::endl;
  }
}

std::vector<std::vector<float>> GetTrainDataValues(FannTrainData &data) {
  std::vector<std::vector<float>> output;
  
  unsigned input_size = fann_num_input_train_data(data.get());
  unsigned output_size = fann_num_output_train_data(data.get());
  unsigned num_samples = fann_length_train_data(data.get());

  for (unsigned sample = 0; sample < num_samples; ++sample) {
    std::vector<float> row;
    float *data_input = fann_get_train_input(data.get(), sample);
    float *data_output = fann_get_train_output(data.get(), sample);
    row.insert(row.end(), data_input, data_input + input_size);
    row.insert(row.end(), data_output, data_output + output_size);
    output.push_back(row);
  }
  
  return output;
}
