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

#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "./../crossvalidate.h"
#include "./../data.h"
#include "./../fann_types.h"
#include "./../fann_extension.h"

FannTrainData GenerateData(int samples) {
  auto data = FannTrainData(fann_create_train(samples, 2, 1));
   for (int sample = 0; sample < samples; ++sample) {
     float values[] = {0.0f, 1.0f, 0.0f};
     float *values_ptr = &values[sample < samples / 2 ? 0 : 1];
     fann_set_train_data(data.get(), sample, values_ptr, values_ptr);
   }
  return data;
}

static const char *test_path = "test.dat";
static const char *fann_data = R"(2 2 1
0 1
0
1 0
1)";

TEST_CASE("LoadTrainData", "[Data]") {
  std::ofstream test_file(test_path);
  test_file << fann_data;
  test_file.close();
  std::shared_ptr<void> _(nullptr, [](...){ remove(test_path); });
  
  FannTrainData data = LoadTrainData({test_path, test_path});
  REQUIRE(fann_length_train_data(data.get()) == 4);
}

TEST_CASE("StratifyTrainData", "[Data]") {
  FannTrainData data = GenerateData(100);
  
  std::vector<FannTrainData> statified_data = StratifyTrainData(
      data, 2, [](float *input, float *output) {
    return *output >= 0.5f ? 0 : 1;
  });
  
  REQUIRE(statified_data.size() == 2);
  REQUIRE(fann_length_train_data(statified_data[0].get()) == 50);
  REQUIRE(fann_length_train_data(statified_data[1].get()) == 50);
}

TEST_CASE("WriteCsv", "[Data]") {
  std::vector<std::vector<float>> values {
    {0.8f, 0.0f},
    {1.0f, 0.2f},
  };
  
  WriteCsv(test_path, values, {"a", "b"});
  std::shared_ptr<void> _(nullptr, [](...){ remove(test_path); });

  std::ifstream test_file(test_path);
  std::stringstream data;
  data << test_file.rdbuf();
  std::string csv = data.str();
  
  REQUIRE(csv == "a,b\n0.800000,0.000000\n1.000000,0.200000\n");
}

TEST_CASE("GetTrainDataValues", "[Data]") {
  FannTrainData data = GenerateData(2);
  std::vector<std::vector<float>> values = GetTrainDataValues(data);
  
  REQUIRE(values.size() == 2);
  REQUIRE(values[0].size() == 3);
  REQUIRE(values[0][0] == 0);
  REQUIRE(values[0][1] == 1);
  REQUIRE(values[0][2] == 0);
  REQUIRE(values[1].size() == 3);
  REQUIRE(values[1][0] == 1);
  REQUIRE(values[1][1] == 0);
  REQUIRE(values[1][2] == 1);
}

TEST_CASE("CrossValidation", "[crossvalidate]") {
  auto data = std::vector<FannTrainData>();
  data.emplace_back(GenerateData(100));
  CrossValidation(data,
                  [](FannTrainData &train, FannTrainData &test,
                     int fold, int repeat) {
    REQUIRE(fann_length_train_data(train.get()) == 90);
    REQUIRE(fann_length_train_data(test.get()) == 10);
    REQUIRE(fold < 10);
    REQUIRE(fold >= 0);
    REQUIRE(repeat < 2);
    REQUIRE(repeat >= 0);
  }, 10, 2);
}


