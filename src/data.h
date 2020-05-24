/*
  data.h
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

#ifndef DATA_H_
#define DATA_H_

#include <functional>
#include <string>
#include <vector>

#include "fann_types.h"

/**
  \rst
  Loads a list of files that contain FANN formatted training data and returns a
  ``FannTrainData`` object.

  ***Example**::

    LoadTrainData({"file1.dat", "file2.dat"});
  \endrst
*/
FannTrainData LoadTrainData(std::vector<std::string> files);

/**
  \rst
  Splits a ``FannTrainData`` object using a user supplied function. The
  ``stratify_func`` is called for each sample with pointers to the feature
  and label data. The value returned indicates the zero-indexed group the sample
  will be allocated to and should not exceed the ``groups`` argument.

  ***Example**::

    std::vector<FannTrainData> statified_data = StratifyTrainData(
        data, 2, [](float *feature, float *label) {
      return *label >= 0.5f ? 0 : 1;
    });
  \endrst
*/
std::vector<FannTrainData> StratifyTrainData(
    FannTrainData &data,
    unsigned groups,
    const std::function <unsigned(float*, float*)>& stratify_func);

/**
  \rst
  Writes floating point data to a CSV file. The ``header`` argument should be
  set with a label for each column in the provided ``data``.

  ***Example**::

    WriteCsv(file_path, values_to_write, {"column1", "column2"});
  \endrst
*/
void WriteCsv(std::string path,
              std::vector<std::vector<float>> data,
              std::vector<std::string> header);

/**
  \rst
  Extracts floating point data from ``FannTrainData`` object.

  ***Example**::

    std::vector<std::vector<float>> values = GetTrainDataValues(data);
  \endrst
*/
std::vector<std::vector<float>> GetTrainDataValues(FannTrainData &data);

#endif // DATA_H_
