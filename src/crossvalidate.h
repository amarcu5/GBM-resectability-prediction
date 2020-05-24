/*
  crossvalidate.h
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

#ifndef CROSSVALIDATE_H_
#define CROSSVALIDATE_H_

#include <vector>
#include <functional>

#include "fann_types.h"

using CVFunc = std::function<void(FannTrainData&, FannTrainData&)>;
using CVFuncExt = std::function<void(FannTrainData&, FannTrainData&, int, int)>;

/**
  \rst
  Performs stratified k-fold repeated cross validation. The ``process_data``
  function is called for each round with relevant data and optionally the
  current fold and repeat.

  ***Example**::

    CrossValidation(data,
                    [](FannTrainData &training, FannTrainData &validation,
                       int fold, int repeat) {
      // Develop a model on training set and evaluate on held out
      // validation set
    }, 10, 2);
  \endrst
*/
void CrossValidation(std::vector<FannTrainData> &data,
                     const CVFuncExt& process_data, int folds, int repeats = 1);

/** \cond PRIVATE */
void CrossValidation(std::vector<FannTrainData> &data,
                     const CVFunc& process_data, int folds, int repeats = 1);
/** \endcond */

#endif // CROSSVALIDATE_H_
