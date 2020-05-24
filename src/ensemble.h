/*
  ensemble.h
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
#ifndef ENSEMBLE_H_
#define ENSEMBLE_H_

#include <vector>

#include "fann_types.h"

/** An ensemble of ``FannNetwork`` objects. */
class Ensemble {
 public:
  /** Create an empty ensemble. */
  Ensemble();
  
  /** Add a network to the ensemble. */
  void Add(FannNetwork network);
  
  /** Make a single predict using the ensemble. */
  std::vector<float> Run(float *input);
  
  /** Make predictions for an entire data set. */
  std::vector<std::vector<float>> Predict(FannTrainData &data);
  
  /** Remove all networks from the ensemble. */
  void Reset();
  
 private:
  std::vector<FannNetwork> networks_;
};

#endif // ENSEMBLE_H_
