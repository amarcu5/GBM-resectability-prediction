/*
  train.h
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

#ifndef TRAIN_H_
#define TRAIN_H_

#include "fann_types.h"
#include "network.h"

/**
  \rst
  Trains a ``FannNetwork`` object using early stopping. Returns mean squared
  error (MSE) which is used as the loss function. Training and validation data
  supplied are ``FannTrainData`` objects.

  ***Example**::

    float error = TrainNetwork(network, training_data, validation_data)
 \endrst
*/
float TrainNetwork(FannNetwork &network,
                   FannTrainData &training_data,
                   FannTrainData &validation_data);

#endif // TRAIN_H_
