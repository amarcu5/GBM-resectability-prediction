/*
  fann_types.h
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

#ifndef FANN_TYPES_H_
#define FANN_TYPES_H_

#include <fann.h>

#include <memory>

/** \cond PRIVATE */
struct TrainDataDeleter {
  void operator()(fann_train_data* ptr) const {
    fann_destroy_train(ptr);
  }
};

struct NetworkDeleter {
  void operator()(fann* ptr) const {
    fann_destroy(ptr);
  }
};
/** \endcond */

typedef std::unique_ptr<fann_train_data, TrainDataDeleter> FannTrainData;
typedef std::unique_ptr<fann, NetworkDeleter> FannNetwork;

#endif // FANN_TYPES_H_
