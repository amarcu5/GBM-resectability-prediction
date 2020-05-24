/*
  fann_extension.h
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

#ifndef FANN_EXTENSION_H_
#define FANN_EXTENSION_H_

#include <fann.h>

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

/**
  Sets the input and desired output values into the specified position in the
  training data structure
*/
int fann_set_train_data(struct fann_train_data* data,
                        unsigned num,
                        fann_type* input,
                        fann_type* output);

/**
  Gets the training input data at the given position
*/
fann_type *fann_get_train_input(struct fann_train_data * data,
                                unsigned position);

/**
  Gets the training output data at the given position
*/
fann_type *fann_get_train_output(struct fann_train_data * data,
                                 unsigned position);
  
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // FANN_EXTENSION_H_
