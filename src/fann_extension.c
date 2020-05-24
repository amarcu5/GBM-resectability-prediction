/*
  fann_extension.c
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

#include "fann_extension.h"

int fann_set_train_data(struct fann_train_data* data,
                        unsigned num,
                        fann_type* input,
                        fann_type* output) {
#ifdef DEBUG
  if (num >= data->num_data) {
    fann_error(NULL, FANN_E_INDEX_OUT_OF_BOUND);
    return -1;
  }
#endif
  
  for (unsigned i = 0; i < data->num_input; ++i) {
    data->input[num][i] = input[i];
  }
  for (unsigned i = 0; i < data->num_output; ++i) {
    data->output[num][i] = output[i];
  }
  
  return 0;
}

fann_type *fann_get_train_input(struct fann_train_data * data,
                                unsigned position) {
  
  if (position >= data->num_data) {
    return NULL;
  }
  
  return data->input[position];
}

fann_type *fann_get_train_output(struct fann_train_data * data,
                                 unsigned position) {
  
  if (position >= data->num_data) {
    return NULL;
  }
  
  return data->output[position];
}
