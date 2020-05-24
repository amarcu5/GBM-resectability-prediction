/*
  config.h
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

#ifndef CONFIG_H_
#define CONFIG_H_

/** Number of folds in outer cross validation loop. */
extern const int kCrossValidationOuterFolds;
/** Number of folds in inner cross validation loop. */
extern const int kCrossValidationInnerFolds;
/** Number of repeats in outer cross validation loop. */
extern const int kCrossValidationOuterRepeats;
/** Number of repeats in inner cross validation loop. */
extern const int kCrossValidationInnerRepeats;

/** Size of the population of network descriptors for each generation. */
extern const int kNetworksPerGeneration;
/** Number of fittest network descriptors to breed in each generation. */
extern const int kNetworksMatingPerGeneration;
/** Total number of generations. */
extern const int kMaxGenerations;
/** Initial mutation probability. */
extern const float kBigMutationStartChance;
/** Final mutation probability. */
extern const float kBigMutationEndChance;
/** Exponentional coefficient used to anneal initial to final mutation rate. */
extern const float kBigMutationCoefficient;

/** Maximum number of EPOCH. */
extern const int kTrainMaxEpochs;
/** Stop after number of EPOCH without improvement to error. */
extern const int kTrainEarlyStoppingCount;

/** Size of final ensemble in multiples of 10. */
extern const int kEnsembleSize;

#endif // CONFIG_H_
