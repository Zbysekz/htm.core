/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

#include <algorithm> // std::generate
#include <iostream>
#include <vector>

#include "nupic/algorithms/Anomaly.hpp"
#include "nupic/algorithms/Cells4.hpp"  //TODO use TM instead
#include "nupic/algorithms/SpatialPooler.hpp"
#include "nupic/encoders/ScalarEncoder.hpp"

#include "nupic/utils/VectorHelpers.hpp"
#include "nupic/utils/Random.hpp"
#include "nupic/utils/Time.hpp"

namespace examples {

using namespace std;
using namespace nupic;
using namespace nupic::utils;
using nupic::ScalarEncoder;
using nupic::algorithms::spatial_pooler::SpatialPooler;
using nupic::algorithms::Cells4::Cells4;
using nupic::algorithms::anomaly::Anomaly;
using nupic::algorithms::anomaly::AnomalyMode;

// work-load
void run() {
  const UInt COLS = 2048; // number of columns in SP, TP
  const UInt DIM_INPUT = 10000;
  const UInt CELLS = 10; // cells per column in TP
#ifdef NDEBUG
  const UInt EPOCHS = 5000; // number of iterations (calls to SP/TP compute() )
#else
  const UInt EPOCHS = 2; // make test faster in Debug
#endif

  cout << "starting test. DIM_INPUT=" << DIM_INPUT
       << ", DIM=" << COLS << ", CELLS=" << CELLS << endl;
  cout << "EPOCHS = " << EPOCHS << endl;


  // initialize SP, TP, Anomaly, AnomalyLikelihood
  clock_t tInit = clock();
  ScalarEncoder enc(133, -100.0, 100.0, DIM_INPUT, 0.0, 0.0, false);
  SpatialPooler sp(vector<UInt>{DIM_INPUT}, vector<UInt>{COLS});
  Cells4 tp(COLS, CELLS, 12, 8, 15, 5, .5f, .8f, 1.0f, .1f, .1f, 0.0f,
            false, 42, true, false);
  Anomaly an(5, AnomalyMode::LIKELIHOOD);
  tInit = clock() - tInit;

  // data for processing input
  vector<UInt> input(DIM_INPUT);
  vector<UInt> outSP(COLS); // active array, output of SP/TP
  vector<UInt> outTP(tp.nCells());
  vector<Real> rIn(COLS); // input for TP (must be Reals)
  vector<Real> rOut(tp.nCells());
  Real res = 0.0; //for anomaly:
  vector<UInt> prevPred_(outSP.size());
  Random rnd;

  // Start a stopwatch timer
  printf("starting:  %d iterations.", EPOCHS);
  clock_t tAll = clock();
  clock_t tRng = 0u;
  clock_t tEnc = 0u;
  clock_t tSP  = 0u;
  clock_t tTP  = 0u;
  clock_t tAn  = 0u;


  //run
  for (UInt e = 0; e < EPOCHS; e++) {
    //Input
//    generate(input.begin(), input.end(), [&] () { return rnd.getUInt32(2); });
    tRng -= clock();
    const Real r = (Real)(rnd.getUInt32(100) - rnd.getUInt32(100)*rnd.getReal64()); //rnd from range -100..100
    tRng += clock();

    //Encode
    tEnc -= clock();
    enc.encodeIntoArray(r, input.data());
    tEnc += clock();

    //SP
    tSP -= clock();
    fill(outSP.begin(), outSP.end(), 0);
    sp.compute(input.data(), true, outSP.data());
    sp.stripUnlearnedColumns(outSP.data());
    tSP += clock();

    //TP
    tTP -= clock();
    rIn = VectorHelpers::castVectorType<UInt, Real>(outSP);
    tp.compute(rIn.data(), rOut.data(), true, true);
    outTP = VectorHelpers::castVectorType<Real, UInt>(rOut);
    tTP += clock();

    //Anomaly
    tAn -= clock();
    res = an.compute(outSP /*active*/, prevPred_ /*prev predicted*/);
    prevPred_ = outTP; //to be used as predicted T-1
    tAn += clock();

    // print
    if (e == EPOCHS - 1) {
      tAll = clock() - tAll;

      cout << "Epoch = " << e << endl;
      cout << "Anomaly = " << res << endl;
      VectorHelpers::print_vector(VectorHelpers::binaryToSparse<UInt>(outSP), ",", "SP= ");
      VectorHelpers::print_vector(VectorHelpers::binaryToSparse<UInt>(VectorHelpers::cellsToColumns(outTP, CELLS)), ",", "TP= ");
      NTA_CHECK(outSP[69] == 0) << "A value in SP computed incorrectly";
      NTA_CHECK(outTP[42] == 0) << "Incorrect value in TP";
      cout << "==============TIMERS============" << endl;
      cout << "Init:\t"   << (float)tInit / CLOCKS_PER_SEC << endl;
      cout << "Random:\t" << (float)tRng  / CLOCKS_PER_SEC << endl;
      cout << "Encode:\t" << (float)tEnc  / CLOCKS_PER_SEC << endl;
      cout << "SP:\t"     << (float)tSP   / CLOCKS_PER_SEC << endl;
      cout << "TP:\t"     << (float)tTP   / CLOCKS_PER_SEC << endl;
      cout << "AN:\t"     << (float)tAn   / CLOCKS_PER_SEC << endl;
      float timeTotal      = (float)tAll  / CLOCKS_PER_SEC;
      cout << "Total elapsed time = " << timeTotal << " seconds" << endl;

      #ifdef NDEBUG
       #ifdef _MSC_VER
          const size_t CI_avg_time = (size_t)floor(14 * getSpeed()); //sec
        #else
          const size_t CI_avg_time = (size_t)floor(7 * getSpeed()); //sec
        #endif
        NTA_CHECK(timeTotal <= CI_avg_time) << //we'll see how stable the time result in CI is, if usable
          "HelloSPTP test slower than expected! (" << timeTotal << ",should be "<< CI_avg_time;
      #endif
    }
  } //end for

} //end run()
} //-ns
