// Copyright 2019 The Inception Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
)

// Result an experiment result
type Result struct {
	Costs     []float32
	Converged bool
	Misses    int
}

// Statistics aggregation of results
type Statistics struct {
	Count     int
	Converged int
	Epochs    int
}

// Aggregate adds the results to the statistics
func (s *Statistics) Aggregate(result Result) {
	s.Count++
	if result.Converged {
		s.Converged++
		s.Epochs += len(result.Costs)
	}
}

// String generates a string for the statistics
func (s *Statistics) String() string {
	return fmt.Sprintf("%f %f", float64(s.Converged)/float64(s.Count), float64(s.Epochs)/float64(s.Converged))
}

// Optimizer an optimizer type
type Optimizer int

const (
	// OptimizerMomentum basic optimizer
	OptimizerMomentum Optimizer = iota
	// OptimizerAdam the adam opptimizer
	OptimizerAdam
)

func pow(x, y float32) float32 {
	return float32(math.Pow(float64(x), float64(y)))
}

func sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

var (
	seed                      = flag.Int64("seed", 9, "the seed to use")
	runXORExperiment          = flag.Bool("xor", false, "run the xor experiment")
	runXORRepeatedExperiment  = flag.Bool("xorRepeated", false, "run the xor experiment repeatedly")
	runIrisExperiment         = flag.Bool("iris", false, "run the iris experiment")
	runIrisRepeatedExperiment = flag.Bool("irisRepeated", false, "run the iris experiment repeatedly")
)

func main() {
	flag.Parse()

	if *runXORRepeatedExperiment {
		RunXORRepeatedExperiment()
		return
	}

	if *runXORExperiment {
		RunXORExperiment(*seed)
		return
	}

	if *runIrisRepeatedExperiment {
		RunIrisRepeatedExperiment()
		return
	}

	if *runIrisExperiment {
		RunIrisExperiment(*seed)
		return
	}

	flag.Usage()
}
