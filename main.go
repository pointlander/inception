// Copyright 2019 The Inception Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"image/color"
	"math"

	"github.com/pointlander/gradient/tf32"
)

// Result an experiment result
type Result struct {
	Costs     []float32
	Converged bool
	Misses    int
}

// Statistics aggregation of results
type Statistics struct {
	Mode      string
	Optimizer Optimizer
	Batch     int
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

// ConvergenceProbability the probability of convergence
func (s *Statistics) ConvergenceProbability() float64 {
	return float64(s.Converged) / float64(s.Count)
}

// AverageEpochs the average epochs
func (s *Statistics) AverageEpochs() float64 {
	return float64(s.Epochs) / float64(s.Converged)
}

// String generates a string for the statistics
func (s *Statistics) String() string {
	return fmt.Sprintf("%f %f", s.ConvergenceProbability(), s.AverageEpochs())
}

// Optimizer an optimizer type
type Optimizer int

const (
	// OptimizerStatic is a static learning optimizer
	OptimizerStatic Optimizer = iota
	// OptimizerMomentum basic optimizer
	OptimizerMomentum
	// OptimizerAdam the adam optimizer
	OptimizerAdam
)

// Optimizers the optimizers
var Optimizers = [...]Optimizer{
	OptimizerStatic,
	OptimizerMomentum,
	OptimizerAdam,
}

// Converts the optimzer to a string
func (o Optimizer) String() string {
	switch o {
	case OptimizerStatic:
		return "static"
	case OptimizerMomentum:
		return "momentum"
	case OptimizerAdam:
		return "adam"
	}
	return "unknown"
}

var colors = [...]color.RGBA{
	{R: 0x00, G: 0x3f, B: 0x5c, A: 255},
	{R: 0x44, G: 0x4e, B: 0x86, A: 255},
	{R: 0x95, G: 0x51, B: 0x96, A: 255},
	{R: 0xdd, G: 0x51, B: 0x82, A: 255},
	{R: 0xff, G: 0x6e, B: 0x54, A: 255},
	{R: 0xff, G: 0xa6, B: 0x00, A: 255},
}

func pow(x, y float32) float32 {
	return float32(math.Pow(float64(x), float64(y)))
}

func sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func cos(a float32) float32 {
	return float32(math.Cos(float64(a)))
}

// DCT2 create dct2 coefficient matrices
// https://edoras.sdsu.edu/doc/matlab/toolbox/images/transfo7.html
func DCT2(size int) (t, tt tf32.V) {
	m := float32(size)
	t, tt = tf32.NewV(size, size), tf32.NewV(size, size)
	for p := 0; p < size; p++ {
		if p == 0 {
			for q := 0; q < size; q++ {
				t.X = append(t.X, 1/sqrt(m))
			}
			continue
		}
		for q := 0; q < size; q++ {
			t.X = append(t.X, sqrt(2/m)*cos(math.Pi*(2*float32(q)+1)*float32(p)/(2*m)))
		}
	}

	for q := 0; q < size; q++ {
		for p := 0; p < size; p++ {
			if p == 0 {
				tt.X = append(tt.X, 1/sqrt(m))
				continue
			}
			tt.X = append(tt.X, sqrt(2/m)*cos(math.Pi*(2*float32(q)+1)*float32(p)/(2*m)))
		}
	}
	return
}

var (
	seed                      = flag.Int64("seed", 9, "the seed to use")
	runXORExperiment          = flag.Bool("xor", false, "run the xor experiment")
	runXORParallelExperiment  = flag.Bool("xorParallel", false, "run the xor experiment parallelly")
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

	if *runXORParallelExperiment {
		XORParallelExperiment(*seed, 16)
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
