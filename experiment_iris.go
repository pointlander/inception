// Copyright 2019 The Inception Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"sync"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf32"
)

var (
	datum iris.Datum
	once  sync.Once
)

func load() {
	var err error
	datum, err = iris.Load()
	if err != nil {
		panic(err)
	}
	max := 0.0
	for _, item := range datum.Fisher {
		for _, measure := range item.Measures {
			if measure > max {
				max = measure
			}
		}
	}
	for _, item := range datum.Fisher {
		for i, measure := range item.Measures {
			item.Measures[i] = measure / max
		}
	}

	max = 0.0
	for _, item := range datum.Bezdek {
		for _, measure := range item.Measures {
			if measure > max {
				max = measure
			}
		}
	}
	for _, item := range datum.Bezdek {
		for i, measure := range item.Measures {
			item.Measures[i] = measure / max
		}
	}
}

// IrisExperiment iris neural network experiment
func IrisExperiment(seed int64, width, depth int, optimizer Optimizer, batch, inception, dct, context bool) Result {
	once.Do(load)

	rnd, costs, converged, misses := rand.New(rand.NewSource(seed)), make([]float32, 0, 1000), false, 0
	random32 := func(a, b float32) float32 {
		return (b-a)*rnd.Float32() + a
	}

	batchSize := 10

	var input, output tf32.V
	if batch {
		input, output = tf32.NewV(4, batchSize), tf32.NewV(3, batchSize)
	} else {
		input, output = tf32.NewV(4), tf32.NewV(3)
	}
	w1, b1, w2, b2 := tf32.NewV(4, width), tf32.NewV(width), tf32.NewV(width, 3), tf32.NewV(3)
	parameters, zero := []*tf32.V{&w1, &b1, &w2, &b2}, []*tf32.V{}
	m1, m2, m1a, m2a := w1.Meta(), w2.Meta(), b1.Meta(), b2.Meta()
	if dct {
		t1, tt1 := DCT2(4)
		t2, tt2 := DCT2(width)
		t3, tt3 := DCT2(width)
		t4, tt4 := DCT2(3)
		w1b, b1b, w2b, b2b := tf32.NewV(4, width), tf32.NewV(width), tf32.NewV(width, 3), tf32.NewV(3)
		m1 = tf32.Add(tf32.Mul(tt1.Meta(), tf32.T(tf32.Mul(m1, t1.Meta()))), w1b.Meta())
		m1a = tf32.Add(tf32.Mul(tt2.Meta(), tf32.T(tf32.Mul(m1a, t2.Meta()))), b1b.Meta())
		m2 = tf32.Add(tf32.Mul(tt3.Meta(), tf32.T(tf32.Mul(m2, t3.Meta()))), w2b.Meta())
		m2a = tf32.Add(tf32.Mul(tt4.Meta(), tf32.T(tf32.Mul(m2a, t4.Meta()))), b2b.Meta())
		zero = append(zero, &t1, &tt1, &t2, &tt2, &t3, &tt3, &t4, &tt4)
		parameters = append(parameters, &w1b, &b1b, &w2b, &b2b)
	} else if inception {
		for i := 0; i < depth; i++ {
			a, b := tf32.NewV(4, 4), tf32.NewV(4, width)
			m1 = tf32.Add(tf32.Mul(a.Meta(), b.Meta()), m1)
			parameters = append(parameters, &a, &b)
		}
		for i := 0; i < depth; i++ {
			a, b := tf32.NewV(width, width), tf32.NewV(width)
			m1a = tf32.Add(tf32.Mul(a.Meta(), b.Meta()), m1a)
			parameters = append(parameters, &a, &b)
		}
		for i := 0; i < depth; i++ {
			a, b := tf32.NewV(width, width), tf32.NewV(width, 3)
			m2 = tf32.Add(tf32.Mul(a.Meta(), b.Meta()), m2)
			parameters = append(parameters, &a, &b)
		}
		for i := 0; i < depth; i++ {
			a, b := tf32.NewV(3, 3), tf32.NewV(3)
			m2a = tf32.Add(tf32.Mul(a.Meta(), b.Meta()), m2a)
			parameters = append(parameters, &a, &b)
		}
	}

	var deltas, m, v [][]float32
	for _, p := range parameters {
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, random32(-1, 1))
		}
		switch optimizer {
		case OptimizerMomentum:
			deltas = append(deltas, make([]float32, len(p.X)))
		case OptimizerAdam:
			m = append(m, make([]float32, len(p.X)))
			v = append(v, make([]float32, len(p.X)))
		}
	}

	l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(m1, input.Meta()), m1a))
	l2 := tf32.Softmax(tf32.Add(tf32.Mul(m2, l1), m2a))
	cost := tf32.Avg(tf32.CrossEntropy(l2, output.Meta()))

	type Datum struct {
		iris         *iris.Iris
		deltas, m, v [][]float32
	}
	data := make([]Datum, len(datum.Fisher))
	table := make([]*Datum, len(data))
	for i := range data {
		data[i].iris = &datum.Fisher[i]
		if context {
			for _, p := range parameters {
				switch optimizer {
				case OptimizerMomentum:
					data[i].deltas = append(data[i].deltas, make([]float32, len(p.X)))
				case OptimizerAdam:
					data[i].m = append(data[i].m, make([]float32, len(p.X)))
					data[i].v = append(data[i].v, make([]float32, len(p.X)))
				}
			}
		}
		table[i] = &data[i]
	}

	rnd = rand.New(rand.NewSource(seed))
	// momentum parameters
	alpha, eta := float32(.1), float32(.1)
	// adam parameters
	a, beta1, beta2, epsilon := float32(.001), float32(.9), float32(.999), float32(1E-8)
	optimize := func(i int) {
		norm := float32(0)
		for _, p := range parameters {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 1 {
			scaling := 1 / norm
			for k, p := range parameters {
				for l, d := range p.D {
					d *= scaling
					switch optimizer {
					case OptimizerMomentum:
						deltas[k][l] = alpha*deltas[k][l] - eta*d
						p.X[l] += deltas[k][l]
					case OptimizerAdam:
						m[k][l] = beta1*m[k][l] + (1-beta1)*d
						v[k][l] = beta2*v[k][l] + (1-beta2)*d*d
						t := float32(i + 1)
						mCorrected := m[k][l] / (1 - pow(beta1, t))
						vCorrected := v[k][l] / (1 - pow(beta2, t))
						p.X[l] -= a * mCorrected / (sqrt(vCorrected) + epsilon)
					}
				}
			}
		} else {
			for k, p := range parameters {
				for l, d := range p.D {
					switch optimizer {
					case OptimizerMomentum:
						deltas[k][l] = alpha*deltas[k][l] - eta*d
						p.X[l] += deltas[k][l]
					case OptimizerAdam:
						m[k][l] = beta1*m[k][l] + (1-beta1)*d
						v[k][l] = beta2*v[k][l] + (1-beta2)*d*d
						t := float32(i + 1)
						mCorrected := m[k][l] / (1 - pow(beta1, t))
						vCorrected := v[k][l] / (1 - pow(beta2, t))
						p.X[l] -= a * mCorrected / (sqrt(vCorrected) + epsilon)
					}
				}
			}
		}
	}

	length := len(table)
	if batch {
		for i := 0; i < 10000; i++ {
			for i := range table {
				j := i + rnd.Intn(length-i)
				table[i], table[j] = table[j], table[i]
			}
			total := float32(0.0)
			for j := 0; j < length; j += batchSize {
				for _, p := range parameters {
					p.Zero()
				}
				for _, p := range zero {
					p.Zero()
				}

				inputs, outputs := make([]float32, 0, 4*batchSize), make([]float32, 0, 3*batchSize)
				for k := 0; k < batchSize; k++ {
					index := (j + k) % length
					for _, measure := range table[index].iris.Measures {
						inputs = append(inputs, float32(measure))
					}
					out := make([]float32, 3)
					out[iris.Labels[table[index].iris.Label]] = 1
					outputs = append(outputs, out...)
				}
				input.Set(inputs)
				output.Set(outputs)
				total += tf32.Gradient(cost).X[0]
				optimize(i)
			}
			costs = append(costs, total)
			if total < 13/float32(batchSize) {
				converged = true
				break
			}
		}
	} else {
		for i := 0; i < 10000; i++ {
			for i := range table {
				j := i + rnd.Intn(length-i)
				table[i], table[j] = table[j], table[i]
			}
			total := float32(0.0)
			for j := range table {
				for _, p := range parameters {
					p.Zero()
				}
				for _, p := range zero {
					p.Zero()
				}
				in := make([]float32, len(table[j].iris.Measures))
				for k, measure := range table[j].iris.Measures {
					in[k] = float32(measure)
				}
				input.Set(in)
				out := make([]float32, 3)
				out[iris.Labels[table[j].iris.Label]] = 1
				output.Set(out)
				total += tf32.Gradient(cost).X[0]
				if context {
					switch optimizer {
					case OptimizerMomentum:
						deltas = table[j].deltas
					case OptimizerAdam:
						m = table[j].m
						v = table[j].v
					}
				}
				optimize(i)
			}
			costs = append(costs, total)
			if total < 13 {
				converged = true
				break
			}
		}
	}

	if converged {
		for i := range data {
			in := make([]float32, len(data[i].iris.Measures))
			for i, measure := range data[i].iris.Measures {
				in[i] = float32(measure)
			}
			input.Set(in)
			var output tf32.V
			l2(func(a *tf32.V) {
				output = *a
			})
			max, actual := float32(0.0), 0
			expected := iris.Labels[data[i].iris.Label]
			for j, value := range output.X {
				if value > max {
					max, actual = value, j
				}
			}
			if expected != actual {
				misses++
			}
		}
	}

	return Result{
		Costs:     costs,
		Converged: converged,
		Misses:    misses,
	}
}

// RunIrisRepeatedExperiment runs multiple iris experiments
func RunIrisRepeatedExperiment() {
	experiment := func(seed int64, inception, context bool, results chan<- Result) {
		results <- IrisExperiment(seed, 3, 4, OptimizerAdam, true, inception, false, context)
	}
	normalStats, inceptionStats := Statistics{}, Statistics{}
	normalResults, inceptionResults := make(chan Result, 8), make(chan Result, 8)
	for i := 1; i <= 256; i++ {
		go experiment(int64(i), false, false, normalResults)
		go experiment(int64(i), true, false, inceptionResults)
	}
	for normalStats.Count < 256 || inceptionStats.Count < 256 {
		select {
		case result := <-normalResults:
			normalStats.Aggregate(result)
		case result := <-inceptionResults:
			inceptionStats.Aggregate(result)
		}
	}
	fmt.Printf("normal: %s\n", normalStats.String())
	fmt.Printf("inception: %s\n", inceptionStats.String())
}

// RunIrisExperiment runs an iris experiment once
func RunIrisExperiment(seed int64) {
	normal := IrisExperiment(seed, 3, 4, OptimizerAdam, true, false, false, false)
	inception := IrisExperiment(seed, 3, 4, OptimizerAdam, true, true, false, false)

	pointsNormal := make(plotter.XYs, 0, len(normal.Costs))
	for i, cost := range normal.Costs {
		pointsNormal = append(pointsNormal, plotter.XY{X: float64(i), Y: float64(cost)})
	}

	pointsInception := make(plotter.XYs, 0, len(inception.Costs))
	for i, cost := range inception.Costs {
		pointsInception = append(pointsInception, plotter.XY{X: float64(i), Y: float64(cost)})
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "iris epochs"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "cost"

	normalScatter, err := plotter.NewScatter(pointsNormal)
	if err != nil {
		panic(err)
	}
	normalScatter.GlyphStyle.Radius = vg.Length(1)
	normalScatter.GlyphStyle.Shape = draw.CircleGlyph{}
	normalScatter.GlyphStyle.Color = color.RGBA{R: 255, A: 255}

	inceptionScatter, err := plotter.NewScatter(pointsInception)
	if err != nil {
		panic(err)
	}
	inceptionScatter.GlyphStyle.Radius = vg.Length(1)
	inceptionScatter.GlyphStyle.Shape = draw.CircleGlyph{}
	inceptionScatter.GlyphStyle.Color = color.RGBA{B: 255, A: 255}

	p.Add(normalScatter, inceptionScatter)
	p.Legend.Top = true
	p.Legend.Add("normal", normalScatter)
	p.Legend.Add("inception", inceptionScatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost_iris.png")
	if err != nil {
		panic(err)
	}
}
