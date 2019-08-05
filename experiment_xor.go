// Copyright 2019 The Inception Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
	"sort"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tf32"
)

// XORExperiment xor neural network experiment
func XORExperiment(seed int64, width, depth int, optimizer Optimizer, batch, inception, dct, context bool) Result {
	rnd, costs, converged := rand.New(rand.NewSource(seed)), make([]float32, 0, 1000), false
	random32 := func(a, b float32) float32 {
		return (b-a)*rnd.Float32() + a
	}

	var input, output tf32.V
	if batch {
		input, output = tf32.NewV(2, 4), tf32.NewV(1, 4)
	} else {
		input, output = tf32.NewV(2), tf32.NewV(1)
	}
	w1, b1, w2, b2 := tf32.NewV(2, width), tf32.NewV(width), tf32.NewV(width), tf32.NewV(1)
	parameters, zero := []*tf32.V{&w1, &b1, &w2, &b2}, []*tf32.V{}
	m1, m2, m1a, m2a := w1.Meta(), w2.Meta(), b1.Meta(), b2.Meta()
	if dct {
		t1, tt1 := DCT2(2)
		t2, tt2 := DCT2(width)
		t3, tt3 := DCT2(width)
		t4, tt4 := DCT2(1)
		w1b, b1b, w2b, b2b := tf32.NewV(2, width), tf32.NewV(width), tf32.NewV(width), tf32.NewV(1)
		m1 = tf32.Add(tf32.Mul(tt1.Meta(), tf32.T(tf32.Mul(m1, t1.Meta()))), w1b.Meta())
		m1a = tf32.Add(tf32.Mul(tt2.Meta(), tf32.T(tf32.Mul(m1a, t2.Meta()))), b1b.Meta())
		m2 = tf32.Add(tf32.Mul(tt3.Meta(), tf32.T(tf32.Mul(m2, t3.Meta()))), w2b.Meta())
		m2a = tf32.Add(tf32.Mul(tt4.Meta(), tf32.T(tf32.Mul(m2a, t4.Meta()))), b2b.Meta())
		zero = append(zero, &t1, &tt1, &t2, &tt2, &t3, &tt3, &t4, &tt4)
		parameters = append(parameters, &w1b, &b1b, &w2b, &b2b)
	} else if inception {
		for i := 0; i < depth; i++ {
			a, b := tf32.NewV(2, 2), tf32.NewV(2, width)
			m1 = tf32.Add(tf32.Mul(a.Meta(), b.Meta()), m1)
			parameters = append(parameters, &a, &b)
		}
		for i := 0; i < depth; i++ {
			a, b := tf32.NewV(width, width), tf32.NewV(width)
			m1a = tf32.Add(tf32.Mul(a.Meta(), b.Meta()), m1a)
			parameters = append(parameters, &a, &b)
		}
		for i := 0; i < depth; i++ {
			a, b := tf32.NewV(width, width), tf32.NewV(width)
			m2 = tf32.Add(tf32.Mul(a.Meta(), b.Meta()), m2)
			parameters = append(parameters, &a, &b)
		}
		for i := 0; i < depth; i++ {
			a, b := tf32.NewV(1), tf32.NewV(1)
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
	l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(m2, l1), m2a))
	cost := tf32.Avg(tf32.Quadratic(l2, output.Meta()))

	type Datum struct {
		input        []float32
		output       []float32
		deltas, m, v [][]float32
	}
	data := [...]Datum{
		{
			input:  []float32{0, 0},
			output: []float32{0},
		},
		{
			input:  []float32{1, 0},
			output: []float32{1},
		},
		{
			input:  []float32{0, 1},
			output: []float32{1},
		},
		{
			input:  []float32{1, 1},
			output: []float32{0},
		},
	}
	table := make([]*Datum, len(data))
	for i := range data {
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
	alpha, eta := float32(.1), float32(.6)
	// adam parameters
	a, beta1, beta2, epsilon := float32(.001), float32(.9), float32(.999), float32(1E-8)
	optimize := func(i int) {
		for k, p := range parameters {
			for l, d := range p.D {
				switch optimizer {
				case OptimizerStatic:
					p.X[l] -= eta * d
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

	if batch {
		inputs, outputs := make([]float32, 0, 16), make([]float32, 0, 4)
		for i := range table {
			inputs = append(inputs, table[i].input...)
			outputs = append(outputs, table[i].output...)
		}
		input.Set(inputs)
		output.Set(outputs)
		for i := 0; i < 10000; i++ {
			for _, p := range parameters {
				p.Zero()
			}
			for _, p := range zero {
				p.Zero()
			}
			total := tf32.Gradient(cost).X[0]
			optimize(i)
			costs = append(costs, total)
			if total < .01 {
				converged = true
				break
			}
		}
	} else {
		for i := 0; i < 10000; i++ {
			for i := range table {
				j := i + rnd.Intn(len(data)-i)
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
				input.Set(table[j].input)
				output.Set(table[j].output)
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
			if total < .1 {
				converged = true
				break
			}
		}
	}

	if converged {
		for i := range data {
			input.X[0], input.X[1] = data[i].input[0], data[i].input[1]
			var output tf32.V
			l2(func(a *tf32.V) {
				output = *a
			})
			if data[i].output[0] == 1 && output.X[0] < .5 {
				panic(fmt.Sprintf("%v output should be 1 %f %v %v", context, output.X[0], data[i].input, data[i].output))
			} else if data[i].output[0] == 0 && output.X[0] >= .5 {
				panic(fmt.Sprintf("%v output should be 0 %f %v %v", context, output.X[0], data[i].input, data[i].output))
			}
		}
	}

	return Result{
		Costs:     costs,
		Converged: converged,
	}
}

// RunXORRepeatedExperiment runs multiple xor experiments
func RunXORRepeatedExperiment() {
	run := func(optimizer Optimizer) (normalStats, inceptionStats Statistics) {
		normalStats.Mode, inceptionStats.Mode = "normal", "inception"
		normalStats.Optimizer, inceptionStats.Optimizer = optimizer, optimizer
		experiment := func(seed int64, inception, context bool, results chan<- Result) {
			results <- XORExperiment(seed, 3, 16, optimizer, true, inception, false, context)
		}
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
		return
	}

	statistics := []Statistics{}
	for _, optimizer := range Optimizers {
		normalStats, inceptionStats := run(optimizer)
		statistics = append(statistics, normalStats, inceptionStats)
	}
	sort.Slice(statistics, func(i, j int) bool {
		return statistics[i].AverageEpochs() < statistics[j].AverageEpochs()
	})

	fmt.Println("| Mode | Optimizer | Converged | Epochs |")
	fmt.Println("| ---- | --------- | --------- | ------ |")
	for _, statistic := range statistics {
		fmt.Printf("| %s | %s | %f | %f |\n",
			statistic.Mode,
			statistic.Optimizer.String(),
			statistic.ConvergenceProbability(),
			statistic.AverageEpochs())
	}
}

// RunXORExperiment runs an xor experiment once
func RunXORExperiment(seed int64) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "xor epochs"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "cost"
	p.Legend.Top = true

	index := 0
	for _, optimizer := range Optimizers {
		normal := XORExperiment(seed, 3, 16, optimizer, true, false, false, false)
		inception := XORExperiment(seed, 3, 16, optimizer, true, true, false, false)

		pointsNormal := make(plotter.XYs, 0, len(normal.Costs))
		for i, cost := range normal.Costs {
			pointsNormal = append(pointsNormal, plotter.XY{X: float64(i), Y: float64(cost)})
		}

		pointsInception := make(plotter.XYs, 0, len(inception.Costs))
		for i, cost := range inception.Costs {
			pointsInception = append(pointsInception, plotter.XY{X: float64(i), Y: float64(cost)})
		}

		normalScatter, err := plotter.NewScatter(pointsNormal)
		if err != nil {
			panic(err)
		}
		normalScatter.GlyphStyle.Radius = vg.Length(1)
		normalScatter.GlyphStyle.Shape = draw.CircleGlyph{}
		normalScatter.GlyphStyle.Color = colors[index]
		normalScatter.GlyphStyle.Radius = 2
		index++

		inceptionScatter, err := plotter.NewScatter(pointsInception)
		if err != nil {
			panic(err)
		}
		inceptionScatter.GlyphStyle.Radius = vg.Length(1)
		inceptionScatter.GlyphStyle.Shape = draw.CircleGlyph{}
		inceptionScatter.GlyphStyle.Color = colors[index]
		inceptionScatter.GlyphStyle.Radius = 2
		index++

		p.Add(normalScatter, inceptionScatter)
		p.Legend.Add(fmt.Sprintf("normal %s", optimizer.String()), normalScatter)
		p.Legend.Add(fmt.Sprintf("inception %s", optimizer.String()), inceptionScatter)
	}

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost_xor.png")
	if err != nil {
		panic(err)
	}
}
