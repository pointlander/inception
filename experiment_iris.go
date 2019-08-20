// Copyright 2019 The Inception Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
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

// IrisNetwork is an irs neural network
type IrisNetwork struct {
	Rnd           *rand.Rand
	Iris          []*iris.Iris
	BatchSize     int
	Input, Output *tf32.V
	Parameters    []*tf32.V
	Genome        [][]*tf32.V
	Cost          tf32.Meta
	Fitness       float32
}

// NewIrisNetwork creates a new iris network
func NewIrisNetwork(rnd *rand.Rand, seed int64, width, depth int) IrisNetwork {
	once.Do(load)

	random32 := func(a, b float32) float32 {
		if rnd == nil {
			return 0
		}
		return (b-a)*rnd.Float32() + a
	}
	batchSize := 10

	input, output := tf32.NewV(4, batchSize), tf32.NewV(3, batchSize)
	w1, b1, w2, b2 := tf32.NewV(4, width), tf32.NewV(width), tf32.NewV(width, 3), tf32.NewV(3)
	parameters := []*tf32.V{&w1, &b1, &w2, &b2}
	genome := make([][]*tf32.V, 4)

	m1, m2, m1a, m2a := w1.Meta(), w2.Meta(), b1.Meta(), b2.Meta()
	for i := 0; i < depth; i++ {
		a, b := tf32.NewV(4, 4), tf32.NewV(4, width)
		m1 = tf32.Add(tf32.Mul(a.Meta(), b.Meta()), m1)
		parameters = append(parameters, &a, &b)
		genome[0] = append(genome[0], &a, &b)
	}
	for i := 0; i < depth; i++ {
		a, b := tf32.NewV(width, width), tf32.NewV(width)
		m1a = tf32.Add(tf32.Mul(a.Meta(), b.Meta()), m1a)
		parameters = append(parameters, &a, &b)
		genome[1] = append(genome[1], &a, &b)
	}
	for i := 0; i < depth; i++ {
		a, b := tf32.NewV(width, width), tf32.NewV(width, 3)
		m2 = tf32.Add(tf32.Mul(a.Meta(), b.Meta()), m2)
		parameters = append(parameters, &a, &b)
		genome[2] = append(genome[2], &a, &b)
	}
	for i := 0; i < depth; i++ {
		a, b := tf32.NewV(3, 3), tf32.NewV(3)
		m2a = tf32.Add(tf32.Mul(a.Meta(), b.Meta()), m2a)
		parameters = append(parameters, &a, &b)
		genome[3] = append(genome[3], &a, &b)
	}

	for _, p := range parameters {
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, random32(-1, 1))
		}
	}

	l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(m1, input.Meta()), m1a))
	l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(m2, l1), m2a))
	cost := tf32.Avg(tf32.CrossEntropy(l2, output.Meta()))

	data := make([]*iris.Iris, len(datum.Fisher))
	for i := range data {
		data[i] = &datum.Fisher[i]
	}

	return IrisNetwork{
		Rnd:        rand.New(rand.NewSource(seed)),
		Iris:       data,
		BatchSize:  batchSize,
		Input:      &input,
		Output:     &output,
		Parameters: parameters,
		Genome:     genome,
		Cost:       cost,
	}
}

// Fit get the fitness of the network
func (i *IrisNetwork) Fit() float32 {
	length := len(i.Iris)
	total := float32(0.0)
	for j := 0; j < length; j += i.BatchSize {
		inputs, outputs := make([]float32, 0, 4*i.BatchSize), make([]float32, 0, 3*i.BatchSize)
		for k := 0; k < i.BatchSize; k++ {
			index := (j + k) % length
			for _, measure := range i.Iris[index].Measures {
				inputs = append(inputs, float32(measure))
			}
			out := make([]float32, 3)
			out[iris.Labels[i.Iris[index].Label]] = 1
			outputs = append(outputs, out...)
		}
		i.Input.Set(inputs)
		i.Output.Set(outputs)
		total += tf32.Gradient(i.Cost).X[0]
	}
	i.Fitness = total
	return total
}

// Mutate mutates the network with gradient descent
func (i *IrisNetwork) Mutate() float32 {
	length := len(i.Iris)
	for a := range i.Iris {
		b := a + i.Rnd.Intn(length-a)
		i.Iris[a], i.Iris[b] = i.Iris[b], i.Iris[a]
	}

	total := float32(0.0)
	for j := 0; j < length; j += i.BatchSize {
		for _, p := range i.Parameters {
			p.Zero()
		}

		inputs, outputs := make([]float32, 0, 4*i.BatchSize), make([]float32, 0, 3*i.BatchSize)
		for k := 0; k < i.BatchSize; k++ {
			index := (j + k) % length
			for _, measure := range i.Iris[index].Measures {
				inputs = append(inputs, float32(measure))
			}
			out := make([]float32, 3)
			out[iris.Labels[i.Iris[index].Label]] = 1
			outputs = append(outputs, out...)
		}
		i.Input.Set(inputs)
		i.Output.Set(outputs)
		total += tf32.Gradient(i.Cost).X[0]
		eta, norm := float32(.1), float32(0)
		for _, p := range i.Parameters {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 1 {
			scaling := 1 / norm
			for _, p := range i.Parameters {
				for l, d := range p.D {
					d *= scaling
					p.X[l] -= eta * d
				}
			}
		} else {
			for _, p := range i.Parameters {
				for l, d := range p.D {
					p.X[l] -= eta * d
				}
			}
		}
	}
	i.Fitness = total
	return total
}

// IrisParallelExperiment runs parallel version of experiment
func IrisParallelExperiment(seed int64, depth int) (generatrions int) {
	rnd := rand.New(rand.NewSource(seed))
	networks := make([]IrisNetwork, 100)
	for i := range networks {
		networks[i] = NewIrisNetwork(rnd, seed+int64(i), 3, depth)
	}
	done := make(chan float32, 8)
	fit := func(n *IrisNetwork) {
		done <- n.Fit()
	}
	mutate := func(n *IrisNetwork) {
		done <- n.Mutate()
	}

	generatrions, rnd = 1000, rand.New(rand.NewSource(seed))
	for i := 0; i < 1000; i++ {
		for j := range networks {
			go mutate(&networks[j])
		}
		for j := 0; j < 100; j++ {
			<-done
		}

		tf32.Static.InferenceOnly = true
		for j := range networks {
			go fit(&networks[j])
		}
		for j := 0; j < 100; j++ {
			<-done
		}
		tf32.Static.InferenceOnly = false
		sort.Slice(networks, func(i, j int) bool {
			return networks[i].Fitness < networks[j].Fitness
		})
		//fmt.Println(i, networks[0].Fitness)
		if networks[0].Fitness < 13/float32(networks[0].BatchSize) {
			generatrions = i
			break
		}

		index := 50
		for j := 0; j < 25; j++ {
			a, b := rnd.Intn(50), rnd.Intn(50)
			for a == b {
				b = rnd.Intn(50)
			}

			childa := &networks[index]
			for k, p := range childa.Parameters {
				copy(p.X, networks[a].Parameters[k].X)
			}
			index++

			childb := &networks[index]
			for k, p := range childb.Parameters {
				copy(p.X, networks[b].Parameters[k].X)
			}
			index++

			set, x, y := rnd.Intn(4), rnd.Intn(depth), rnd.Intn(depth)
			childa.Genome[set][x*2].X, childb.Genome[set][y*2].X =
				childb.Genome[set][y*2].X, childa.Genome[set][x*2].X
			childa.Genome[set][x*2+1].X, childb.Genome[set][y*2+1].X =
				childb.Genome[set][y*2+1].X, childa.Genome[set][x*2+1].X
		}
	}
	return
}

// RunIrisRepeatedParallelExperiment runs iris prarallel experiment repeatedly
func RunIrisRepeatedParallelExperiment() {
	total := 0
	for i := 0; i < 256; i++ {
		generations := IrisParallelExperiment(int64(i)+1, 4)
		total += generations
		fmt.Println(i, generations, float64(total)/float64(i+1))
	}
	fmt.Printf("generations=%f\n", float64(total)/256)
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
		} else {
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
	run := func(optimizer Optimizer, batch bool) (normalStats, inceptionStats Statistics) {
		normalStats.Mode, inceptionStats.Mode = "normal", "inception"
		normalStats.Optimizer, inceptionStats.Optimizer = optimizer, optimizer
		if batch {
			normalStats.Batch, inceptionStats.Batch = 10, 10
		} else {
			normalStats.Batch, inceptionStats.Batch = 1, 1
		}
		experiment := func(seed int64, inception, context bool, results chan<- Result) {
			results <- IrisExperiment(seed, 3, 4, optimizer, batch, inception, false, context)
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
		normalStats, inceptionStats := run(optimizer, false)
		statistics = append(statistics, normalStats, inceptionStats)

		normalStats, inceptionStats = run(optimizer, true)
		statistics = append(statistics, normalStats, inceptionStats)
	}
	sort.Slice(statistics, func(i, j int) bool {
		return statistics[i].AverageEpochs() < statistics[j].AverageEpochs()
	})

	headers := []string{"Mode", "Optimizer", "Batch", "Converged", "Epochs"}
	sizes, results := make([]int, 5), make([][5]string, len(statistics))
	for i, header := range headers {
		sizes[i] = len(header)
	}
	for i, statistic := range statistics {
		results[i][0] = statistic.Mode
		if length := len(results[i][0]); length > sizes[0] {
			sizes[0] = length
		}
		results[i][1] = statistic.Optimizer.String()
		if length := len(results[i][1]); length > sizes[1] {
			sizes[1] = length
		}
		results[i][2] = fmt.Sprintf("%d", statistic.Batch)
		if length := len(results[i][2]); length > sizes[2] {
			sizes[2] = length
		}
		results[i][3] = fmt.Sprintf("%f", statistic.ConvergenceProbability())
		if length := len(results[i][3]); length > sizes[3] {
			sizes[3] = length
		}
		results[i][4] = fmt.Sprintf("%f", statistic.AverageEpochs())
		if length := len(results[i][4]); length > sizes[4] {
			sizes[4] = length
		}
	}

	fmt.Printf("| ")
	for i, header := range headers {
		fmt.Printf("%s", header)
		spaces := sizes[i] - len(header)
		for spaces > 0 {
			fmt.Printf(" ")
			spaces--
		}
		fmt.Printf(" | ")
	}
	fmt.Printf("\n| ")
	for i, header := range headers {
		dashes := len(header)
		if sizes[i] > dashes {
			dashes = sizes[i]
		}
		for dashes > 0 {
			fmt.Printf("-")
			dashes--
		}
		fmt.Printf(" | ")
	}
	fmt.Printf("\n")
	for _, row := range results {
		fmt.Printf("| ")
		for i, entry := range row {
			spaces := sizes[i] - len(entry)
			fmt.Printf("%s", entry)
			for spaces > 0 {
				fmt.Printf(" ")
				spaces--
			}
			fmt.Printf(" | ")
		}
		fmt.Printf("\n")
	}
}

// RunIrisExperiment runs an iris experiment once
func RunIrisExperiment(seed int64) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "iris epochs"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "cost"
	p.Legend.Top = true

	index := 0
	for _, optimizer := range Optimizers {
		normal := IrisExperiment(seed, 3, 4, optimizer, true, false, false, false)
		inception := IrisExperiment(seed, 3, 4, optimizer, true, true, false, false)

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

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost_iris.png")
	if err != nil {
		panic(err)
	}
}
