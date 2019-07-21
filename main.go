// Copyright 2019 The Inception Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"image/color"
	"math"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf32"
)

var datum iris.Datum

func init() {
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
}

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

// XORExperiment xor neural network experiment
func XORExperiment(seed int64, width int, inception, context bool) Result {
	rnd, costs, converged := rand.New(rand.NewSource(seed)), make([]float32, 0, 1000), false
	random32 := func(a, b float32) float32 {
		return (b-a)*rnd.Float32() + a
	}

	input, output := tf32.NewV(2), tf32.NewV(1)
	w1, w1a, w1b := tf32.NewV(2, width), tf32.NewV(2, 2), tf32.NewV(2, width)
	b1, b1a, b1b := tf32.NewV(width), tf32.NewV(width, width), tf32.NewV(width)
	w2, w2a, w2b := tf32.NewV(width), tf32.NewV(width, width), tf32.NewV(width)
	b2, b2a, b2b := tf32.NewV(1), tf32.NewV(1), tf32.NewV(1)
	parameters := []*tf32.V{&w1, &w1a, &w1b, &b1, &b1a, &b1b, &w2, &w2a, &w2b, &b2, &b2a, &b2b}
	var deltas [][]float32
	for _, p := range parameters {
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, random32(-1, 1))
		}
		deltas = append(deltas, make([]float32, len(p.X)))
	}
	m1, m2, m1a, m2a := w1.Meta(), w2.Meta(), b1.Meta(), b2.Meta()
	if inception {
		m1 = tf32.Add(tf32.Mul(w1a.Meta(), m1), w1b.Meta())
		m1a = tf32.Add(tf32.Mul(b1a.Meta(), m1a), b1b.Meta())
		m2 = tf32.Add(tf32.Mul(w2a.Meta(), m2), w2b.Meta())
		m2a = tf32.Add(tf32.Mul(b2a.Meta(), m2a), b2b.Meta())
	}
	l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(m1, input.Meta()), m1a))
	l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(m2, l1), m2a))
	cost := tf32.Quadratic(l2, output.Meta())

	type Datum struct {
		input  []float32
		output []float32
		deltas [][]float32
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
		for _, p := range parameters {
			data[i].deltas = append(data[i].deltas, make([]float32, len(p.X)))
		}
		table[i] = &data[i]
	}

	alpha, eta := float32(.1), float32(.6)
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
			input.Set(table[j].input)
			output.Set(table[j].output)
			total += tf32.Gradient(cost).X[0]
			deltas := deltas
			if context {
				deltas = table[j].deltas
			}
			for k, p := range parameters {
				for l, d := range p.D {
					deltas[k][l] = alpha*deltas[k][l] - eta*d
					p.X[l] += deltas[k][l]
				}
			}
		}
		costs = append(costs, total)
		if total < .001 {
			converged = true
			break
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

// IrisExperiment iris neural network experiment
func IrisExperiment(seed int64, width int, inception, context bool) Result {
	rnd, costs, converged, misses := rand.New(rand.NewSource(seed)), make([]float32, 0, 1000), false, 0
	random32 := func(a, b float32) float32 {
		return (b-a)*rnd.Float32() + a
	}

	input, output := tf32.NewV(4), tf32.NewV(3)
	w1, w1a, w1b := tf32.NewV(4, width), tf32.NewV(4, 4), tf32.NewV(4, width)
	b1, b1a, b1b := tf32.NewV(width), tf32.NewV(width, width), tf32.NewV(width)
	w2, w2a, w2b := tf32.NewV(width, 3), tf32.NewV(width, width), tf32.NewV(width, 3)
	b2, b2a, b2b := tf32.NewV(3), tf32.NewV(3, 3), tf32.NewV(3)
	parameters := []*tf32.V{&w1, &w1a, &w1b, &b1, &b1a, &b1b, &w2, &w2a, &w2b, &b2, &b2a, &b2b}
	var deltas [][]float32
	for _, p := range parameters {
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, random32(-1, 1))
		}
		deltas = append(deltas, make([]float32, len(p.X)))
	}
	m1, m2, m1a, m2a := w1.Meta(), w2.Meta(), b1.Meta(), b2.Meta()
	if inception {
		m1 = tf32.Add(tf32.Mul(w1a.Meta(), m1), w1b.Meta())
		m1a = tf32.Add(tf32.Mul(b1a.Meta(), m1a), b1b.Meta())
		m2 = tf32.Add(tf32.Mul(w2a.Meta(), m2), w2b.Meta())
		m2a = tf32.Add(tf32.Mul(b2a.Meta(), m2a), b2b.Meta())
	}
	l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(m1, input.Meta()), m1a))
	l2 := tf32.Softmax(tf32.Add(tf32.Mul(m2, l1), m2a))
	cost := tf32.CrossEntropy(l2, output.Meta())

	type Datum struct {
		iris   *iris.Iris
		deltas [][]float32
	}
	data := make([]Datum, len(datum.Fisher))
	table := make([]*Datum, len(data))
	for i := range data {
		data[i].iris = &datum.Fisher[i]
		for _, p := range parameters {
			data[i].deltas = append(data[i].deltas, make([]float32, len(p.X)))
		}
		table[i] = &data[i]
	}

	alpha, eta := float32(.1), float32(.1)
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
			in := make([]float32, len(table[j].iris.Measures))
			for i, measure := range table[j].iris.Measures {
				in[i] = float32(measure)
			}
			input.Set(in)
			out := make([]float32, 3)
			out[iris.Labels[table[j].iris.Label]] = 1
			output.Set(out)
			total += tf32.Gradient(cost).X[0]
			norm := float32(0)
			for _, p := range parameters {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = float32(math.Sqrt(float64(norm)))
			deltas := deltas
			if context {
				deltas = table[j].deltas
			}
			if norm > 1 {
				scaling := 1 / norm
				for k, p := range parameters {
					for l, d := range p.D {
						deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
						p.X[l] += deltas[k][l]
					}
				}
			} else {
				for k, p := range parameters {
					for l, d := range p.D {
						deltas[k][l] = alpha*deltas[k][l] - eta*d
						p.X[l] += deltas[k][l]
					}
				}
			}
		}
		costs = append(costs, total)
		if total < 10 {
			converged = true
			break
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

// RunXORRepeatedExperiment runs multiple xor experiments
func RunXORRepeatedExperiment() {
	experiment := func(seed int64, inception, context bool, results chan<- Result) {
		results <- XORExperiment(seed, 3, inception, context)
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

// RunXORExperiment runs an xor experiment once
func RunXORExperiment(seed int64) {
	normal := XORExperiment(seed, 3, false, false)
	inception := XORExperiment(seed, 3, true, false)

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

	p.Title.Text = "xor epochs"
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

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost_xor.png")
	if err != nil {
		panic(err)
	}
}

// RunIrisRepeatedExperiment runs multiple iris experiments
func RunIrisRepeatedExperiment() {
	experiment := func(seed int64, inception, context bool, results chan<- Result) {
		results <- IrisExperiment(seed, 3, inception, context)
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
	normal := IrisExperiment(seed, 3, false, false)
	inception := IrisExperiment(seed, 3, true, false)

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
