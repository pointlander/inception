// Copyright 2019 The Inception Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"testing"

	"github.com/pointlander/gradient/tf32"
)

func TestDCT(t *testing.T) {
	T, Tt := DCT2(8)
	random32 := func(a, b float32) float32 {
		return (b-a)*rand.Float32() + a
	}
	round := func(a float32) float32 {
		return float32(math.Round(float64(a)*1000) / 1000)
	}
	x := make([]float32, 8*8)
	for i := range x {
		x[i] = random32(-1, 1)
	}
	t.Log(x)
	x1, x2 := tf32.NewV(8, 8), tf32.NewV(8, 8)
	x1.Set(x)
	dct := tf32.Mul(T.Meta(), tf32.T(tf32.Mul(x1.Meta(), Tt.Meta())))
	idct := tf32.Mul(Tt.Meta(), tf32.T(tf32.Mul(x2.Meta(), T.Meta())))
	dct(func(a *tf32.V) {
		x2.Set(a.X)
		idct(func(a *tf32.V) {
			t.Log(a.X)
			for i, value := range x {
				if round(value) != round(a.X[i]) {
					t.Fatal("values should be equal", value, a.X[i])
				}
			}
		})
	})
}
