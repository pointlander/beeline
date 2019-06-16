// Copyright 2019 The BeeLine Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestDual(t *testing.T) {
	x, y := Dual{Val: 5, Der: 1}, Dual{Val: 6}
	f := Mul(Pow(x, 2), y)
	if math.Round(float64(f.Der)) != 60.0 {
		t.Fatal("derivative should be 60")
	}
}

func TestNetwork(t *testing.T) {
	rand.Seed(1)
	network := NewNetwork(2, 2, 1)
	data := []TrainingData{
		{
			[]float32{0, 0}, []float32{0},
		},
		{
			[]float32{1, 0}, []float32{1},
		},
		{
			[]float32{0, 1}, []float32{1},
		},
		{
			[]float32{1, 1}, []float32{0},
		},
	}
	iterations := network.Train(data, false, .001, .4, .6, 1)
	t.Log(iterations)
	state := network.NewNetState()
	for _, item := range data {
		for i, input := range item.Inputs {
			state.State[0][i].Val = input
		}
		state.Inference()
		output := state.State[2][0].Val > .5
		expected := item.Outputs[0] > .5
		if output != expected {
			t.Fatal(state.State[2][0].Val, item)
		}
	}
}

func TestNetworkCCNOT(t *testing.T) {
	rand.Seed(1)
	network := NewNetwork(3, 3, 1)
	data := []TrainingData{
		{
			[]float32{0, 0, 0}, []float32{0},
		},
		{
			[]float32{1, 0, 0}, []float32{0},
		},
		{
			[]float32{0, 1, 0}, []float32{0},
		},
		{
			[]float32{1, 1, 0}, []float32{1},
		},
		{
			[]float32{0, 0, 1}, []float32{1},
		},
		{
			[]float32{1, 0, 1}, []float32{1},
		},
		{
			[]float32{0, 1, 1}, []float32{1},
		},
		{
			[]float32{1, 1, 1}, []float32{0},
		},
	}
	iterations := network.Train(data, false, .001, .4, .6, 1)
	t.Log(iterations)
	state := network.NewNetState()
	for _, item := range data {
		for i, input := range item.Inputs {
			state.State[0][i].Val = input
		}
		state.Inference()
		output := state.State[2][0].Val > .5
		expected := item.Outputs[0] > .5
		if output != expected {
			t.Fatal(state.State[2][0].Val, item)
		}
	}
}
