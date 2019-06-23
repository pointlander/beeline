// Copyright 2019 The BeeLine Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/datum/mnist"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const Pixels = mnist.Width * mnist.Height

func mnistNetwork() {
	datum, err := mnist.Load()
	if err != nil {
		panic(err)
	}
	fmt.Println(len(datum.Train.Images), len(datum.Test.Images))
	load := func(set mnist.Set) []TrainingData {
		data := make([]TrainingData, len(set.Images))
		for i, image := range set.Images {
			inputs, outputs := make([]float32, Pixels), make([]float32, 10)
			for j, pixel := range image {
				if pixel > 0 {
					inputs[j] = 1
				}
			}
			outputs[set.Labels[i]] = 1
			data[i] = TrainingData{
				Inputs:  inputs,
				Outputs: outputs,
			}
		}
		return data
	}
	trainData := load(datum.Train)

	network := NewNetwork(OptionNone(Pixels), OptionSigmoid(Pixels/4), OptionSigmoid(10))
	epochs := network.Train(trainData, true, .001, .4, .6, 1)
	fmt.Println(len(epochs))
}

func beeline(shared bool, name string) {
	rand.Seed(1)

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	data := make([]TrainingData, 0, 256)
	for _, line := range datum.Fisher {
		inputs := make([]float32, 4)
		for i := range inputs {
			inputs[i] = float32(line.Measures[i])
		}
		data = append(data, TrainingData{
			Inputs: inputs,
			Output: iris.Labels[line.Label],
		})
	}
	var maxValues [4]float32
	for _, item := range data {
		for i, value := range item.Inputs {
			if value > maxValues[i] {
				maxValues[i] = value
			}
		}
	}
	for _, item := range data {
		for i, value := range item.Inputs {
			item.Inputs[i] = value / maxValues[i]
			if math.IsNaN(float64(item.Inputs[i])) {
				panic("bad input")
			}
		}
	}

	network := NewNetwork(OptionNone(4), OptionSigmoid(2), OptionSoftmax(3), OptionShared(shared))
	epochs := network.Train(data, true, 10, .1, .9, 1)
	fmt.Println("iterations=", len(epochs))

	state, fails := network.NewNetState(), 0
	for _, item := range data {
		for i, value := range item.Inputs {
			state.State[0][i] = Dual{Val: value}
		}
		state.Inference()
		is, max := 0, float32(0.0)
		for i, value := range state.State[2] {
			if value.Val > max {
				is, max = i, value.Val
			}
		}
		if item.Output != is {
			fails++
		}
	}
	fmt.Println("fails=", fails)

	points := make(plotter.XYs, 0, len(epochs))
	for i, epoch := range epochs {
		points = append(points, plotter.XY{X: float64(i), Y: epoch})
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "epochs"
	p.X.Label.Text = "time"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("epochs_%s.png", name))
	if err != nil {
		panic(err)
	}
}

func main() {
	beeline(true, "shared")
	beeline(false, "normal")
}
