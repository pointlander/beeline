// Copyright 2019 The BeeLine Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"

	"github.com/petar/GoMNIST"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const Pixels = GoMNIST.Width * GoMNIST.Height

func mnist() {
	train, test, err := GoMNIST.Load("./data/")
	if err != nil {
		panic(err)
	}
	fmt.Println(len(train.Images), len(test.Images))
	load := func(set *GoMNIST.Set) []TrainingData {
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
	trainData := load(train)

	network := NewNetwork(OptionNone(Pixels), OptionSigmoid(Pixels/4), OptionSigmoid(10))
	epochs := network.Train(trainData, true, .001, .4, .6, 1)
	fmt.Println(len(epochs))
}

var labelMap = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

func beeline(shared bool, name string) {
	rand.Seed(1)

	input, err := os.Open("data/iris.csv")
	if err != nil {
		panic(err)
	}
	defer input.Close()
	reader := csv.NewReader(input)
	data := make([]TrainingData, 0, 256)
	line, err := reader.Read()
	for err == nil {
		inputs := make([]float32, 4)
		for i := range inputs {
			value, err := strconv.ParseFloat(line[i], 32)
			if err != nil {
				panic(err)
			}
			inputs[i] = float32(value)
		}
		data = append(data, TrainingData{
			Inputs: inputs,
			Output: labelMap[line[4]],
		})
		line, err = reader.Read()
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
