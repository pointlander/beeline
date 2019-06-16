// Copyright 2019 The BeeLine Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"

	"github.com/petar/GoMNIST"
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

	network := NewNetwork(Pixels, Pixels/4, 10)
	iterations := network.Train(trainData, .001, .4, .6)
	fmt.Println(iterations)

}

var labelMap = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

func main() {
	input, err := os.Open("data/iris.csv")
	if err != nil {
		panic(err)
	}
	defer input.Close()
	reader := csv.NewReader(input)
	data := make([]TrainingData, 0, 256)
	line, err := reader.Read()
	for err == nil {
		inputs, outputs := make([]float32, 4), make([]float32, 3)
		for i := range inputs {
			value, err := strconv.ParseFloat(line[i], 32)
			if err != nil {
				panic(err)
			}
			inputs[i] = float32(value)
		}
		outputs[labelMap[line[4]]] = 1
		data = append(data, TrainingData{
			Inputs:  inputs,
			Outputs: outputs,
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
	fmt.Println(data)

	network := NewNetwork(4, 5, 3)
	iterations := network.Train(data, 4, .3, .7)
	fmt.Println(iterations)

	state, fails := network.NewNetState(), 0
	for _, item := range data {
		for i, value := range item.Inputs {
			state.State[0][i] = Dual{Val: value}
		}
		state.Inference()
		should := 0
		for i, value := range item.Outputs {
			if value > 0 {
				should = i
				break
			}
		}
		is, max := 0, float32(0.0)
		for i, value := range state.State[2] {
			if value.Val > max {
				is, max = i, value.Val
			}
		}
		if should != is {
			fails++
		}
	}
	fmt.Println(fails)
}