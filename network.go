// Copyright 2019 The BeeLine Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Weight struct {
	Weight          Dual
	Delta, Gradient float32
}

type Network struct {
	Sizes  []int
	Layers [][]Weight
	Biases [][]Weight
}

func random32(a, b float32) float32 {
	return (b-a)*rand.Float32() + a
}

func NewNetwork(sizes ...int) Network {
	last, layers, biases := sizes[0], make([][]Weight, len(sizes)-1), make([][]Weight, len(sizes)-1)
	for i, size := range sizes[1:] {
		layers[i] = make([]Weight, last*size)
		for j := range layers[i] {
			layers[i][j].Weight.Val = random32(-1, 1) / float32(math.Sqrt(float64(last)))
			if Debug {
				layers[i][j].Weight.Expr = &Expr{
					Name: fmt.Sprintf("w%d_%d", i, j),
					Val:  &layers[i][j].Weight.Val,
					Der:  &layers[i][j].Weight.Der,
				}
			}
		}
		biases[i] = make([]Weight, size)
		for j := range biases[i] {
			biases[i][j].Weight.Val = random32(-1, 1) / float32(math.Sqrt(float64(last)))
			if Debug {
				biases[i][j].Weight.Expr = &Expr{
					Name: fmt.Sprintf("b%d_%d", i, j),
					Val:  &biases[i][j].Weight.Val,
					Der:  &biases[i][j].Weight.Der,
				}
			}
		}
		last = size
	}
	return Network{
		Sizes:  sizes,
		Layers: layers,
		Biases: biases,
	}
}

type NetState struct {
	*Network
	State [][]Dual
}

func (n *Network) NewNetState() NetState {
	state := make([][]Dual, len(n.Sizes))
	for i, size := range n.Sizes {
		state[i] = make([]Dual, size)
	}
	if Debug {
		for i := range state[0] {
			state[0][i].Expr = &Expr{
				Name: fmt.Sprintf("i%d", i),
				Val:  &state[0][i].Val,
				Der:  &state[0][i].Der,
			}
		}
	}
	return NetState{
		Network: n,
		State:   state,
	}
}

func (n *NetState) Inference() {
	for i, layer := range n.Layers {
		w := 0
		for j := 0; j < n.Sizes[i+1]; j++ {
			var sum Dual
			for _, activation := range n.State[i] {
				sum = Add(sum, Mul(activation, layer[w].Weight))
				sum = Add(sum, n.Biases[i][j].Weight)
				w++
			}
			n.State[i+1][j] = Sigmoid(sum)
		}
	}
}

type TrainingData struct {
	Inputs, Outputs []float32
}

func (n *Network) Train(data []TrainingData, verbose bool, target float64, alpha, eta, threshold float32) int {
	size := len(data)
	iterations, state, randomized := 0, n.NewNetState(), make([]TrainingData, size)
	copy(randomized, data)
	for {
		for i, sample := range randomized {
			j := i + rand.Intn(size-i)
			randomized[i], randomized[j] = randomized[j], sample
		}

		total := 0.0
		for _, item := range randomized {
			cost := 0.0
			for j, input := range item.Inputs {
				state.State[0][j].Val = input
			}
			for _, layer := range n.Layers {
				for j := range layer {
					layer[j].Weight.Der = 1.0
					state.Inference()
					var sum Dual
					for k, output := range item.Outputs {
						right := Dual{Val: output}
						if Debug {
							right.Expr = &Expr{
								Name: fmt.Sprintf("o%d", k),
								Val:  &right.Val,
								Der:  &right.Der,
							}
						}
						sub := Sub(state.State[len(state.State)-1][k], right)
						sum = Add(sum, Mul(sub, sub))
					}
					sum = Mul(Half, sum)
					layer[j].Weight.Der = 0.0
					layer[j].Gradient = sum.Der
					cost = float64(sum.Val)
				}
			}
			for _, bias := range n.Biases {
				for j := range bias {
					bias[j].Weight.Der = 1.0
					state.Inference()
					var sum Dual
					for k, output := range item.Outputs {
						right := Dual{Val: output}
						if Debug {
							right.Expr = &Expr{
								Name: fmt.Sprintf("o%d", k),
								Val:  &right.Val,
								Der:  &right.Der,
							}
						}
						sub := Sub(state.State[len(state.State)-1][k], right)
						sum = Add(sum, Mul(sub, sub))
					}
					sum = Mul(Half, sum)
					bias[j].Weight.Der = 0.0
					bias[j].Gradient = sum.Der
					cost = float64(sum.Val)
				}
			}
			total += cost
			norm := float32(0)
			for _, layer := range n.Layers {
				for j := range layer {
					value := layer[j].Gradient
					norm += value * value
				}
			}
			for _, bias := range n.Biases {
				for j := range bias {
					value := bias[j].Gradient
					norm += value * value
				}
			}
			norm = float32(math.Sqrt(float64(norm)))
			if norm > threshold {
				scaling := threshold / norm
				for _, layer := range n.Layers {
					for j := range layer {
						layer[j].Delta = alpha*layer[j].Delta - eta*layer[j].Gradient*scaling
						layer[j].Weight.Val += layer[j].Delta
					}
				}
				for _, bias := range n.Biases {
					for j := range bias {
						bias[j].Delta = alpha*bias[j].Delta - eta*bias[j].Gradient*scaling
						bias[j].Weight.Val += bias[j].Delta
					}
				}
			} else {
				for _, layer := range n.Layers {
					for j := range layer {
						layer[j].Delta = alpha*layer[j].Delta - eta*layer[j].Gradient
						layer[j].Weight.Val += layer[j].Delta
					}
				}
				for _, bias := range n.Biases {
					for j := range bias {
						bias[j].Delta = alpha*bias[j].Delta - eta*bias[j].Gradient
						bias[j].Weight.Val += bias[j].Delta
					}
				}
			}
		}
		iterations++
		if verbose {
			fmt.Println(iterations, total)
		}
		if total < target {
			break
		}
	}

	return iterations
}
