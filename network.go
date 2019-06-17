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
	Tag             int
	Weight          Dual
	Delta, Gradient float32
}

type Meta struct {
	Size      int
	Transform Transform
}

type Share struct {
	Sum   float32
	Count int
}

type Network struct {
	Tags   int
	Meta   []Meta
	Layers [][]Weight
	Biases [][]Weight
	Shares map[int]Share
}

func random32(a, b float32) float32 {
	return (b-a)*rand.Float32() + a
}

type Option func(*Network) error

func OptionNone(size int) Option {
	return func(network *Network) error {
		network.Meta = append(network.Meta, Meta{
			Size: size,
		})
		return nil
	}
}

func OptionSigmoid(size int) Option {
	return func(network *Network) error {
		network.Meta = append(network.Meta, Meta{
			Size:      size,
			Transform: SigmoidTransform,
		})
		return nil
	}
}

func OptionSoftmax(size int) Option {
	return func(network *Network) error {
		network.Meta = append(network.Meta, Meta{
			Size:      size,
			Transform: SoftmaxTransform,
		})
		return nil
	}
}

func OptionShared(shared bool) Option {
	if shared {
		return func(network *Network) error {
			network.Shares = make(map[int]Share)
			network.Shares[network.Tags] = Share{}
			network.Tags++
			network.Shares[network.Tags] = Share{}
			network.Tags++
			return nil
		}
	}
	return func(network *Network) error {
		return nil
	}
}

func NewNetwork(options ...Option) Network {
	network := Network{}
	for _, option := range options {
		err := option(&network)
		if err != nil {
			panic(err)
		}
	}
	meta := network.Meta

	var shares []float32
	if network.Shares != nil {
		shares = make([]float32, network.Tags)
		shares[0] = 1
		shares[1] = -1
	}

	last, layers, biases := meta[0].Size, make([][]Weight, len(meta)-1), make([][]Weight, len(meta)-1)
	for i, m := range meta[1:] {
		size := m.Size
		layers[i] = make([]Weight, last*size)
		for j := range layers[i] {
			if network.Shares != nil {
				tag := rand.Intn(network.Tags)
				layers[i][j].Tag = tag
				layers[i][j].Weight.Val = shares[tag]
			} else {
				layers[i][j].Weight.Val = random32(-1, 1) / float32(math.Sqrt(float64(last)))
			}
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
			if network.Shares != nil {
				tag := rand.Intn(network.Tags)
				biases[i][j].Tag = tag
				biases[i][j].Weight.Val = shares[tag]
			} else {
				biases[i][j].Weight.Val = random32(-1, 1) / float32(math.Sqrt(float64(last)))
			}
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
	network.Layers = layers
	network.Biases = biases
	return network
}

type NetState struct {
	*Network
	State [][]Dual
}

func (n *Network) NewNetState() NetState {
	state := make([][]Dual, len(n.Meta))
	for i, m := range n.Meta {
		state[i] = make([]Dual, m.Size)
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
	meta := n.Meta[1:]
	for i, layer := range n.Layers {
		w := 0
		for j := 0; j < meta[i].Size; j++ {
			var sum Dual
			for _, activation := range n.State[i] {
				sum = Add(sum, Mul(activation, layer[w].Weight))
				w++
			}
			n.State[i+1][j] = Add(sum, n.Biases[i][j].Weight)
		}
		meta[i].Transform(n.State[i+1])
	}
}

type TrainingData struct {
	Inputs, Outputs []float32
	Input, Output   int
}

func (n *NetState) QuadraticCost(item TrainingData) float64 {
	cost := 0.0
	for _, layer := range n.Layers {
		for j := range layer {
			layer[j].Weight.Der = 1.0
			n.Inference()
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
				sub := Sub(n.State[len(n.State)-1][k], right)
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
			n.Inference()
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
				sub := Sub(n.State[len(n.State)-1][k], right)
				sum = Add(sum, Mul(sub, sub))
			}
			sum = Mul(Half, sum)
			bias[j].Weight.Der = 0.0
			bias[j].Gradient = sum.Der
			cost = float64(sum.Val)
		}
	}
	return cost
}

func (n *NetState) CrossEntropyCost(item TrainingData) float64 {
	cost := 0.0
	for _, layer := range n.Layers {
		for j := range layer {
			layer[j].Weight.Der = 1.0
			n.Inference()
			LogTransform(n.State[len(n.State)-1])
			NegTransform(n.State[len(n.State)-1])
			loss := n.State[len(n.State)-1][item.Output]
			layer[j].Weight.Der = 0.0
			layer[j].Gradient = loss.Der
			cost = float64(loss.Val)
		}
	}
	for _, bias := range n.Biases {
		for j := range bias {
			bias[j].Weight.Der = 1.0
			n.Inference()
			LogTransform(n.State[len(n.State)-1])
			NegTransform(n.State[len(n.State)-1])
			loss := n.State[len(n.State)-1][item.Output]
			bias[j].Weight.Der = 0.0
			bias[j].Gradient = loss.Der
			cost = float64(loss.Val)
		}
	}
	return cost
}

func (n *Network) Train(data []TrainingData, verbose bool, target float64, alpha, eta, threshold float32) []float64 {
	size := len(data)
	epochs, state, randomized := make([]float64, 0, 8), n.NewNetState(), make([]TrainingData, size)
	copy(randomized, data)
	for {
		for i, sample := range randomized {
			j := i + rand.Intn(size-i)
			randomized[i], randomized[j] = randomized[j], sample
		}

		total := 0.0
		for _, item := range randomized {
			for j, input := range item.Inputs {
				state.State[0][j].Val = input
			}
			if len(item.Outputs) == 0 {
				total += state.CrossEntropyCost(item)
			} else {
				total += state.QuadraticCost(item)
			}
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
			if shares := n.Shares; shares != nil {
				for j := range shares {
					shares[j] = Share{}
				}
				for _, layer := range n.Layers {
					for j := range layer {
						tag := layer[j].Tag
						value, share := layer[j].Gradient, shares[tag]
						share.Sum += value
						share.Count++
						shares[tag] = share
					}
				}
				for _, bias := range n.Biases {
					for j := range bias {
						tag := bias[j].Tag
						value, share := bias[j].Gradient, shares[tag]
						share.Sum += value
						share.Count++
						shares[tag] = share
					}
				}
				for _, layer := range n.Layers {
					for j := range layer {
						tag := layer[j].Tag
						share := shares[tag]
						layer[j].Gradient = share.Sum / float32(share.Count)
					}
				}
				for _, bias := range n.Biases {
					for j := range bias {
						tag := bias[j].Tag
						share := shares[tag]
						bias[j].Gradient = share.Sum / float32(share.Count)
					}
				}
			}
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
		epochs = append(epochs, total)
		if shares, tags := n.Shares, n.Tags; shares != nil {
			next := make(map[int]int, tags)
			for i := 0; i < tags; i++ {
				share := shares[i]
				if share.Count < 2 {
					continue
				}
				next[i] = rand.Intn(n.Tags)
				n.Tags++
			}
			for _, layer := range n.Layers {
				for j := range layer {
					tag, ok := next[layer[j].Tag]
					if !ok {
						continue
					}
					if rand.Intn(2) == 0 {
						layer[j].Tag = tag
					}
				}
			}
			for _, bias := range n.Biases {
				for j := range bias {
					tag, ok := next[bias[j].Tag]
					if !ok {
						continue
					}
					if rand.Intn(2) == 0 {
						bias[j].Tag = tag
					}
				}
			}
			if verbose {
				fmt.Println(len(epochs), total, tags)
			}
		} else if verbose {
			fmt.Println(len(epochs), total)
		}
		if total < target {
			break
		}
	}

	return epochs
}
