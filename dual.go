// Copyright 2019 The BeeLine Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
)

const Debug = false

var (
	Half = Dual{Val: 0.5}
	One  = Dual{Val: 1.0}
)

type Expr struct {
	Name     string
	Val, Der *float32
	Exprs    []*Expr
}

func (e *Expr) String() string {
	switch len(e.Exprs) {
	case 0:
		return fmt.Sprintf("%s(%f,%f)", e.Name, *e.Val, *e.Der)
	case 1:
		return fmt.Sprintf("%s(%s)", e.Name, e.Exprs[0].String())
	case 2:
		if e.Exprs[0] == nil {
			return e.Exprs[1].String()
		}
		return fmt.Sprintf("(%s %s %s)", e.Exprs[0].String(), e.Name, e.Exprs[1].String())
	}
	return ""
}

type Dual struct {
	Expr     *Expr
	Val, Der float32
}

func Check(d Dual) {
	if math.IsNaN(float64(d.Val)) {
		panic(fmt.Errorf("Val %s isNaN", d.Expr))
	} else if math.IsInf(float64(d.Val), 0) {
		panic(fmt.Errorf("Val %s isInf", d.Expr))
	}
	if math.IsNaN(float64(d.Der)) {
		panic(fmt.Errorf("Der %s isNaN", d.Expr))
	} else if math.IsInf(float64(d.Der), 0) {
		panic(fmt.Errorf("Der %s isInf", d.Expr))
	}

}

func (d *Dual) Clip() {
	if der := float64(d.Der); math.IsNaN(der) || math.IsInf(der, 0) {
		d.Der = 0
	} else if der > 256 || der < -256 {
		d.Der /= 256
	}
}

func Add(u, v Dual) Dual {
	if Debug {
		Check(u)
		Check(v)
	}
	value := Dual{
		Val: u.Val + v.Val,
		Der: u.Der + v.Der,
	}
	value.Clip()
	if Debug {
		value.Expr = &Expr{
			Name:  "+",
			Exprs: []*Expr{u.Expr, v.Expr},
		}
	}
	return value
}

func Sub(u, v Dual) Dual {
	if Debug {
		Check(u)
		Check(v)
	}
	value := Dual{
		Val: u.Val - v.Val,
		Der: u.Der - v.Der,
	}
	value.Clip()
	if Debug {
		value.Expr = &Expr{
			Name:  "-",
			Exprs: []*Expr{u.Expr, v.Expr},
		}
	}
	return value
}

func Mul(u, v Dual) Dual {
	if Debug {
		Check(u)
		Check(v)
	}
	value := Dual{
		Val: u.Val * v.Val,
		Der: u.Der*v.Val + u.Val*v.Der,
	}
	value.Clip()
	if Debug {
		value.Expr = &Expr{
			Name:  "*",
			Exprs: []*Expr{u.Expr, v.Expr},
		}
	}
	return value
}

func Div(u, v Dual) Dual {
	if Debug {
		Check(u)
		Check(v)
		if v.Val == 0 {
			panic(fmt.Errorf("Val %s isZero", v.Expr))
		}
		if v.Val*v.Val == 0 {
			panic(fmt.Errorf("Val*Val %s isZero", v.Expr))
		}
	}
	value := Dual{
		Val: u.Val / v.Val,
		Der: (u.Der*v.Val - u.Val*v.Der) / (v.Val * v.Val),
	}
	value.Clip()
	if Debug {
		value.Expr = &Expr{
			Name:  "/",
			Exprs: []*Expr{u.Expr, v.Expr},
		}
	}
	return value
}

func Sin(d Dual) Dual {
	if Debug {
		Check(d)
	}
	value := Dual{
		Val: float32(math.Sin(float64(d.Val))),
		Der: d.Der * float32(math.Cos(float64(d.Val))),
	}
	value.Clip()
	if Debug {
		value.Expr = &Expr{
			Name:  "sin",
			Exprs: []*Expr{d.Expr},
		}
	}
	return value
}

func Cos(d Dual) Dual {
	if Debug {
		Check(d)
	}
	value := Dual{
		Val: float32(math.Cos(float64(d.Val))),
		Der: -d.Der * float32(math.Sin(float64(d.Val))),
	}
	value.Clip()
	if Debug {
		value.Expr = &Expr{
			Name:  "cos",
			Exprs: []*Expr{d.Expr},
		}
	}
	return value
}

func Exp(d Dual) Dual {
	if Debug {
		Check(d)
	}
	exp := float32(math.Exp(float64(d.Val)))
	value := Dual{
		Val: exp,
		Der: d.Der * exp,
	}
	value.Clip()
	if Debug {
		value.Expr = &Expr{
			Name:  "exp",
			Exprs: []*Expr{d.Expr},
		}
	}
	return value
}

func Sigmoid(d Dual) Dual {
	if Debug {
		Check(d)
	}
	e := Exp(d)
	value := Div(e, Add(e, One))
	if Debug {
		value.Expr = &Expr{
			Name:  "sigmoid",
			Exprs: []*Expr{d.Expr},
		}
	}
	return value
}

func Log(d Dual) Dual {
	if Debug {
		Check(d)
	}
	value := Dual{
		Val: float32(math.Log(float64(d.Val))),
		Der: d.Der / d.Val,
	}
	value.Clip()
	if Debug {
		value.Expr = &Expr{
			Name:  "log",
			Exprs: []*Expr{d.Expr},
		}
	}
	return value
}

func Abs(d Dual) Dual {
	if Debug {
		Check(d)
	}
	var sign float32
	val := float32(math.Abs(float64(d.Val)))
	if d.Val != 0.0 {
		sign = d.Val / val
	}
	value := Dual{
		Val: val,
		Der: d.Der * sign,
	}
	value.Clip()
	if Debug {
		value.Expr = &Expr{
			Name:  "abs",
			Exprs: []*Expr{d.Expr},
		}
	}
	return value
}

func Pow(d Dual, p float32) Dual {
	if Debug {
		Check(d)
	}
	value := Dual{
		Val: float32(math.Pow(float64(d.Val), float64(p))),
		Der: p * d.Der * float32(math.Pow(float64(d.Val), float64(p-1.0))),
	}
	value.Clip()
	if Debug {
		right := &Expr{Name: fmt.Sprintf("%f", p)}
		value.Expr = &Expr{
			Name:  "^",
			Exprs: []*Expr{d.Expr, right},
		}
	}
	return value
}